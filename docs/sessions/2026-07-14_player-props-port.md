# Session log — 2026-07-14 — Player-props port (V1 → V2), start to finish

Full record of a single working session porting player props from the frozen V1
project into V2's structure, per the migration plan in `docs/AUDIT_2026-07-13.md`.
Captures both what was asked and what was delivered, in order, plus every model
trained and every artifact created. Saved so the conversation itself can be
cleared without losing the trail.

---

## 1. Boxscore collector design (`R/14_collect_boxscores.R`)

**Ask:** Design a per-batter per-game collector — the foundation everything
downstream (props features, labels) needs. Verified facts going in: season-grain
`batter_stats_*.csv` isn't enough for props; `pitcher_gamelogs` has FanGraphs
*rate* stats but no raw K count; V1's platoon-split join logic is leakage-safe
and portable as-is; V1's H2H logic is a confirmed lookahead leak that must be
rebuilt as-of-date. Instructed to propose before coding — baseballr function(s),
schema/grain, crosswalk tie-in, market coverage — and wait for sign-off.

**What was proposed and decided:** No baseballr function covers per-game
boxscore data at a tractable call volume (`fg_batter_game_logs` is one call per
player-*season*, already rejected for the same reason in `R/10`;
`mlb_player_game_stats` is one call per player-*game*, ~40k calls/season).
Recommendation: call the MLB Stats API boxscore endpoint directly (a documented
exception to the baseballr-first rule, precedented by `R/04`) — one call per
`game_pk` returns every player's full batting *and* pitching line in one shot,
~2,430 calls/season backfilled once, ~15/day in-season after. Proposed two
outputs (`batter_gamelogs_%d.csv`, `pitcher_boxscore_%d.csv`), both keyed on
MLBAM id (joins the crosswalk with no name matching), which also closes the
pitcher-strikeouts data gap for free since the same fetch carries raw K counts.

**User:** *"Signed off — two files, name it 14_collect_boxscores.R, implement."*

**Delivered:** `R/14_collect_boxscores.R` — direct MLB Stats API boxscore
fetch, incremental cache (same pattern as `R/12`), grain is one row per
`(game_pk, player)`. Verified live against one real boxscore response before
writing the transmute logic. Backfilled 2021–2026 in the background while other
work continued.

---

## 2. Validation, ADR, and the team-file request

Spot-validated the in-progress cache at 500 games: zero missing MLBAM ids, zero
`total_bases` inconsistencies, zero unmapped teams, correct starter counts.

**User (mid-backfill):** *"please make an ADR"* (gave the exact template).
Wrote `docs/adr/ADR-001-boxscore-collector-source.md` documenting the source
decision, four alternatives ruled out, pros/cons/tradeoffs, and the live
verification.

Backfill completed cleanly for all six seasons (2021–2026): 275,613 batter-game
rows, 116,803 pitcher-game rows total.

**User:** *"I want to make sure the data doesnt get messed up please make
seperate files one for players x stats and one for teams x stats"* — then, when
asked to clarify what "team stats" should contain, said the recommended
answer (a team-game rollup summed from the boxscore, doubling as a
reconciliation check against the independently-collected `game_logs`) was
*"an example"* — confirming direction without treating it as a literal spec.

**Delivered:** Extended `R/14` to write a third output, `team_boxscore_%d.csv`
(team × game, summed from the two player files), with a built-in reconciliation
check against `game_logs_%d.csv`'s independently-sourced `runs_scored`/
`runs_allowed`. Registered in `R/00_run_all.R`.

**A real mistake, caught and explained:** Edited the R script on disk while the
original backfill process still had it open for reading — `Rscript` parses
top-level expressions incrementally rather than buffering the whole file
upfront, so the edit shifted byte offsets underneath the running process and it
crashed with a syntax error right at the very end. No data was lost (every
year's player-grain files had already written successfully before the crash),
but it's a real footgun worth remembering: **never edit a script while a
background run against that same file is still in flight.**

Reran cleanly as an isolated process. All six `team_boxscore_%d.csv` files
wrote successfully — and the reconciliation check immediately surfaced
something real: **3.0–4.6% of team-games** showed a 1-run mismatch between
summed individual batting stats and the official score.

**Root-cause investigation (not hand-waved):** Traced one concrete example
(game_pk 716354, 2023, WSH @ ATL). Ruled out the two plausible collector bugs
with direct evidence — not the `pa > 0` filter (every batter row had `pa ≥ 1`)
and not extra innings (regulation 9 innings, linescore matched `game_logs`
exactly via `mlb_game_linescore()`). Conclusion: a genuine MLB Stats API quirk
— in a small share of games, one run isn't attributed to any individual
batter's `runs` counting stat, even though it's correct in the official score
and in the opposing pitcher's runs-allowed (which matched >99.9% of the time).
Documented in code comments and the ADR; `game_logs`/pitching-side runs-allowed
remain the authoritative team-run source.

Ran a full six-year structural validation pass afterward: zero duplicate keys,
zero NA in key columns, zero unmapped teams, zero `total_bases`/`ab`/`pa`
violations, zero impossible pitching stats, across all 275,613 + 116,803 rows.
One minor, expected finding: 16 batters / 17 pitchers in the 2026 (current,
in-progress) season not yet in `player_id_crosswalk.csv` — almost certainly
recent rookie debuts the crosswalk snapshot hasn't picked up yet.

**User:** *"okay now please give me a detailed outline of all the stats we
have"* — produced a full inventory of all 14 R collectors' outputs, organized
by team-level / pitcher-level / batter-level / identity-bio-context / odds,
with exact columns and sources for each. (Not reproduced here — see that
inventory or just re-read `data/raw/` + the R collectors directly; it reflects
live state better than a frozen copy would.)

---

## 3. Planning the props feature/model port

**User (voice-dictated, lightly garbled):** asked for the boxscore data to be
wired into `features/props_builder.py`, and for a structured plan going
forward before continuing — "if it looks good I will want you to continue to
build out whatever you think is good."

**Plan presented** (grounded by first reading `features/builder.py` and
`scripts/01_build_features.py` in full, to mirror the actual existing
leakage-safe patterns rather than described-from-memory ones):

- **Phase 0**: `features/props_builder.py`, batter markets, hits first as a
  vertical slice — rolling form (`.shift(1)` + rolling mean, mirroring
  `build_rolling_features`), prior-year season stats (`join_year = year+1`,
  mirroring `join_team_batting`), opposing-pitcher stats + rolling form
  (mirroring `join_sp_stats`/`join_pitcher_rolling_form`), park/umpire reuse.
- **Phase 2**: same pattern for pitcher strikeouts, against `pitcher_boxscore`.
- **Three flagged open questions, explicitly not decided unilaterally**:
  1. **Platoon splits** — V1's join logic is leakage-safe, but the underlying
     split-stat data doesn't exist anywhere in V2 yet. Needs a new source
     before it can be ported.
  2. **True batter-vs-pitcher H2H** — box scores only show who *played*, not
     who faced whom at the plate; needs play-by-play-level data
     (`statcast_search_batters` is a candidate). Bigger lift, low sample size
     per matchup anyway; recommended deferring.
  3. **Prop betting lines/odds** — nothing collects these yet. Odds API quota
     is small, prop markets have far more line-combinations than moneyline,
     and historical retroactive prop odds for backtesting almost certainly
     aren't available on the current tier. Recommended validating the raw-stat
     model first (zero API cost) before spending any odds credits on this.
- Naming: `props_` prefix instead of renumbering the `scripts/01–07` sequence.

**User:** *"sounds good!"*

---

## 4. Phase 0 — batter feature builder

Built `features/props_builder.py` (batter section) and
`scripts/props_build_features.py`. Ran end-to-end: 275,613 rows × 79 columns.

**A real calibration decision, not just execution:** the first run excluded
season-level batter quality features (wRC+, wOBA, ISO — arguably the most
standard, valuable batter-quality signal) at 82.4% coverage against a
moneyline-inherited 90% threshold. Checked `scripts/02_train.py`'s actual
imputation convention first (train-only-median imputation for anything that
*passes* `coverage_check`) rather than guessing, confirmed that 82.4% is real,
expected rookie/call-up turnover — not a data problem — and lowered the props
coverage bar to 75%. Reran: 56/56 candidate features passed.

**Leakage verified concretely, not just asserted:** pulled a real batter's row
(Albert Pujols, 2021-06-09), hand-recomputed `roll_h_7`/`roll_h_20` from only
the strictly-prior games in Python, confirmed exact match to the stored
values, and confirmed the stored value *differs* from what a leaky
(today-included) version would produce.

Full project smoke test (`.claude/skills/run-mlb-model-v2/smoke.py`) passed
throughout — moneyline pipeline never touched or broken.

**User:** *"sounds good"* → continue without stopping except at the three
flagged questions.

---

## 5. Phase 2 — pitcher feature builder

Extended `features/props_builder.py` with a pitcher section: rolling form
directly from `pitcher_boxscore` (keyed on `pitcher_mlbam_id` — no FanGraphs
name-matching needed at all, unlike the moneyline pitcher-rolling join),
season stats, bio, opposing-team batting context, and — the most direct
strikeouts signal — the actual opposing lineup's average wRC+/K% that game
(`join_opposing_lineup_quality`, with a team-average fallback for
prediction-time unknown lineups). Renamed the batter-side constants
(`LABEL_COLS` → `BATTER_LABEL_COLS`, etc.) since the module now serves two
entities.

Built `scripts/props_build_pitcher_features.py`. Ran: 27,181 starter-rows × 60
columns, 38/42 features passed coverage (the 4 that correctly failed — Statcast
spin rate/extension/xFIP/barrel% against, at 58.2% — reflect `R/05`'s own
50-IP-per-season scoping, a real limitation, not a bug; left excluded rather
than force-included).

Same concrete leakage check repeated for pitchers (Adam Wainwright,
2022-05-26) — exact match confirmed against hand-computed strictly-prior
values. Confirmed relief appearances were fully excluded from the rolling
grain (`is_starter == 1` only in the output).

---

## 6. Accidental detour

**User:** *"claude rc"* — ambiguous, asked for clarification rather than
guessing.

**User (local commands):** ran `/plugin install mcp-server-dev` and
`/mcp-server-dev:build-mcp-server`, loading that skill's full discovery
questions. Flagged this as a likely accidental context switch (unrelated to
the MLB work) rather than launching into it unprompted.

**User:** *"no keep going I should have used desktop app"* — confirmed
accidental; resumed the MLB props work.

---

## 7. Phase 3 — model training

Read `scripts/02_train.py` and `scripts/model_classes.py` in full before
writing anything, to ground the design rather than assume.

**A design decision made and flagged, not silently substituted:** moneyline is
binary classification (one fixed threshold — home team wins or not). Props
markets are counts, and no real sportsbook O/U line exists yet (open question
#3, above). Forcing an arbitrary threshold (e.g. "hits ≥ 1") to mirror the
classifier pattern would mean training against an invented line — worse, and
less reusable later, than predicting the expected count directly. Built
**Poisson regression** instead (XGB/LGBM, well-suited to non-negative skewed
counts): once real lines exist, a predicted mean converts to P(over any line)
via the Poisson CDF for any player, which a fixed-threshold classifier
couldn't do. Kept everything else mirrored: same `TRAIN_YEARS`/`TEST_YEAR`
split from `config.py`, same Optuna + `TimeSeriesSplit` tuning discipline, same
train-median imputation.

Built `scripts/props_train.py` — generic, `--matrix {batter,pitcher} --market
<label_col>` — plus a naive-baseline comparison (does the tuned model actually
beat just using the player's own rolling average?) that moneyline's evaluation
doesn't need but a props model does, since that's the real bar for whether the
modeling effort adds value.

**A real bug, hit and fixed:** first training run crashed —
`TypeError: Cannot perform reduction 'median' with string dtype`. The
handedness columns (`batter_bats`, `opp_sp_throws`, `pitcher_throws`) were raw
strings (`'L'`/`'R'`/`'S'`) sitting in the numeric feature list. Fixed with
fixed-category one-hot encoding (`batter_bats_L/R/S`,
`opp_sp_throws_L/R`, `pitcher_throws_L/R`) — explicitly *not*
`pd.get_dummies`, which would produce inconsistent columns between train/test
slices if a category happened to be absent from one split. Rebuilt both
feature matrices, reran the smoke test, retrained.

### Trained models

| Market | Grain | Model | Holdout year | RMSE (model) | RMSE (naive baseline) | Correlation (model) | Correlation (baseline) | Beats baseline? |
|---|---|---|---|---|---|---|---|---|
| Batter hits (`h`) | batter × game | avg(XGB, LGBM), Poisson objective | 2024 | 0.8440 | 0.8644 (`roll_h_20`) | 0.196 | 0.141 | Yes |
| Pitcher strikeouts (`k`) | pitcher × game (starts only) | avg(XGB, LGBM), Poisson objective | 2024 | 2.2426 | 2.3579 (`roll_k_10`) | 0.425 | 0.354 | Yes |

Hits shows a much weaker edge than strikeouts — expected, not a red flag: an
individual batter's hit total in one game is dominated by high-variance,
low-sample-size noise (4–5 at-bats), while a starting pitcher's strikeout total
draws on a much larger, more stable underlying skill signal (25+ batters
faced). Both are genuine, measurable improvements over using the player's own
recent rolling average directly — the real bar, not just "did training
complete."

Trained hyperparameters (Optuna, 60 trials, `TimeSeriesSplit(n_splits=3)`,
Poisson deviance objective):

- **Batter hits** — XGB: `max_depth=6, learning_rate=0.0214, n_estimators=366,
  subsample=0.603, colsample_bytree=0.567, min_child_weight=2, reg_alpha=0.0009,
  reg_lambda=0.0079` (CV Poisson deviance 1.0306). LGBM: `max_depth=8,
  learning_rate=0.0995, n_estimators=192, subsample=0.707,
  colsample_bytree=0.530, min_child_samples=12, reg_alpha=4.464,
  reg_lambda=0.0625, num_leaves=12` (CV Poisson deviance 1.0309).
- **Pitcher strikeouts** — XGB: `max_depth=3, learning_rate=0.0392,
  n_estimators=377, subsample=0.631, colsample_bytree=0.758,
  min_child_weight=9, reg_alpha=0.616, reg_lambda=0.261` (CV Poisson deviance
  1.1817). LGBM: `max_depth=5, learning_rate=0.0415, n_estimators=286,
  subsample=0.968, colsample_bytree=0.574, min_child_samples=29,
  reg_alpha=0.00037, reg_lambda=0.00148, num_leaves=12` (CV Poisson deviance
  1.1824).

---

## 8. Artifacts inventory (as of end of session)

**R collector:**
- `R/14_collect_boxscores.R` (new) — writes `batter_gamelogs_%d.csv`,
  `pitcher_boxscore_%d.csv`, `team_boxscore_%d.csv`, `boxscore_cache_%d.csv`
  for 2021–2026.
- `R/00_run_all.R` (modified) — registered as step 14.

**Docs:**
- `docs/adr/ADR-001-boxscore-collector-source.md` — the source decision,
  updated twice more (team-file addition, run-attribution-gap finding).
- `docs/sessions/2026-07-14_player-props-port.md` — this file.

**Feature modules:**
- `features/props_builder.py` (new) — batter + pitcher feature construction,
  single source of truth for both, mirroring `features/builder.py`'s patterns.

**Orchestrator scripts:**
- `scripts/props_build_features.py` — batter feature matrix →
  `data/processed/props_feature_matrix.csv` (275,613 × 84 after the
  handedness fix) + `props_feature_names.json`.
- `scripts/props_build_pitcher_features.py` — pitcher feature matrix →
  `data/processed/pitcher_props_feature_matrix.csv` (27,181 × 62) +
  `pitcher_props_feature_names.json`.
- `scripts/props_train.py` — generic Poisson-regression trainer →
  `models/props/{market}_xgb.joblib`, `{market}_lgbm.joblib`,
  `{market}_medians.joblib`, `{market}_feature_names.json`, and
  `outputs/props_train_report_{matrix}_{market}.json`.

**Trained model files on disk:**
- `models/props/h_xgb.joblib`, `models/props/h_lgbm.joblib`,
  `models/props/h_medians.joblib`, `models/props/h_feature_names.json`
- `models/props/k_xgb.joblib`, `models/props/k_lgbm.joblib`,
  `models/props/k_medians.joblib`, `models/props/k_feature_names.json`
- `outputs/props_train_report_batter_h.json`,
  `outputs/props_train_report_pitcher_k.json`

Moneyline pipeline (`scripts/01`–`07`, `features/builder.py`, `models/*`)
untouched throughout — full smoke test passed after every change.

---

## 9. Where this leaves things

**Done:** the whole raw-data-to-trained-model path for two vertical-slice
markets (batter hits, pitcher strikeouts), fully leakage-verified, beating
naive baselines on a real holdout.

**Mechanical, no new design needed:** the other 6 batter markets (HR, TB,
RBI, runs, walks, SB) and remaining pitcher markets (walks, hits/runs
allowed) — same feature matrices already contain those label columns, just
run `props_train.py` with a different `--market`.

**Still open, needs a decision before touching:**
1. Platoon-split data source (batter vs. LHP/RHP season stats) — doesn't
   exist in V2 yet.
2. True batter-vs-pitcher H2H — needs play-by-play data, bigger lift, low
   per-matchup sample size.
3. Prop-odds/lines collection — needed before real edge-detection/ROI
   backtesting is possible at all; budget and historical-availability
   constraints flagged, not yet decided.
