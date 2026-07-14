# CLAUDE.md — MLB Model V2

Operating guide for Claude Code in this repo. Read this before making changes.

## What this project is

A daily **MLB moneyline betting model**. Pipeline:

```
baseballr (R) → data/raw/*.csv → features/builder.py → data/processed/feature_matrix.csv
  → calibrated XGB/LGBM ensemble → daily predictions with de-vigged edge detection
  → quarter-Kelly bet logging with closing-line-value (CLV) tracking
```

There is **no GUI and no server** — the "app" is a chain of numbered scripts driven by two
Windows Task Scheduler jobs. Scope today is **moneyline only** (player props, totals/run-line,
and a dashboard are planned; see Roadmap).

## Two-project context (important)

There are two folders on disk. **This one (`D:/mlb_model_v2`) is the chosen backbone.**

- **`D:/mlb_model_v2` (V2, this repo)** — the project to build on. baseballr-based R
  ingestion + clean layered Python pipeline. Smaller but well-engineered, leakage-aware,
  reproducible. All new work happens here.
- **`D:/mlb_model` (V1)** — **FROZEN, read-only parts donor.** More built out (player props,
  totals/run-line, line-movement, a Streamlit dashboard) but architecturally tangled: a nested
  duplicate `MLB/` project, four copies of `savant_stats_fetch`, ~20 loose `check_/fix_/debug_`
  scripts. **Never develop in V1.** Only read it to mine logic worth porting *into V2's
  structure*. Do not port the nested `MLB/` project, the savant fetcher duplicates, or root
  debug scripts.

Rationale for choosing V2 lives in the audit that produced this file (see "Continuing in the
terminal" in the chat that created it).

## Hard rules (do not violate)

1. **Preserve raw data.** Never overwrite or delete `data/raw/` snapshots. Collection is
   incremental (`skip_completed_season` in `R/utils.R`) — finished seasons are immutable and
   reused; only the current season re-collects. Keep it that way.
2. **No lookahead bias.** Every feature must be available at prediction time. All rolling
   features use `.shift(1)` so a game never sees its own result. Historical odds/lineups are
   dated snapshots — never backfill a game with data that didn't exist pre-first-pitch.
3. **One feature module.** `features/builder.py` is the single source of truth imported by
   **both** `01_build_features.py` (training) and `04_predict.py` (prediction). Never duplicate
   or fork feature logic between train and predict — that causes train/serve skew.
4. **Match existing conventions** (below). Don't introduce a second config system, a second
   team-normalization map, or a new logging style.
5. **Design before code** for anything nontrivial: propose the file layout + data lineage +
   leakage risks and wait for sign-off before implementing. Challenge assumptions; don't just
   agree.

## Repo layout

```
R/                 13 numbered baseballr collectors, one concern each; 00_run_all.R orchestrates
  utils.R          single source of truth: team normalization, accent stripping, season helpers
features/builder.py  single source of truth for feature construction (train + predict share it)
scripts/
  01_build_features.py  raw CSVs -> data/processed/feature_matrix.csv
  02_train.py           trains 2021-23, isotonic-calibrated on OOF, evals untouched 2024 holdout
  03_backtest.py        holdout metrics + ROI vs real closing lines
  04_predict.py         daily slate: probables, lineups+umpire, de-vigged consensus, edge flags
  05_bankroll.py        quarter-Kelly logging, settlement, CLV, drawdown pause
  06_runner.py          8 AM daily orchestrator
  07_capture_closing_lines.py  captures real closing lines; --pregame-predict re-runs 2h out
  model_classes.py, odds_utils.py, wait_for_odds_and_predict.py
config/config.py   paths, betting params, team maps, league-average imputes
data/raw/          collector output (gitignored, preserved) ; data/processed/  feature matrix
models/            calibrated ensemble + train medians ; outputs/  predictions, bet_log, backtest
docs/              stats catalog ; app/  searchable stat-definitions app
```

## Environment (Windows)

- **Python must use the project venv**: `venv\Scripts\python.exe` (system Python lacks deps).
  Deps in `requirements.txt` (pandas, scikit-learn, xgboost, lightgbm, optuna, loguru, etc.).
- **R 4.3.2** at `C:\Program Files\R\R-4.3.2\bin\Rscript.exe`; packages pinned via `renv.lock`
  (`renv::restore()` to install). `.Rprofile` auto-activates renv.
- **`.env`** (never commit): `ODDS_API_KEY=...` (required; ~500 credits/month — spend
  sparingly), `SPORTSDATAIO_API_KEY=...` (optional).
- Task Scheduler: `MLB2-Morning` 08:00 → `06_runner.py`; `MLB2-CloseCapture` 08:30 →
  `07_capture_closing_lines.py --pregame-predict`.

## Running & testing

**After any change to `scripts/` or `features/builder.py`, run the smoke driver:**

```
venv\Scripts\python.exe .claude\skills\run-mlb-model-v2\smoke.py
```

Fast (~30s, offline): import-loads all pipeline scripts (catches syntax/import errors), scores
the 2024 holdout (asserts AUC > 0.55, Brier < 0.25), runs the backtest end-to-end.
`SMOKE PASSED` + exit 0 = healthy. Flags: `--predict` (full E2E, network, ~2 min),
`--capture-once` (spends 1 Odds API credit). The `run-mlb-model-v2` skill documents direct
invocation and the human-path stage-by-stage commands.

**The smoke driver only tests the Python side.** After editing any `R/` file, parse/behavior-
check it separately with Rscript, e.g.:

```
& "C:\Program Files\R\R-4.3.2\bin\Rscript.exe" -e "source('R/utils.R'); cat('ok\n')"
```

Script filenames start with digits, so `import` by name fails — load via `importlib` (the smoke
driver shows the pattern). `features/builder.py`, `model_classes.py`, and `odds_utils.py` import
normally once repo root + `scripts/` are on `sys.path`.

## Conventions

- **Collectors**: numbered `R/NN_collect_*.R`, one data concern each, all `source("R/utils.R")`,
  write to `data/raw/`, log via `log_msg()`, skip finished seasons via `skip_completed_season()`.
- **Team codes**: normalize everything to the standard 30-team set with `normalize_team()` (R) /
  `TEAM_ABBREV_MAP` in `config/config.py` (Python). Never invent a third map.
- **Paths & params**: from `config/config.py` (Python) — no hardcoded paths. Season ranges are
  derived dynamically (`current_season()`), never hardcoded.
- **Logging**: `loguru` in Python, `log_msg()` in R.
- **baseballr is the primary ingestion layer.** Use a direct API/scrape only when baseballr can't
  provide the data reliably, and document why in a comment (see `04_collect_statcast_team.R`).

## Current state & roadmap

- **Done & running daily**: moneyline pipeline end-to-end (collect → features → calibrated
  ensemble → predict → edge flags → Kelly bet log → CLV).
- **Next (port from frozen V1, one clean module at a time)**:
  1. Player props (V1 `scripts/08,2x_*`) → new `R/` collectors + `features/props_builder.py`
     mirroring the moneyline pattern.
  2. Point the dashboard (V1 `app.py`) at V2 outputs.
  3. Fold V1 line-movement ideas into existing `07_capture_closing_lines.py`.
- **Lightweight governance to adopt** (not heavyweight): a one-page data-source registry, a
  feature/leakage checklist, and a running tech-debt list. Skip per-dataset data dictionaries
  and ADRs unless a collaborator joins.

## Gotchas

- Don't run V1 and V2 in the same Python env — different deps and different `config`.
- `read_csv` type inference on re-read can clash with fresh baseballr responses; collectors
  normalize types before `bind_rows` (see `R/01_collect_gamelogs.R`). Preserve that when editing.
- Odds API quota is small — don't add code paths that call it in loops or on every run.
