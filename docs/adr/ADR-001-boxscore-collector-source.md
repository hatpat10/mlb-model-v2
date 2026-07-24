# ADR-001: Direct MLB Stats API boxscore endpoint for per-game player stats

Date: 2026-07-14
Status: Accepted
Decision Owner: Patrick Shirley
Related ADRs: none

## Context

V2 is porting player props from the frozen V1 project (`D:/mlb_model`) into V2's structure,
per the migration plan in `docs/AUDIT_2026-07-13.md`. Prop markets (hits, HR, total bases,
RBI, runs, SB, batter/pitcher strikeouts) need **per-game** counting stats for both batters
and pitchers. Neither exists in `data/raw/` today:

- `batter_stats_*.csv` (`R/10_collect_batter_stats.R`) is season-grain only — that script's own
  header already rejected `fg_batter_game_logs()` as a per-game source because FanGraphs only
  exposes it per-player-per-season, which would mean 1,500+ individual API calls with no
  per-game incremental caching.
- `pitcher_gamelogs_*.csv` (`R/11_collect_pitcher_gamelogs.R`) is per-game but FanGraphs
  *rate* stats (ERA, FIP, K%) — no raw strikeout or batters-faced count, which the
  pitcher-strikeouts prop market needs directly.

Per `CLAUDE.md`, `baseballr` is the primary ingestion layer for this project, and any
deviation must be documented (precedent: `R/04_collect_statcast_team.R`).

## Problem Statement

Which data source and call pattern should `R/14_collect_boxscores.R` use to fetch per-game
batting and pitching counting stats for every player, at a call volume that supports both a
one-time multi-season backfill and a daily incremental run?

## Decision

Call the MLB Stats API boxscore endpoint directly —
`https://statsapi.mlb.com/api/v1/game/{game_pk}/boxscore` via `jsonlite::fromJSON()` — one
request per `game_pk`, returning every player's full batting and pitching line for that game
in a single response. Cache incrementally to `boxscore_cache_%d.csv` (same pattern as
`R/12_collect_lineups.R`: flush every 100 games, `normalize_types()` to character before
`bind_rows()`, skip already-cached `game_pk`s on rerun). Split the cache into two final
outputs per season: `batter_gamelogs_%d.csv` (rows with `plateAppearances > 0`) and
`pitcher_boxscore_%d.csv` (rows with `battersFaced > 0`), both keyed by MLBAM person id.

## Alternatives Considered

1. **`baseballr::fg_batter_game_logs(playerid, year)`** — per-player-per-season FanGraphs
   call. Already rejected in `R/10_collect_batter_stats.R` for the batting side; the same
   objection applies to any pitching equivalent.
2. **`baseballr::mlb_player_game_stats(person_id, game_pk)`** — correct data and correct
   (MLBAM) id space, but one call per player-*game* — roughly 40,000+ calls/season.
3. **ESPN via `espn_mlb_game_player_box` / `espn_mlb_player_gamelog`** — keyed by ESPN player
   ids, not MLBAM, which would force name-based matching — the exact problem
   `R/08_collect_player_ids.R` (the Chadwick crosswalk) exists to eliminate.
4. **A baseballr boxscore wrapper** — does not exist in the installed baseballr 2.0.0
   (confirmed by enumerating `ls("package:baseballr")`; no `mlb_boxscore*` function).

## Pros

- One API call returns the full-game batting *and* pitching lines for all ~40 players in a
  game — the pitcher-strikeouts data gap (raw K, batters-faced) closes in the same fetch that
  builds the batter props data, no separate collector or second backfill needed.
- Native MLBAM person ids — joins `player_id_crosswalk.csv`, `lineups_%d.csv`, and
  `batter_stats_%d.csv` directly, no name matching.
- Call volume matches an already-accepted cost profile: ~2,430 calls/season one-time backfill,
  then ~15/day in-season — identical to the `mlb_batting_orders()` pull in
  `R/12_collect_lineups.R`.
- Reuses the incremental-cache pattern from `R/12` verbatim, so no new caching idiom enters
  the codebase.

## Cons

- Bypasses `baseballr`, the project's stated primary ingestion layer — a second collector
  (after `R/04`) that hits a raw endpoint instead.
- Undocumented/unversioned public endpoint: MLB Stats API has no formal external contract:
  field names or response shape could change without notice, unlike a maintained R package
  that would absorb an upstream break itself.
- One JSON response bundles two different "concerns" (batting + pitching), stretching the
  collector's single-concern convention slightly — mitigated by writing them to two separate
  output files, but the fetch/cache layer is still shared.

## Tradeoffs

Accepting a raw, unversioned HTTP dependency and a documented exception to the baseballr-first
rule, in exchange for closing both the batter-per-game gap and the pitcher-strikeouts gap in
a single collector at a call volume the project has already validated as acceptable (R/12's
precedent). The alternative that stays fully inside baseballr (`mlb_player_game_stats`) would
cost ~15-20x more API calls for the same data.

## Impact

- New file: `R/14_collect_boxscores.R`.
- New raw outputs: `data/raw/batter_gamelogs_%d.csv` and `data/raw/pitcher_boxscore_%d.csv`
  (PLAYER x game), `data/raw/team_boxscore_%d.csv` (TEAM x game, summed from the two player
  files — added 2026-07-14 so a downstream join can never silently mix player-game and
  team-game grain; it also doubles as a reconciliation check against the independently-sourced
  `game_logs_%d.csv` from `R/01`, logging a warning if summed runs don't match exactly), and
  `data/raw/boxscore_cache_%d.csv` (intermediate, not a final feature input).
- `R/00_run_all.R`: registered as step 14, all three final outputs added to `OUTPUT_GLOBS`,
  enhancement-script count updated to 08-14.
- No existing collector, `features/builder.py`, or training/prediction script is modified —
  this is additive, laying groundwork for the future `features/props_builder.py`.

## Implementation Notes

- Verified live against one real boxscore response (game_pk 745444, 2024) before writing the
  transmute logic, confirming field names (`totalBases`, `battersFaced`, `battingOrder`,
  `gamesStarted`, etc.) match what's used below.
- `battingOrder` is kept as the raw statsapi-encoded integer (e.g. 100 = leadoff starter, 401 =
  substitute batting in the 4-slot) rather than normalized, so downstream feature code can
  distinguish starters from pinch-hitters without a separate lineups join.
- Backfilled 2021-2026 and spot-validated the 2021 cache: 21.1 batting rows/game, 8.7 pitching
  rows/game, exactly 2.0 starters/game, zero missing MLBAM ids, zero `totalBases` values
  inconsistent with `H + 2B + 2*3B + 3*HR` across 10,555 sampled batting rows, zero unmapped
  team names via `normalize_team()`.
- **Known upstream data quirk (found 2026-07-14 via the `team_boxscore_%d.csv` reconciliation
  check against `game_logs_%d.csv`):** in ~3-4.5% of team-games, individual batters' summed
  `runs` undercounts the official team score by exactly 1. Traced one example (game_pk 716354,
  2023) to rule out both plausible collector bugs — not caused by the `pa > 0` filter (every
  batter row had `pa >= 1`) and not extra innings (regulation 9 innings, linescore matched
  `game_logs` exactly). Own-pitching runs-allowed (`p_r`) matches `game_logs` almost exactly
  (>99.9%), so it — and `game_logs` itself — is the authoritative team-run source; an individual
  batter's own `r` stays correct for that player, but team-level batting-side sums can be off by
  one run in a small share of games. Logged as a non-fatal NOTE (not WARNING) per run.

## Future Considerations

Revisit if: the MLB Stats API endpoint shape changes and breaks the parser (watch for
`flatten_boxscore()` returning `NULL` or NA-heavy rows in `00_run_all.R`'s summary row counts);
a future baseballr release adds a boxscore wrapper (re-evaluate whether switching is worth the
churn once this collector is load-bearing for the props model); or call volume changes
materially (e.g. adding minor-league or doubleheader granularity) such that the ~2,430
calls/season backfill assumption no longer holds. If the batting-runs undercount rate above
ever needs a root cause (e.g. because a props feature starts summing batter `r` per team), the
next investigative step is pulling the MLB Stats API's play-by-play endpoint for a handful of
mismatched `game_pk`s to identify the specific scoring-play type the boxscore endpoint fails to
attribute.
