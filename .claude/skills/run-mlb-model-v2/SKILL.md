---
name: run-mlb-model-v2
description: Run, smoke-test, or verify the MLB betting model pipeline — launch daily predictions, run the backtest, capture closing lines, retrain, or check that a code change didn't break any script. Use for "run the model", "run the pipeline", "test my change", "smoke test", "predict today", "run the backtest".
---

# Run MLB_MODEL_V2

Daily MLB moneyline pipeline (Python + R, Windows). No GUI, no server — the
"app" is a chain of scripts driven by two Task Scheduler jobs. All paths below
are relative to the repo root (`D:\MLB_MODEL_V2`) and all Python must use the
project venv: `venv\Scripts\python.exe` (system Python lacks the deps).

## Run (agent path) — the smoke driver

```
venv\Scripts\python.exe .claude\skills\run-mlb-model-v2\smoke.py
```

Fast, offline, ~30s: checks env/data/model files, import-loads all 10 pipeline
scripts (catches syntax/import errors anywhere), scores the 2024 holdout
through the calibrated models (asserts AUC > 0.55, Brier < 0.25), and runs
`scripts/03_backtest.py` end-to-end. Exit 0 + `SMOKE PASSED` = healthy.
**Run this after any change to `scripts/` or `features/builder.py`.**

Full E2E (network: MLB Stats API + The Odds API, ~2 min — overwrites
`outputs/predictions_<date>.csv`):

```
venv\Scripts\python.exe .claude\skills\run-mlb-model-v2\smoke.py --predict
```

Closing-line snapshot (spends 1 Odds API credit, ~500/month quota):

```
venv\Scripts\python.exe .claude\skills\run-mlb-model-v2\smoke.py --capture-once
```

## Direct invocation

Script filenames start with digits, so `import` by name fails — load via
importlib (this is also how the smoke driver does it):

```
venv\Scripts\python.exe -c "import importlib.util, sys; sys.path[:0]=['scripts','.']; spec=importlib.util.spec_from_file_location('p','scripts/04_predict.py'); m=importlib.util.module_from_spec(spec); spec.loader.exec_module(m); print(len(m.fetch_schedule_with_probables('2026-07-11')), 'games')"
```

`features/builder.py`, `scripts/model_classes.py`, and `scripts/odds_utils.py`
import normally once repo root + `scripts/` are on `sys.path`.

## Run (human path)

Individual stages, in dependency order (each is standalone):

```
venv\Scripts\python.exe scripts\01_build_features.py        # raw CSVs -> feature_matrix (~2 min)
venv\Scripts\python.exe scripts\02_train.py                 # retrain + 2024 holdout eval (~15 min, Optuna)
venv\Scripts\python.exe scripts\03_backtest.py              # holdout metrics + ROI (~10 s)
venv\Scripts\python.exe scripts\04_predict.py --date 2026-07-11
venv\Scripts\python.exe scripts\05_bankroll.py --settle
```

R data refresh (needs R 4.3 at `C:\Program Files\R\R-4.3.2\bin\Rscript.exe`;
minutes normally, hours on a multi-day backlog): `Rscript R\00_run_all.R`

Production scheduling: Task Scheduler jobs `MLB2-Morning` (08:00, runs
`06_runner.py`) and `MLB2-CloseCapture` (08:30, runs
`07_capture_closing_lines.py --pregame-predict`, sleeps until ~2h before first
pitch). Don't launch `07_capture_closing_lines.py` without `--once` manually
while the task is live — it's a long-running watcher and you'd double-spend
API quota.

## Test

There is no unit-test suite. The smoke driver is the test. For a deeper check
after model-affecting changes, run `03_backtest.py` and compare AUC/Brier in
`outputs/backtest_2024.json` against the committed baseline (AUC≈0.588,
Brier≈0.244).

## Gotchas

- **Unpickling models requires `scripts/` on `sys.path` first** — the
  joblib files reference classes in `scripts/model_classes.py`; loading them
  from anywhere else raises `ModuleNotFoundError: model_classes`.
- **`04_predict.py` silently overwrites `outputs/predictions_<date>.csv`**
  (and `.xlsx`) for the date it's given. Re-running later in the day is by
  design (better lineups/odds), but don't run it casually mid-slate if the
  file has already been used to log bets.
- **Odds API quota is shared and small** (~500 credits/month on the free
  tier): every `04_predict.py` run, capture snapshot, and `wait_for_odds`
  poll spends 1. The scheduled pipeline uses ~13/day already.
- **`predictions_<date>.csv` legitimately has empty odds columns** when run
  before sportsbooks post lines (early morning) or after games start — "no
  market odds available" in game cards is not a bug.
- **`game_pk` dtype flips between int64 and str** depending on which CSV it
  came from; every merge in the codebase casts to `str` first. Do the same
  in any new join or you'll get silent zero-row merges.
- **Console output shows `�` for em-dashes** under Windows cp1252 — cosmetic
  (loguru output is UTF-8). For Task Scheduler logs the tasks set
  `PYTHONUTF8=1` and `LOGURU_COLORIZE=0`.
- **Old V1 pipeline lives at `D:\mlb_model`** with its own (now disabled)
  tasks `MLB-Morning`/`MLB-Settle`/`MLB-CloseOddsSnapshot`. Don't re-enable —
  it shares the Odds API key and keeps a separate conflicting bet log.

## Troubleshooting

- `ModuleNotFoundError: model_classes` when loading `.joblib` →
  `sys.path.insert(0, "scripts")` before `joblib.load` (see Gotchas).
- `Disable/Register-ScheduledTask : Access is denied` → the V1 tasks were
  created elevated; use an admin PowerShell.
- Smoke `venv interpreter FAIL — running under C:\...` → you invoked system
  Python; rerun with `venv\Scripts\python.exe`.
- `03_backtest.py` reports SYNTHETIC ROI → no `data/raw/odds_close_*.csv`
  overlap with the holdout year; real closing lines only accumulate from
  2026-07-11 onward via the capture task.
