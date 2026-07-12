# MLB Model V2

MLB moneyline betting model: R data collection → Python feature build → calibrated
XGB/LGBM ensemble → daily predictions with edge detection against de-vigged market
consensus → quarter-Kelly bet logging with CLV tracking.

## Layout

- `R/` — data collectors (MLB Stats API via baseballr, FanGraphs, Statcast).
  `00_run_all.R` orchestrates; results land in `data/raw/` (gitignored).
- `features/builder.py` — single source of truth for feature construction,
  shared by training and prediction so the two can never diverge.
- `scripts/`
  - `01_build_features.py` — raw CSVs → `data/processed/feature_matrix.csv`
  - `02_train.py` — trains on 2021–2023, isotonic-calibrated on OOF preds,
    evaluated on the untouched 2024 holdout (AUC / Brier / ECE)
  - `03_backtest.py` — holdout metrics + ROI vs real closing lines when
    `data/raw/odds_close_*.csv` exist (synthetic Elo proxy otherwise, clearly labeled)
  - `04_predict.py` — daily slate: probables, posted lineups + home-plate umpire
    (live from MLB boxscore), de-vigged consensus odds, edge flags
  - `05_bankroll.py` — quarter-Kelly bet logging, settlement, CLV, drawdown pause
  - `06_runner.py` — 8 AM daily orchestrator (refresh → features → early predict → settle)
  - `07_capture_closing_lines.py` — wakes before each start time to capture real
    closing lines; `--pregame-predict` re-runs predict + bet logging 2h before first pitch
- `config/config.py` — paths, betting parameters, team maps, league-average imputes.

## Daily automation (Windows Task Scheduler)

| Task | Time | Runs |
|---|---|---|
| MLB2-Morning | 08:00 | `06_runner.py` |
| MLB2-CloseCapture | 08:30 | `07_capture_closing_lines.py --pregame-predict` |

## Setup

1. `python -m venv venv` then `venv\Scripts\python.exe -m pip install -r requirements.txt`; R 4.3 + `renv::restore()`.
2. `.env` (never committed): `ODDS_API_KEY=...` (required),
   `SPORTSDATAIO_API_KEY=...` (optional).
3. `Rscript R/00_run_all.R`, then scripts 01 → 02 → 04.
