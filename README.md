# MLB Model V2

MLB moneyline betting model: R data collection → Python feature build → calibrated
XGB/LGBM ensemble → daily predictions with edge detection against de-vigged market
consensus → quarter-Kelly bet logging with CLV tracking.

The system deliberately separates a frozen research benchmark from the live
production bundle. A positive result is not treated as proof of profitability
until the real-price validation thresholds and confidence-interval gate pass.

## Layout

- `R/` — data collectors (MLB Stats API via baseballr, FanGraphs, Statcast).
  `00_run_all.R` orchestrates; results land in `data/raw/` (gitignored).
- `features/builder.py` — single source of truth for feature construction,
  shared by training and prediction so the two can never diverge.
- `scripts/`
  - `01_build_features.py` — raw CSVs → `data/processed/feature_matrix.csv`
  - `02_train.py` — `--mode benchmark` preserves the 2021–2023/2024 holdout;
    `--mode production` trains through the latest completed season and publishes
    an atomic, versioned model bundle
  - `03_backtest.py` — frozen holdout metrics plus sequential capped-bankroll ROI
    against real quoted prices, market coverage, drawdown, and a bootstrap interval
  - `04_predict.py` — daily slate: probables, posted lineups + home-plate umpire
    (live from MLB boxscore), de-vigged consensus odds, edge flags
  - `05_bankroll.py` — auditable quarter-Kelly decisions; per-bet/daily/open-exposure
    caps; ledger-derived bankroll recovery; settlement, CLV, and drawdown pause
  - `06_runner.py` — fail-closed daily orchestrator with raw/feature/model gates
  - `07_capture_closing_lines.py` — wakes before each start time to capture real
    closing lines; `--pregame-predict` re-runs predict + bet logging 2h before first pitch
- `config/config.py` — paths, betting parameters, team maps, league-average imputes.
- `scripts/pipeline_health.py` — schema, key, freshness, row-count, and artifact-hash gates.
- `tests/` — temporal invariance, odds matching, risk, and ledger tests.

Prediction files carry the complete lineage chain: feature/data build, model
version, prediction run, Odds API event, representative bookmaker quote, live
lineup/umpire availability, and the stake rule that constrained any logged bet.
Posted lineups are required before a row can become bet-eligible.

## Daily automation (Windows Task Scheduler)

| Task | Time | Runs |
|---|---|---|
| MLB2-Morning | 08:00 | `06_runner.py` |
| MLB2-CloseCapture | 08:30 | `07_capture_closing_lines.py --pregame-predict` |

## Setup

1. `python -m venv venv` then `venv\Scripts\python.exe -m pip install -r requirements.txt`; R 4.3 + `renv::restore()`.
2. `.env` (never committed): `ODDS_API_KEY=...` (required),
   `SPORTSDATAIO_API_KEY=...` (optional).
3. `Rscript R/00_run_all.R`, then:
   `scripts/01_build_features.py` → `scripts/02_train.py --mode production` →
   `scripts/04_predict.py --date YYYY-MM-DD`.
4. Verify with
   `venv\Scripts\python.exe .claude\skills\run-mlb-model-v2\smoke.py`.
