# MLB Model V2

MLB moneyline betting model: R data collection → Python feature build → honestly
selected probability model → daily predictions with edge detection against de-vigged market
consensus → quarter-Kelly bet logging with CLV tracking.

The system deliberately separates a frozen research benchmark from the live
production bundle. A positive result is not treated as proof of profitability.
Retrospective closing-price results and actual forward paper/live results are
reported separately, and only the auditable forward ledger can validate deployment.

## Layout

- `R/` — data collectors (MLB Stats API via baseballr, FanGraphs, Statcast).
  `00_run_all.R` orchestrates; results land in `data/raw/` (gitignored).
- `features/builder.py` — single source of truth for feature construction,
  shared by training and prediction so the two can never diverge.
- `scripts/`
  - `01_build_features.py` — raw CSVs → `data/processed/feature_matrix.csv`
  - `02_train.py` — `--mode benchmark` preserves the 2021–2023/2024 holdout;
    `--mode production` trains through the latest completed season and publishes
    an atomic, versioned model bundle. Hyperparameters stop before the evaluation
    season; the untouched next season selects among raw/calibrated single models,
    averages, and the weighted ensemble before the final future model is fit
  - `03_backtest.py` — frozen holdout metrics plus close-time retrospective ROI,
    model-versus-market metrics, coverage, drawdown, and date-block uncertainty
  - `04_predict.py` — daily slate: probables, posted lineups + home-plate umpire
    (live from MLB boxscore), de-vigged consensus odds, best side-specific quoted
    prices, manifest-selected model scoring, edge flags, and an append-only
    decision/no-bet history
  - `05_bankroll.py` — auditable quarter-Kelly decisions; per-bet/daily/open-exposure
    caps; ledger-derived bankroll recovery; settlement, CLV, and drawdown pause
  - `06_runner.py` — fail-closed daily orchestrator with raw/feature/model gates
  - `07_capture_closing_lines.py` — wakes before each start time to capture real
    closing lines; `--pregame-predict` polls the free MLB lineup feed from T-2h and
    decides when lineups post, with a final T-15m cutoff
  - `08_forward_performance.py` — actual settled paper/live ROI, CLV, drawdown,
    date-block uncertainty, lineage coverage, and explicit deployment gates
- `config/config.py` — paths, betting parameters, team maps, league-average imputes.
- `scripts/pipeline_health.py` — schema, key, freshness, row-count, and artifact-hash gates.
- `tests/` — temporal invariance, odds matching, risk, and ledger tests.

Prediction and ledger files carry the complete lineage chain: feature/data build,
model version, prediction run, Odds API event, side-specific executable bookmaker
quote, market dispersion, live lineup/umpire availability, and the stake rule that
constrained any logged bet. Posted lineups are required before a row can become
bet-eligible. `outputs/decision_log.csv` and `outputs/bet_log.csv` are durable,
tracked history; regenerated prediction workbooks are not evidence records.

## Daily automation (Windows Task Scheduler)

| Task | Time | Runs |
|---|---|---|
| MLB2-Morning | 08:00 | `06_runner.py` |
| MLB2-CloseCapture | 08:30 | `07_capture_closing_lines.py --pregame-predict` |

## Setup

1. `python -m venv venv` then `venv\Scripts\python.exe -m pip install -r requirements.txt`; R 4.3 + `renv::restore()`.
2. Copy `.env.example` to `.env` and set `ODDS_API_KEY` (required) plus
   `SPORTSDATAIO_API_KEY` (optional). Keep `BETTING_EXECUTION_MODE=paper` while
   validation is false. If switching to `live`, `BETTING_BOOKMAKERS` is required;
   prices will then be selected only from those executable book keys.
3. `Rscript R/00_run_all.R`, then:
   `scripts/01_build_features.py` → `scripts/02_train.py --mode production` →
   `scripts/04_predict.py --date YYYY-MM-DD`.
4. Verify with
   `venv\Scripts\python.exe .claude\skills\run-mlb-model-v2\smoke.py`.
