# -*- coding: utf-8 -*-
"""Backfill durable game identifiers into legacy bankroll rows.

Prediction snapshots are authoritative because they preserve the game_pk
known at decision time. A date-window lookup is used only as an explicit,
warning-producing fallback and remains safe to re-run.
"""
import sys
from pathlib import Path

import pandas as pd
from loguru import logger

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config.config import PATHS  # noqa: E402
sys.path.insert(0, str(Path(__file__).resolve().parent))
from artifact_utils import atomic_write_csv  # noqa: E402

RAW = PATHS["raw"]
OUTPUTS = PATHS["outputs"]
BET_LOG_PATH = OUTPUTS / "bet_log.csv"
WINDOW_DAYS = 5


def resolve_from_predictions(row):
    pred_path = OUTPUTS / f"predictions_{row['date']}.csv"
    if not pred_path.exists():
        return None, None, None
    preds = pd.read_csv(pred_path)
    prow = preds[(preds["home_team"] == row["home_team"]) & (preds["away_team"] == row["away_team"])]
    if prow.empty:
        return None, None, None
    return prow["game_pk"].iloc[0], prow["home_win_prob"].iloc[0], prow["edge"].iloc[0]


def resolve_from_game_logs(row, home_games):
    logged_date = pd.to_datetime(row["date"])
    window = home_games[
        (home_games["home_team"] == row["home_team"]) & (home_games["away_team"] == row["away_team"])
        & ((home_games["date"] - logged_date).abs() <= pd.Timedelta(days=WINDOW_DAYS))
    ]
    if window.empty:
        return None
    window = window.copy()
    window["_dist"] = (window["date"] - logged_date).abs()
    nearest = window[window["_dist"] == window["_dist"].min()]
    if len(nearest) != 1:
        return None
    return nearest.iloc[0]["game_pk"]


def main():
    bet_log = pd.read_csv(BET_LOG_PATH)
    if "game_pk" not in bet_log.columns:
        bet_log["game_pk"] = pd.NA
    for col in ("decision_id", "model_home_prob", "edge", "decision_timestamp"):
        if col not in bet_log.columns:
            bet_log[col] = pd.NA

    game_logs = pd.read_csv(RAW / "game_logs_all.csv")
    home_games = game_logs[game_logs["is_home"] == 1][["date", "team", "opponent", "game_pk"]].rename(
        columns={"team": "home_team", "opponent": "away_team"})
    home_games["date"] = pd.to_datetime(home_games["date"])

    n_from_predictions = 0
    n_from_fallback = 0
    n_unresolved = 0
    for idx, row in bet_log.iterrows():
        if pd.notna(row["game_pk"]):
            continue

        game_pk, model_prob, edge = resolve_from_predictions(row)
        if game_pk is not None:
            n_from_predictions += 1
        else:
            game_pk = resolve_from_game_logs(row, home_games)
            if game_pk is None:
                n_unresolved += 1
                logger.warning(f"No game_pk match for {row['home_team']} vs {row['away_team']} logged {row['date']} "
                               f"(no predictions file and no game-log match within +/-{WINDOW_DAYS}d).")
                continue
            n_from_fallback += 1
            logger.warning(f"{row['home_team']} vs {row['away_team']} logged {row['date']}: using ambiguity-prone "
                           f"nearest-date fallback game_pk={game_pk}; verify doubleheaders manually.")

        bet_log.loc[idx, "game_pk"] = game_pk
        bet_log.loc[idx, "decision_id"] = f"{row['date']}_{game_pk}_{row['side']}"
        bet_log.loc[idx, "decision_timestamp"] = f"{row['date']}T00:00:00"
        if model_prob is not None:
            bet_log.loc[idx, "model_home_prob"] = model_prob
            bet_log.loc[idx, "edge"] = edge

    atomic_write_csv(bet_log, BET_LOG_PATH)
    logger.info(f"Resolved game_pk: {n_from_predictions} from predictions, {n_from_fallback} from nearest-date "
                f"fallback, {n_unresolved} unresolved. Wrote {BET_LOG_PATH}")


if __name__ == "__main__":
    main()
