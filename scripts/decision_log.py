# -*- coding: utf-8 -*-
"""Durable append-only decision history, including explicit no-bet reasons."""
from pathlib import Path

import numpy as np
import pandas as pd

from artifact_utils import atomic_write_csv, exclusive_lock


DECISION_LOG_COLUMNS = [
    "decision_id", "run_id", "generated_at_utc", "date", "game_pk",
    "start_utc", "home_team", "away_team", "home_starter", "away_starter",
    "home_win_prob", "no_vig_home_implied", "edge", "home_ml", "away_ml",
    "bet_flag", "bet_side", "decision_eligible", "ineligibility_reason",
    "decision_status", "starters_confirmed", "lineup_available", "umpire_available",
    "market_available", "n_books", "book_prob_std", "book_prob_range",
    "home_bookmaker_key", "home_bookmaker_title", "away_bookmaker_key",
    "away_bookmaker_title", "price_selection_method", "odds_event_id",
    "odds_snapshot_utc", "model_version", "model_mode", "data_version",
    "feature_build_id", "model_training_data_version", "model_feature_build_id",
    "execution_mode",
]


def append_decisions(predictions: pd.DataFrame, path: Path, execution_mode: str) -> int:
    """Atomically append one row per run/game, de-duplicated by decision_id."""
    if predictions.empty:
        return 0
    frame = predictions.copy()
    frame["decision_id"] = frame["run_id"].astype(str) + "_" + frame["game_pk"].astype(str)
    frame["execution_mode"] = execution_mode
    frame["decision_status"] = np.select(
        [~frame["decision_eligible"].astype(bool), frame["bet_flag"].astype(bool)],
        ["ineligible", "bet_candidate"],
        default="eligible_no_edge",
    )
    for column in DECISION_LOG_COLUMNS:
        if column not in frame.columns:
            frame[column] = pd.NA
    frame = frame[DECISION_LOG_COLUMNS]

    lock_path = path.parent / ".decision_log.lock"
    with exclusive_lock(lock_path):
        existing = pd.read_csv(path, dtype={"game_pk": str}) if path.exists() else pd.DataFrame(columns=DECISION_LOG_COLUMNS)
        for column in DECISION_LOG_COLUMNS:
            if column not in existing.columns:
                existing[column] = pd.NA
        known = set(existing["decision_id"].dropna().astype(str))
        new_rows = frame[~frame["decision_id"].astype(str).isin(known)]
        if new_rows.empty:
            return 0
        combined = pd.concat([existing[DECISION_LOG_COLUMNS], new_rows], ignore_index=True)
        atomic_write_csv(combined, path)
    return int(len(new_rows))
