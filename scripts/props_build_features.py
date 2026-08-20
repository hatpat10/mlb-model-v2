# -*- coding: utf-8 -*-
"""Loads all R-collected raw CSVs needed for player props, builds the
batter-props feature set via features/props_builder.py, runs
coverage_check() (reused from features/builder.py), and writes
data/processed/props_feature_matrix.csv + props_feature_names.json.

Batter markets only for now (hits, HR, total bases, RBI, runs, walks,
strikeouts, stolen bases) — see props_builder.BATTER_LABEL_COLS for the raw
per-game counting stats each row carries as its own target.
"""
import sys
import json
from datetime import date as _date
from pathlib import Path

import pandas as pd
from loguru import logger

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config.config import PATHS, TRAIN_YEARS  # noqa: E402
from features import props_builder, builder as moneyline_builder  # noqa: E402
from features.pitcher_utils import filter_verified_starts  # noqa: E402
from features.builder import coverage_check  # noqa: E402
sys.path.insert(0, str(Path(__file__).resolve().parent))
from artifact_utils import atomic_write_csv, atomic_write_json  # noqa: E402

RAW = PATHS["raw"]
PROCESSED = PATHS["processed"]
LOGS = PATHS["logs"]

logger.remove()
logger.add(sys.stderr, level="INFO")
logger.add(LOGS / "props_build_features.log", level="DEBUG", rotation="5 MB")


def _load_concat(pattern: str, years) -> pd.DataFrame:
    frames = []
    for year in years:
        path = RAW / pattern.format(year=year)
        if path.exists():
            frames.append(pd.read_csv(path))
        else:
            logger.warning(f"missing raw file: {path}")
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)


def main():
    logger.info("Loading raw R-collected CSVs for player props from data/raw ...")
    cur = _date.today().year

    batter_gamelogs = _load_concat("batter_gamelogs_{year}.csv", range(2021, cur + 1))
    if batter_gamelogs.empty:
        logger.error("no batter_gamelogs_*.csv found — run R/14_collect_boxscores.R first.")
        sys.exit(1)
    logger.info(f"batter_gamelogs: {len(batter_gamelogs)} rows")

    game_logs_path = RAW / "game_logs_all.csv"
    if not game_logs_path.exists():
        logger.error(f"{game_logs_path} not found — run R/01_collect_gamelogs.R first.")
        sys.exit(1)
    game_logs = pd.read_csv(game_logs_path)

    batter_stats = _load_concat("batter_stats_{year}.csv", range(2020, cur + 1))
    fg_sp = _load_concat("fg_sp_stats_{year}.csv", range(2020, cur + 1))
    statcast_sp = _load_concat("statcast_sp_{year}.csv", range(2020, cur + 1))
    pitcher_gamelogs = _load_concat("pitcher_gamelogs_{year}.csv", range(2020, cur + 1))
    game_logs_path = RAW / "game_logs_all.csv"
    game_logs = pd.read_csv(game_logs_path) if game_logs_path.exists() else pd.DataFrame()

    player_bio_path = RAW / "player_bio.csv"
    player_bio = pd.read_csv(player_bio_path) if player_bio_path.exists() else pd.DataFrame()

    park_factors_path = RAW / "park_factors.csv"
    park_factors = pd.read_csv(park_factors_path) if park_factors_path.exists() else pd.DataFrame(
        columns=["team", "year", "park_factor"])

    umpire_assign_path = RAW / "umpire_assignments.csv"
    umpire_game_log_path = RAW / "umpire_game_log.csv"
    if umpire_assign_path.exists() and umpire_game_log_path.exists():
        umpire_assignments = pd.read_csv(umpire_assign_path)
        umpire_game_log = pd.read_csv(umpire_game_log_path)
    else:
        logger.warning("umpire assignment/game-log files not found — umpire_run_factor will be all-NaN")
        umpire_assignments = umpire_game_log = pd.DataFrame()

    logger.info("Building batter-props features via features/props_builder.py ...")
    df = props_builder.build_batter_rolling_form(batter_gamelogs)

    if not batter_stats.empty:
        df = props_builder.join_batter_season_stats(df, batter_stats)
    else:
        logger.warning("batter_stats data unavailable — skipping join_batter_season_stats")

    df = props_builder.attach_opposing_starter(df, game_logs)

    if not player_bio.empty:
        df = props_builder.join_batter_bio_and_platoon(df, player_bio)
    else:
        logger.warning("player_bio data unavailable — skipping join_batter_bio_and_platoon")

    if not fg_sp.empty:
        df = props_builder.join_opposing_pitcher_stats(df, fg_sp, statcast_sp)
    else:
        logger.warning("fg_sp_stats data unavailable — skipping join_opposing_pitcher_stats")

    if not pitcher_gamelogs.empty:
        if game_logs.empty:
            raise FileNotFoundError("game_logs_all.csv is required to verify starter-only pitcher appearances")
        pitcher_gamelogs = filter_verified_starts(pitcher_gamelogs, game_logs)
        df = props_builder.join_opposing_pitcher_rolling(df, pitcher_gamelogs)
    else:
        logger.warning("pitcher_gamelogs data unavailable — skipping join_opposing_pitcher_rolling")

    if not park_factors.empty:
        df = props_builder.join_park_factors(df, park_factors)
    else:
        logger.warning("park_factors data unavailable — skipping join_park_factors")

    if not umpire_assignments.empty and not umpire_game_log.empty:
        df = moneyline_builder.join_umpires(df, umpire_assignments, umpire_game_log)
    else:
        df["umpire_run_factor"] = pd.NA

    logger.info(f"Batter-props feature table: {len(df)} rows, {len(df.columns)} columns")

    # Lower bar than moneyline's team-level 95% default: real roster churn
    # (rookies/call-ups without a qualifying 50+ PA prior season) legitimately
    # costs season_* coverage at the player level in a way that almost never
    # happens for team-level joins. Anything that passes still gets
    # train-median-imputed at training time (scripts/02_train.py's pattern),
    # so this bar only needs to catch genuinely broken/sparse columns, not
    # penalize real, expected missingness.
    feature_cols = [c for c in props_builder.BATTER_FEATURE_COLS if c in df.columns]
    passing_cols = coverage_check(
        df, feature_cols, min_coverage=0.75, train_years=TRAIN_YEARS,
        coverage_overrides={"park_factor": 0.60},
    )

    PROCESSED.mkdir(parents=True, exist_ok=True)
    out_path = PROCESSED / "props_feature_matrix.csv"
    atomic_write_csv(df, out_path)
    logger.info(f"Wrote {out_path} ({len(df)} rows, {len(df.columns)} columns)")

    names_path = PROCESSED / "props_feature_names.json"
    atomic_write_json({
        "all_candidate_features": feature_cols,
        "passing_features": passing_cols,
        "label_cols": props_builder.BATTER_LABEL_COLS,
    }, names_path)
    logger.info(f"Wrote {names_path} ({len(passing_cols)}/{len(feature_cols)} features passed coverage_check)")


if __name__ == "__main__":
    main()
