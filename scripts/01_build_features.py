# -*- coding: utf-8 -*-
"""Loads all R-collected raw CSVs, builds the full feature set via
features/builder.py, pivots to one row per game (home-team perspective),
runs coverage_check(), and writes data/processed/feature_matrix.csv +
data/processed/feature_names.json.
"""
import sys
import json
from pathlib import Path

import numpy as np
import pandas as pd
from loguru import logger

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config.config import PATHS, ALL_DATA_YEARS  # noqa: E402
from features import builder  # noqa: E402

RAW = PATHS["raw"]
PROCESSED = PATHS["processed"]
LOGS = PATHS["logs"]

logger.remove()
logger.add(sys.stderr, level="INFO")
logger.add(LOGS / "01_build_features.log", level="DEBUG", rotation="5 MB")


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
    logger.info("Loading raw R-collected CSVs from data/raw ...")

    game_logs_path = RAW / "game_logs_all.csv"
    if not game_logs_path.exists():
        logger.error(f"{game_logs_path} not found — run R/01_collect_gamelogs.R first.")
        sys.exit(1)
    game_logs = pd.read_csv(game_logs_path)
    logger.info(f"game_logs_all: {len(game_logs)} rows")

    # Ranges run through the season currently being played — current-season
    # files are collected now (for in-season/as-of features and next year's
    # prior-year joins), and the prior-year join keys make any same-season
    # rows inert to existing features until explicitly used.
    from datetime import date as _date
    cur = _date.today().year
    fg_batting = _load_concat("fg_team_batting_{year}.csv", range(2020, cur + 1))
    fg_sp = _load_concat("fg_sp_stats_{year}.csv", range(2020, cur + 1))
    statcast_team = _load_concat("statcast_team_batting_{year}.csv", range(2020, cur + 1))
    statcast_sp = _load_concat("statcast_sp_{year}.csv", range(2020, cur + 1))
    team_fielding = _load_concat("team_fielding_{year}.csv", range(2020, cur + 1))
    batter_stats = _load_concat("batter_stats_{year}.csv", range(2020, cur + 1))
    lineups = _load_concat("lineups_{year}.csv", range(2021, cur + 1))
    pitcher_gamelogs = _load_concat("pitcher_gamelogs_{year}.csv", range(2020, cur + 1))

    player_bio_path = RAW / "player_bio.csv"
    player_bio = pd.read_csv(player_bio_path) if player_bio_path.exists() else pd.DataFrame()

    park_factors_path = RAW / "park_factors.csv"
    park_factors = pd.read_csv(park_factors_path) if park_factors_path.exists() else pd.DataFrame(
        columns=["team", "year", "park_factor"])

    umpire_assign_path = RAW / "umpire_assignments.csv"
    umpire_factors_path = RAW / "umpire_factors.csv"
    if umpire_assign_path.exists() and umpire_factors_path.exists():
        assign = pd.read_csv(umpire_assign_path)
        factors = pd.read_csv(umpire_factors_path)
        umpire_lookup = assign.merge(factors[["umpire_name", "umpire_run_factor"]], on="umpire_name", how="left")
        umpire_lookup["game_pk"] = umpire_lookup["game_pk"].astype(str)
    else:
        logger.warning("umpire assignment/factor files not found — umpire_run_factor will be all-NaN")
        umpire_lookup = pd.DataFrame(columns=["game_pk", "umpire_run_factor"])

    logger.info("Building features via features/builder.py ...")
    df = builder.build_rolling_features(game_logs)
    df = builder.build_elo_ratings(df)
    df = builder.build_schedule_features(df)

    if not fg_batting.empty:
        df = builder.join_team_batting(df, fg_batting)
    else:
        logger.warning("fg_team_batting data unavailable — skipping join_team_batting")

    if not fg_sp.empty:
        df = builder.join_team_pitching(df, fg_sp)
        df = builder.join_sp_stats(df, fg_sp, statcast_sp)
    else:
        logger.warning("fg_sp_stats data unavailable — skipping join_team_pitching and join_sp_stats")

    if not statcast_team.empty:
        df = builder.join_statcast_batting(df, statcast_team)
    else:
        logger.warning("statcast_team_batting data unavailable — skipping join_statcast_batting")

    if not team_fielding.empty:
        df = builder.join_team_fielding(df, team_fielding)
    else:
        logger.warning("team_fielding data unavailable — skipping join_team_fielding")

    if not lineups.empty and not batter_stats.empty:
        df = builder.join_lineup_quality(df, lineups, batter_stats)
    else:
        logger.warning("lineups/batter_stats data unavailable — skipping join_lineup_quality")

    if not player_bio.empty and not lineups.empty:
        df = builder.join_player_bio(df, player_bio, lineups)
    else:
        logger.warning("player_bio/lineups data unavailable — skipping join_player_bio")

    if not pitcher_gamelogs.empty:
        df = builder.join_pitcher_rolling_form(df, pitcher_gamelogs)
    else:
        logger.warning("pitcher_gamelogs data unavailable — skipping join_pitcher_rolling_form")

    if not park_factors.empty:
        df = builder.join_park_factors(df, park_factors)
    else:
        logger.warning("park_factors data unavailable — skipping join_park_factors")

    if not umpire_lookup.empty:
        df = builder.join_umpires(df, umpire_lookup)
    else:
        df["umpire_run_factor"] = np.nan

    logger.info(f"Long-format feature table: {len(df)} rows, {len(df.columns)} columns")

    logger.info("Pivoting to one row per game (home-team perspective) ...")
    game_df, feature_cols = builder.pivot_to_game_level(df)
    logger.info(f"Pivoted game-level table: {len(game_df)} rows, {len(game_df.columns)} columns")

    passing_cols = builder.coverage_check(game_df, feature_cols, min_coverage=0.95,
                                           train_years=[y for y in ALL_DATA_YEARS if y <= 2025])

    PROCESSED.mkdir(parents=True, exist_ok=True)
    out_path = PROCESSED / "feature_matrix.csv"
    game_df.to_csv(out_path, index=False)
    logger.info(f"Wrote {out_path} ({len(game_df)} rows, {len(game_df.columns)} columns)")

    names_path = PROCESSED / "feature_names.json"
    with open(names_path, "w") as f:
        json.dump({"all_candidate_features": feature_cols, "passing_features": passing_cols}, f, indent=2)
    logger.info(f"Wrote {names_path} ({len(passing_cols)}/{len(feature_cols)} features passed coverage_check)")


if __name__ == "__main__":
    main()
