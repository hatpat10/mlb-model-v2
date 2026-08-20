# -*- coding: utf-8 -*-
"""Loads all R-collected raw CSVs needed for pitcher props, builds the
pitcher-props feature set via features/props_builder.py, runs
coverage_check() (reused from features/builder.py), and writes
data/processed/pitcher_props_feature_matrix.csv + pitcher_props_feature_names.json.

Strikeouts is the headline market (see props_builder.PITCHER_LABEL_COLS for
the full set of raw per-start counting stats each row carries as its own
target: outs, batters_faced, k, bb, h, hr, r, er, pitches). Starts only
(is_starter==1, enforced inside build_pitcher_rolling_form) — relief
appearances aren't in scope for this market.
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
from features.builder import coverage_check  # noqa: E402
sys.path.insert(0, str(Path(__file__).resolve().parent))
from artifact_utils import atomic_write_csv, atomic_write_json  # noqa: E402

RAW = PATHS["raw"]
PROCESSED = PATHS["processed"]
LOGS = PATHS["logs"]

logger.remove()
logger.add(sys.stderr, level="INFO")
logger.add(LOGS / "props_build_pitcher_features.log", level="DEBUG", rotation="5 MB")


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
    logger.info("Loading raw R-collected CSVs for pitcher props from data/raw ...")
    cur = _date.today().year

    pitcher_boxscore = _load_concat("pitcher_boxscore_{year}.csv", range(2021, cur + 1))
    if pitcher_boxscore.empty:
        logger.error("no pitcher_boxscore_*.csv found — run R/14_collect_boxscores.R first.")
        sys.exit(1)
    logger.info(f"pitcher_boxscore: {len(pitcher_boxscore)} rows")

    fg_sp = _load_concat("fg_sp_stats_{year}.csv", range(2020, cur + 1))
    statcast_sp = _load_concat("statcast_sp_{year}.csv", range(2020, cur + 1))
    fg_team_batting = _load_concat("fg_team_batting_{year}.csv", range(2020, cur + 1))
    batter_stats = _load_concat("batter_stats_{year}.csv", range(2020, cur + 1))
    lineups = _load_concat("lineups_{year}.csv", range(2021, cur + 1))

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

    logger.info("Building pitcher-props features via features/props_builder.py ...")
    df = props_builder.build_pitcher_rolling_form(pitcher_boxscore)
    logger.info(f"  {len(df)} starter rows (relief appearances excluded)")

    if not fg_sp.empty:
        df = props_builder.join_pitcher_season_stats(df, fg_sp, statcast_sp)
    else:
        logger.warning("fg_sp_stats data unavailable — skipping join_pitcher_season_stats")

    if not player_bio.empty:
        df = props_builder.join_pitcher_bio(df, player_bio)
    else:
        logger.warning("player_bio data unavailable — skipping join_pitcher_bio")

    if not fg_team_batting.empty:
        df = props_builder.join_opponent_team_batting(df, fg_team_batting)
    else:
        logger.warning("fg_team_batting data unavailable — skipping join_opponent_team_batting")

    if not lineups.empty and not batter_stats.empty and "opp_team_wrc_plus" in df.columns:
        df = props_builder.join_opposing_lineup_quality(df, lineups, batter_stats)
    else:
        logger.warning("lineups/batter_stats/opp_team_wrc_plus unavailable — skipping join_opposing_lineup_quality")

    if not park_factors.empty:
        df = props_builder.join_park_factors(df, park_factors)
    else:
        logger.warning("park_factors data unavailable — skipping join_park_factors")

    if not umpire_assignments.empty and not umpire_game_log.empty:
        df = moneyline_builder.join_umpires(df, umpire_assignments, umpire_game_log)
    else:
        df["umpire_run_factor"] = pd.NA

    logger.info(f"Pitcher-props feature table: {len(df)} rows, {len(df.columns)} columns")

    # Same relaxed bar as props_build_features.py — real roster churn
    # (call-ups making their first start) legitimately costs season-stat
    # coverage at the player level; anything that passes still gets
    # train-median-imputed at training time (scripts/02_train.py's pattern).
    feature_cols = [c for c in props_builder.PITCHER_FEATURE_COLS if c in df.columns]
    passing_cols = coverage_check(
        df, feature_cols, min_coverage=0.75, train_years=TRAIN_YEARS,
        coverage_overrides={"park_factor": 0.60},
    )

    PROCESSED.mkdir(parents=True, exist_ok=True)
    out_path = PROCESSED / "pitcher_props_feature_matrix.csv"
    atomic_write_csv(df, out_path)
    logger.info(f"Wrote {out_path} ({len(df)} rows, {len(df.columns)} columns)")

    names_path = PROCESSED / "pitcher_props_feature_names.json"
    atomic_write_json({
        "all_candidate_features": feature_cols,
        "passing_features": passing_cols,
        "label_cols": props_builder.PITCHER_LABEL_COLS,
    }, names_path)
    logger.info(f"Wrote {names_path} ({len(passing_cols)}/{len(feature_cols)} features passed coverage_check)")


if __name__ == "__main__":
    main()
