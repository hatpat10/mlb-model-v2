# -*- coding: utf-8 -*-
"""Fail-closed data and feature quality gates for daily orchestration."""
import argparse
import json
import sys
from datetime import date, datetime
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config.config import PATHS  # noqa: E402
sys.path.insert(0, str(Path(__file__).resolve().parent))
from artifact_utils import sha256_file  # noqa: E402

RAW = PATHS["raw"]
PROCESSED = PATHS["processed"]


def validate_raw(as_of: date | None = None) -> list[str]:
    as_of = as_of or date.today()
    errors = []
    path = RAW / "game_logs_all.csv"
    if not path.exists() or path.stat().st_size == 0:
        return ["game_logs_all.csv is missing or empty"]
    df = pd.read_csv(path, dtype={"game_pk": str})
    required = {"date", "game_pk", "team", "opponent", "is_home", "win", "home_starter", "away_starter", "year"}
    missing = required - set(df.columns)
    if missing:
        errors.append(f"game_logs_all.csv missing columns: {sorted(missing)}")
        return errors
    dates = pd.to_datetime(df["date"], errors="coerce")
    if dates.isna().any():
        errors.append(f"game_logs_all.csv has {int(dates.isna().sum())} invalid dates")
    if dates.notna().any() and dates.max().date() > as_of:
        errors.append(f"game_logs_all.csv contains future games through {dates.max().date()}")
    if dates.notna().any() and 4 <= as_of.month <= 9 and (as_of - dates.max().date()).days > 7:
        errors.append(f"game_logs_all.csv is stale: latest game is {dates.max().date()} (more than 7 days old)")
    if df.duplicated(["game_pk", "team"]).any():
        errors.append("duplicate (game_pk, team) rows in game_logs_all.csv")
    legs = df.groupby("game_pk").agg(rows=("team", "size"), home_rows=("is_home", "sum"))
    bad_legs = legs[(legs["rows"] != 2) | (legs["home_rows"] != 1)]
    if not bad_legs.empty:
        errors.append(f"{len(bad_legs)} games do not have exactly one home and one away row")
    if not set(pd.to_numeric(df["win"], errors="coerce").dropna().unique()).issubset({0, 1}):
        errors.append("win contains values outside {0,1}")
    for name in ("park_factors.csv", "umpire_assignments.csv", "umpire_game_log.csv"):
        required_path = RAW / name
        if not required_path.exists() or required_path.stat().st_size == 0:
            errors.append(f"{name} is missing or empty")
    return errors


def validate_features() -> list[str]:
    errors = []
    matrix_path = PROCESSED / "feature_matrix.csv"
    names_path = PROCESSED / "feature_names.json"
    manifest_path = PROCESSED / "feature_manifest.json"
    for path in (matrix_path, names_path, manifest_path):
        if not path.exists() or path.stat().st_size == 0:
            errors.append(f"{path.name} is missing or empty")
    if errors:
        return errors
    df = pd.read_csv(matrix_path, dtype={"game_pk": str})
    with open(names_path, encoding="utf-8") as handle:
        features = json.load(handle).get("passing_features", [])
    with open(manifest_path, encoding="utf-8") as handle:
        manifest = json.load(handle)
    missing = [feature for feature in features if feature not in df.columns]
    if missing:
        errors.append(f"feature matrix missing passing features: {missing}")
    if df["game_pk"].duplicated().any():
        errors.append("feature matrix has duplicate game_pk rows")
    if len(df) < 1000:
        errors.append(f"feature matrix unexpectedly small: {len(df)} rows")
    if manifest.get("row_count") != len(df):
        errors.append("feature manifest row_count does not match feature_matrix.csv")
    if manifest.get("feature_matrix_sha256") != sha256_file(matrix_path):
        errors.append("feature_matrix.csv hash does not match feature manifest")
    if manifest.get("passing_features") != features:
        errors.append("feature_names.json and feature_manifest.json disagree on passing features")
    target = pd.to_numeric(df.get("home_win"), errors="coerce")
    if not set(target.dropna().unique()).issubset({0, 1}) or target.nunique() < 2:
        errors.append("home_win target is invalid or has only one class")
    nonfinite = [column for column in features if np.isinf(pd.to_numeric(df[column], errors="coerce")).any()]
    if nonfinite:
        errors.append(f"passing features contain infinity: {nonfinite}")
    return errors


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", choices=("raw", "features", "all"), default="all")
    parser.add_argument("--date", type=lambda value: datetime.strptime(value, "%Y-%m-%d").date())
    args = parser.parse_args()
    errors = []
    if args.stage in ("raw", "all"):
        errors.extend(validate_raw(args.date))
    if args.stage in ("features", "all"):
        errors.extend(validate_features())
    if errors:
        for error in errors:
            print(f"FAIL: {error}", file=sys.stderr)
        raise SystemExit(1)
    print(f"Pipeline health gate passed: {args.stage}")


if __name__ == "__main__":
    main()
