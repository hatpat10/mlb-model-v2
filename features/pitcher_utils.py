# -*- coding: utf-8 -*-
"""Shared pitcher-log hygiene helpers."""
import unicodedata

import pandas as pd


def _name_key(value):
    if not isinstance(value, str):
        return None
    ascii_name = "".join(c for c in unicodedata.normalize("NFKD", value) if not unicodedata.combining(c))
    return " ".join(ascii_name.lower().split())


def filter_verified_starts(pitcher_gamelogs: pd.DataFrame, game_logs: pd.DataFrame) -> pd.DataFrame:
    """Keep only appearances whose name/date appears as an actual MLB starter."""
    required = {"date", "is_home", "home_starter", "away_starter"}
    if not required.issubset(game_logs.columns):
        raise ValueError(f"game logs lack starter verification columns: {sorted(required - set(game_logs.columns))}")
    home_games = game_logs[game_logs["is_home"] == 1]
    actual = pd.concat([
        home_games[["date", "home_starter"]].rename(columns={"home_starter": "pitcher_name"}),
        home_games[["date", "away_starter"]].rename(columns={"away_starter": "pitcher_name"}),
    ], ignore_index=True)
    actual["_date"] = pd.to_datetime(actual["date"], errors="coerce").dt.normalize()
    actual["_pitcher"] = actual["pitcher_name"].map(_name_key)
    actual = actual[["_date", "_pitcher"]].dropna().drop_duplicates()

    logs = pitcher_gamelogs.copy()
    logs["_date"] = pd.to_datetime(logs["date"], errors="coerce").dt.normalize()
    logs["_pitcher"] = logs["pitcher_name"].map(_name_key)
    logs = logs.merge(actual.assign(_verified_start=True), on=["_date", "_pitcher"], how="left")
    return logs[logs["_verified_start"].eq(True)].drop(columns=["_date", "_pitcher", "_verified_start"])
