# -*- coding: utf-8 -*-
"""Single source of truth for MLB moneyline feature construction.

Every function here operates on the "long" game log format — one row per
team per game (see R/01_collect_gamelogs.R) — except where noted. Both
scripts/01_build_features.py (training) and scripts/04_predict.py (daily
prediction) import from this module; feature logic must never be
duplicated in either caller.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from loguru import logger

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config.config import LEAGUE_AVG_SP, STADIUM_COORDS, TEAM_ABBREV_MAP  # noqa: E402

# Per-team features that need both a home_ and away_ prefixed copy when the
# long (team-per-game) table is pivoted to one row per game. Shared by
# scripts/01_build_features.py (training) and scripts/04_predict.py (daily
# prediction) via pivot_to_game_level() below.
TEAM_LEVEL_COLS = [
    "win_rate_roll_5", "win_rate_roll_10", "win_rate_roll_20",
    "run_diff_roll_10", "runs_scored_roll_10", "runs_allowed_roll_10",
    "pyth_win_pct", "pyth_win_pct_roll_20", "pyth_win_pct_roll_40",
    "rest_days", "travel_penalty", "consec_away",
    "wrc_plus", "woba", "bb_pct", "k_pct", "iso",
    "team_fip", "team_xfip", "team_k_pct", "team_bb_pct",
    "barrel_pct", "hard_hit_pct", "xwoba", "exit_velo_avg",
    "drs", "oaa", "def_runs",
    "lineup_wrc_plus", "platoon_advantage",
]

# Game-level features (already identical on both the home- and away-team
# perspective rows) that just need to be carried through as-is.
GAME_LEVEL_COLS = [
    "game_pk", "date", "year", "month", "day_of_week",
    "home_starter", "away_starter",
    "home_sp_known", "away_sp_known",
    "home_sp_era", "home_sp_fip", "home_sp_xfip", "home_sp_siera",
    "home_sp_k_pct", "home_sp_bb_pct", "home_sp_whip", "home_sp_k9", "home_sp_velo", "home_sp_whiff_pct",
    "away_sp_era", "away_sp_fip", "away_sp_xfip", "away_sp_siera",
    "away_sp_k_pct", "away_sp_bb_pct", "away_sp_whip", "away_sp_k9", "away_sp_velo", "away_sp_whiff_pct",
    "sp_era_adv", "sp_fip_adv", "sp_k_pct_adv",
    "home_sp_era_roll3", "away_sp_era_roll3", "home_sp_fip_roll3", "away_sp_fip_roll3",
    "park_factor", "park_factor_h", "umpire_run_factor",
]


def normalize_team_abbrev(series: pd.Series) -> pd.Series:
    """Map any known alternate team code onto the standard 30-team set."""
    return series.astype(str).str.strip().str.upper().map(TEAM_ABBREV_MAP).fillna(series)


def _haversine_miles(lat1, lon1, lat2, lon2):
    r = 3958.8
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = np.sin(dlat / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2
    return 2 * r * np.arcsin(np.sqrt(a))


def build_rolling_features(df: pd.DataFrame) -> pd.DataFrame:
    """Rolling team-form features, all computed with .shift(1) so a game's
    features never see that game's own result. Windows reset naturally at
    the start of a team's history (rolling with min_periods=1 on shifted
    data) but do NOT reset across season boundaries — a team's most recent
    games from last season are still its most recent form entering a new one.
    """
    df = df.sort_values(["team", "date"]).reset_index(drop=True)
    g = df.groupby("team", group_keys=False)

    def _roll(col, window):
        shifted = g[col].shift(1)
        return shifted.groupby(df["team"]).transform(lambda s: s.rolling(window, min_periods=1).mean())

    def _roll_sum(shifted_series, window):
        return shifted_series.groupby(df["team"]).transform(lambda s: s.rolling(window, min_periods=1).sum())

    df["win_rate_roll_5"] = _roll("win", 5)
    df["win_rate_roll_10"] = _roll("win", 10)
    df["win_rate_roll_20"] = _roll("win", 20)

    df["_run_diff"] = df["runs_scored"] - df["runs_allowed"]
    df["run_diff_roll_10"] = _roll("_run_diff", 10)
    df["runs_scored_roll_10"] = _roll("runs_scored", 10)
    df["runs_allowed_roll_10"] = _roll("runs_allowed", 10)
    df = df.drop(columns=["_run_diff"])

    # Season-to-date pythagorean win pct (expanding, shifted) and rolling versions.
    rs_shift = g["runs_scored"].shift(1)
    ra_shift = g["runs_allowed"].shift(1)

    cum_rs = rs_shift.groupby(df["team"]).transform(lambda s: s.cumsum())
    cum_ra = ra_shift.groupby(df["team"]).transform(lambda s: s.cumsum())
    df["pyth_win_pct"] = (cum_rs ** 2) / (cum_rs ** 2 + cum_ra ** 2).replace(0, np.nan)

    for window in (20, 40):
        rs_roll = _roll_sum(rs_shift, window)
        ra_roll = _roll_sum(ra_shift, window)
        df[f"pyth_win_pct_roll_{window}"] = (rs_roll ** 2) / (rs_roll ** 2 + ra_roll ** 2).replace(0, np.nan)

    return df


def build_elo_ratings(df: pd.DataFrame, k: int = 20) -> pd.DataFrame:
    """Per-team Elo, regressed 33% toward 1500 at the start of each season.
    Processes games in chronological order (one pass per unique game_pk) and
    returns elo_rating / opp_elo_rating / elo_diff / elo_win_prob for every
    team-game row, using each team's PRE-game rating (never post-game).
    """
    df = df.copy()
    games = (
        df[["game_pk", "date", "year", "team", "opponent", "is_home", "win"]]
        .drop_duplicates(subset=["game_pk", "team"])
        .sort_values(["date", "game_pk"])
    )

    ratings = {}
    last_season = {}
    pre_game_elo = {}  # (game_pk, team) -> elo before this game

    for game_pk, grp in games.groupby("game_pk", sort=False):
        home_row = grp[grp["is_home"] == 1]
        away_row = grp[grp["is_home"] == 0]
        if home_row.empty or away_row.empty:
            continue
        home_team = home_row["team"].iloc[0]
        away_team = away_row["team"].iloc[0]
        year = home_row["year"].iloc[0]
        home_win_raw = home_row["win"].iloc[0]

        for t in (home_team, away_team):
            if t not in ratings:
                ratings[t] = 1500.0
                last_season[t] = year
            elif last_season[t] != year:
                ratings[t] = ratings[t] * 0.67 + 1500.0 * 0.33
                last_season[t] = year

        home_elo = ratings[home_team]
        away_elo = ratings[away_team]
        pre_game_elo[(game_pk, home_team)] = home_elo
        pre_game_elo[(game_pk, away_team)] = away_elo

        # Unplayed games (e.g. today's slate at prediction time) have no
        # result yet — still record their pre-game ratings above, but skip
        # the rating update since there's no outcome to update on.
        if pd.isna(home_win_raw):
            continue

        expected_home = 1.0 / (1.0 + 10 ** ((away_elo - home_elo) / 400.0))
        delta = k * (int(home_win_raw) - expected_home)
        ratings[home_team] = home_elo + delta
        ratings[away_team] = away_elo - delta

    elo_df = pd.Series(pre_game_elo, name="elo_rating").rename_axis(["game_pk", "team"]).reset_index()
    df = df.merge(elo_df, on=["game_pk", "team"], how="left")

    opp_elo_df = elo_df.rename(columns={"team": "opponent", "elo_rating": "opp_elo_rating"})
    df = df.merge(opp_elo_df, on=["game_pk", "opponent"], how="left")

    df["elo_diff"] = df["elo_rating"] - df["opp_elo_rating"]
    df["elo_win_prob"] = 1.0 / (1.0 + 10 ** (-df["elo_diff"] / 400.0))
    return df


def build_schedule_features(df: pd.DataFrame) -> pd.DataFrame:
    """is_home passthrough, rest_days (capped 7), consec_away (capped 7),
    and a binary travel_penalty flag (>1000 miles between consecutive games
    within 48 hours), using STADIUM_COORDS to locate each game by its host
    (home) team's park.
    """
    df = df.sort_values(["team", "date"]).reset_index(drop=True)
    df["date"] = pd.to_datetime(df["date"])

    host_team = np.where(df["is_home"] == 1, df["team"], df["opponent"])
    coords = pd.DataFrame(STADIUM_COORDS).T.rename(columns={0: "lat", 1: "lon"})
    host_coords = coords.reindex(host_team).reset_index(drop=True)
    df["_lat"] = host_coords["lat"].values
    df["_lon"] = host_coords["lon"].values

    g = df.groupby("team", group_keys=False)
    prev_date = g["date"].shift(1)
    rest_days = (df["date"] - prev_date).dt.days
    df["rest_days"] = rest_days.clip(upper=7)

    prev_lat = g["_lat"].shift(1)
    prev_lon = g["_lon"].shift(1)
    dist = _haversine_miles(df["_lat"], df["_lon"], prev_lat, prev_lon)
    df["travel_penalty"] = ((dist > 1000) & (rest_days <= 2)).astype(int)
    df.loc[prev_date.isna(), "travel_penalty"] = 0

    # Consecutive-away streak entering this game (not counting this game itself).
    streak = np.zeros(len(df))
    for team, idx in df.groupby("team").groups.items():
        idx = list(idx)
        run = 0
        vals = df.loc[idx, "is_home"].values
        out = np.zeros(len(idx))
        for i in range(len(idx)):
            out[i] = min(run, 7)
            run = run + 1 if vals[i] == 0 else 0
        streak[idx] = out
    df["consec_away"] = streak

    return df.drop(columns=["_lat", "_lon"])


def join_team_batting(df: pd.DataFrame, fg_batting: pd.DataFrame) -> pd.DataFrame:
    """Prior-year FanGraphs team batting: wrc_plus, woba, bb_pct, k_pct, iso."""
    fgb = fg_batting.copy()
    fgb["team"] = normalize_team_abbrev(fgb["team"])
    fgb["join_year"] = fgb["year"].astype(int) + 1
    cols = ["team", "join_year", "wrc_plus", "woba", "bb_pct", "k_pct", "iso"]
    fgb = fgb[cols].drop_duplicates(subset=["team", "join_year"])
    df = df.merge(fgb, left_on=["team", "year"], right_on=["team", "join_year"], how="left")
    return df.drop(columns=["join_year"])


def join_team_pitching(df: pd.DataFrame, fg_sp: pd.DataFrame) -> pd.DataFrame:
    """Prior-year team-level pitching (FIP, xFIP, K%, BB%), derived by
    aggregating fg_sp_stats to the team level (IP-weighted) since no direct
    team-pitching source is collected.
    """
    sp = fg_sp.copy()
    sp["team"] = normalize_team_abbrev(sp["team"])

    def _wmean(x, w):
        w = w.reindex(x.index)
        return np.average(x, weights=w) if w.sum() > 0 else x.mean()

    team_pitch = (
        sp.groupby(["team", "year"])
        .apply(lambda x: pd.Series({
            "team_fip": _wmean(x["fip"], x["ip"]),
            "team_xfip": _wmean(x["xfip"], x["ip"]),
            "team_k_pct": _wmean(x["k_pct"], x["ip"]),
            "team_bb_pct": _wmean(x["bb_pct"], x["ip"]),
        }))
        .reset_index()
    )
    team_pitch["join_year"] = team_pitch["year"].astype(int) + 1
    team_pitch = team_pitch.drop(columns=["year"]).drop_duplicates(subset=["team", "join_year"])
    df = df.merge(team_pitch, left_on=["team", "year"], right_on=["team", "join_year"], how="left")
    return df.drop(columns=["join_year"])


def join_sp_stats(df: pd.DataFrame, fg_sp: pd.DataFrame, statcast_sp: pd.DataFrame) -> pd.DataFrame:
    """Join home/away starter stats (fg_sp + statcast_sp) by pitcher_name +
    (game_year - 1), with league-average imputation when a starter isn't
    found and home_sp_known/away_sp_known flags marking real vs imputed.
    """
    fg = fg_sp.copy()
    fg["pitcher_name"] = fg["pitcher_name"].astype(str)
    sc = statcast_sp.copy()
    sc["pitcher_name"] = sc["pitcher_name"].astype(str)

    merged = fg.merge(sc[["pitcher_name", "year", "velo_avg", "whiff_pct"]], on=["pitcher_name", "year"], how="outer")
    merged["join_year"] = merged["year"].astype(int) + 1

    raw_stat_cols = ["era", "fip", "xfip", "siera", "k_pct", "bb_pct", "whip", "k9", "velo_avg", "whiff_pct"]

    df = df.copy()
    for side, starter_col in (("home", "home_starter"), ("away", "away_starter")):
        prefixed_cols = {c: f"{side}_sp_{c}" if c != "velo_avg" else f"{side}_sp_velo" for c in raw_stat_cols}
        stat_cols = list(prefixed_cols.values())

        side_lookup = merged.rename(columns={"join_year": "year_lu", **prefixed_cols})
        side_lookup = side_lookup.drop_duplicates(subset=["pitcher_name", "year_lu"])
        side_lookup = side_lookup.rename(columns={"pitcher_name": starter_col, "year_lu": "_yr"})

        df = df.merge(
            side_lookup[[starter_col, "_yr"] + stat_cols],
            left_on=[starter_col, "year"], right_on=[starter_col, "_yr"], how="left",
        )
        df = df.drop(columns=["_yr"])

        df[f"{side}_sp_known"] = df[stat_cols].notna().any(axis=1).astype(int)
        years = df["year"].fillna(-1).astype(int)
        for raw_col, prefixed_col in prefixed_cols.items():
            league_default = years.map(lambda y: LEAGUE_AVG_SP.get(y, LEAGUE_AVG_SP[max(LEAGUE_AVG_SP)]).get(raw_col, np.nan))
            df[prefixed_col] = df[prefixed_col].fillna(league_default)

    df["sp_era_adv"] = df["away_sp_era"] - df["home_sp_era"]
    df["sp_fip_adv"] = df["away_sp_fip"] - df["home_sp_fip"]
    df["sp_k_pct_adv"] = df["home_sp_k_pct"] - df["away_sp_k_pct"]
    return df


def join_statcast_batting(df: pd.DataFrame, statcast_team: pd.DataFrame) -> pd.DataFrame:
    """Prior-year team Statcast batting quality: barrel_pct, hard_hit_pct,
    xwoba, exit_velo_avg.
    """
    sct = statcast_team.copy()
    sct["team"] = normalize_team_abbrev(sct["team"])
    sct["join_year"] = sct["year"].astype(int) + 1
    cols = ["team", "join_year", "barrel_pct", "hard_hit_pct", "xwoba", "exit_velo_avg"]
    sct = sct[cols].drop_duplicates(subset=["team", "join_year"])
    df = df.merge(sct, left_on=["team", "year"], right_on=["team", "join_year"], how="left")
    return df.drop(columns=["join_year"])


def join_team_fielding(df: pd.DataFrame, team_fielding: pd.DataFrame) -> pd.DataFrame:
    """Prior-year team defensive quality: drs, oaa, def_runs. Closes the
    original pipeline's blind spot — offense and pitching were covered,
    defense wasn't.
    """
    tf = team_fielding.copy()
    tf["team"] = normalize_team_abbrev(tf["team"])
    tf["join_year"] = tf["year"].astype(int) + 1
    cols = ["team", "join_year", "drs", "oaa", "def_runs"]
    tf = tf[cols].drop_duplicates(subset=["team", "join_year"])
    df = df.merge(tf, left_on=["team", "year"], right_on=["team", "join_year"], how="left")
    return df.drop(columns=["join_year"])


def join_lineup_quality(df: pd.DataFrame, lineups: pd.DataFrame, batter_stats: pd.DataFrame) -> pd.DataFrame:
    """Actual starting-lineup quality (PA-unweighted average prior-year
    wRC+ of the 9 batters who actually started), matched by game_pk +
    is_home so it lines up with `df`'s own team-perspective rows.

    At prediction time no lineup is known yet (lineups are usually posted
    only a few hours pre-game and nothing in this pipeline fetches them
    live), so this falls back to the team's season-average wRC+ — already
    present as the plain `wrc_plus` column from join_team_batting(), which
    must run before this. Historical training rows get the real lineup
    signal; prediction rows degrade gracefully to the team average.
    """
    if "wrc_plus" not in df.columns:
        raise ValueError("join_lineup_quality requires join_team_batting() to have run first")

    bs = batter_stats.copy()
    bs["join_year"] = bs["year"].astype(int) + 1
    bs = bs[["batter_mlbam_id", "join_year", "wrc_plus"]].rename(columns={"wrc_plus": "batter_wrc_plus"})
    bs = bs.drop_duplicates(subset=["batter_mlbam_id", "join_year"])

    lu = lineups.copy()
    lu["game_pk"] = lu["game_pk"].astype(str)
    lu = lu.merge(bs, left_on=["batter_mlbam_id", "year"], right_on=["batter_mlbam_id", "join_year"], how="left")
    lu["batter_wrc_plus"] = lu["batter_wrc_plus"].fillna(100.0)  # league-average wRC+ for unknown/rookie batters

    lineup_quality = (
        lu.groupby(["game_pk", "is_home"])["batter_wrc_plus"].mean()
        .reset_index().rename(columns={"batter_wrc_plus": "lineup_wrc_plus"})
    )

    df = df.copy()
    df["game_pk"] = df["game_pk"].astype(str)
    df = df.merge(lineup_quality, on=["game_pk", "is_home"], how="left")
    df["lineup_wrc_plus"] = df["lineup_wrc_plus"].fillna(df["wrc_plus"])
    return df


def join_player_bio(df: pd.DataFrame, player_bio: pd.DataFrame, lineups: pd.DataFrame) -> pd.DataFrame:
    """Starter handedness (home_sp_throws/away_sp_throws) and a
    platoon_advantage feature: the fraction of a team's actual starting
    lineup that has the platoon advantage (bats opposite-handed) against
    the OPPOSING starter. home_starter/away_starter and player_bio's
    full_name both come from the MLB Stats API, so — unlike the
    FanGraphs-vs-MLB matching join_sp_stats() has to do — a direct name
    match here is reliable.
    """
    bio = player_bio.copy()
    throw_lookup = bio[["full_name", "throw_hand"]].dropna().drop_duplicates(subset=["full_name"])

    df = df.merge(
        throw_lookup.rename(columns={"full_name": "home_starter", "throw_hand": "home_sp_throws"}),
        on="home_starter", how="left",
    )
    df = df.merge(
        throw_lookup.rename(columns={"full_name": "away_starter", "throw_hand": "away_sp_throws"}),
        on="away_starter", how="left",
    )

    lu = lineups.copy()
    lu["game_pk"] = lu["game_pk"].astype(str)
    bat_lookup = bio[["mlbam_id", "bat_side"]].dropna().drop_duplicates(subset=["mlbam_id"])
    lu = lu.merge(bat_lookup, left_on="batter_mlbam_id", right_on="mlbam_id", how="left")
    lu["bats_left"] = (lu["bat_side"] == "L").astype(float)
    lineup_hand = (
        lu.groupby(["game_pk", "is_home"])["bats_left"].mean()
        .reset_index().rename(columns={"bats_left": "lineup_pct_left"})
    )

    df["game_pk"] = df["game_pk"].astype(str)
    df = df.merge(lineup_hand, on=["game_pk", "is_home"], how="left")
    df["lineup_pct_left"] = df["lineup_pct_left"].fillna(0.40)  # roughly league-average LHB share

    opp_sp_throws = np.where(df["is_home"] == 1, df.get("away_sp_throws"), df.get("home_sp_throws"))
    # Opposing lefty -> platoon advantage is having more RHB (1 - pct_left); opposing righty -> more LHB.
    df["platoon_advantage"] = np.where(opp_sp_throws == "L", 1 - df["lineup_pct_left"], df["lineup_pct_left"])
    return df.drop(columns=["lineup_pct_left"])


def join_pitcher_rolling_form(df: pd.DataFrame, pitcher_gamelogs: pd.DataFrame) -> pd.DataFrame:
    """Each starter's rolling ERA/FIP over their last 3 starts, as of (but
    not including) the game in `df` — captures a pitcher coming off a rough
    stretch, which a static season average can't. Uses a merge_asof (each
    game gets the starter's most recently *completed* rolling value) since
    this is inherently a date-based lookup, not a year-based one like the
    other joins.
    """
    pg = pitcher_gamelogs.copy()
    pg["date"] = pd.to_datetime(pg["date"])
    pg = pg.sort_values(["pitcher_name", "date"])

    g = pg.groupby("pitcher_name", group_keys=False)
    shifted_era = g["era"].shift(1)
    shifted_fip = g["fip"].shift(1)
    pg["era_roll3"] = shifted_era.groupby(pg["pitcher_name"]).transform(lambda s: s.rolling(3, min_periods=1).mean())
    pg["fip_roll3"] = shifted_fip.groupby(pg["pitcher_name"]).transform(lambda s: s.rolling(3, min_periods=1).mean())

    rolling_lookup = pg[["pitcher_name", "date", "era_roll3", "fip_roll3"]].dropna(subset=["date"])

    df = df.copy()
    df["_date_dt"] = pd.to_datetime(df["date"])

    for side, starter_col in (("home", "home_starter"), ("away", "away_starter")):
        side_lookup = rolling_lookup.rename(columns={"pitcher_name": starter_col})
        matched = pd.merge_asof(
            df[["_date_dt", starter_col]].reset_index().sort_values("_date_dt"),
            side_lookup.sort_values("date"),
            left_on="_date_dt", right_on="date", by=starter_col, direction="backward",
            allow_exact_matches=False,
        ).set_index("index")
        df[f"{side}_sp_era_roll3"] = matched["era_roll3"]
        df[f"{side}_sp_fip_roll3"] = matched["fip_roll3"]

    return df.drop(columns=["_date_dt"])


def join_park_factors(df: pd.DataFrame, park_factors: pd.DataFrame) -> pd.DataFrame:
    """park_factor (and park_factor_h, if available) for the game's host
    (home) team. Host is `team` when is_home==1, else `opponent`.
    """
    pf = park_factors.copy()
    pf["team"] = normalize_team_abbrev(pf["team"])
    keep_cols = ["team", "year", "park_factor"]
    if "park_factor_h" in pf.columns:
        keep_cols.append("park_factor_h")
    else:
        logger.warning("park_factor_h not present in park_factors data (fg_park scrape unavailable) — omitted.")
    pf = pf[keep_cols].drop_duplicates(subset=["team", "year"])

    df = df.copy()
    df["_host_team"] = np.where(df["is_home"] == 1, df["team"], df["opponent"])
    df = df.merge(pf.rename(columns={"team": "_host_team"}), on=["_host_team", "year"], how="left")
    return df.drop(columns=["_host_team"])


def join_umpires(df: pd.DataFrame, umpire_factors: pd.DataFrame) -> pd.DataFrame:
    """Join umpire_run_factor per game via a (game_pk -> umpire_run_factor)
    lookup table (already merged from umpire_assignments + umpire_factors).
    """
    uf = umpire_factors[["game_pk", "umpire_run_factor"]].drop_duplicates(subset=["game_pk"])
    uf["game_pk"] = uf["game_pk"].astype(str)
    df = df.copy()
    df["game_pk"] = df["game_pk"].astype(str)
    return df.merge(uf, on="game_pk", how="left")


def pivot_to_game_level(df: pd.DataFrame) -> pd.DataFrame:
    """Pivots the long (team-per-game) table to one row per game, home-team
    perspective: TEAM_LEVEL_COLS get home_/away_ prefixes from the
    respective perspective row, GAME_LEVEL_COLS (already identical on both
    rows) are carried through once, and home_win is taken from the
    is_home==1 row's `win` (NaN for future/unplayed games, e.g. at
    prediction time). Shared by 01_build_features.py and 04_predict.py so
    the two never diverge on how a game row is assembled.
    """
    present_team_cols = [c for c in TEAM_LEVEL_COLS if c in df.columns]
    present_game_cols = [c for c in GAME_LEVEL_COLS if c in df.columns and c != "game_pk"]

    home = df[df["is_home"] == 1].copy()
    away = df[df["is_home"] == 0].copy()

    home_select = (
        ["game_pk"] + present_team_cols + present_game_cols
        + ["team", "opponent", "win", "elo_rating", "opp_elo_rating", "elo_diff", "elo_win_prob"]
    )
    home_renamed = home[home_select].rename(
        columns={**{c: f"home_{c}" for c in present_team_cols}, **{
            "team": "home_team", "opponent": "away_team", "win": "home_win",
            "elo_rating": "home_elo_rating", "opp_elo_rating": "away_elo_rating",
        }}
    )
    away_renamed = away[["game_pk"] + present_team_cols].rename(columns={c: f"away_{c}" for c in present_team_cols})

    game_df = home_renamed.merge(away_renamed, on="game_pk", how="inner")

    feature_cols = sorted(set(
        [f"home_{c}" for c in present_team_cols] + [f"away_{c}" for c in present_team_cols]
        + [c for c in present_game_cols if c not in (
            "game_pk", "date", "year", "month", "day_of_week", "home_starter", "away_starter")]
        + ["elo_diff", "elo_win_prob"]
    ))
    return game_df, feature_cols


def coverage_check(df: pd.DataFrame, feature_cols: list, min_coverage: float = 0.95,
                    train_years=(2021, 2022, 2023, 2024, 2025)) -> list:
    """Checks non-null coverage of each candidate feature column, restricted
    to `train_years` rows. Prints a coverage table, logs a WARNING for any
    column under `min_coverage`, and returns only the columns that pass
    (never silently drops without logging).
    """
    scoped = df[df["year"].isin(train_years)]
    n = len(scoped)
    passed = []
    logger.info(f"Coverage check on {n} rows spanning years {sorted(train_years)}:")
    for col in feature_cols:
        if col not in scoped.columns:
            logger.warning(f"  {col}: MISSING from dataframe entirely — excluded.")
            continue
        coverage = scoped[col].notna().mean() if n > 0 else 0.0
        status = "PASS" if coverage >= min_coverage else "FAIL"
        logger.info(f"  {col:30s} {coverage:6.1%}  [{status}]")
        if coverage >= min_coverage:
            passed.append(col)
        else:
            logger.warning(f"  {col}: coverage {coverage:.1%} < {min_coverage:.0%} threshold — excluded from model.")
    return passed
