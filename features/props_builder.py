# -*- coding: utf-8 -*-
"""Single source of truth for player-props feature construction: batter
markets (hits, HR, total bases, RBI, runs, walks, strikeouts, stolen
bases) and pitcher markets (strikeouts, walks, hits/runs allowed).

Grain is PLAYER x GAME (one row per batter/pitcher per game, see
R/14_collect_boxscores.R's batter_gamelogs_%d.csv / pitcher_boxscore_%d.csv)
— simpler than features/builder.py's team-perspective long format, since a
prop is one-sided and needs no home/away pivot.

A player's own counting stats on a given date serve double duty: they are
that row's prediction TARGET, and — via the .shift(1) in
build_batter_rolling_form()/build_pitcher_rolling_form() — the raw material
for EVERY LATER row's rolling-form features. Every function below that
reads those columns as a feature source shifts by player first; a row's
own label never leaks into its own features. See
scripts/props_build_features.py (batter) and
scripts/props_build_pitcher_features.py (pitcher) for the orchestration
that will eventually feed both training and daily prediction for props,
exactly as 01_build_features.py / 04_predict.py share features/builder.py
today.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from loguru import logger

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config.config import LEAGUE_AVG_SP, TEAM_ABBREV_MAP, STADIUM_COORDS  # noqa: E402

# Raw per-game counting stats a batter's own row carries. These are BOTH
# this row's prediction targets AND (after build_batter_rolling_form's
# shift) the source of every other row's rolling features — never treat
# them as features for the SAME row.
BATTER_LABEL_COLS = ["pa", "ab", "h", "doubles", "triples", "hr", "tb", "rbi", "r", "bb", "so", "sb", "hbp"]

BATTER_ROLLING_STATS = ["pa", "ab", "h", "doubles", "triples", "hr", "tb", "rbi", "r", "bb", "so", "sb", "hbp"]
BATTER_ROLLING_WINDOWS = (7, 20)

# All engineered/joined columns a props model can train on. Kept explicit
# (rather than "everything not in LABEL_COLS") so a coverage_check() run
# and a training script have one obvious list to iterate, mirroring
# features/builder.py's TEAM_LEVEL_COLS / GAME_LEVEL_COLS.
BATTER_FEATURE_COLS = (
    [f"roll_{stat}_{w}" for stat in BATTER_ROLLING_STATS for w in BATTER_ROLLING_WINDOWS]
    + ["season_pa", "season_wrc_plus", "season_woba", "season_obp", "season_slg", "season_iso",
       "season_bb_pct", "season_k_pct", "season_barrel_pct", "season_hard_hit_pct", "season_xwoba"]
    + ["batter_age", "batter_bats_L", "batter_bats_R", "batter_bats_S",
       "opp_sp_throws_L", "opp_sp_throws_R", "platoon_advantage"]
    + ["opp_sp_era", "opp_sp_fip", "opp_sp_xfip", "opp_sp_siera", "opp_sp_k_pct", "opp_sp_bb_pct",
       "opp_sp_whip", "opp_sp_k9", "opp_sp_velo", "opp_sp_whiff_pct", "opp_sp_known",
       "opp_sp_era_roll3", "opp_sp_fip_roll3"]
    + ["park_factor", "umpire_run_factor"]
)

PITCHER_LABEL_COLS = ["outs", "batters_faced", "k", "bb", "h", "hr", "r", "er", "pitches"]

PITCHER_ROLLING_STATS = ["k", "bb", "h", "hr", "r", "er", "outs", "batters_faced", "pitches"]
PITCHER_ROLLING_WINDOWS = (3, 10)

PITCHER_FEATURE_COLS = (
    [f"roll_{stat}_{w}" for stat in PITCHER_ROLLING_STATS for w in PITCHER_ROLLING_WINDOWS]
    + ["sp_era", "sp_fip", "sp_xfip", "sp_siera", "sp_k_pct", "sp_bb_pct", "sp_whip", "sp_k9",
       "sp_velo", "sp_whiff_pct", "sp_known",
       "sp_spin_rate_avg", "sp_extension_avg", "sp_xfip_statcast", "sp_barrel_pct_against"]
    + ["pitcher_age", "pitcher_throws_L", "pitcher_throws_R"]
    + ["opp_lineup_wrc_plus", "opp_lineup_k_pct", "opp_team_wrc_plus", "opp_team_k_pct", "opp_team_bb_pct"]
    + ["park_factor", "umpire_run_factor"]
)


def normalize_team_abbrev(series: pd.Series) -> pd.Series:
    return series.astype(str).str.strip().str.upper().map(TEAM_ABBREV_MAP).fillna(series)


def build_batter_rolling_form(df: pd.DataFrame) -> pd.DataFrame:
    """Rolling per-game batter counting stats (mean over the last N games),
    computed with .shift(1) so a game's features never see that game's own
    line — the same pattern as features/builder.py's build_rolling_features,
    grouped by batter_mlbam_id instead of team. Windows do not reset across
    season boundaries (a batter's most recent games from last season are
    still their most recent form entering a new one).
    """
    df = df.sort_values(["batter_mlbam_id", "date"]).reset_index(drop=True)
    g = df.groupby("batter_mlbam_id", group_keys=False)

    for stat in BATTER_ROLLING_STATS:
        shifted = g[stat].shift(1)
        for window in BATTER_ROLLING_WINDOWS:
            df[f"roll_{stat}_{window}"] = (
                shifted.groupby(df["batter_mlbam_id"]).transform(lambda s: s.rolling(window, min_periods=1).mean())
            )

    return df


def join_batter_season_stats(df: pd.DataFrame, batter_stats: pd.DataFrame) -> pd.DataFrame:
    """Prior-year FanGraphs individual batter stats. Columns are prefixed
    `season_` (including `season_pa`) so they're never confused with this
    row's own in-game pa/ab from LABEL_COLS.
    """
    bs = batter_stats.copy()
    bs["join_year"] = bs["year"].astype(int) + 1
    keep = {
        "pa": "season_pa", "wrc_plus": "season_wrc_plus", "woba": "season_woba",
        "obp": "season_obp", "slg": "season_slg", "iso": "season_iso",
        "bb_pct": "season_bb_pct", "k_pct": "season_k_pct",
        "barrel_pct": "season_barrel_pct", "hard_hit_pct": "season_hard_hit_pct",
        "xwoba": "season_xwoba",
    }
    bs = bs[["batter_mlbam_id", "join_year"] + list(keep.keys())].rename(columns=keep)
    bs = bs.drop_duplicates(subset=["batter_mlbam_id", "join_year"])

    df = df.merge(bs, left_on=["batter_mlbam_id", "year"], right_on=["batter_mlbam_id", "join_year"], how="left")
    return df.drop(columns=["join_year"])


def attach_opposing_starter(df: pd.DataFrame, game_logs: pd.DataFrame) -> pd.DataFrame:
    """Looks up the opposing starting pitcher's name for each batter-game
    row via game_logs' home_starter/away_starter (one row per game_pk,
    both columns identical across that game's two team-perspective rows).
    Every later opposing-pitcher join keys off this column.
    """
    starters = (
        game_logs[["game_pk", "home_starter", "away_starter"]]
        .assign(game_pk=lambda x: x["game_pk"].astype(str))
        .drop_duplicates(subset=["game_pk"])
    )
    df = df.copy()
    df["game_pk"] = df["game_pk"].astype(str)
    df = df.merge(starters, on="game_pk", how="left")
    df["opp_starter_name"] = np.where(df["is_home"] == 1, df["away_starter"], df["home_starter"])
    return df.drop(columns=["home_starter", "away_starter"])


def join_batter_bio_and_platoon(df: pd.DataFrame, player_bio: pd.DataFrame) -> pd.DataFrame:
    """Batter age (as of game date) and bat side via batter_mlbam_id, plus
    the opposing starter's throw hand via attach_opposing_starter()'s
    opp_starter_name (MLB Stats API name on both sides, so — like
    features/builder.py's join_player_bio — a direct name match is
    reliable). platoon_advantage is 1 when the batter bats opposite-handed
    from the opposing starter (switch hitters always get the advantage).
    Requires attach_opposing_starter() to have run first.
    """
    if "opp_starter_name" not in df.columns:
        raise ValueError("join_batter_bio_and_platoon requires attach_opposing_starter() to have run first")

    bio = player_bio.copy()
    bio["birth_date"] = pd.to_datetime(bio["birth_date"])

    bat_bio = bio[["mlbam_id", "birth_date", "bat_side"]].dropna(subset=["mlbam_id"]).drop_duplicates(subset=["mlbam_id"])
    df = df.merge(bat_bio, left_on="batter_mlbam_id", right_on="mlbam_id", how="left")
    df["batter_age"] = (pd.to_datetime(df["date"]) - df["birth_date"]).dt.days / 365.25
    df = df.rename(columns={"bat_side": "batter_bats"}).drop(columns=["mlbam_id", "birth_date"])

    throw_lookup = bio[["full_name", "throw_hand"]].dropna().drop_duplicates(subset=["full_name"])
    df = df.merge(
        throw_lookup.rename(columns={"full_name": "opp_starter_name", "throw_hand": "opp_sp_throws"}),
        on="opp_starter_name", how="left",
    )

    both_known = df["batter_bats"].notna() & df["opp_sp_throws"].notna()
    opposite = df["batter_bats"] != df["opp_sp_throws"]
    df["platoon_advantage"] = np.where(both_known, ((df["batter_bats"] == "S") | opposite).astype(float), np.nan)

    # One-hot with FIXED categories (not pd.get_dummies, which would produce
    # inconsistent columns between train/test slices if a category happens
    # to be absent from one) — batter_bats/opp_sp_throws stay in the output
    # as human-readable strings, but a regressor needs numeric input, and
    # these _L/_R/_S indicators are what BATTER_FEATURE_COLS actually lists.
    df["batter_bats_L"] = (df["batter_bats"] == "L").astype(float)
    df["batter_bats_R"] = (df["batter_bats"] == "R").astype(float)
    df["batter_bats_S"] = (df["batter_bats"] == "S").astype(float)
    df["opp_sp_throws_L"] = (df["opp_sp_throws"] == "L").astype(float)
    df["opp_sp_throws_R"] = (df["opp_sp_throws"] == "R").astype(float)
    return df


def join_opposing_pitcher_stats(df: pd.DataFrame, fg_sp: pd.DataFrame, statcast_sp: pd.DataFrame) -> pd.DataFrame:
    """Prior-year season stats (fg_sp + statcast_sp) for the opposing
    starter, with league-average imputation and an opp_sp_known flag —
    same pattern as features/builder.py's join_sp_stats, but for one side
    only. Matches on pitcher_name, which means the same FanGraphs-vs-MLB
    Stats API name-matching imperfection join_sp_stats already has;
    opp_sp_known distinguishes a real match from an imputed one.
    Requires attach_opposing_starter() to have run first.
    """
    if "opp_starter_name" not in df.columns:
        raise ValueError("join_opposing_pitcher_stats requires attach_opposing_starter() to have run first")

    fg = fg_sp.copy()
    fg["pitcher_name"] = fg["pitcher_name"].astype(str)
    sc = statcast_sp.copy()
    sc["pitcher_name"] = sc["pitcher_name"].astype(str)

    merged = fg.merge(sc[["pitcher_name", "year", "velo_avg", "whiff_pct"]], on=["pitcher_name", "year"], how="outer")
    merged["join_year"] = merged["year"].astype(int) + 1

    raw_stat_cols = ["era", "fip", "xfip", "siera", "k_pct", "bb_pct", "whip", "k9", "velo_avg", "whiff_pct"]
    prefixed_cols = {c: f"opp_sp_{c}" if c != "velo_avg" else "opp_sp_velo" for c in raw_stat_cols}
    stat_cols = list(prefixed_cols.values())

    side_lookup = merged.rename(columns={"join_year": "_yr", **prefixed_cols})
    side_lookup = side_lookup.drop_duplicates(subset=["pitcher_name", "_yr"])
    side_lookup = side_lookup.rename(columns={"pitcher_name": "opp_starter_name"})

    df = df.merge(
        side_lookup[["opp_starter_name", "_yr"] + stat_cols],
        left_on=["opp_starter_name", "year"], right_on=["opp_starter_name", "_yr"], how="left",
    )
    df = df.drop(columns=["_yr"])

    df["opp_sp_known"] = df[stat_cols].notna().any(axis=1).astype(int)
    years = df["year"].fillna(-1).astype(int)
    for raw_col, prefixed_col in prefixed_cols.items():
        league_default = years.map(lambda y: LEAGUE_AVG_SP.get(y, LEAGUE_AVG_SP[max(LEAGUE_AVG_SP)]).get(raw_col, np.nan))
        df[prefixed_col] = df[prefixed_col].fillna(league_default)

    return df


def join_opposing_pitcher_rolling(df: pd.DataFrame, pitcher_gamelogs: pd.DataFrame) -> pd.DataFrame:
    """The opposing starter's rolling ERA/FIP over their last 3 starts, as
    of (but not including) this game — same merge_asof pattern as
    features/builder.py's join_pitcher_rolling_form, one side only.
    Requires attach_opposing_starter() to have run first.
    """
    if "opp_starter_name" not in df.columns:
        raise ValueError("join_opposing_pitcher_rolling requires attach_opposing_starter() to have run first")

    pg = pitcher_gamelogs.copy()
    pg["date"] = pd.to_datetime(pg["date"])
    pg = pg.sort_values(["pitcher_name", "date"])

    g = pg.groupby("pitcher_name", group_keys=False)
    pg["era_roll3"] = g["era"].transform(lambda s: s.rolling(3, min_periods=1).mean())
    pg["fip_roll3"] = g["fip"].transform(lambda s: s.rolling(3, min_periods=1).mean())

    rolling_lookup = pg[["pitcher_name", "date", "era_roll3", "fip_roll3"]].dropna(subset=["date"])
    rolling_lookup = rolling_lookup.rename(columns={"pitcher_name": "opp_starter_name"})

    df = df.copy()
    df["_date_dt"] = pd.to_datetime(df["date"])
    matched = pd.merge_asof(
        df[["_date_dt", "opp_starter_name"]].reset_index().sort_values("_date_dt"),
        rolling_lookup.sort_values("date"),
        left_on="_date_dt", right_on="date", by="opp_starter_name", direction="backward",
        allow_exact_matches=False,
    ).set_index("index")
    df["opp_sp_era_roll3"] = matched["era_roll3"]
    df["opp_sp_fip_roll3"] = matched["fip_roll3"]
    return df.drop(columns=["_date_dt"])


def join_park_factors(df: pd.DataFrame, park_factors: pd.DataFrame) -> pd.DataFrame:
    """park_factor for the game's host (home) team. Host is `team` when
    is_home==1, else `opp` — same logic as features/builder.py's
    join_park_factors, adapted to this module's `opp` column name
    (batter_gamelogs_%d.csv calls it `opp`, game_logs_%d.csv calls it
    `opponent`).
    """
    pf = park_factors.copy()
    pf["team"] = normalize_team_abbrev(pf["team"])
    pf = pf[["team", "year", "park_factor"]].drop_duplicates(subset=["team", "year"])
    pf["year"] = pf["year"] + 1

    df = df.copy()
    df["_host_team"] = np.where(df["is_home"] == 1, df["team"], df["opp"])
    df = df.merge(pf.rename(columns={"team": "_host_team"}), on=["_host_team", "year"], how="left")
    return df.drop(columns=["_host_team"])


def join_umpires(df: pd.DataFrame, umpire_lookup: pd.DataFrame) -> pd.DataFrame:
    """Join umpire_run_factor per game_pk — identical to
    features/builder.py's join_umpires (game_pk is a common key in both
    modules), kept here so props_builder.py has no runtime dependency on
    the moneyline module.
    """
    uf = umpire_lookup[["game_pk", "umpire_run_factor"]].drop_duplicates(subset=["game_pk"])
    uf["game_pk"] = uf["game_pk"].astype(str)
    df = df.copy()
    df["game_pk"] = df["game_pk"].astype(str)
    return df.merge(uf, on="game_pk", how="left")


# --------------------------------------------------------------------------
# Pitcher markets (strikeouts, walks, hits/runs allowed). Grain is
# pitcher_boxscore_%d.csv (pitcher x game). join_park_factors() and
# join_umpires() above are reused as-is for pitchers — both already operate
# on the generic team/opp/is_home/game_pk columns pitcher_boxscore shares
# with batter_gamelogs.
# --------------------------------------------------------------------------

def build_pitcher_rolling_form(pitcher_boxscore: pd.DataFrame) -> pd.DataFrame:
    """Rolling per-start pitcher counting stats (mean over the last N
    starts), computed with .shift(1) exactly like build_batter_rolling_form,
    grouped by pitcher_mlbam_id. Restricted to actual starts (is_starter==1)
    before rolling — pitcher_boxscore_%d.csv also carries relief
    appearances, and mixing sporadic relief workload into a starter's
    rolling window would blur what is really a different role. Unlike
    features/builder.py's join_pitcher_rolling_form (which has to match on
    pitcher_name against a separate FanGraphs game-log source),
    pitcher_boxscore is keyed on the same MLBAM id this whole table already
    uses, so this rolling form needs no name matching at all.
    """
    df = pitcher_boxscore[pitcher_boxscore["is_starter"] == 1].copy()
    df = df.sort_values(["pitcher_mlbam_id", "date"]).reset_index(drop=True)
    g = df.groupby("pitcher_mlbam_id", group_keys=False)

    for stat in PITCHER_ROLLING_STATS:
        shifted = g[stat].shift(1)
        for window in PITCHER_ROLLING_WINDOWS:
            df[f"roll_{stat}_{window}"] = (
                shifted.groupby(df["pitcher_mlbam_id"]).transform(lambda s: s.rolling(window, min_periods=1).mean())
            )

    return df


def join_pitcher_season_stats(df: pd.DataFrame, fg_sp: pd.DataFrame, statcast_sp: pd.DataFrame) -> pd.DataFrame:
    """Prior-year season stats for the pitcher's OWN performance (fg_sp +
    statcast_sp), matched on pitcher_name + (year+1) — same
    FanGraphs-vs-MLB-Stats-API name-matching caveat as
    features/builder.py's join_sp_stats, tracked via the sp_known flag.
    FanGraphs-covered fields get the same LEAGUE_AVG_SP fallback
    imputation as join_sp_stats/join_opposing_pitcher_stats; the
    Statcast-only quality metrics (spin rate, extension, barrel% against,
    Statcast-derived xFIP) have no league-average constant to fall back to
    and are left for scripts/02_train.py-style median imputation at
    training time.
    """
    fg = fg_sp.copy()
    fg["pitcher_name"] = fg["pitcher_name"].astype(str)
    sc = statcast_sp.copy()
    sc["pitcher_name"] = sc["pitcher_name"].astype(str)

    merged = fg.merge(sc, on=["pitcher_name", "year"], how="outer")
    merged["join_year"] = merged["year"].astype(int) + 1

    imputed_raw_cols = ["era", "fip", "xfip", "siera", "k_pct", "bb_pct", "whip", "k9", "velo_avg", "whiff_pct"]
    extra_raw_cols = ["spin_rate_avg", "extension_avg", "xfip_statcast", "barrel_pct_against"]
    prefixed = {c: f"sp_{c}" if c != "velo_avg" else "sp_velo" for c in imputed_raw_cols + extra_raw_cols}
    stat_cols = list(prefixed.values())

    side_lookup = merged.rename(columns={"join_year": "_yr", **prefixed})
    side_lookup = side_lookup.drop_duplicates(subset=["pitcher_name", "_yr"])

    df = df.merge(
        side_lookup[["pitcher_name", "_yr"] + stat_cols],
        left_on=["pitcher_name", "year"], right_on=["pitcher_name", "_yr"], how="left",
    ).drop(columns=["_yr"])

    imputed_prefixed_cols = [prefixed[c] for c in imputed_raw_cols]
    df["sp_known"] = df[imputed_prefixed_cols].notna().any(axis=1).astype(int)
    years = df["year"].fillna(-1).astype(int)
    for raw_col in imputed_raw_cols:
        prefixed_col = prefixed[raw_col]
        league_default = years.map(lambda y: LEAGUE_AVG_SP.get(y, LEAGUE_AVG_SP[max(LEAGUE_AVG_SP)]).get(raw_col, np.nan))
        df[prefixed_col] = df[prefixed_col].fillna(league_default)

    return df


def join_pitcher_bio(df: pd.DataFrame, player_bio: pd.DataFrame) -> pd.DataFrame:
    """Pitcher age (as of game date) and throw hand via pitcher_mlbam_id.
    pitcher_throws stays as a human-readable string; pitcher_throws_L/_R
    (fixed categories, not pd.get_dummies) are what PITCHER_FEATURE_COLS
    actually lists, since a regressor needs numeric input.
    """
    bio = player_bio.copy()
    bio["birth_date"] = pd.to_datetime(bio["birth_date"])
    lookup = bio[["mlbam_id", "birth_date", "throw_hand"]].dropna(subset=["mlbam_id"]).drop_duplicates(subset=["mlbam_id"])

    df = df.merge(lookup, left_on="pitcher_mlbam_id", right_on="mlbam_id", how="left")
    df["pitcher_age"] = (pd.to_datetime(df["date"]) - df["birth_date"]).dt.days / 365.25
    df = df.rename(columns={"throw_hand": "pitcher_throws"}).drop(columns=["mlbam_id", "birth_date"])
    df["pitcher_throws_L"] = (df["pitcher_throws"] == "L").astype(float)
    df["pitcher_throws_R"] = (df["pitcher_throws"] == "R").astype(float)
    return df


def join_opponent_team_batting(df: pd.DataFrame, fg_team_batting: pd.DataFrame) -> pd.DataFrame:
    """Prior-year team-level batting tendency for the opponent the pitcher
    is facing (wRC+, K%, BB%) — team-wide context/fallback alongside
    join_opposing_lineup_quality()'s game-specific actual-lineup signal.
    Must run before join_opposing_lineup_quality() so that function's
    fallback (opp_team_wrc_plus / opp_team_k_pct) is available.
    """
    fgb = fg_team_batting.copy()
    fgb["team"] = normalize_team_abbrev(fgb["team"])
    fgb["join_year"] = fgb["year"].astype(int) + 1
    cols = ["team", "join_year", "wrc_plus", "k_pct", "bb_pct"]
    fgb = fgb[cols].drop_duplicates(subset=["team", "join_year"]).rename(columns={
        "team": "_opp_team_key", "wrc_plus": "opp_team_wrc_plus",
        "k_pct": "opp_team_k_pct", "bb_pct": "opp_team_bb_pct",
    })
    df = df.merge(fgb, left_on=["opp", "year"], right_on=["_opp_team_key", "join_year"], how="left")
    return df.drop(columns=["_opp_team_key", "join_year"])


def join_opposing_lineup_quality(df: pd.DataFrame, lineups: pd.DataFrame, batter_stats: pd.DataFrame) -> pd.DataFrame:
    """Actual opposing starting-lineup quality the pitcher faces that game:
    PA-unweighted average prior-year wRC+ and K% of the 9 batters who
    actually started against them — mirrors features/builder.py's
    join_lineup_quality, but computed for the lineup on the OTHER side of
    this pitcher's own team/opp/is_home. A high-strikeout-prone opposing
    lineup is one of the most direct signals for a strikeouts prop.

    At prediction time no lineup is known yet (same caveat as
    join_lineup_quality), so this falls back to join_opponent_team_batting's
    prior-year TEAM average, which must run first.
    """
    if "opp_team_wrc_plus" not in df.columns:
        raise ValueError("join_opposing_lineup_quality requires join_opponent_team_batting() to have run first")

    bs = batter_stats.copy()
    bs["join_year"] = bs["year"].astype(int) + 1
    bs = bs[["batter_mlbam_id", "join_year", "wrc_plus", "k_pct"]].rename(
        columns={"wrc_plus": "batter_wrc_plus", "k_pct": "batter_k_pct"})
    bs = bs.drop_duplicates(subset=["batter_mlbam_id", "join_year"])

    lu = lineups.copy()
    lu["game_pk"] = lu["game_pk"].astype(str)
    lu = lu.merge(bs, left_on=["batter_mlbam_id", "year"], right_on=["batter_mlbam_id", "join_year"], how="left")

    lineup_quality = (
        lu.groupby(["game_pk", "is_home"])[["batter_wrc_plus", "batter_k_pct"]].mean()
        .reset_index().rename(columns={"batter_wrc_plus": "opp_lineup_wrc_plus", "batter_k_pct": "opp_lineup_k_pct"})
    )

    df = df.copy()
    df["game_pk"] = df["game_pk"].astype(str)
    df["_opp_is_home"] = 1 - df["is_home"]
    df = df.merge(
        lineup_quality.rename(columns={"is_home": "_opp_is_home"}),
        on=["game_pk", "_opp_is_home"], how="left",
    )
    df = df.drop(columns=["_opp_is_home"])

    df["opp_lineup_wrc_plus"] = df["opp_lineup_wrc_plus"].fillna(df["opp_team_wrc_plus"])
    df["opp_lineup_k_pct"] = df["opp_lineup_k_pct"].fillna(df["opp_team_k_pct"])
    return df
