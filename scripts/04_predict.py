# -*- coding: utf-8 -*-
"""Daily prediction pipeline: fetch today's schedule + probable starters
from the MLB Stats API, build features via features/builder.py (the same
functions used in training), score with the calibrated XGB+LGBM average,
pull moneyline odds from The Odds API, and flag edges against the
no-vig market-implied probability.

Usage: python scripts/04_predict.py --date YYYY-MM-DD
"""
import sys
import json
import argparse
import unicodedata
from pathlib import Path
from datetime import datetime
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
import requests
import joblib
from loguru import logger
from dotenv import load_dotenv
import os

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from config.config import PATHS, MIN_EDGE_MONEYLINE, MAX_EDGE_MONEYLINE, TEAM_ABBREV_MAP  # noqa: E402
from model_classes import WeightedEnsembleClassifier, PreFitCalibratedClassifier  # noqa: E402
from odds_utils import aggregate_h2h_event  # noqa: E402
from features import builder  # noqa: E402

RAW = PATHS["raw"]
MODELS = PATHS["models"]
OUTPUTS = PATHS["outputs"]
LOGS = PATHS["logs"]

MLB_API = "https://statsapi.mlb.com/api/v1"
ODDS_API = "https://api.the-odds-api.com/v4"
MIN_BOOKMAKERS = 3

logger.remove()
logger.add(sys.stderr, level="INFO")
logger.add(LOGS / "04_predict.log", level="DEBUG", rotation="5 MB")


def strip_accents(name: str) -> str:
    if not isinstance(name, str):
        return name
    return "".join(c for c in unicodedata.normalize("NFKD", name) if not unicodedata.combining(c))


def fetch_schedule_with_probables(date: str) -> list:
    url = f"{MLB_API}/schedule"
    params = {"sportId": 1, "date": date, "hydrate": "probablePitcher,team,venue"}
    resp = requests.get(url, params=params, timeout=30)
    resp.raise_for_status()
    data = resp.json()

    games = []
    for date_entry in data.get("dates", []):
        for g in date_entry.get("games", []):
            if g.get("gameType") != "R":
                continue
            teams = g["teams"]
            home = teams["home"]["team"]
            away = teams["away"]["team"]
            home_probable = teams["home"].get("probablePitcher", {})
            away_probable = teams["away"].get("probablePitcher", {})
            games.append({
                "game_pk": str(g["gamePk"]),
                "date": date,
                "home_team_name": home.get("name"),
                "away_team_name": away.get("name"),
                "home_team_abbr": home.get("abbreviation"),
                "away_team_abbr": away.get("abbreviation"),
                "home_starter": strip_accents(home_probable.get("fullName")),
                "away_starter": strip_accents(away_probable.get("fullName")),
                "venue_name": g.get("venue", {}).get("name"),
            })
    return games


def build_today_rows(games: list, date: str) -> pd.DataFrame:
    dt = pd.to_datetime(date)
    rows = []
    for g in games:
        home = builder.normalize_team_abbrev(pd.Series([g["home_team_abbr"] or g["home_team_name"]])).iloc[0]
        away = builder.normalize_team_abbrev(pd.Series([g["away_team_abbr"] or g["away_team_name"]])).iloc[0]
        for team, opp, is_home in [(home, away, 1), (away, home, 0)]:
            rows.append({
                "date": date, "game_pk": g["game_pk"], "team": team, "opponent": opp,
                "is_home": is_home, "runs_scored": np.nan, "runs_allowed": np.nan, "win": np.nan,
                "home_starter": g["home_starter"], "away_starter": g["away_starter"],
                "year": dt.year, "month": dt.month, "day_of_week": dt.day_name()[:3],
            })
    return pd.DataFrame(rows)


def fetch_moneyline_odds() -> pd.DataFrame:
    api_key = os.getenv("ODDS_API_KEY")
    if not api_key or api_key == "your_odds_api_key_here":
        logger.warning("ODDS_API_KEY not set in .env — skipping live odds fetch.")
        return pd.DataFrame()

    url = f"{ODDS_API}/sports/baseball_mlb/odds"
    params = {"apiKey": api_key, "regions": "us", "markets": "h2h", "oddsFormat": "american"}
    try:
        resp = requests.get(url, params=params, timeout=30)
        resp.raise_for_status()
    except requests.RequestException as e:
        logger.warning(f"Odds API request failed: {e}")
        return pd.DataFrame()

    rows = []
    for event in resp.json():
        # Per-book de-vig then median across books — see odds_utils for why
        # this must never be an average of raw American prices, and why
        # fewer than MIN_BOOKMAKERS books means no usable consensus.
        agg = aggregate_h2h_event(event, min_bookmakers=MIN_BOOKMAKERS)
        if agg is None:
            continue

        # The odds feed returns every upcoming event, not just today's — if the
        # same two teams play on consecutive days (a common series), both
        # events match on team names alone. Convert commence_time to the US
        # Eastern civil date (how MLB defines "the game date," even for late
        # West Coast starts) so the merge can be scoped to the right day.
        commence = event.get("commence_time")
        event_date = None
        if commence:
            event_date = (
                datetime.fromisoformat(commence.replace("Z", "+00:00"))
                .astimezone(ZoneInfo("America/New_York"))
                .strftime("%Y-%m-%d")
            )
        rows.append({
            "home_team_name": event.get("home_team"), "away_team_name": event.get("away_team"),
            "event_date": event_date,
            "home_ml": agg["home_ml"], "away_ml": agg["away_ml"],
            "no_vig_home_implied": agg["no_vig_home_implied"],
        })
    return pd.DataFrame(rows)


def fetch_boxscore_lineups_and_umps(games: list, date: str):
    """Pull each game's MLB boxscore for the two pieces of pre-game info the
    morning pipeline is otherwise blind to: the posted starting lineup
    (battingOrder) and the home-plate umpire assignment.

    Both are empty early in the morning and fill in as game time approaches
    (lineups typically 2-4 hours before first pitch, umpires around the same
    time or earlier) — so a run close to game time gets the real values the
    model was trained on, while an 8 AM run degrades gracefully to the same
    team-average / train-median fallbacks as before. Returns
    (lineups_df, umps_df); either may be empty.
    """
    lineup_rows, ump_rows = [], []
    year = pd.to_datetime(date).year
    for g in games:
        try:
            resp = requests.get(f"{MLB_API}/game/{g['game_pk']}/boxscore", timeout=30)
            resp.raise_for_status()
            box = resp.json()
        except requests.RequestException as e:
            logger.warning(f"boxscore fetch failed for game {g['game_pk']}: {e}")
            continue

        for side, is_home in (("home", 1), ("away", 0)):
            batting_order = box.get("teams", {}).get(side, {}).get("battingOrder") or []
            for pid in batting_order:
                lineup_rows.append({
                    "game_pk": str(g["game_pk"]), "batter_mlbam_id": int(pid),
                    "is_home": is_home, "year": year,
                })

        for official in box.get("officials", []) or []:
            if official.get("officialType") == "Home Plate":
                name = strip_accents((official.get("official") or {}).get("fullName"))
                if name:
                    ump_rows.append({"game_pk": str(g["game_pk"]), "umpire_name": name})

    return pd.DataFrame(lineup_rows), pd.DataFrame(ump_rows)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--date", default=datetime.now().strftime("%Y-%m-%d"))
    args = parser.parse_args()
    date = args.date

    load_dotenv(PATHS["root"] / ".env")

    logger.info(f"Fetching schedule + probable starters for {date} ...")
    games = fetch_schedule_with_probables(date)
    if not games:
        logger.warning(f"No regular-season games found for {date}.")
        return
    logger.info(f"{len(games)} games scheduled.")

    history_path = RAW / "game_logs_all.csv"
    if not history_path.exists():
        logger.error(f"{history_path} not found — run the R collection pipeline first.")
        sys.exit(1)
    history = pd.read_csv(history_path)

    today_rows = build_today_rows(games, date)
    combined = pd.concat([history, today_rows], ignore_index=True)

    logger.info("Building features via features/builder.py (same functions as training) ...")
    df = builder.build_rolling_features(combined)
    df = builder.build_elo_ratings(df)
    df = builder.build_schedule_features(df)

    fg_batting = pd.concat(
        [pd.read_csv(p) for p in sorted(RAW.glob("fg_team_batting_*.csv"))], ignore_index=True
    ) if list(RAW.glob("fg_team_batting_*.csv")) else pd.DataFrame()
    fg_sp = pd.concat(
        [pd.read_csv(p) for p in sorted(RAW.glob("fg_sp_stats_*.csv"))], ignore_index=True
    ) if list(RAW.glob("fg_sp_stats_*.csv")) else pd.DataFrame()
    statcast_team = pd.concat(
        [pd.read_csv(p) for p in sorted(RAW.glob("statcast_team_batting_*.csv"))], ignore_index=True
    ) if list(RAW.glob("statcast_team_batting_*.csv")) else pd.DataFrame()
    statcast_sp = pd.concat(
        [pd.read_csv(p) for p in sorted(RAW.glob("statcast_sp_*.csv"))], ignore_index=True
    ) if list(RAW.glob("statcast_sp_*.csv")) else pd.DataFrame()
    team_fielding = pd.concat(
        [pd.read_csv(p) for p in sorted(RAW.glob("team_fielding_*.csv"))], ignore_index=True
    ) if list(RAW.glob("team_fielding_*.csv")) else pd.DataFrame()
    batter_stats = pd.concat(
        [pd.read_csv(p) for p in sorted(RAW.glob("batter_stats_*.csv"))], ignore_index=True
    ) if list(RAW.glob("batter_stats_*.csv")) else pd.DataFrame()
    lineups = pd.concat(
        [pd.read_csv(p) for p in sorted(RAW.glob("lineups_*.csv")) if "cache" not in p.name], ignore_index=True
    ) if list(RAW.glob("lineups_*.csv")) else pd.DataFrame()
    pitcher_gamelogs = pd.concat(
        [pd.read_csv(p) for p in sorted(RAW.glob("pitcher_gamelogs_*.csv"))], ignore_index=True
    ) if list(RAW.glob("pitcher_gamelogs_*.csv")) else pd.DataFrame()
    player_bio_path = RAW / "player_bio.csv"
    player_bio = pd.read_csv(player_bio_path) if player_bio_path.exists() else pd.DataFrame()

    if not fg_batting.empty:
        df = builder.join_team_batting(df, fg_batting)
    if not fg_sp.empty:
        df = builder.join_team_pitching(df, fg_sp)
        df = builder.join_sp_stats(df, fg_sp, statcast_sp)
    if not statcast_team.empty:
        df = builder.join_statcast_batting(df, statcast_team)
    if not team_fielding.empty:
        df = builder.join_team_fielding(df, team_fielding)
    # Posted lineups and umpire assignments for today's slate, straight from
    # the MLB boxscore feed. Empty in an early-morning run (falls back to
    # team-season wRC+ / train-median umpire factor exactly as before); real
    # values in a run within ~2-4 hours of first pitch — the same
    # information the model saw for its historical training rows.
    today_lineups, today_umps = fetch_boxscore_lineups_and_umps(games, date)
    logger.info(f"Live pre-game info: {today_lineups['game_pk'].nunique() if not today_lineups.empty else 0}"
                f"/{len(games)} games with posted lineups, "
                f"{len(today_umps)}/{len(games)} with home-plate umpire assigned.")
    if not today_lineups.empty:
        lineups = pd.concat([lineups, today_lineups], ignore_index=True) if not lineups.empty else today_lineups

    if not lineups.empty and not batter_stats.empty:
        df = builder.join_lineup_quality(df, lineups, batter_stats)
    if not player_bio.empty and not lineups.empty:
        df = builder.join_player_bio(df, player_bio, lineups)
    if not pitcher_gamelogs.empty:
        df = builder.join_pitcher_rolling_form(df, pitcher_gamelogs)

    park_factors_path = RAW / "park_factors.csv"
    if park_factors_path.exists():
        df = builder.join_park_factors(df, pd.read_csv(park_factors_path))

    # umpire_run_factor: today's live assignments (if posted yet) mapped
    # through the same historical umpire_factors table used in training.
    # Unassigned games stay NaN and fall back to train-set medians below,
    # same as any other missing feature.
    factors_path = RAW / "umpire_factors.csv"
    if factors_path.exists() and not today_umps.empty:
        factors = pd.read_csv(factors_path)
        ump_lookup = today_umps.merge(factors[["umpire_name", "umpire_run_factor"]], on="umpire_name", how="left")
        n_known = ump_lookup["umpire_run_factor"].notna().sum()
        logger.info(f"Umpire run factors resolved for {n_known}/{len(ump_lookup)} assigned games.")
        df = builder.join_umpires(df, ump_lookup)
    else:
        if not factors_path.exists():
            logger.warning("umpire_factors.csv not found — umpire_run_factor left NaN (train-median fallback).")
        df["umpire_run_factor"] = df.get("umpire_run_factor", np.nan)

    today_mask = df["date"].astype(str) == date
    today_long = df[today_mask].copy()

    with open(MODELS / "feature_names.json") as f:
        features = json.load(f)["passing_features"]
    medians = joblib.load(MODELS / "train_medians.joblib")
    xgb_cal = joblib.load(MODELS / "xgb_calibrated.joblib")
    lgbm_cal = joblib.load(MODELS / "lgbm_calibrated.joblib")

    game_df, _ = builder.pivot_to_game_level(today_long)

    for col in features:
        if col not in game_df.columns:
            game_df[col] = np.nan
    X_today = game_df[features].fillna(medians)

    home_prob = (xgb_cal.predict_proba(X_today)[:, 1] + lgbm_cal.predict_proba(X_today)[:, 1]) / 2.0
    game_df["home_win_prob"] = home_prob

    # game_df only carries team abbreviations, but the odds feed matches on
    # full team names, so bring those in from the original schedule payload first.
    game_df["game_pk"] = game_df["game_pk"].astype(str)
    game_meta = pd.DataFrame(games)[["game_pk", "home_team_name", "away_team_name"]]
    game_meta["game_pk"] = game_meta["game_pk"].astype(str)
    game_df = game_df.merge(game_meta, on="game_pk", how="left")

    odds = fetch_moneyline_odds()
    if not odds.empty:
        # The odds feed returns every upcoming event, not just `date` — if the
        # same two teams play on consecutive days, matching on team names
        # alone would fan one game out into multiple rows with mismatched
        # odds attached. Scope to this date's events before merging.
        odds_today = odds[odds["event_date"] == date].drop(columns=["event_date"])
        n_before = len(game_df)
        game_df = game_df.merge(odds_today, on=["home_team_name", "away_team_name"], how="left")
        if len(game_df) != n_before:
            logger.warning(f"Odds merge produced {len(game_df)} rows from {n_before} games — "
                            f"dropping extra matches per game_pk to avoid mismatched odds.")
            game_df = game_df.drop_duplicates(subset=["game_pk"], keep="first")
        # no_vig_home_implied already comes from fetch_moneyline_odds() as the
        # median of each book's own no-vig probability — do not recompute it
        # from home_ml/away_ml here, since those are just one representative
        # book's raw price and would be a less accurate, noisier estimate.
    else:
        game_df["home_ml"] = np.nan
        game_df["away_ml"] = np.nan
        game_df["no_vig_home_implied"] = np.nan

    game_df["edge"] = game_df["home_win_prob"] - game_df["no_vig_home_implied"]
    game_df["bet_flag"] = (
        game_df["edge"].abs().ge(MIN_EDGE_MONEYLINE) & game_df["edge"].abs().lt(MAX_EDGE_MONEYLINE)
    )
    game_df["bet_side"] = np.where(game_df["bet_flag"] & (game_df["edge"] > 0), "HOME",
                            np.where(game_df["bet_flag"] & (game_df["edge"] < 0), "AWAY", ""))

    logger.info("========== GAME CARDS ==========")
    for _, row in game_df.iterrows():
        logger.info(
            f"{row['away_team']} @ {row['home_team']}  |  "
            f"SP: {row['away_starter']} vs {row['home_starter']}  |  "
            f"model_home_win_prob={row['home_win_prob']:.3f}  "
            f"market_no_vig_home={row['no_vig_home_implied']:.3f}  "
            f"edge={row['edge']:+.3f}  "
            f"{'*** BET ' + row['bet_side'] + ' ***' if row['bet_flag'] else ''}"
            if pd.notna(row["no_vig_home_implied"]) else
            f"{row['away_team']} @ {row['home_team']}  |  SP: {row['away_starter']} vs {row['home_starter']}  |  "
            f"model_home_win_prob={row['home_win_prob']:.3f}  (no market odds available)"
        )

    OUTPUTS.mkdir(parents=True, exist_ok=True)
    csv_path = OUTPUTS / f"predictions_{date}.csv"
    xlsx_path = OUTPUTS / f"predictions_{date}.xlsx"
    game_df.to_csv(csv_path, index=False)
    game_df.to_excel(xlsx_path, index=False)
    logger.info(f"Saved {csv_path} and {xlsx_path}")


if __name__ == "__main__":
    main()
