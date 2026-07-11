# -*- coding: utf-8 -*-
"""Captures REAL closing moneylines for a day's MLB slate from The Odds API
and writes data/raw/odds_close_<date>.csv — the file that (a) lets
scripts/05_bankroll.py --settle compute CLV for every logged bet and
(b) accumulates into the real-odds backtest path in scripts/03_backtest.py,
replacing the synthetic Elo proxy for every season captured this way.

Strategy: games on a slate share a handful of distinct start times, so
instead of polling on a fixed interval all day (expensive against The Odds
API's monthly quota), the script wakes once per distinct start time, 5
minutes before first pitch, snapshots the full market, and updates the
stored line for every game that hasn't started yet. A game's stored row
therefore always ends up being the last pre-start snapshot — its closing
line. Cost: (# distinct start times + 1) API calls per day, typically
8-13, ~300-400/month.

The CSV is rewritten after every snapshot, so a crash mid-day keeps every
line captured up to that point (later games just get their most recent
pre-crash snapshot instead of a true close).

Usage:
  python scripts/07_capture_closing_lines.py                 # watch today's slate
  python scripts/07_capture_closing_lines.py --date YYYY-MM-DD
  python scripts/07_capture_closing_lines.py --once          # single snapshot now, then exit
  python scripts/07_capture_closing_lines.py --pregame-predict
      # additionally, 2 hours before the day's first pitch, re-run
      # 04_predict.py + 05_bankroll.py --log-bets so bets are logged with
      # posted lineups/umpires and near-final lines (the 8 AM run's
      # already-logged games are skipped by 05_bankroll's dupe guard).
"""
import sys
import os
import time
import argparse
import subprocess
from pathlib import Path
from datetime import datetime, timedelta, timezone
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
import requests
from loguru import logger
from dotenv import load_dotenv

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from config.config import PATHS  # noqa: E402
from odds_utils import aggregate_h2h_event, MIN_BOOKMAKERS  # noqa: E402
from features import builder  # noqa: E402

RAW = PATHS["raw"]
LOGS = PATHS["logs"]
ROOT = PATHS["root"]
PYTHON = ROOT / "venv" / "Scripts" / "python.exe"

MLB_API = "https://statsapi.mlb.com/api/v1"
ODDS_API = "https://api.the-odds-api.com/v4"
SNAPSHOT_LEAD = timedelta(minutes=5)    # wake this long before each start time
STARTED_GRACE = timedelta(minutes=2)    # still accept a snapshot this soon after start
PREGAME_PREDICT_LEAD = timedelta(hours=2)
MAX_RUNTIME_HOURS = 18

logger.remove()
logger.add(sys.stderr, level="INFO")
logger.add(LOGS / "07_capture_closing_lines.log", level="DEBUG", rotation="5 MB")


def fetch_slate(date: str) -> pd.DataFrame:
    """Today's regular-season games from the MLB Stats API: game_pk, team
    names + normalized abbreviations, and first-pitch time (UTC)."""
    resp = requests.get(f"{MLB_API}/schedule", params={"sportId": 1, "date": date, "hydrate": "team"}, timeout=30)
    resp.raise_for_status()
    rows = []
    for date_entry in resp.json().get("dates", []):
        for g in date_entry.get("games", []):
            if g.get("gameType") != "R":
                continue
            home = g["teams"]["home"]["team"]
            away = g["teams"]["away"]["team"]
            rows.append({
                "game_pk": str(g["gamePk"]),
                "home_team_name": home.get("name"),
                "away_team_name": away.get("name"),
                "home_team": home.get("abbreviation"),
                "away_team": away.get("abbreviation"),
                "start_utc": datetime.fromisoformat(g["gameDate"].replace("Z", "+00:00")),
            })
    slate = pd.DataFrame(rows)
    if not slate.empty:
        slate["home_team"] = builder.normalize_team_abbrev(slate["home_team"])
        slate["away_team"] = builder.normalize_team_abbrev(slate["away_team"])
    return slate


def fetch_odds_events(api_key: str) -> list:
    resp = requests.get(
        f"{ODDS_API}/sports/baseball_mlb/odds",
        params={"apiKey": api_key, "regions": "us", "markets": "h2h", "oddsFormat": "american"},
        timeout=30,
    )
    resp.raise_for_status()
    remaining = resp.headers.get("x-requests-remaining")
    if remaining is not None:
        logger.debug(f"Odds API quota remaining: {remaining}")
    return resp.json()


def match_events_to_slate(events: list, slate: pd.DataFrame, date: str) -> dict:
    """Map each odds event onto a game_pk: same US-Eastern civil date (how
    MLB defines the game date, even for late West Coast starts), same two
    team names, and — for doubleheaders, where both legs share teams and
    date — the leg whose scheduled first pitch is nearest the event's
    commence_time. Greedy nearest-first so the two legs can't both claim
    the same event. Returns {game_pk: aggregated-odds dict}.
    """
    candidates = []
    for event in events:
        commence_raw = event.get("commence_time")
        if not commence_raw:
            continue
        commence = datetime.fromisoformat(commence_raw.replace("Z", "+00:00"))
        event_date = commence.astimezone(ZoneInfo("America/New_York")).strftime("%Y-%m-%d")
        if event_date != date:
            continue
        matches = slate[(slate["home_team_name"] == event.get("home_team"))
                        & (slate["away_team_name"] == event.get("away_team"))]
        if matches.empty:
            logger.warning(f"Odds event has no schedule match: {event.get('away_team')} @ {event.get('home_team')}")
            continue
        agg = aggregate_h2h_event(event, min_bookmakers=MIN_BOOKMAKERS)
        if agg is None:
            continue
        for _, m in matches.iterrows():
            gap = abs((m["start_utc"] - commence).total_seconds())
            candidates.append((gap, m["game_pk"], agg))

    matched = {}
    used_pks = set()
    for gap, game_pk, agg in sorted(candidates, key=lambda c: c[0]):
        if game_pk in used_pks:
            continue
        used_pks.add(game_pk)
        matched[game_pk] = agg
    return matched


def out_path_for(date: str) -> Path:
    return RAW / f"odds_close_{date}.csv"


def write_snapshot(date: str, slate: pd.DataFrame, matched: dict, now_utc: datetime) -> int:
    """Update stored rows for every game that hasn't started yet (within
    grace), preserving rows of already-started games untouched — so each
    game's row converges to its last pre-start snapshot. Returns the number
    of rows updated.
    """
    path = out_path_for(date)
    existing = pd.read_csv(path, dtype={"game_pk": str}) if path.exists() else pd.DataFrame()

    updated = 0
    new_rows = {}
    for _, g in slate.iterrows():
        if g["game_pk"] not in matched:
            continue
        if now_utc > g["start_utc"] + STARTED_GRACE:
            continue  # game already underway — its stored close must not move
        agg = matched[g["game_pk"]]
        new_rows[g["game_pk"]] = {
            "date": date,
            "game_pk": g["game_pk"],
            "home_team": g["home_team"],
            "away_team": g["away_team"],
            "home_team_name": g["home_team_name"],
            "away_team_name": g["away_team_name"],
            "commence_time_utc": g["start_utc"].isoformat(),
            "snapshot_time_utc": now_utc.isoformat(),
            "n_books": agg["n_books"],
            "home_ml_close": agg["home_ml"],
            "away_ml_close": agg["away_ml"],
            "home_no_vig_prob": agg["no_vig_home_implied"],
        }
        updated += 1

    if not new_rows and existing.empty:
        return 0

    kept = existing[~existing["game_pk"].isin(new_rows)] if not existing.empty else pd.DataFrame()
    out = pd.concat([kept, pd.DataFrame(list(new_rows.values()))], ignore_index=True)
    out = out.sort_values("commence_time_utc").reset_index(drop=True)
    out.to_csv(path, index=False)
    return updated


def run_pregame_predict(date: str):
    logger.info("Running pre-first-pitch re-predict (posted lineups/umps + near-final lines) ...")
    subprocess.run([str(PYTHON), "scripts/04_predict.py", "--date", date], cwd=str(ROOT), check=False)
    subprocess.run([str(PYTHON), "scripts/05_bankroll.py", "--date", date, "--log-bets"], cwd=str(ROOT), check=False)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--date", default=datetime.now().strftime("%Y-%m-%d"))
    parser.add_argument("--once", action="store_true",
                        help="take one snapshot now, update not-yet-started games, exit")
    parser.add_argument("--pregame-predict", action="store_true",
                        help="also re-run 04_predict + 05_bankroll --log-bets 2h before first pitch")
    args = parser.parse_args()
    date = args.date

    load_dotenv(ROOT / ".env")
    api_key = os.getenv("ODDS_API_KEY")
    if not api_key or api_key == "your_odds_api_key_here":
        logger.error("ODDS_API_KEY not set in .env — cannot capture closing lines.")
        sys.exit(1)

    slate = fetch_slate(date)
    if slate.empty:
        logger.warning(f"No regular-season games scheduled for {date} — nothing to capture.")
        return
    logger.info(f"{len(slate)} games on the {date} slate, "
                f"{slate['start_utc'].nunique()} distinct start times.")

    def snapshot(label: str):
        now_utc = datetime.now(timezone.utc)
        try:
            events = fetch_odds_events(api_key)
        except requests.RequestException as e:
            logger.warning(f"[{label}] odds fetch failed: {e}")
            return
        matched = match_events_to_slate(events, slate, date)
        n = write_snapshot(date, slate, matched, now_utc)
        logger.info(f"[{label}] snapshot: odds for {len(matched)}/{len(slate)} games, "
                    f"{n} not-yet-started rows updated -> {out_path_for(date).name}")

    if args.once:
        snapshot("once")
        return

    # Wake plan: one baseline snapshot now, then one wake per distinct start
    # time (5 min before it), plus the optional pre-game re-predict wake.
    now = datetime.now(timezone.utc)
    wakes = []  # (when_utc, kind)
    for start in sorted(slate["start_utc"].unique()):
        start = pd.Timestamp(start).to_pydatetime()
        wake_at = start - SNAPSHOT_LEAD
        if wake_at > now:
            wakes.append((wake_at, "close"))
        else:
            logger.warning(f"Start time {start:%H:%M UTC} already within {SNAPSHOT_LEAD} — "
                           f"baseline snapshot will serve as its closing line.")
    if args.pregame_predict:
        first_pitch = pd.Timestamp(slate["start_utc"].min()).to_pydatetime()
        predict_at = first_pitch - PREGAME_PREDICT_LEAD
        if predict_at > now:
            wakes.append((predict_at, "predict"))
        else:
            logger.warning("First pitch is less than 2h away — skipping the pre-game re-predict wake "
                           "(run scripts/04_predict.py manually if wanted).")
    wakes.sort(key=lambda w: w[0])

    snapshot("baseline")

    deadline = now + timedelta(hours=MAX_RUNTIME_HOURS)
    for wake_at, kind in wakes:
        wait = (wake_at - datetime.now(timezone.utc)).total_seconds()
        if wait > 0:
            if wake_at > deadline:
                logger.warning(f"Wake at {wake_at:%H:%M UTC} exceeds {MAX_RUNTIME_HOURS}h runtime cap — stopping.")
                break
            logger.info(f"Sleeping {wait/60:.0f} min until {wake_at:%H:%M UTC} ({kind}) ...")
            time.sleep(wait)
        if kind == "predict":
            run_pregame_predict(date)
        else:
            snapshot(f"close@{wake_at:%H:%M}")

    path = out_path_for(date)
    if path.exists():
        final = pd.read_csv(path)
        logger.info(f"Done: {len(final)}/{len(slate)} games captured in {path.name}. "
                    f"CLV will populate on the next 05_bankroll.py --settle.")
    else:
        logger.warning(f"Done, but no odds were ever captured for {date}.")


if __name__ == "__main__":
    main()
