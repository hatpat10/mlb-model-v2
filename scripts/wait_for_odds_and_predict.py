# -*- coding: utf-8 -*-
"""One-shot watcher: polls The Odds API until lines for --date are posted,
then runs 04_predict.py and 05_bankroll.py --log-bets for that date and
exits. Not part of the regular pipeline — a manual catch-up tool for when
04_predict.py was run before sportsbooks opened a day's lines.

Usage: python scripts/wait_for_odds_and_predict.py --date YYYY-MM-DD
"""
import sys
import os
import time
import argparse
import subprocess
from pathlib import Path
from datetime import datetime, timezone
from zoneinfo import ZoneInfo

import requests
import pandas as pd
from loguru import logger
from dotenv import load_dotenv

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config.config import PATHS  # noqa: E402
sys.path.insert(0, str(Path(__file__).resolve().parent))
from odds_utils import fetch_slate  # noqa: E402

ROOT = PATHS["root"]
PYTHON = ROOT / "venv" / "Scripts" / "python.exe"
POLL_SECONDS = 1800  # 30 min — plenty of quota headroom, no need to poll tighter
MAX_HOURS = 20

logger.remove()
logger.add(sys.stderr, level="INFO")


def odds_posted_for(date: str) -> bool:
    api_key = os.getenv("ODDS_API_KEY")
    resp = requests.get(
        "https://api.the-odds-api.com/v4/sports/baseball_mlb/odds",
        params={"apiKey": api_key, "regions": "us", "markets": "h2h"},
        timeout=30,
    )
    resp.raise_for_status()
    for event in resp.json():
        commence = event.get("commence_time")
        if not commence:
            continue
        event_date = (
            datetime.fromisoformat(commence.replace("Z", "+00:00"))
            .astimezone(ZoneInfo("America/New_York"))
            .strftime("%Y-%m-%d")
        )
        if event_date == date and len(event.get("bookmakers", [])) >= 3:
            return True
    return False


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--date", required=True)
    args = parser.parse_args()
    date = args.date

    load_dotenv(ROOT / ".env")
    if not os.getenv("ODDS_API_KEY"):
        logger.error("ODDS_API_KEY is not configured.")
        sys.exit(1)

    deadline = time.time() + MAX_HOURS * 3600
    attempt = 0
    while time.time() < deadline:
        attempt += 1
        logger.info(f"[attempt {attempt}] checking whether odds are posted for {date} ...")
        try:
            if odds_posted_for(date):
                logger.info(f"Odds are up for {date} — running predict + log-bets now.")
                invoked_at = datetime.now(timezone.utc)
                prediction = subprocess.run([str(PYTHON), "scripts/04_predict.py", "--date", date], cwd=str(ROOT))
                pred_path = ROOT / "outputs" / f"predictions_{date}.csv"
                if prediction.returncode != 0 or not pred_path.exists():
                    logger.error("Prediction failed or produced no artifact; refusing to log bets.")
                    sys.exit(1)
                written_at = datetime.fromtimestamp(pred_path.stat().st_mtime, tz=timezone.utc)
                if written_at < invoked_at:
                    logger.error("Prediction artifact was not refreshed by this invocation; refusing to log bets.")
                    sys.exit(1)
                slate = fetch_slate(date)
                now = datetime.now(timezone.utc)
                eligible = slate.loc[pd.to_datetime(slate["start_utc"], utc=True) > now, "game_pk"].astype(str).tolist()
                if not eligible:
                    logger.error("Odds appeared, but no games remain unstarted; no decisions will be logged.")
                    sys.exit(1)
                logging = subprocess.run([
                    str(PYTHON), "scripts/05_bankroll.py", "--date", date, "--log-bets",
                    "--game-pks", ",".join(eligible),
                ], cwd=str(ROOT))
                if logging.returncode != 0:
                    logger.error("Bankroll logging failed.")
                    sys.exit(logging.returncode)
                logger.info(f"Done — outputs/predictions_{date}.csv and bet_log.csv are up to date with real odds.")
                return
        except requests.RequestException as e:
            logger.warning(f"Odds API check failed: {e} — will retry.")
        logger.info(f"Not posted yet — sleeping {POLL_SECONDS}s.")
        time.sleep(POLL_SECONDS)

    logger.error(f"Gave up after {MAX_HOURS}h — odds for {date} never appeared with >=3 books.")
    sys.exit(1)


if __name__ == "__main__":
    main()
