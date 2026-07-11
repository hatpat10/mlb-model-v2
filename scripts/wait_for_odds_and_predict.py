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
from datetime import datetime
from zoneinfo import ZoneInfo

import requests
from loguru import logger
from dotenv import load_dotenv

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config.config import PATHS  # noqa: E402

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

    deadline = time.time() + MAX_HOURS * 3600
    attempt = 0
    while time.time() < deadline:
        attempt += 1
        logger.info(f"[attempt {attempt}] checking whether odds are posted for {date} ...")
        try:
            if odds_posted_for(date):
                logger.info(f"Odds are up for {date} — running predict + log-bets now.")
                subprocess.run([str(PYTHON), "scripts/04_predict.py", "--date", date], cwd=str(ROOT), check=False)
                subprocess.run([str(PYTHON), "scripts/05_bankroll.py", "--date", date, "--log-bets"],
                                cwd=str(ROOT), check=False)
                logger.info(f"Done — outputs/predictions_{date}.csv and bet_log.csv are up to date with real odds.")
                return
        except requests.RequestException as e:
            logger.warning(f"Odds API check failed: {e} — will retry.")
        logger.info(f"Not posted yet — sleeping {POLL_SECONDS}s.")
        time.sleep(POLL_SECONDS)

    logger.warning(f"Gave up after {MAX_HOURS}h — odds for {date} never appeared with >=3 books.")


if __name__ == "__main__":
    main()
