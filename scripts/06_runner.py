# -*- coding: utf-8 -*-
"""Daily orchestrator, intended to run under Windows Task Scheduler at 8 AM:
  1. Rscript R/00_run_all.R          (refresh raw data)
  2. scripts/01_build_features.py    (rebuild feature matrix)
  3. scripts/04_predict.py --date today   (early read — no bets logged)
  4. scripts/05_bankroll.py --settle      (settle yesterday's bets)
  5. fallback bet-logging safety net (see check_and_fallback_log_bets below)
  6. git commit+push of the betting-history files (bet log, bankroll state,
     closing lines) — the only pipeline outputs that can't be regenerated.
A failure in any step is logged and does not stop the remaining steps.

Bet LOGGING intentionally does not happen here on a normal on-time run: at
8 AM neither lineups nor umpires are posted and lines are far from close.
It happens instead in scripts/07_capture_closing_lines.py --pregame-predict
(scheduled separately), which re-runs 04_predict + 05_bankroll --log-bets 2
hours before the day's first pitch, when the model sees the same
information it was trained on.

That design assumes this job actually starts near 8 AM. When the machine
was fully powered off overnight (WakeToRun only wakes from sleep, not a
shutdown — Task Scheduler's StartWhenAvailable then fires this job hours
late, once someone turns the PC back on), the afternoon pass can be starved
of lead time too, and the day's bets silently never get logged at all
(observed 2026-08-03 through 2026-08-05). check_and_fallback_log_bets below
is the safety net for that: it never fires on a normal-timing day.
"""
import sys
import subprocess
from pathlib import Path
from datetime import datetime, timezone, timedelta

import pandas as pd
from loguru import logger

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from config.config import PATHS  # noqa: E402
from odds_utils import fetch_slate  # noqa: E402

ROOT = PATHS["root"]
LOGS = PATHS["logs"]
OUTPUTS = PATHS["outputs"]
PYTHON = ROOT / "venv" / "Scripts" / "python.exe"
RSCRIPT_CANDIDATES = [
    r"C:\Program Files\R\R-4.3.2\bin\Rscript.exe",
    "Rscript",
]
# Mirrors scripts/07_capture_closing_lines.py's PREGAME_PREDICT_LEAD: the
# lead time the normal near-game-time bet-logging pass needs before first
# pitch. Used here only to decide whether that pass still has a fair chance
# to run — not to duplicate its snapshot/predict logic.
PREGAME_PREDICT_LEAD = timedelta(hours=2)

today = datetime.now().strftime("%Y-%m-%d")
log_path = LOGS / f"daily_{today}.log"

logger.remove()
logger.add(sys.stderr, level="INFO")
logger.add(log_path, level="DEBUG")


def find_rscript():
    for candidate in RSCRIPT_CANDIDATES:
        try:
            subprocess.run([candidate, "--version"], capture_output=True, check=True, cwd=str(ROOT))
            return candidate
        except (FileNotFoundError, subprocess.CalledProcessError):
            continue
    return None


def run_step(name, cmd, timeout=3600):
    logger.info(f"---------- starting: {name} ----------")
    logger.debug(f"command: {' '.join(str(c) for c in cmd)}")
    try:
        result = subprocess.run(cmd, cwd=str(ROOT), capture_output=True, text=True, timeout=timeout)
        for line in result.stdout.splitlines():
            logger.debug(f"[{name}] {line}")
        for line in result.stderr.splitlines():
            logger.debug(f"[{name}] {line}")
        if result.returncode != 0:
            logger.error(f"{name} FAILED (exit code {result.returncode}) — continuing to next step.")
            return False
        logger.info(f"---------- completed: {name} ----------")
        return True
    except subprocess.TimeoutExpired:
        logger.error(f"{name} TIMED OUT — continuing to next step.")
        return False
    except Exception as e:
        logger.error(f"{name} raised an exception: {e} — continuing to next step.")
        return False


def backup_betting_history():
    """Commit + push the irreplaceable betting-history files (bet log,
    bankroll state, captured closing lines) to the GitHub remote. Everything
    else in data/models/outputs is regenerable and stays gitignored. No-op
    when nothing changed since the last run.
    """
    import glob as _glob
    files = ["outputs/bet_log.csv", "models/bankroll_state.json"]
    files += sorted(_glob.glob("data/raw/odds_close_*.csv"))
    existing = [f for f in files if (ROOT / f).exists()]
    if not existing:
        logger.info("backup: no betting-history files exist yet — skipping.")
        return True
    try:
        subprocess.run(["git", "add", "--"] + existing, cwd=str(ROOT), check=True, capture_output=True)
        staged = subprocess.run(["git", "diff", "--cached", "--quiet"], cwd=str(ROOT))
        if staged.returncode == 0:
            logger.info("backup: betting history unchanged — nothing to commit.")
            return True
        subprocess.run(["git", "commit", "-m", f"Betting history backup {today}"],
                        cwd=str(ROOT), check=True, capture_output=True)
        push = subprocess.run(["git", "push"], cwd=str(ROOT), capture_output=True, text=True, timeout=120)
        if push.returncode != 0:
            logger.error(f"backup: commit created but push failed: {push.stderr.strip()[-200:]}")
            return False
        logger.info(f"backup: committed and pushed {len(existing)} betting-history file(s).")
        return True
    except Exception as e:
        logger.error(f"backup: git step failed: {e}")
        return False


def check_and_fallback_log_bets():
    """Safety net for a late-starting run: if today's predictions exist but
    no bets have been logged yet for today, and there's still enough lead
    time before first pitch for the normal near-game-time pass
    (07_capture_closing_lines.py --pregame-predict) to do its job, do
    nothing — that pass is better-informed (posted lineups/umpires) and
    should win. Otherwise that pass either already missed its window or
    won't get a fair one, so log bets now with whatever this run has rather
    than silently losing the whole day (bet_log.csv's existing dupe guard
    makes this safe even if both passes end up trying). Always logs loudly
    either way so a missed day is never silent again.
    """
    pred_path = OUTPUTS / f"predictions_{today}.csv"
    if not pred_path.exists():
        logger.error(f"No predictions_{today}.csv — predict step failed or produced nothing; "
                      "cannot evaluate bet-logging fallback.")
        return

    bet_log_path = OUTPUTS / "bet_log.csv"
    already_logged = False
    if bet_log_path.exists():
        bl = pd.read_csv(bet_log_path, dtype={"date": str})
        already_logged = (bl["date"] == today).any()
    if already_logged:
        logger.info(f"Bets already logged for {today} — fallback not needed.")
        return

    try:
        slate = fetch_slate(today)
    except Exception as e:
        logger.error(f"Fallback bet-logging: could not fetch today's slate ({e}) — cannot evaluate.")
        return
    if slate.empty:
        logger.info(f"No regular-season games scheduled for {today} — nothing to log.")
        return

    now = datetime.now(timezone.utc)
    first_pitch = pd.Timestamp(slate["start_utc"].min()).to_pydatetime()
    not_yet_started = (pd.to_datetime(slate["start_utc"]) > now).sum()

    if now < first_pitch - PREGAME_PREDICT_LEAD:
        logger.info(f"No bets logged for {today} yet, but first pitch ({first_pitch:%H:%M UTC}) is still "
                     f"{(first_pitch - now)} away — the normal near-game-time pass still has a fair chance. "
                     "Not triggering the fallback.")
        return

    if not_yet_started == 0:
        logger.error(f"FALLBACK BET-LOGGING MISSED: no bets were logged for {today} and all "
                      f"{len(slate)} game(s) have already started — this betting day is lost. "
                      "This run started too late (check whether the machine was powered off overnight).")
        return

    logger.error(f"FALLBACK BET-LOGGING: no bets logged for {today} and first pitch was "
                  f"{first_pitch:%H:%M UTC} — this run started too late for the normal near-game-time "
                  f"pass to have a fair shot (check whether the machine was powered off overnight). "
                  f"Logging bets now for the {not_yet_started}/{len(slate)} game(s) not yet started, "
                  "using this run's data instead of losing the day entirely.")
    subprocess.run([str(PYTHON), "scripts/05_bankroll.py", "--date", today, "--log-bets"],
                    cwd=str(ROOT), check=False)


def main():
    logger.info(f"========== Daily run: {today} ==========")
    results = {}

    rscript = find_rscript()
    if rscript is None:
        logger.error("Rscript executable not found — skipping R data refresh step.")
        results["R data refresh"] = False
    else:
        # A normal daily increment finishes in minutes, but a multi-day
        # backlog (e.g. the pipeline missed a few days) can take hours —
        # give this step much more room than the others before giving up.
        results["R data refresh"] = run_step("R data refresh", [rscript, "R/00_run_all.R"], timeout=14400)

    results["build features"] = run_step("build features", [str(PYTHON), "scripts/01_build_features.py"])
    results["predict"] = run_step("predict", [str(PYTHON), "scripts/04_predict.py", "--date", today])
    if results["predict"]:
        check_and_fallback_log_bets()
    results["settle bets"] = run_step("settle bets", [str(PYTHON), "scripts/05_bankroll.py", "--settle"])
    results["backup betting history"] = backup_betting_history()

    logger.info("========== SUMMARY ==========")
    for name, ok in results.items():
        logger.info(f"  {name:20s} {'OK' if ok else 'FAILED'}")
    logger.info(f"Full log: {log_path}")


if __name__ == "__main__":
    main()
