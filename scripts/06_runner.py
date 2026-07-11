# -*- coding: utf-8 -*-
"""Daily orchestrator, intended to run under Windows Task Scheduler at 8 AM:
  1. Rscript R/00_run_all.R          (refresh raw data)
  2. scripts/01_build_features.py    (rebuild feature matrix)
  3. scripts/04_predict.py --date today   (early read — no bets logged)
  4. scripts/05_bankroll.py --settle      (settle yesterday's bets)
A failure in any step is logged and does not stop the remaining steps.

Bet LOGGING intentionally does not happen here: at 8 AM neither lineups nor
umpires are posted and lines are far from close. It happens instead in
scripts/07_capture_closing_lines.py --pregame-predict (scheduled separately),
which re-runs 04_predict + 05_bankroll --log-bets 2 hours before the day's
first pitch, when the model sees the same information it was trained on.
"""
import sys
import subprocess
from pathlib import Path
from datetime import datetime

from loguru import logger

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config.config import PATHS  # noqa: E402

ROOT = PATHS["root"]
LOGS = PATHS["logs"]
PYTHON = ROOT / "venv" / "Scripts" / "python.exe"
RSCRIPT_CANDIDATES = [
    r"C:\Program Files\R\R-4.3.2\bin\Rscript.exe",
    "Rscript",
]

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
    results["settle bets"] = run_step("settle bets", [str(PYTHON), "scripts/05_bankroll.py", "--settle"])

    logger.info("========== SUMMARY ==========")
    for name, ok in results.items():
        logger.info(f"  {name:20s} {'OK' if ok else 'FAILED'}")
    logger.info(f"Full log: {log_path}")


if __name__ == "__main__":
    main()
