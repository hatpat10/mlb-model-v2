# -*- coding: utf-8 -*-
"""Bankroll tracker: logs new flagged bets from a day's predictions with
quarter-Kelly sizing, and settles previously-logged bets once results (and,
where available, real closing lines for CLV) are known.

Usage:
  python scripts/05_bankroll.py --date YYYY-MM-DD --log-bets [--resume] [--force]
  python scripts/05_bankroll.py --settle
"""
import sys
import json
import argparse
import glob
from pathlib import Path

import numpy as np
import pandas as pd
from loguru import logger

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config.config import PATHS, KELLY_FRACTION, DRAWDOWN_PAUSE_THRESHOLD  # noqa: E402

RAW = PATHS["raw"]
OUTPUTS = PATHS["outputs"]
MODELS = PATHS["models"]
LOGS = PATHS["logs"]

STATE_PATH = MODELS / "bankroll_state.json"
BET_LOG_PATH = OUTPUTS / "bet_log.csv"
STARTING_BANKROLL = 10000.0
BET_LOG_COLUMNS = [
    "date", "home_team", "away_team", "side", "bet_size", "odds", "no_vig_prob",
    "result", "pnl", "closing_no_vig_prob", "clv", "status",
]

logger.remove()
logger.add(sys.stderr, level="INFO")
logger.add(LOGS / "05_bankroll.log", level="DEBUG", rotation="5 MB")


def load_state():
    if STATE_PATH.exists():
        with open(STATE_PATH) as f:
            return json.load(f)
    return {
        "bankroll": STARTING_BANKROLL,
        "peak_bankroll": STARTING_BANKROLL,
        "starting_bankroll": STARTING_BANKROLL,
        "is_paused": False,
    }


def save_state(state):
    MODELS.mkdir(parents=True, exist_ok=True)
    with open(STATE_PATH, "w") as f:
        json.dump(state, f, indent=2)


def load_bet_log():
    if BET_LOG_PATH.exists():
        df = pd.read_csv(BET_LOG_PATH)
    else:
        df = pd.DataFrame(columns=BET_LOG_COLUMNS)
    # "result"/"status" start out all-NaN for pending bets, which pandas
    # infers as float64 — writing a string like "loss" into that column
    # later then raises TypeError instead of upcasting. Force object dtype
    # up front so settlement can always write strings into these columns.
    for col in ("result", "status"):
        if col in df.columns:
            df[col] = df[col].astype(object)
    return df


def american_to_decimal(odds):
    odds = np.asarray(odds, dtype=float)
    return np.where(odds < 0, 1 + 100 / -odds, 1 + odds / 100)


def kelly_fraction(prob, decimal_odds):
    net = decimal_odds - 1
    f = np.where(net > 0, (prob * decimal_odds - 1) / net, 0.0)
    return np.clip(f, 0.0, 1.0)


def cmd_log_bets(date, resume, force):
    state = load_state()

    if resume:
        state["is_paused"] = False
        save_state(state)
        logger.info("Drawdown pause cleared via --resume.")

    drawdown = 1 - (state["bankroll"] / state["peak_bankroll"]) if state["peak_bankroll"] > 0 else 0.0
    if state["is_paused"] and not force:
        logger.error(
            f"BLOCKED: bankroll is paused (drawdown={drawdown:.1%} >= {DRAWDOWN_PAUSE_THRESHOLD:.0%}). "
            f"Re-run with --resume to clear the pause, or --force to override this one run."
        )
        sys.exit(1)
    if state["is_paused"] and force:
        logger.warning(f"Drawdown pause overridden via --force for this run only (drawdown={drawdown:.1%}). "
                        f"Pause state remains set for future runs.")

    pred_path = OUTPUTS / f"predictions_{date}.csv"
    if not pred_path.exists():
        logger.error(f"{pred_path} not found — run scripts/04_predict.py --date {date} first.")
        sys.exit(1)
    preds = pd.read_csv(pred_path)
    flagged = preds[preds["bet_flag"] == True]  # noqa: E712
    if flagged.empty:
        save_state(state)
        logger.info(f"No bets flagged for {date}.")
        return

    bet_log = load_bet_log()
    already_logged = set(
        zip(bet_log.loc[bet_log["date"] == date, "home_team"], bet_log.loc[bet_log["date"] == date, "away_team"])
    )
    if already_logged:
        logger.warning(f"{len(already_logged)} bet(s) already logged for {date} — skipping to avoid double-staking. "
                        f"Delete the row(s) from {BET_LOG_PATH} first if you need to re-log.")

    new_rows = []
    for _, row in flagged.iterrows():
        if (row["home_team"], row["away_team"]) in already_logged:
            continue
        side = row["bet_side"]
        if side == "HOME":
            prob, odds = row["home_win_prob"], row["home_ml"]
        else:
            prob, odds = 1 - row["home_win_prob"], row["away_ml"]

        decimal_odds = american_to_decimal(odds)
        f_full = kelly_fraction(prob, decimal_odds)
        stake = float(KELLY_FRACTION * f_full * state["bankroll"])

        new_rows.append({
            "date": date, "home_team": row["home_team"], "away_team": row["away_team"],
            "side": side, "bet_size": round(stake, 2), "odds": odds,
            "no_vig_prob": row.get("no_vig_home_implied", np.nan) if side == "HOME"
            else 1 - row.get("no_vig_home_implied", np.nan),
            "result": np.nan, "pnl": np.nan, "closing_no_vig_prob": np.nan, "clv": np.nan,
            "status": "pending",
        })

    if not new_rows:
        save_state(state)
        logger.info(f"All flagged bets for {date} were already logged — nothing new to add.")
        return

    new_rows_df = pd.DataFrame(new_rows)
    new_rows_df["result"] = new_rows_df["result"].astype(object)
    bet_log = pd.concat([bet_log, new_rows_df], ignore_index=True)
    bet_log.to_csv(BET_LOG_PATH, index=False)
    save_state(state)
    logger.info(f"Logged {len(new_rows)} bets for {date} to {BET_LOG_PATH} (total stake={sum(r['bet_size'] for r in new_rows):.2f})")


def load_closing_lines():
    files = sorted(glob.glob(str(RAW / "odds_close_*.csv")))
    if not files:
        return None
    return pd.concat([pd.read_csv(f) for f in files], ignore_index=True)


def cmd_settle():
    bet_log = load_bet_log()
    pending = bet_log[bet_log["status"] == "pending"]
    if pending.empty:
        logger.info("No pending bets to settle.")
        return

    game_logs_path = RAW / "game_logs_all.csv"
    if not game_logs_path.exists():
        logger.error(f"{game_logs_path} not found — cannot settle bets without results.")
        return
    game_logs = pd.read_csv(game_logs_path)
    results = game_logs[game_logs["is_home"] == 1][["date", "team", "opponent", "win"]].rename(
        columns={"team": "home_team", "opponent": "away_team", "win": "home_win"})

    closing = load_closing_lines()

    state = load_state()
    n_settled = 0
    for idx, row in pending.iterrows():
        match = results[(results["date"] == row["date"]) & (results["home_team"] == row["home_team"]) &
                         (results["away_team"] == row["away_team"])]
        if match.empty:
            continue  # game hasn't been played/collected yet

        home_won = bool(match["home_win"].iloc[0])
        bet_won = home_won if row["side"] == "HOME" else not home_won
        decimal_odds = american_to_decimal(row["odds"])
        pnl = row["bet_size"] * (decimal_odds - 1) if bet_won else -row["bet_size"]

        closing_no_vig = np.nan
        if closing is not None and "date" in closing.columns:
            cmatch = closing[closing["date"] == row["date"]]
            crow = cmatch[(cmatch["home_team"] == row["home_team"]) & (cmatch["away_team"] == row["away_team"])]
            if not crow.empty:
                closing_no_vig = crow["home_no_vig_prob"].iloc[0] if row["side"] == "HOME" \
                    else 1 - crow["home_no_vig_prob"].iloc[0]

        clv = closing_no_vig - row["no_vig_prob"] if pd.notna(closing_no_vig) else np.nan

        bet_log.loc[idx, "result"] = "win" if bet_won else "loss"
        bet_log.loc[idx, "pnl"] = round(float(pnl), 2)
        bet_log.loc[idx, "closing_no_vig_prob"] = closing_no_vig
        bet_log.loc[idx, "clv"] = clv
        bet_log.loc[idx, "status"] = "settled"

        state["bankroll"] += pnl
        n_settled += 1

    if n_settled == 0:
        logger.info("No pending bets had known results yet.")
        return

    state["peak_bankroll"] = max(state["peak_bankroll"], state["bankroll"])
    drawdown = 1 - (state["bankroll"] / state["peak_bankroll"]) if state["peak_bankroll"] > 0 else 0.0
    if drawdown >= DRAWDOWN_PAUSE_THRESHOLD and not state["is_paused"]:
        state["is_paused"] = True
        logger.warning(f"Drawdown {drawdown:.1%} >= {DRAWDOWN_PAUSE_THRESHOLD:.0%} threshold — bankroll PAUSED. "
                        f"--log-bets will be blocked until --resume.")

    bet_log.to_csv(BET_LOG_PATH, index=False)
    save_state(state)
    logger.info(f"Settled {n_settled} bets. Bankroll={state['bankroll']:.2f} "
                f"(peak={state['peak_bankroll']:.2f}, drawdown={drawdown:.1%})")

    if closing is None:
        logger.warning("No data/raw/odds_close_*.csv found — CLV could not be computed for any settled bet.")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--date")
    parser.add_argument("--log-bets", action="store_true")
    parser.add_argument("--settle", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    if args.log_bets:
        if not args.date:
            logger.error("--log-bets requires --date YYYY-MM-DD")
            sys.exit(1)
        cmd_log_bets(args.date, args.resume, args.force)
    if args.settle:
        cmd_settle()
    if not args.log_bets and not args.settle:
        logger.error("Specify --log-bets and/or --settle")
        sys.exit(1)


if __name__ == "__main__":
    main()
