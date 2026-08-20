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
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
from loguru import logger

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from config.config import PATHS, DRAWDOWN_PAUSE_THRESHOLD  # noqa: E402
from artifact_utils import atomic_write_csv, atomic_write_json, exclusive_lock, utc_now_iso  # noqa: E402
from betting_strategy import american_to_decimal, size_stake  # noqa: E402

RAW = PATHS["raw"]
OUTPUTS = PATHS["outputs"]
MODELS = PATHS["models"]
LOGS = PATHS["logs"]

STATE_PATH = MODELS / "bankroll_state.json"
BET_LOG_PATH = OUTPUTS / "bet_log.csv"
STARTING_BANKROLL = 10000.0
POSTPONED_GRACE_DAYS = 5
BET_LOG_COLUMNS = [
    "date", "home_team", "away_team", "side", "bet_size", "odds", "no_vig_prob",
    "result", "pnl", "closing_no_vig_prob", "clv", "status",
    "game_pk", "decision_id", "model_home_prob", "edge", "decision_timestamp",
    "run_id", "model_version", "model_mode", "data_version", "feature_build_id",
    "model_training_data_version", "model_feature_build_id",
    "odds_event_id", "bookmaker_key", "bookmaker_title", "odds_snapshot_utc",
    "price_selection_method", "book_prob_std", "book_prob_range", "execution_mode",
    "lineup_available", "umpire_available", "full_kelly", "uncapped_bet_size",
    "stake_limiting_rule", "daily_staked_before", "open_staked_before",
    "available_bankroll_before",
    "settled_timestamp",
    "void_reason",
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
    state["updated_at_utc"] = utc_now_iso()
    atomic_write_json(state, STATE_PATH)


def load_bet_log():
    if BET_LOG_PATH.exists():
        df = pd.read_csv(BET_LOG_PATH)
        for col in BET_LOG_COLUMNS:
            if col not in df.columns:
                df[col] = np.nan
    else:
        df = pd.DataFrame(columns=BET_LOG_COLUMNS)
    # "result"/"status" start out all-NaN for pending bets, which pandas
    # infers as float64 — writing a string like "loss" into that column
    # later then raises TypeError instead of upcasting. Force object dtype
    # up front so settlement can always write strings into these columns.
    for col in ("result", "status", "decision_id"):
        if col in df.columns:
            df[col] = df[col].astype(object)
    if "game_pk" in df.columns:
        df["game_pk"] = df["game_pk"].apply(lambda value: str(int(value)) if pd.notna(value) and str(value).endswith(".0") else str(value))
    return df


def reconcile_state_from_ledger(state, bet_log):
    """Make bankroll/peak derivable from the durable settled-bet ledger.

    This heals either side of an interrupted two-file write: settlement can
    never be double-counted or disappear merely because the process stopped
    between writing bet_log.csv and bankroll_state.json.
    """
    starting = float(state.get("starting_bankroll", STARTING_BANKROLL))
    settled_pnl = pd.to_numeric(
        bet_log.loc[bet_log["status"].eq("settled"), "pnl"], errors="coerce"
    ).fillna(0.0)
    equity = starting + settled_pnl.cumsum()
    state["starting_bankroll"] = starting
    state["bankroll"] = float(equity.iloc[-1]) if not equity.empty else starting
    state["peak_bankroll"] = float(max(starting, equity.max())) if not equity.empty else starting
    return state


def cmd_log_bets(date, resume, force, game_pks=None):
    state = load_state()
    bet_log = load_bet_log()
    state = reconcile_state_from_ledger(state, bet_log)

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
    preds = pd.read_csv(pred_path, dtype={"game_pk": str})
    required = {"date", "game_pk", "home_team", "away_team", "bet_flag", "decision_eligible", "bet_side", "home_win_prob", "home_ml", "away_ml", "run_id", "generated_at_utc"}
    missing = required - set(preds.columns)
    if missing:
        logger.error(f"Prediction artifact is not auditable; missing columns: {sorted(missing)}")
        sys.exit(1)
    if preds["run_id"].nunique(dropna=False) != 1:
        logger.error("Prediction artifact contains zero or multiple run_id values.")
        sys.exit(1)
    if set(preds["date"].astype(str)) != {date}:
        logger.error(f"Prediction artifact date does not match requested logging date {date}.")
        sys.exit(1)
    generated = pd.to_datetime(preds["generated_at_utc"], utc=True, errors="coerce")
    if generated.isna().any() or (pd.Timestamp.now(tz="UTC") - generated.max()).total_seconds() > 6 * 3600:
        logger.error("Prediction artifact is invalid or more than six hours old; refusing to log bets.")
        sys.exit(1)
    if game_pks:
        requested = {str(pk) for pk in game_pks}
        preds = preds[preds["game_pk"].isin(requested)]
        missing_requested = requested - set(preds["game_pk"])
        if missing_requested:
            logger.error(f"Requested game_pk values missing from prediction artifact: {sorted(missing_requested)}")
            sys.exit(1)
    flagged = preds[(preds["bet_flag"] == True) & (preds["decision_eligible"] == True)]  # noqa: E712
    if flagged.empty:
        save_state(state)
        logger.info(f"No bets flagged for {date}.")
        return

    already_logged = set(bet_log.loc[bet_log["date"].astype(str) == date, "game_pk"].dropna().astype(str))
    if already_logged:
        logger.warning(f"{len(already_logged)} bet(s) already logged for {date} — skipping to avoid double-staking. "
                        f"Delete the row(s) from {BET_LOG_PATH} first if you need to re-log.")

    daily_staked = float(pd.to_numeric(bet_log.loc[bet_log["date"].astype(str) == date, "bet_size"], errors="coerce").fillna(0).sum())
    open_staked = float(pd.to_numeric(bet_log.loc[bet_log["status"].isin(["pending", "postponed"]), "bet_size"], errors="coerce").fillna(0).sum())
    new_rows = []
    for _, row in flagged.iterrows():
        game_pk = str(row["game_pk"])
        if game_pk in already_logged:
            continue
        side = row["bet_side"]
        if side == "HOME":
            prob, odds = row["home_win_prob"], row["home_ml"]
        else:
            prob, odds = 1 - row["home_win_prob"], row["away_ml"]
        side_prefix = side.lower()
        bookmaker_key = row.get(f"{side_prefix}_bookmaker_key")
        bookmaker_title = row.get(f"{side_prefix}_bookmaker_title")
        if pd.isna(bookmaker_key):
            bookmaker_key = row.get("bookmaker_key")
        if pd.isna(bookmaker_title):
            bookmaker_title = row.get("bookmaker_title")

        sizing = size_stake(prob, odds, state["bankroll"], daily_staked=daily_staked, open_staked=open_staked)
        stake = sizing.stake
        if stake <= 0:
            logger.warning(f"Skipping {game_pk} {side}: stake is zero ({sizing.limiting_rule}).")
            continue

        new_rows.append({
            "date": date, "home_team": row["home_team"], "away_team": row["away_team"],
            "side": side, "bet_size": round(stake, 2), "odds": odds,
            "no_vig_prob": row.get("no_vig_home_implied", np.nan) if side == "HOME"
            else 1 - row.get("no_vig_home_implied", np.nan),
            "result": np.nan, "pnl": np.nan, "closing_no_vig_prob": np.nan, "clv": np.nan,
            "status": "pending",
            "game_pk": game_pk, "decision_id": f"{date}_{game_pk}_{side}",
            "model_home_prob": row.get("home_win_prob", np.nan), "edge": row.get("edge", np.nan),
            "decision_timestamp": row["generated_at_utc"],
            "run_id": row["run_id"], "model_version": row.get("model_version"),
            "model_mode": row.get("model_mode"), "data_version": row.get("data_version"),
            "feature_build_id": row.get("feature_build_id"), "odds_event_id": row.get("odds_event_id"),
            "model_training_data_version": row.get("model_training_data_version"),
            "model_feature_build_id": row.get("model_feature_build_id"),
            "bookmaker_key": bookmaker_key, "bookmaker_title": bookmaker_title,
            "odds_snapshot_utc": row.get("odds_snapshot_utc"),
            "price_selection_method": row.get("price_selection_method"),
            "book_prob_std": row.get("book_prob_std"), "book_prob_range": row.get("book_prob_range"),
            "execution_mode": row.get("execution_mode") if pd.notna(row.get("execution_mode")) else "paper",
            "lineup_available": row.get("lineup_available"), "umpire_available": row.get("umpire_available"),
            "full_kelly": sizing.full_kelly, "uncapped_bet_size": round(sizing.uncapped_stake, 2),
            "stake_limiting_rule": sizing.limiting_rule,
            "daily_staked_before": round(daily_staked, 2), "open_staked_before": round(open_staked, 2),
            "available_bankroll_before": round(sizing.available_bankroll, 2),
            "settled_timestamp": np.nan,
        })
        daily_staked += stake
        open_staked += stake

    if not new_rows:
        save_state(state)
        logger.info(f"All flagged bets for {date} were already logged — nothing new to add.")
        return

    new_rows_df = pd.DataFrame(new_rows)
    new_rows_df["result"] = new_rows_df["result"].astype(object)
    bet_log = pd.concat([bet_log, new_rows_df], ignore_index=True)
    atomic_write_csv(bet_log, BET_LOG_PATH)
    save_state(state)
    logger.info(f"Logged {len(new_rows)} bets for {date} to {BET_LOG_PATH} (total stake={sum(r['bet_size'] for r in new_rows):.2f})")


def load_closing_lines():
    files = sorted(glob.glob(str(RAW / "odds_close_*.csv")))
    if not files:
        return None
    return pd.concat([pd.read_csv(f, dtype={"game_pk": str}) for f in files], ignore_index=True)


def cmd_settle():
    bet_log = load_bet_log()
    pending = bet_log[bet_log["status"].isin(["pending", "postponed"])]
    if pending.empty:
        logger.info("No pending bets to settle.")
        return

    game_logs_path = RAW / "game_logs_all.csv"
    if not game_logs_path.exists():
        logger.error(f"{game_logs_path} not found — cannot settle bets without results.")
        sys.exit(1)
    game_logs = pd.read_csv(game_logs_path, dtype={"game_pk": str})
    results = game_logs[game_logs["is_home"] == 1][["game_pk", "win"]].rename(columns={"win": "home_win"})

    closing = load_closing_lines()

    state = reconcile_state_from_ledger(load_state(), bet_log)
    n_settled = 0
    n_postponed = 0
    now = datetime.now()
    for idx, row in pending.iterrows():
        match = results[results["game_pk"] == row["game_pk"]]
        if match.empty:
            if row["status"] == "pending" and pd.notna(row.get("decision_timestamp")):
                decided = pd.to_datetime(row["decision_timestamp"], utc=True, errors="coerce")
                if pd.notna(decided) and pd.Timestamp.now(tz="UTC") - decided >= timedelta(days=POSTPONED_GRACE_DAYS):
                    bet_log.loc[idx, "status"] = "postponed"
                    n_postponed += 1
                    logger.warning(f"Bet {row.get('decision_id', idx)} ({row['home_team']} vs {row['away_team']}, "
                                   f"logged {row['date']}) has no result after {POSTPONED_GRACE_DAYS}+ days — "
                                   f"flagging as postponed. Will settle automatically once its game_pk appears.")
            continue  # game hasn't been played/collected yet

        home_won = bool(match["home_win"].iloc[0])
        bet_won = home_won if row["side"] == "HOME" else not home_won
        decimal_odds = american_to_decimal(row["odds"])
        pnl = row["bet_size"] * (decimal_odds - 1) if bet_won else -row["bet_size"]

        closing_no_vig = np.nan
        if closing is not None and "game_pk" in closing.columns:
            crow = closing[closing["game_pk"] == row["game_pk"]]
            if not crow.empty:
                closing_no_vig = crow["home_no_vig_prob"].iloc[0] if row["side"] == "HOME" \
                    else 1 - crow["home_no_vig_prob"].iloc[0]

        clv = closing_no_vig - row["no_vig_prob"] if pd.notna(closing_no_vig) else np.nan

        bet_log.loc[idx, "result"] = "win" if bet_won else "loss"
        bet_log.loc[idx, "pnl"] = round(float(pnl), 2)
        bet_log.loc[idx, "closing_no_vig_prob"] = closing_no_vig
        bet_log.loc[idx, "clv"] = clv
        bet_log.loc[idx, "status"] = "settled"
        bet_log.loc[idx, "settled_timestamp"] = utc_now_iso()

        state["bankroll"] += pnl
        n_settled += 1

    if n_postponed:
        atomic_write_csv(bet_log, BET_LOG_PATH)
    if n_settled == 0:
        logger.info(f"No pending bets had known results yet ({n_postponed} newly flagged postponed).")
        return

    state["peak_bankroll"] = max(state["peak_bankroll"], state["bankroll"])
    drawdown = 1 - (state["bankroll"] / state["peak_bankroll"]) if state["peak_bankroll"] > 0 else 0.0
    if drawdown >= DRAWDOWN_PAUSE_THRESHOLD and not state["is_paused"]:
        state["is_paused"] = True
        logger.warning(f"Drawdown {drawdown:.1%} >= {DRAWDOWN_PAUSE_THRESHOLD:.0%} threshold — bankroll PAUSED. "
                        f"--log-bets will be blocked until --resume.")

    atomic_write_csv(bet_log, BET_LOG_PATH)
    save_state(state)
    logger.info(f"Settled {n_settled} bets. Bankroll={state['bankroll']:.2f} "
                f"(peak={state['peak_bankroll']:.2f}, drawdown={drawdown:.1%})")

    if closing is None:
        logger.warning("No data/raw/odds_close_*.csv found — CLV could not be computed for any settled bet.")


def cmd_void(game_pks, reason):
    """Explicitly void cancelled/no-action games without changing bankroll."""
    bet_log = load_bet_log()
    targets = {str(game_pk) for game_pk in game_pks}
    mask = bet_log["game_pk"].astype(str).isin(targets) & bet_log["status"].isin(["pending", "postponed"])
    if not mask.any():
        logger.error(f"No pending/postponed bets matched game_pk values {sorted(targets)}")
        sys.exit(1)
    bet_log.loc[mask, "result"] = "void"
    bet_log.loc[mask, "pnl"] = 0.0
    bet_log.loc[mask, "status"] = "void"
    bet_log.loc[mask, "settled_timestamp"] = utc_now_iso()
    if "void_reason" not in bet_log.columns:
        bet_log["void_reason"] = np.nan
    bet_log.loc[mask, "void_reason"] = reason
    atomic_write_csv(bet_log, BET_LOG_PATH)
    save_state(reconcile_state_from_ledger(load_state(), bet_log))
    logger.warning(f"Voided {int(mask.sum())} bet(s): {reason}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--date")
    parser.add_argument("--log-bets", action="store_true")
    parser.add_argument("--settle", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--game-pks", help="comma-separated game_pk values eligible for this logging pass")
    parser.add_argument("--void-game-pks", help="comma-separated cancelled/no-action game_pk values to void")
    parser.add_argument("--void-reason", default="cancelled/no action")
    args = parser.parse_args()

    with exclusive_lock(MODELS / "bankroll.lock"):
        if args.log_bets:
            if not args.date:
                logger.error("--log-bets requires --date YYYY-MM-DD")
                sys.exit(1)
            game_pks = [value.strip() for value in args.game_pks.split(",") if value.strip()] if args.game_pks else None
            cmd_log_bets(args.date, args.resume, args.force, game_pks=game_pks)
        if args.settle:
            cmd_settle()
        if args.void_game_pks:
            cmd_void([value.strip() for value in args.void_game_pks.split(",") if value.strip()], args.void_reason)
    if not args.log_bets and not args.settle and not args.void_game_pks:
        logger.error("Specify --log-bets, --settle, and/or --void-game-pks")
        sys.exit(1)


if __name__ == "__main__":
    main()
