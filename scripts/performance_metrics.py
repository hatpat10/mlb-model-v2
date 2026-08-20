# -*- coding: utf-8 -*-
"""Shared, slate-aware performance metrics for backtests and forward logs."""
from __future__ import annotations

import numpy as np
import pandas as pd


def date_block_roi_ci(records: pd.DataFrame, n_bootstrap: int = 5000, seed: int = 42):
    """Bootstrap ROI by betting date, preserving within-slate correlation.

    Individual bets on one slate share weather, market, lineup and exposure
    conditions, so treating them as independent observations produces an
    unjustifiably narrow interval. Resampling whole dates is conservative and
    matches how bankroll exposure is actually managed.
    """
    if records.empty:
        return None
    daily = records.assign(
        pnl=pd.to_numeric(records["pnl"], errors="coerce"),
        stake=pd.to_numeric(records["stake"], errors="coerce"),
    ).dropna(subset=["date", "pnl", "stake"]).groupby("date", sort=True)[["pnl", "stake"]].sum()
    daily = daily[daily["stake"] > 0]
    if len(daily) < 2:
        return None

    rng = np.random.default_rng(seed)
    indices = rng.integers(0, len(daily), size=(n_bootstrap, len(daily)))
    pnl = daily["pnl"].to_numpy()[indices].sum(axis=1)
    stake = daily["stake"].to_numpy()[indices].sum(axis=1)
    roi = np.divide(pnl, stake, out=np.full_like(pnl, np.nan, dtype=float), where=stake > 0)
    roi = roi[np.isfinite(roi)]
    if not len(roi):
        return None
    return [float(value) for value in np.quantile(roi, [0.025, 0.975])]


def summarize_settled_bets(bets: pd.DataFrame, starting_bankroll: float = 10000.0) -> dict:
    """Summarize already-settled ledger rows without inventing missing data."""
    if bets.empty:
        return {
            "n_bets": 0, "n_betting_days": 0, "wins": 0, "losses": 0,
            "win_rate": None, "total_staked": 0.0, "total_pnl": 0.0,
            "roi": None, "starting_bankroll": float(starting_bankroll),
            "final_bankroll": float(starting_bankroll), "max_drawdown": None,
            "roi_ci_95_date_block": None, "clv_coverage": 0.0,
            "mean_clv": None, "median_clv": None,
        }

    frame = bets.copy()
    frame["date"] = pd.to_datetime(frame["date"], errors="coerce").dt.strftime("%Y-%m-%d")
    frame["stake"] = pd.to_numeric(frame["bet_size"], errors="coerce")
    frame["pnl"] = pd.to_numeric(frame["pnl"], errors="coerce")
    frame = frame.dropna(subset=["date", "stake", "pnl"])
    frame = frame[frame["stake"] > 0]
    total_staked = float(frame["stake"].sum())
    total_pnl = float(frame["pnl"].sum())

    daily_pnl = frame.groupby("date", sort=True)["pnl"].sum()
    equity = float(starting_bankroll) + daily_pnl.cumsum()
    running_peak = pd.concat([
        pd.Series([float(starting_bankroll)]), equity.reset_index(drop=True)
    ], ignore_index=True).cummax().iloc[1:].to_numpy()
    drawdowns = 1.0 - equity.to_numpy() / running_peak if len(equity) else np.array([])

    result = frame.get("result", pd.Series(index=frame.index, dtype=object)).astype(str).str.lower()
    wins = int(result.eq("win").sum())
    losses = int(result.eq("loss").sum())
    clv = pd.to_numeric(frame.get("clv", pd.Series(index=frame.index, dtype=float)), errors="coerce")
    known_clv = clv.dropna()
    records = frame[["date", "pnl", "stake"]]
    return {
        "n_bets": int(len(frame)),
        "n_betting_days": int(frame["date"].nunique()),
        "wins": wins,
        "losses": losses,
        "win_rate": float(wins / (wins + losses)) if wins + losses else None,
        "total_staked": total_staked,
        "total_pnl": total_pnl,
        "roi": float(total_pnl / total_staked) if total_staked else None,
        "starting_bankroll": float(starting_bankroll),
        "final_bankroll": float(starting_bankroll + total_pnl),
        "max_drawdown": float(max(drawdowns, default=0.0)),
        "roi_ci_95_date_block": date_block_roi_ci(records),
        "clv_coverage": float(len(known_clv) / len(frame)) if len(frame) else 0.0,
        "mean_clv": float(known_clv.mean()) if len(known_clv) else None,
        "median_clv": float(known_clv.median()) if len(known_clv) else None,
    }
