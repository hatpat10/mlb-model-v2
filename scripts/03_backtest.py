# -*- coding: utf-8 -*-
"""Backtests the calibrated model on the 2024 holdout: reports AUC/accuracy/
Brier, then computes moneyline ROI against closing lines — using REAL
closing-line snapshots from data/raw/odds_close_*.csv if present, otherwise
falling back to a clearly-labeled SYNTHETIC proxy (the Elo-implied win
probability) so the report can never be mistaken for a real-market edge.
"""
import sys
import glob
import json
from pathlib import Path

import numpy as np
import pandas as pd
import joblib
from loguru import logger
from sklearn.metrics import roc_auc_score, accuracy_score, brier_score_loss

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from config.config import PATHS, TEST_YEAR, MIN_EDGE_MONEYLINE, MAX_EDGE_MONEYLINE  # noqa: E402
from model_classes import WeightedEnsembleClassifier, PreFitCalibratedClassifier  # noqa: E402

RAW = PATHS["raw"]
PROCESSED = PATHS["processed"]
MODELS = PATHS["models"]
OUTPUTS = PATHS["outputs"]
LOGS = PATHS["logs"]

logger.remove()
logger.add(sys.stderr, level="INFO")
logger.add(LOGS / "03_backtest.log", level="DEBUG", rotation="5 MB")


def load_holdout_predictions():
    with open(MODELS / "feature_names.json") as f:
        features = json.load(f)["passing_features"]
    medians = joblib.load(MODELS / "train_medians.joblib")
    xgb_cal = joblib.load(MODELS / "xgb_calibrated.joblib")
    lgbm_cal = joblib.load(MODELS / "lgbm_calibrated.joblib")

    df = pd.read_csv(PROCESSED / "feature_matrix.csv")
    test_df = df[df["year"] == TEST_YEAR].reset_index(drop=True)
    X_test = test_df[features].fillna(medians)

    home_prob = (xgb_cal.predict_proba(X_test)[:, 1] + lgbm_cal.predict_proba(X_test)[:, 1]) / 2.0
    test_df = test_df.copy()
    test_df["model_home_prob"] = home_prob
    return test_df


def evaluate_accuracy(test_df):
    y = test_df["home_win"].values
    p = test_df["model_home_prob"].values
    auc = roc_auc_score(y, p)
    acc = accuracy_score(y, (p >= 0.5).astype(int))
    brier = brier_score_loss(y, p)
    logger.info(f"2024 holdout — AUC={auc:.4f}  Accuracy={acc:.4f}  Brier={brier:.4f}")
    return auc, acc, brier


def load_real_closing_lines():
    files = sorted(glob.glob(str(RAW / "odds_close_*.csv")))
    if not files:
        return None
    frames = [pd.read_csv(f) for f in files]
    odds = pd.concat(frames, ignore_index=True)
    odds["game_pk"] = odds["game_pk"].astype(str)
    return odds


def moneyline_roi(test_df, home_market_prob, label):
    """Bets whenever |model_prob - market_prob| is within
    [MIN_EDGE_MONEYLINE, MAX_EDGE_MONEYLINE) — MAX is a hard cap: an edge
    that large against the market is far more likely a data/model bug than
    a real inefficiency, so those games are skipped, not bet.
    """
    edge = test_df["model_home_prob"].values - home_market_prob
    bet_home = (edge >= MIN_EDGE_MONEYLINE) & (edge < MAX_EDGE_MONEYLINE)
    bet_away = (-edge >= MIN_EDGE_MONEYLINE) & (-edge < MAX_EDGE_MONEYLINE)

    home_win = test_df["home_win"].values
    pnl = np.zeros(len(test_df))

    home_decimal_odds = np.divide(1.0, home_market_prob, out=np.full_like(home_market_prob, np.nan), where=home_market_prob > 0)
    away_decimal_odds = np.divide(1.0, (1 - home_market_prob), out=np.full_like(home_market_prob, np.nan), where=(1 - home_market_prob) > 0)

    pnl[bet_home & (home_win == 1)] = (home_decimal_odds - 1)[bet_home & (home_win == 1)]
    pnl[bet_home & (home_win == 0)] = -1
    pnl[bet_away & (home_win == 0)] = (away_decimal_odds - 1)[bet_away & (home_win == 0)]
    pnl[bet_away & (home_win == 1)] = -1

    n_bets = int((bet_home | bet_away).sum())
    total_pnl = pnl[bet_home | bet_away].sum()
    roi = total_pnl / n_bets if n_bets > 0 else np.nan

    logger.info(f"ROI vs {label} closing lines: {n_bets} bets placed / {len(test_df)} games, "
                f"total PnL={total_pnl:.2f} units, ROI={roi:.2%}" if n_bets > 0 else
                f"ROI vs {label} closing lines: 0 bets placed (no games met the edge threshold)")
    return {"label": label, "n_bets": n_bets, "total_pnl": float(total_pnl), "roi": float(roi) if n_bets > 0 else None}


def main():
    test_df = load_holdout_predictions()
    auc, acc, brier = evaluate_accuracy(test_df)

    real_odds = load_real_closing_lines()
    results = {"auc": auc, "accuracy": acc, "brier": brier}

    if real_odds is not None:
        # feature_matrix game_pk reads back as int64; odds files store it as
        # str — without this cast the merge silently matches zero rows.
        test_df = test_df.copy()
        test_df["game_pk"] = test_df["game_pk"].astype(str)
        merged = test_df.merge(real_odds, on="game_pk", how="inner")
        logger.info(f"Found REAL closing-line data for {len(merged)}/{len(test_df)} holdout games.")
        if len(merged) > 0:
            results["roi_real"] = moneyline_roi(merged, merged["home_no_vig_prob"].values, label="REAL")
        else:
            logger.warning("No holdout games matched real closing-line data by game_pk.")
    else:
        logger.warning(
            "No data/raw/odds_close_*.csv found — no real closing-line data was collected for 2024. "
            "Falling back to a SYNTHETIC proxy (Elo-implied win probability) purely to illustrate the "
            "ROI methodology. This is NOT a real market edge and must not be treated as one."
        )
        synthetic_prob = test_df["elo_win_prob"].fillna(0.5).values
        results["roi_synthetic"] = moneyline_roi(test_df, synthetic_prob, label="SYNTHETIC (Elo-implied)")

    out_path = OUTPUTS / f"backtest_{TEST_YEAR}.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    logger.info(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
