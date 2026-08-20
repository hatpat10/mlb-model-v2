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
from config.config import (  # noqa: E402
    PATHS, TEST_YEAR, MIN_EDGE_MONEYLINE, MAX_EDGE_MONEYLINE,
    MIN_STRATEGY_VALIDATION_BETS, MIN_STRATEGY_VALIDATION_COVERAGE,
)
from odds_utils import american_to_decimal_odds  # noqa: E402
from betting_strategy import american_to_decimal, size_stake  # noqa: E402
from artifact_utils import atomic_write_json  # noqa: E402
from model_registry import resolve_production_model_dir  # noqa: E402
from model_scoring import load_model_manifest, score_probability_bundle  # noqa: E402
from performance_metrics import date_block_roi_ci  # noqa: E402

RAW = PATHS["raw"]
PROCESSED = PATHS["processed"]
MODELS = PATHS["models"]
OUTPUTS = PATHS["outputs"]
LOGS = PATHS["logs"]

logger.remove()
logger.add(sys.stderr, level="INFO")
logger.add(LOGS / "03_backtest.log", level="DEBUG", rotation="5 MB")


def load_scored_rows(model_dir=MODELS, start_year=TEST_YEAR):
    with open(model_dir / "feature_names.json") as f:
        features = json.load(f)["passing_features"]
    medians = joblib.load(model_dir / "train_medians.joblib")

    df = pd.read_csv(PROCESSED / "feature_matrix.csv")
    scored = df[df["year"] >= start_year].reset_index(drop=True)
    X_test = scored[features].fillna(medians)

    probability = score_probability_bundle(
        model_dir, X_test, manifest=load_model_manifest(model_dir),
    )
    scored = scored.copy()
    scored["xgb_home_prob"] = probability["xgb_home_probability"]
    scored["lgbm_home_prob"] = probability["lgbm_home_probability"]
    scored["model_disagreement"] = probability["model_disagreement"]
    scored["model_home_prob"] = probability["home_probability"]
    scored["prediction_model_type"] = probability["prediction_model_type"]
    return scored


def evaluate_accuracy(test_df, label="2024 holdout"):
    y = test_df["home_win"].values
    p = test_df["model_home_prob"].values
    auc = roc_auc_score(y, p)
    acc = accuracy_score(y, (p >= 0.5).astype(int))
    brier = brier_score_loss(y, p)
    logger.info(f"{label} — AUC={auc:.4f}  Accuracy={acc:.4f}  Brier={brier:.4f}")
    return auc, acc, brier


def compare_model_to_market(scored_with_odds):
    """Compare model and no-vig market probabilities on identical games."""
    valid = scored_with_odds[["home_win", "model_home_prob", "home_no_vig_prob"]].dropna()
    if valid.empty or valid["home_win"].nunique() < 2:
        return None
    y = valid["home_win"].astype(int)
    model_prob = valid["model_home_prob"].astype(float)
    market_prob = valid["home_no_vig_prob"].astype(float)
    model_auc = roc_auc_score(y, model_prob)
    market_auc = roc_auc_score(y, market_prob)
    model_brier = brier_score_loss(y, model_prob)
    market_brier = brier_score_loss(y, market_prob)
    return {
        "n_games": int(len(valid)),
        "model_auc": float(model_auc), "market_auc": float(market_auc),
        "model_minus_market_auc": float(model_auc - market_auc),
        "model_brier": float(model_brier), "market_brier": float(market_brier),
        "model_minus_market_brier": float(model_brier - market_brier),
        "brier_note": "Negative model-minus-market is better.",
    }


def disagreement_diagnostics(scored):
    """Measure whether component-model disagreement identifies weak predictions."""
    valid = scored[["home_win", "model_home_prob", "model_disagreement"]].dropna().copy()
    if valid.empty:
        return None
    valid["absolute_error"] = (valid["home_win"] - valid["model_home_prob"]).abs()
    valid["squared_error"] = (valid["home_win"] - valid["model_home_prob"]) ** 2
    correlation = valid["model_disagreement"].corr(valid["absolute_error"])
    labels = ["lowest", "low_mid", "high_mid", "highest"]
    valid["quartile"] = pd.qcut(
        valid["model_disagreement"].rank(method="first"), 4, labels=labels,
    )
    by_quartile = {}
    for label, group in valid.groupby("quartile", observed=True):
        by_quartile[str(label)] = {
            "n_games": int(len(group)),
            "mean_disagreement": float(group["model_disagreement"].mean()),
            "brier": float(group["squared_error"].mean()),
        }
    return {
        "mean": float(valid["model_disagreement"].mean()),
        "p95": float(valid["model_disagreement"].quantile(0.95)),
        "maximum": float(valid["model_disagreement"].max()),
        "correlation_with_absolute_error": float(correlation) if pd.notna(correlation) else None,
        "by_disagreement_quartile": by_quartile,
    }


def price_method_summary(real_odds):
    if "price_selection_method" not in real_odds.columns:
        return {"legacy_representative_book_close": int(len(real_odds))}
    methods = real_odds["price_selection_method"].fillna("legacy_representative_book_close").astype(str)
    return {key: int(value) for key, value in methods.value_counts().items()}


def load_real_closing_lines():
    files = sorted(glob.glob(str(RAW / "odds_close_*.csv")))
    if not files:
        return None
    frames = [pd.read_csv(f) for f in files]
    odds = pd.concat(frames, ignore_index=True)
    odds["game_pk"] = odds["game_pk"].astype(str)
    return odds


def moneyline_roi(test_df, home_fair_prob, label, home_ml_close=None, away_ml_close=None):
    """Bets whenever |model_prob - home_fair_prob| is within
    [MIN_EDGE_MONEYLINE, MAX_EDGE_MONEYLINE) — MAX is a hard cap: an edge
    that large against the market is far more likely a data/model bug than
    a real inefficiency, so those games are skipped, not bet. `home_fair_prob`
    (the de-vigged fair probability) is only ever used to size the EDGE —
    for real closing lines, payout is priced off the actual quoted
    home_ml_close/away_ml_close via american_to_decimal_odds, since pricing
    payout off the fair probability instead (1/home_fair_prob) silently
    strips out the sportsbook's margin and overstates ROI. The synthetic
    Elo-proxy path has no real quoted price, so it has no choice but to
    price off the fair probability too — already disclaimed at the call site.
    """
    edge = test_df["model_home_prob"].values - home_fair_prob
    bet_home = (edge >= MIN_EDGE_MONEYLINE) & (edge < MAX_EDGE_MONEYLINE)
    bet_away = (-edge >= MIN_EDGE_MONEYLINE) & (-edge < MAX_EDGE_MONEYLINE)

    home_win = test_df["home_win"].values
    if home_ml_close is None or away_ml_close is None:
        # The proxy is intentionally unit-staked and cannot validate the
        # deployable sizing strategy because no bookmaker quote existed.
        home_decimal_odds = 1.0 / home_fair_prob
        away_decimal_odds = 1.0 / (1.0 - home_fair_prob)
        pnl = np.zeros(len(test_df))
        pnl[bet_home & (home_win == 1)] = (home_decimal_odds - 1)[bet_home & (home_win == 1)]
        pnl[bet_home & (home_win == 0)] = -1
        pnl[bet_away & (home_win == 0)] = (away_decimal_odds - 1)[bet_away & (home_win == 0)]
        pnl[bet_away & (home_win == 1)] = -1
        n_bets = int((bet_home | bet_away).sum())
        total_staked = float(n_bets)
        total_pnl = float(pnl[bet_home | bet_away].sum())
        final_bankroll = None
        max_drawdown = None
    else:
        simulation = test_df.copy()
        simulation["_fair"] = home_fair_prob
        simulation["_bet_home"] = bet_home
        simulation["_bet_away"] = bet_away
        simulation["_home_ml"] = home_ml_close
        simulation["_away_ml"] = away_ml_close
        simulation = simulation.sort_values(["date", "game_pk"]).reset_index(drop=True)
        bankroll, peak, max_drawdown = 10000.0, 10000.0, 0.0
        bet_records = []
        total_staked, total_pnl, n_bets = 0.0, 0.0, 0
        for _, day in simulation.groupby("date", sort=True):
            daily_staked = 0.0
            day_pnl = 0.0
            for _, row in day.iterrows():
                if not (row["_bet_home"] or row["_bet_away"]):
                    continue
                side_home = bool(row["_bet_home"])
                probability = row["model_home_prob"] if side_home else 1.0 - row["model_home_prob"]
                odds = row["_home_ml"] if side_home else row["_away_ml"]
                if pd.isna(odds):
                    continue
                sizing = size_stake(probability, odds, bankroll, daily_staked=daily_staked, open_staked=daily_staked)
                if sizing.stake <= 0:
                    continue
                won = bool(row["home_win"]) if side_home else not bool(row["home_win"])
                bet_pnl = sizing.stake * (american_to_decimal(odds) - 1.0) if won else -sizing.stake
                daily_staked += sizing.stake
                total_staked += sizing.stake
                total_pnl += bet_pnl
                day_pnl += bet_pnl
                n_bets += 1
                bet_records.append({"date": str(row["date"]), "pnl": bet_pnl, "stake": sizing.stake})
            bankroll += day_pnl
            peak = max(peak, bankroll)
            max_drawdown = max(max_drawdown, 1.0 - bankroll / peak)
        final_bankroll = bankroll
        bet_records = pd.DataFrame(bet_records)
        roi_ci_95 = date_block_roi_ci(bet_records)
    roi = total_pnl / total_staked if total_staked > 0 else np.nan

    logger.info(f"ROI vs {label} closing lines: {n_bets} bets placed / {len(test_df)} games, "
                f"total stake={total_staked:.2f}, PnL={total_pnl:.2f}, ROI={roi:.2%}" if n_bets > 0 else
                f"ROI vs {label} closing lines: 0 bets placed (no games met the edge threshold)")
    return {
        "label": label, "n_bets": n_bets, "total_staked": float(total_staked),
        "total_pnl": float(total_pnl), "roi": float(roi) if n_bets > 0 else None,
        "starting_bankroll": 10000.0 if final_bankroll is not None else None,
        "final_bankroll": float(final_bankroll) if final_bankroll is not None else None,
        "max_drawdown": float(max_drawdown) if max_drawdown is not None else None,
        "roi_ci_95": roi_ci_95 if final_bankroll is not None else None,
        "n_betting_days": int(bet_records["date"].nunique()) if final_bankroll is not None and not bet_records.empty else 0,
        "bootstrap_unit": "betting_date" if final_bankroll is not None else None,
        "evaluation_timing": "retrospective_at_captured_close" if final_bankroll is not None else "synthetic_proxy",
        "forward_performance": False,
    }


def main():
    scored = load_scored_rows()
    test_df = scored[scored["year"] == TEST_YEAR].reset_index(drop=True)
    auc, acc, brier = evaluate_accuracy(test_df)

    real_odds = load_real_closing_lines()
    results = {
        "report_type": "retrospective_model_and_closing_price_evaluation",
        "interpretation": (
            "Captured closing prices are used as both the decision market and payout price. "
            "This is a close-time retrospective, not evidence of prices available at the live T-2h decision."
        ),
        "auc": auc, "accuracy": acc, "brier": brier,
        "prediction_model_type": test_df["prediction_model_type"].iloc[0] if not test_df.empty else None,
        "model_disagreement": disagreement_diagnostics(test_df),
    }

    if real_odds is not None:
        results["closing_price_selection_methods"] = price_method_summary(real_odds)
        # feature_matrix game_pk reads back as int64; odds files store it as
        # str — without this cast the merge silently matches zero rows.
        oos = scored.copy()
        oos["game_pk"] = oos["game_pk"].astype(str)
        odds_for_merge = real_odds.drop_duplicates("game_pk", keep="last").drop(
            columns=[column for column in ("date", "home_team", "away_team") if column in real_odds.columns]
        )
        merged = oos.merge(odds_for_merge, on="game_pk", how="inner")
        collected_dates = set(pd.to_datetime(real_odds["date"]).dt.strftime("%Y-%m-%d")) if "date" in real_odds else set()
        games_on_collected_dates = oos[pd.to_datetime(oos["date"]).dt.strftime("%Y-%m-%d").isin(collected_dates)]
        coverage = len(merged) / len(games_on_collected_dates) if len(games_on_collected_dates) else 0.0
        logger.info(f"Found REAL closing-line data for {len(merged)}/{len(games_on_collected_dates)} "
                    f"out-of-sample games on captured dates ({coverage:.1%} coverage).")
        if len(merged) > 0:
            results["roi_real"] = moneyline_roi(
                merged, merged["home_no_vig_prob"].values, label="REAL",
                home_ml_close=merged["home_ml_close"].values, away_ml_close=merged["away_ml_close"].values,
            )
            results["roi_real"]["market_coverage"] = coverage
            results["market_comparison"] = compare_model_to_market(merged)
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

    # False until a real-price backtest has actually run with enough bets to
    # mean anything — never true from the synthetic Elo-proxy path, and not
    # true just because a real-price run happened to place zero bets.
    production_dir = resolve_production_model_dir(MODELS)
    if real_odds is not None and production_dir is not None:
        with open(production_dir / "manifest.json", encoding="utf-8") as handle:
            production_manifest = json.load(handle)
        production_start = max(production_manifest["training_years"]) + 1
        production_scored = load_scored_rows(production_dir, production_start)
        if not production_scored.empty:
            pa, pc, pb = evaluate_accuracy(production_scored, f"production {production_start}+ out-of-sample")
            results["production_accuracy"] = {
                "auc": pa, "accuracy": pc, "brier": pb,
                "prediction_model_type": production_scored["prediction_model_type"].iloc[0],
            }
            results["production_model_disagreement"] = disagreement_diagnostics(production_scored)
            production_scored["game_pk"] = production_scored["game_pk"].astype(str)
            production_odds = real_odds.drop_duplicates("game_pk", keep="last").drop(
                columns=[column for column in ("date", "home_team", "away_team") if column in real_odds.columns]
            )
            production_merged = production_scored.merge(production_odds, on="game_pk", how="inner")
            collected_dates = set(pd.to_datetime(real_odds["date"]).dt.strftime("%Y-%m-%d")) if "date" in real_odds else set()
            production_denominator = production_scored[
                pd.to_datetime(production_scored["date"]).dt.strftime("%Y-%m-%d").isin(collected_dates)
            ]
            production_coverage = len(production_merged) / len(production_denominator) if len(production_denominator) else 0.0
            if not production_merged.empty:
                results["roi_production_real"] = moneyline_roi(
                    production_merged, production_merged["home_no_vig_prob"].values,
                    label="PRODUCTION REAL",
                    home_ml_close=production_merged["home_ml_close"].values,
                    away_ml_close=production_merged["away_ml_close"].values,
                )
                results["roi_production_real"]["market_coverage"] = production_coverage
                results["roi_production_real"]["model_version"] = production_manifest.get("model_version")
                results["roi_production_real"]["prediction_model_type"] = production_manifest.get("prediction_model_type")
                results["production_market_comparison"] = compare_model_to_market(production_merged)

    real_result = results.get("roi_production_real", results.get("roi_real", {}))
    results["validation_thresholds"] = {
        "minimum_real_bets": MIN_STRATEGY_VALIDATION_BETS,
        "minimum_market_coverage": MIN_STRATEGY_VALIDATION_COVERAGE,
    }
    results["strategy_validated_retrospective"] = bool(
        real_result.get("n_bets", 0) >= MIN_STRATEGY_VALIDATION_BETS
        and real_result.get("market_coverage", 0.0) >= MIN_STRATEGY_VALIDATION_COVERAGE
        and real_result.get("roi_ci_95") is not None
        and real_result["roi_ci_95"][0] > 0
    )
    # Backward-compatible name, but the report now states explicitly that
    # passing these gates only validates a close-time retrospective. The
    # deployment gate lives in 08_forward_performance.py.
    results["strategy_validated"] = results["strategy_validated_retrospective"]

    out_path = OUTPUTS / f"backtest_{TEST_YEAR}.json"
    atomic_write_json(results, out_path)
    logger.info(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
