# -*- coding: utf-8 -*-
"""Trains a per-market player-props regression model: tunes XGBoost and
LightGBM (Poisson objective — every props target is a small non-negative
count, not a 0/1 outcome, so this mirrors 02_train.py's model family and
tuning discipline but not its classifier/isotonic-calibration machinery,
which doesn't apply to a count target) via Optuna + TimeSeriesSplit CV,
evaluates avg(XGB, LGBM) on the untouched TEST_YEAR holdout against both
regression metrics and a naive "just use the player's own rolling average"
baseline, and saves per-market artifacts to models/props/.

Usage:
    python scripts/props_train.py --matrix batter --market h
    python scripts/props_train.py --matrix pitcher --market k

No real sportsbook O/U line exists yet for any prop market (see
docs/adr — that's an open, separate decision), so this predicts the
expected COUNT directly rather than a fixed threshold. Once real lines
are captured, a predicted Poisson mean converts to P(over any line) via
the Poisson CDF for any player — strictly more useful than a model baked
to one arbitrary threshold would have been.
"""
import sys
import json
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import joblib
import optuna
from loguru import logger
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_squared_error, mean_absolute_error, mean_poisson_deviance
from scipy.stats import pearsonr
import xgboost as xgb
import lightgbm as lgb

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config.config import PATHS, TRAIN_YEARS, TEST_YEAR  # noqa: E402

optuna.logging.set_verbosity(optuna.logging.WARNING)

PROCESSED = PATHS["processed"]
MODELS = PATHS["models"] / "props"
OUTPUTS = PATHS["outputs"]
LOGS = PATHS["logs"]
N_SPLITS = 3
N_OPTUNA_TRIALS = 60

# Matrix-specific config: which feature matrix, which id/name columns
# (for reporting), and a same-grain rolling-average column to use as the
# naive baseline for whichever --market is picked.
MATRIX_CONFIG = {
    "batter": {
        "feature_matrix": "props_feature_matrix.csv",
        "feature_names": "props_feature_names.json",
        "id_col": "batter_mlbam_id",
        "name_col": "batter_name",
        "roll_window": 20,
    },
    "pitcher": {
        "feature_matrix": "pitcher_props_feature_matrix.csv",
        "feature_names": "pitcher_props_feature_names.json",
        "id_col": "pitcher_mlbam_id",
        "name_col": "pitcher_name",
        "roll_window": 10,
    },
}

logger.remove()
logger.add(sys.stderr, level="INFO")
logger.add(LOGS / "props_train.log", level="DEBUG", rotation="5 MB")


def load_data(matrix: str, market: str):
    cfg = MATRIX_CONFIG[matrix]
    fm_path = PROCESSED / cfg["feature_matrix"]
    names_path = PROCESSED / cfg["feature_names"]
    if not fm_path.exists() or not names_path.exists():
        logger.error(f"{fm_path.name} / {names_path.name} missing — run the props feature builder first.")
        sys.exit(1)

    df = pd.read_csv(fm_path)
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date").reset_index(drop=True)

    if market not in df.columns:
        logger.error(f"market '{market}' is not a column in {cfg['feature_matrix']}")
        sys.exit(1)

    with open(names_path) as f:
        names = json.load(f)
    features = names["passing_features"]

    train_df = df[df["year"].isin(TRAIN_YEARS)].reset_index(drop=True)
    test_df = df[df["year"] == TEST_YEAR].reset_index(drop=True)
    logger.info(f"train (years {TRAIN_YEARS}): {len(train_df)} rows | test (year {TEST_YEAR}): {len(test_df)} rows")
    return train_df, test_df, features, cfg


def median_impute(train_df, test_df, features, stem):
    medians = train_df[features].median()
    train_X = train_df[features].fillna(medians)
    test_X = test_df[features].fillna(medians)
    joblib.dump(medians, MODELS / f"{stem}_medians.joblib")
    logger.info(f"Median-imputed {len(features)} features using train-only medians.")
    return train_X, test_X


def tune_xgb(X, y, tscv):
    def objective(trial):
        params = {
            "max_depth": trial.suggest_int("max_depth", 2, 6),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
            "n_estimators": trial.suggest_int("n_estimators", 50, 400),
            "subsample": trial.suggest_float("subsample", 0.5, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
            "min_child_weight": trial.suggest_int("min_child_weight", 1, 10),
            "reg_alpha": trial.suggest_float("reg_alpha", 1e-4, 10.0, log=True),
            "reg_lambda": trial.suggest_float("reg_lambda", 1e-4, 10.0, log=True),
        }
        devs = []
        for tr_idx, val_idx in tscv.split(X):
            model = xgb.XGBRegressor(**params, objective="count:poisson", random_state=42)
            model.fit(X.iloc[tr_idx], y.iloc[tr_idx])
            preds = np.clip(model.predict(X.iloc[val_idx]), 1e-6, None)
            devs.append(mean_poisson_deviance(y.iloc[val_idx], preds))
        return float(np.mean(devs))

    study = optuna.create_study(direction="minimize")
    study.optimize(objective, n_trials=N_OPTUNA_TRIALS, show_progress_bar=False)
    logger.info(f"XGBoost: best CV Poisson deviance={study.best_value:.4f}, params={study.best_params}")
    return study.best_params


def tune_lgbm(X, y, tscv):
    def objective(trial):
        params = {
            "max_depth": trial.suggest_int("max_depth", 2, 8),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
            "n_estimators": trial.suggest_int("n_estimators", 50, 400),
            "subsample": trial.suggest_float("subsample", 0.5, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
            "min_child_samples": trial.suggest_int("min_child_samples", 5, 50),
            "reg_alpha": trial.suggest_float("reg_alpha", 1e-4, 10.0, log=True),
            "reg_lambda": trial.suggest_float("reg_lambda", 1e-4, 10.0, log=True),
            "num_leaves": trial.suggest_int("num_leaves", 8, 64),
        }
        devs = []
        for tr_idx, val_idx in tscv.split(X):
            model = lgb.LGBMRegressor(**params, objective="poisson", random_state=42, verbosity=-1)
            model.fit(X.iloc[tr_idx], y.iloc[tr_idx])
            preds = np.clip(model.predict(X.iloc[val_idx]), 1e-6, None)
            devs.append(mean_poisson_deviance(y.iloc[val_idx], preds))
        return float(np.mean(devs))

    study = optuna.create_study(direction="minimize")
    study.optimize(objective, n_trials=N_OPTUNA_TRIALS, show_progress_bar=False)
    logger.info(f"LightGBM: best CV Poisson deviance={study.best_value:.4f}, params={study.best_params}")
    return study.best_params


def regression_report(y_true, y_pred, label):
    rmse = mean_squared_error(y_true, y_pred) ** 0.5
    mae = mean_absolute_error(y_true, y_pred)
    dev = mean_poisson_deviance(y_true, np.clip(y_pred, 1e-6, None))
    corr = pearsonr(y_true, y_pred)[0] if np.std(y_pred) > 0 else float("nan")
    logger.info(f"{label}: RMSE={rmse:.4f}  MAE={mae:.4f}  PoissonDev={dev:.4f}  corr={corr:.4f}")
    return {"rmse": rmse, "mae": mae, "poisson_deviance": dev, "corr": corr}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--matrix", choices=["batter", "pitcher"], required=True)
    parser.add_argument("--market", required=True, help="label column, e.g. h, hr, tb, rbi, r, bb, so, sb (batter) or k, outs, er (pitcher)")
    args = parser.parse_args()

    MODELS.mkdir(parents=True, exist_ok=True)
    OUTPUTS.mkdir(parents=True, exist_ok=True)

    train_df, test_df, features, cfg = load_data(args.matrix, args.market)
    y_train = train_df[args.market].astype(float)
    y_test = test_df[args.market].astype(float)
    stem = f"{args.matrix}_{args.market}"
    X_train, X_test = median_impute(train_df, test_df, features, stem)

    tscv = TimeSeriesSplit(n_splits=N_SPLITS)

    logger.info(f"Tuning XGBoost (Optuna, {N_OPTUNA_TRIALS} trials) for market='{args.market}' ...")
    xgb_params = tune_xgb(X_train, y_train, tscv)
    logger.info(f"Tuning LightGBM (Optuna, {N_OPTUNA_TRIALS} trials) for market='{args.market}' ...")
    lgbm_params = tune_lgbm(X_train, y_train, tscv)

    xgb_model = xgb.XGBRegressor(**xgb_params, objective="count:poisson", random_state=42)
    xgb_model.fit(X_train, y_train)
    lgbm_model = lgb.LGBMRegressor(**lgbm_params, objective="poisson", random_state=42, verbosity=-1)
    lgbm_model.fit(X_train, y_train)

    joblib.dump(xgb_model, MODELS / f"{stem}_xgb.joblib")
    joblib.dump(lgbm_model, MODELS / f"{stem}_lgbm.joblib")
    with open(MODELS / f"{stem}_feature_names.json", "w") as f:
        json.dump({"features": features, "matrix": args.matrix, "market": args.market}, f, indent=2)
    logger.info(f"Saved models/props/{stem}_xgb.joblib, {stem}_lgbm.joblib, {stem}_medians.joblib")

    logger.info(f"========== {TEST_YEAR} HOLDOUT EVALUATION — {args.matrix}/{args.market} ==========")
    xgb_test_pred = np.clip(xgb_model.predict(X_test), 0, None)
    lgbm_test_pred = np.clip(lgbm_model.predict(X_test), 0, None)
    avg_pred = (xgb_test_pred + lgbm_test_pred) / 2.0

    y_test_arr = y_test.values
    results = {}
    for label, pred in [("XGB", xgb_test_pred), ("LGBM", lgbm_test_pred), ("avg(XGB, LGBM)", avg_pred)]:
        results[label] = regression_report(y_test_arr, pred, label)

    # Naive baseline: the player's own rolling average already sitting in
    # the feature matrix. If the tuned model can't beat "just use the
    # rolling average", the modeling effort isn't adding value yet.
    roll_col = f"roll_{args.market}_{cfg['roll_window']}"
    if roll_col in test_df.columns:
        baseline_pred = test_df[roll_col].fillna(y_train.mean()).values
        results["naive rolling-avg baseline"] = regression_report(y_test_arr, baseline_pred, f"naive baseline ({roll_col})")
        beat_baseline = results["avg(XGB, LGBM)"]["rmse"] < results["naive rolling-avg baseline"]["rmse"]
        logger.info(f"Model beats naive rolling-average baseline on RMSE: {beat_baseline}")
    else:
        logger.warning(f"{roll_col} not in test_df — skipping naive baseline comparison")

    report_path = OUTPUTS / f"props_train_report_{args.matrix}_{args.market}.json"
    with open(report_path, "w") as f:
        json.dump({"matrix": args.matrix, "market": args.market, "test_year": TEST_YEAR, "results": results}, f, indent=2)
    logger.info(f"Wrote {report_path}")


if __name__ == "__main__":
    main()
