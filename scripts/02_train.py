# -*- coding: utf-8 -*-
"""Trains LogisticRegression, XGBoost, LightGBM, and RandomForest on
2021-2023, builds a CV-weighted ensemble and isotonic-calibrated XGB/LGBM
models, and reports AUC/accuracy/Brier/ECE on the untouched 2024 holdout.
"""
import sys
import json
from pathlib import Path

import numpy as np
import pandas as pd
import joblib
import optuna
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from loguru import logger
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import roc_auc_score, accuracy_score, brier_score_loss
from sklearn.isotonic import IsotonicRegression
from sklearn.calibration import calibration_curve
import xgboost as xgb
import lightgbm as lgb

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from config.config import PATHS, TRAIN_YEARS, TEST_YEAR, CALIBRATION_YEARS  # noqa: E402
from model_classes import WeightedEnsembleClassifier, PreFitCalibratedClassifier  # noqa: E402

optuna.logging.set_verbosity(optuna.logging.WARNING)

PROCESSED = PATHS["processed"]
MODELS = PATHS["models"]
OUTPUTS = PATHS["outputs"]
LOGS = PATHS["logs"]
N_SPLITS = 3
N_OPTUNA_TRIALS = 60

logger.remove()
logger.add(sys.stderr, level="INFO")
logger.add(LOGS / "02_train.log", level="DEBUG", rotation="5 MB")


def load_data():
    fm_path = PROCESSED / "feature_matrix.csv"
    names_path = PROCESSED / "feature_names.json"
    if not fm_path.exists() or not names_path.exists():
        logger.error("feature_matrix.csv / feature_names.json missing — run scripts/01_build_features.py first.")
        sys.exit(1)

    df = pd.read_csv(fm_path)
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date").reset_index(drop=True)

    with open(names_path) as f:
        names = json.load(f)
    features = names["passing_features"]

    train_df = df[df["year"].isin(TRAIN_YEARS)].reset_index(drop=True)
    test_df = df[df["year"] == TEST_YEAR].reset_index(drop=True)
    logger.info(f"train (years {TRAIN_YEARS}): {len(train_df)} rows | test (year {TEST_YEAR}): {len(test_df)} rows")
    return train_df, test_df, features


def median_impute(train_df, test_df, features):
    medians = train_df[features].median()
    train_X = train_df[features].fillna(medians)
    test_X = test_df[features].fillna(medians)
    joblib.dump(medians, MODELS / "train_medians.joblib")
    logger.info(f"Median-imputed {features.__len__()} features using train-only medians.")
    return train_X, test_X


def tune_logreg(X, y, tscv):
    best_c, best_auc = 1.0, -np.inf
    for c in [0.001, 0.01, 0.1, 1.0, 10.0, 100.0]:
        aucs = []
        for tr_idx, val_idx in tscv.split(X):
            model = LogisticRegression(penalty="l2", C=c, max_iter=2000, solver="lbfgs")
            model.fit(X.iloc[tr_idx], y.iloc[tr_idx])
            preds = model.predict_proba(X.iloc[val_idx])[:, 1]
            aucs.append(roc_auc_score(y.iloc[val_idx], preds))
        mean_auc = np.mean(aucs)
        if mean_auc > best_auc:
            best_auc, best_c = mean_auc, c
    logger.info(f"LogisticRegression: best C={best_c} (CV AUC={best_auc:.4f})")
    return {"C": best_c, "penalty": "l2", "max_iter": 2000, "solver": "lbfgs"}


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
        aucs = []
        for tr_idx, val_idx in tscv.split(X):
            model = xgb.XGBClassifier(**params, eval_metric="logloss", random_state=42)
            model.fit(X.iloc[tr_idx], y.iloc[tr_idx])
            preds = model.predict_proba(X.iloc[val_idx])[:, 1]
            aucs.append(roc_auc_score(y.iloc[val_idx], preds))
        return float(np.mean(aucs))

    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=N_OPTUNA_TRIALS, show_progress_bar=False)
    logger.info(f"XGBoost: best CV AUC={study.best_value:.4f}, params={study.best_params}")
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
        aucs = []
        for tr_idx, val_idx in tscv.split(X):
            model = lgb.LGBMClassifier(**params, random_state=42, verbosity=-1)
            model.fit(X.iloc[tr_idx], y.iloc[tr_idx])
            preds = model.predict_proba(X.iloc[val_idx])[:, 1]
            aucs.append(roc_auc_score(y.iloc[val_idx], preds))
        return float(np.mean(aucs))

    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=N_OPTUNA_TRIALS, show_progress_bar=False)
    logger.info(f"LightGBM: best CV AUC={study.best_value:.4f}, params={study.best_params}")
    return study.best_params


def make_model(name, params):
    if name == "lr":
        return LogisticRegression(**params)
    if name == "xgb":
        return xgb.XGBClassifier(**params, eval_metric="logloss", random_state=42)
    if name == "lgbm":
        return lgb.LGBMClassifier(**params, random_state=42, verbosity=-1)
    if name == "rf":
        return RandomForestClassifier(**params, random_state=42)
    raise ValueError(name)


def oof_predictions(model_specs, X, y, tscv):
    """One shared CV pass: refits every model per fold and collects
    out-of-fold predictions for all of them at once (used both for ensemble
    weighting and for fitting the isotonic calibrators).
    """
    oof = {name: np.full(len(X), np.nan) for name in model_specs}
    for tr_idx, val_idx in tscv.split(X):
        for name, (factory_name, params) in model_specs.items():
            model = make_model(factory_name, params)
            model.fit(X.iloc[tr_idx], y.iloc[tr_idx])
            oof[name][val_idx] = model.predict_proba(X.iloc[val_idx])[:, 1]
    return oof


def expected_calibration_error(y_true, y_prob, n_bins=10):
    bins = np.linspace(0, 1, n_bins + 1)
    bin_ids = np.digitize(y_prob, bins[1:-1])
    ece = 0.0
    n = len(y_true)
    for b in range(n_bins):
        mask = bin_ids == b
        if mask.sum() == 0:
            continue
        bin_conf = y_prob[mask].mean()
        bin_acc = y_true[mask].mean()
        ece += (mask.sum() / n) * abs(bin_conf - bin_acc)
    return ece


def main():
    MODELS.mkdir(parents=True, exist_ok=True)
    OUTPUTS.mkdir(parents=True, exist_ok=True)

    train_df, test_df, features = load_data()
    y_train = train_df["home_win"].astype(int)
    y_test = test_df["home_win"].astype(int)
    X_train, X_test = median_impute(train_df, test_df, features)

    tscv = TimeSeriesSplit(n_splits=N_SPLITS)

    logger.info("Tuning LogisticRegression ...")
    lr_params = tune_logreg(X_train, y_train, tscv)
    logger.info("Tuning XGBoost (Optuna, {} trials) ...".format(N_OPTUNA_TRIALS))
    xgb_params = tune_xgb(X_train, y_train, tscv)
    logger.info("Tuning LightGBM (Optuna, {} trials) ...".format(N_OPTUNA_TRIALS))
    lgbm_params = tune_lgbm(X_train, y_train, tscv)
    rf_params = {"n_estimators": 300, "max_depth": 6, "min_samples_leaf": 5}

    model_specs = {
        "lr": ("lr", lr_params),
        "xgb": ("xgb", xgb_params),
        "lgbm": ("lgbm", lgbm_params),
        "rf": ("rf", rf_params),
    }

    logger.info("Running shared 3-fold OOF pass for ensemble weights + calibration fitting ...")
    oof = oof_predictions(model_specs, X_train, y_train, tscv)

    weights = {}
    for name, preds in oof.items():
        mask = ~np.isnan(preds)
        auc = roc_auc_score(y_train[mask], preds[mask]) if mask.sum() > 0 else 0.5
        weights[name] = max(auc - 0.5, 0.0)
        logger.info(f"  {name}: OOF AUC={auc:.4f}, weight={weights[name]:.4f}")
    if sum(weights.values()) == 0:
        weights = {name: 1.0 for name in model_specs}
    logger.info(f"Ensemble weights (CV-derived, no test-set leakage): {weights}")

    logger.info("Fitting final models on full training set ...")
    final_models = {name: make_model(factory, params) for name, (factory, params) in model_specs.items()}
    for name, model in final_models.items():
        model.fit(X_train, y_train)

    ensemble = WeightedEnsembleClassifier(models=final_models, weights=weights)
    joblib.dump(ensemble, MODELS / "ensemble.joblib")
    logger.info("Saved models/ensemble.joblib")

    calib_mask = train_df["year"].isin(CALIBRATION_YEARS).values
    logger.info(f"Fitting isotonic calibration on {calib_mask.sum()} out-of-fold rows from years {CALIBRATION_YEARS}")

    calibrated_models = {}
    for name in ("xgb", "lgbm"):
        oof_preds = oof[name]
        valid = calib_mask & ~np.isnan(oof_preds)
        iso = IsotonicRegression(out_of_bounds="clip")
        iso.fit(oof_preds[valid], y_train.values[valid])
        calibrated = PreFitCalibratedClassifier(base_estimator=final_models[name], calibrator=iso)
        calibrated_models[name] = calibrated
        joblib.dump(calibrated, MODELS / f"{name}_calibrated.joblib")
        logger.info(f"Saved models/{name}_calibrated.joblib ({valid.sum()} calibration points)")

    with open(PROCESSED / "feature_names.json") as f:
        names_blob = json.load(f)
    with open(MODELS / "feature_names.json", "w") as f:
        json.dump(names_blob, f, indent=2)

    logger.info("========== 2024 HOLDOUT EVALUATION ==========")
    xgb_test_proba = calibrated_models["xgb"].predict_proba(X_test)[:, 1]
    lgbm_test_proba = calibrated_models["lgbm"].predict_proba(X_test)[:, 1]
    avg_calibrated_proba = (xgb_test_proba + lgbm_test_proba) / 2.0
    ensemble_proba = ensemble.predict_proba(X_test)[:, 1]

    y_test_arr = y_test.values
    for label, proba in [
        ("calibrated XGB", xgb_test_proba),
        ("calibrated LGBM", lgbm_test_proba),
        ("avg(calibrated XGB, calibrated LGBM) [used by 04_predict.py]", avg_calibrated_proba),
        ("uncalibrated weighted ensemble", ensemble_proba),
    ]:
        auc = roc_auc_score(y_test_arr, proba)
        acc = accuracy_score(y_test_arr, (proba >= 0.5).astype(int))
        brier = brier_score_loss(y_test_arr, proba)
        ece = expected_calibration_error(y_test_arr, proba)
        logger.info(f"{label}: AUC={auc:.4f}  Acc={acc:.4f}  Brier={brier:.4f}  ECE={ece:.4f}")

    frac_pos, mean_pred = calibration_curve(y_test_arr, avg_calibrated_proba, n_bins=10)
    plt.figure(figsize=(6, 6))
    plt.plot([0, 1], [0, 1], "k--", label="Perfectly calibrated")
    plt.plot(mean_pred, frac_pos, marker="o", label="avg(calibrated XGB, LGBM)")
    plt.xlabel("Mean predicted probability")
    plt.ylabel("Fraction of actual home wins")
    plt.title(f"Calibration curve — {TEST_YEAR} holdout")
    plt.legend()
    plt.tight_layout()
    plot_path = OUTPUTS / f"calibration_plot_{TEST_YEAR}.png"
    plt.savefig(plot_path, dpi=150)
    logger.info(f"Saved calibration plot to {plot_path}")


if __name__ == "__main__":
    main()
