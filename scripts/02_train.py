# -*- coding: utf-8 -*-
"""Trains LogisticRegression, XGBoost, LightGBM, and RandomForest on
2021-2023, builds a CV-weighted ensemble and isotonic-calibrated XGB/LGBM
models, and reports AUC/accuracy/Brier/ECE on the untouched 2024 holdout.
"""
import sys
import json
import argparse
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
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
import xgboost as xgb
import lightgbm as lgb

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from config.config import (  # noqa: E402
    PATHS, TRAIN_YEARS, TEST_YEAR, CALIBRATION_YEARS,
    PRODUCTION_TRAIN_YEARS, PRODUCTION_CALIBRATION_YEARS,
    PRODUCTION_OPTUNA_TRIALS, PRODUCTION_MODEL_SCHEMA_VERSION,
)
from model_classes import WeightedEnsembleClassifier, PreFitCalibratedClassifier  # noqa: E402
from artifact_utils import atomic_write_json, make_run_id, sha256_file, utc_now_iso  # noqa: E402
from model_registry import publish_production_model  # noqa: E402

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


class ExpandingYearSplit:
    """Rolling-origin folds that never train and validate within one season."""

    def __init__(self, years):
        self.years = np.asarray(years, dtype=int)
        unique_years = sorted(np.unique(self.years).tolist())
        if len(unique_years) < 2:
            raise ValueError("ExpandingYearSplit requires at least two seasons")
        self.validation_years = unique_years[1:]

    def split(self, X, y=None, groups=None):
        if len(X) != len(self.years):
            raise ValueError("X length does not match the configured year vector")
        for validation_year in self.validation_years:
            train_index = np.flatnonzero(self.years < validation_year)
            validation_index = np.flatnonzero(self.years == validation_year)
            if len(train_index) and len(validation_index):
                yield train_index, validation_index

    def get_n_splits(self, X=None, y=None, groups=None):
        return len(self.validation_years)


def fold_matrices(X, train_index, validation_index):
    """Impute each validation fold from its own strictly-prior training rows."""
    train = X.iloc[train_index]
    validation = X.iloc[validation_index]
    # Some sources/features did not exist in the earliest seasons. If an
    # entire training fold is missing a feature, there is no historical
    # distribution to borrow from without leakage. Zero is a neutral,
    # constant placeholder; the model cannot learn signal from that column
    # until a later fold has genuine prior observations.
    medians = train.median().fillna(0.0)
    return train.fillna(medians), validation.fillna(medians)


def load_data(mode="benchmark"):
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

    train_years = TRAIN_YEARS if mode == "benchmark" else PRODUCTION_TRAIN_YEARS
    eval_year = TEST_YEAR if mode == "benchmark" else max(PRODUCTION_TRAIN_YEARS)
    train_df = df[df["year"].isin(train_years)].reset_index(drop=True)
    test_df = df[df["year"] == eval_year].reset_index(drop=True)
    logger.info(f"{mode} train (years {train_years}): {len(train_df)} rows | reporting year {eval_year}: {len(test_df)} rows")
    return train_df, test_df, features, train_years, eval_year


def median_impute(train_df, test_df, features, artifact_dir):
    medians = train_df[features].median()
    train_X = train_df[features].fillna(medians)
    test_X = test_df[features].fillna(medians)
    joblib.dump(medians, artifact_dir / "train_medians.joblib")
    logger.info(f"Median-imputed {features.__len__()} features using train-only medians.")
    return train_X, test_X


def tune_logreg(X, y, tscv):
    best_c, best_auc = 1.0, -np.inf
    for c in [0.001, 0.01, 0.1, 1.0, 10.0, 100.0]:
        aucs = []
        for tr_idx, val_idx in tscv.split(X):
            X_tr, X_val = fold_matrices(X, tr_idx, val_idx)
            model = make_pipeline(StandardScaler(), LogisticRegression(C=c, max_iter=5000, solver="lbfgs"))
            model.fit(X_tr, y.iloc[tr_idx])
            preds = model.predict_proba(X_val)[:, 1]
            aucs.append(roc_auc_score(y.iloc[val_idx], preds))
        mean_auc = np.mean(aucs)
        if mean_auc > best_auc:
            best_auc, best_c = mean_auc, c
    logger.info(f"LogisticRegression: best C={best_c} (CV AUC={best_auc:.4f})")
    return {"C": best_c, "max_iter": 5000, "solver": "lbfgs"}


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
            X_tr, X_val = fold_matrices(X, tr_idx, val_idx)
            model = xgb.XGBClassifier(**params, eval_metric="logloss", random_state=42)
            model.fit(X_tr, y.iloc[tr_idx])
            preds = model.predict_proba(X_val)[:, 1]
            aucs.append(roc_auc_score(y.iloc[val_idx], preds))
        return float(np.mean(aucs))

    study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=42))
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
            X_tr, X_val = fold_matrices(X, tr_idx, val_idx)
            model = lgb.LGBMClassifier(**params, random_state=42, verbosity=-1)
            model.fit(X_tr, y.iloc[tr_idx])
            preds = model.predict_proba(X_val)[:, 1]
            aucs.append(roc_auc_score(y.iloc[val_idx], preds))
        return float(np.mean(aucs))

    study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=42))
    study.optimize(objective, n_trials=N_OPTUNA_TRIALS, show_progress_bar=False)
    logger.info(f"LightGBM: best CV AUC={study.best_value:.4f}, params={study.best_params}")
    return study.best_params


def make_model(name, params):
    if name == "lr":
        return make_pipeline(StandardScaler(), LogisticRegression(**params))
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
        X_tr, X_val = fold_matrices(X, tr_idx, val_idx)
        for name, (factory_name, params) in model_specs.items():
            model = make_model(factory_name, params)
            model.fit(X_tr, y.iloc[tr_idx])
            oof[name][val_idx] = model.predict_proba(X_val)[:, 1]
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
    global N_OPTUNA_TRIALS
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=("benchmark", "production"), default="benchmark")
    parser.add_argument("--trials", type=int, default=None)
    args = parser.parse_args()
    N_OPTUNA_TRIALS = args.trials if args.trials is not None else (
        N_OPTUNA_TRIALS if args.mode == "benchmark" else PRODUCTION_OPTUNA_TRIALS
    )

    MODELS.mkdir(parents=True, exist_ok=True)
    OUTPUTS.mkdir(parents=True, exist_ok=True)
    model_version = make_run_id(f"model_{args.mode}")
    artifact_dir = MODELS if args.mode == "benchmark" else MODELS / "production" / model_version
    artifact_dir.mkdir(parents=True, exist_ok=True)

    train_df, test_df, features, train_years, eval_year = load_data(args.mode)
    y_train = train_df["home_win"].astype(int)
    y_test = test_df["home_win"].astype(int)
    X_train, X_test = median_impute(train_df, test_df, features, artifact_dir)

    # Preserve the published benchmark procedure exactly. Production uses
    # true season-level rolling origin: 2021 -> 2022, 2021-22 -> 2023, etc.
    # Raw feature rows are passed to that CV path so each fold's missing-value
    # medians are learned only from its strictly-prior training seasons.
    if args.mode == "benchmark":
        tscv = TimeSeriesSplit(n_splits=N_SPLITS)
        X_cv = X_train
        validation_scheme = f"row_time_series_{N_SPLITS}_fold"
    else:
        tscv = ExpandingYearSplit(train_df["year"].values)
        X_cv = train_df[features]
        validation_scheme = "expanding_season_rolling_origin"
    logger.info(f"Validation scheme: {validation_scheme} ({tscv.get_n_splits()} folds)")

    logger.info("Tuning LogisticRegression ...")
    lr_params = tune_logreg(X_cv, y_train, tscv)
    logger.info("Tuning XGBoost (Optuna, {} trials) ...".format(N_OPTUNA_TRIALS))
    xgb_params = tune_xgb(X_cv, y_train, tscv)
    logger.info("Tuning LightGBM (Optuna, {} trials) ...".format(N_OPTUNA_TRIALS))
    lgbm_params = tune_lgbm(X_cv, y_train, tscv)
    rf_params = {"n_estimators": 300, "max_depth": 6, "min_samples_leaf": 5}

    model_specs = {
        "lr": ("lr", lr_params),
        "xgb": ("xgb", xgb_params),
        "lgbm": ("lgbm", lgbm_params),
        "rf": ("rf", rf_params),
    }

    logger.info(f"Running shared {tscv.get_n_splits()}-fold OOF pass for ensemble weights + calibration fitting ...")
    oof = oof_predictions(model_specs, X_cv, y_train, tscv)

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
    joblib.dump(ensemble, artifact_dir / "ensemble.joblib")
    logger.info(f"Saved {artifact_dir / 'ensemble.joblib'}")

    calibration_years = CALIBRATION_YEARS if args.mode == "benchmark" else PRODUCTION_CALIBRATION_YEARS
    calib_mask = train_df["year"].isin(calibration_years).values
    logger.info(f"Fitting isotonic calibration on {calib_mask.sum()} out-of-fold rows from years {calibration_years}")

    calibrated_models = {}
    for name in ("xgb", "lgbm"):
        oof_preds = oof[name]
        valid = calib_mask & ~np.isnan(oof_preds)
        iso = IsotonicRegression(out_of_bounds="clip")
        iso.fit(oof_preds[valid], y_train.values[valid])
        calibrated = PreFitCalibratedClassifier(base_estimator=final_models[name], calibrator=iso)
        calibrated_models[name] = calibrated
        joblib.dump(calibrated, artifact_dir / f"{name}_calibrated.joblib")
        logger.info(f"Saved {artifact_dir / f'{name}_calibrated.joblib'} ({valid.sum()} calibration points)")

    with open(PROCESSED / "feature_names.json") as f:
        names_blob = json.load(f)
    with open(artifact_dir / "feature_names.json", "w") as f:
        json.dump(names_blob, f, indent=2)

    evaluation_label = f"{eval_year} HOLDOUT" if args.mode == "benchmark" else f"{eval_year} ROLLING-ORIGIN VALIDATION"
    logger.info(f"========== {evaluation_label} ==========")
    if args.mode == "benchmark":
        xgb_eval_proba = calibrated_models["xgb"].predict_proba(X_test)[:, 1]
        lgbm_eval_proba = calibrated_models["lgbm"].predict_proba(X_test)[:, 1]
        average_eval_proba = (xgb_eval_proba + lgbm_eval_proba) / 2.0
        ensemble_eval_proba = ensemble.predict_proba(X_test)[:, 1]
        y_eval = y_test.values
        evaluation_series = [
            ("calibrated XGB", xgb_eval_proba),
            ("calibrated LGBM", lgbm_eval_proba),
            ("avg(calibrated XGB, calibrated LGBM) [used by 04_predict.py]", average_eval_proba),
            ("uncalibrated weighted ensemble", ensemble_eval_proba),
        ]
    else:
        eval_mask = train_df["year"].eq(eval_year).values
        valid_eval = eval_mask & ~np.isnan(oof["xgb"]) & ~np.isnan(oof["lgbm"])
        y_eval = y_train.values[valid_eval]
        xgb_eval_proba = oof["xgb"][valid_eval]
        lgbm_eval_proba = oof["lgbm"][valid_eval]
        average_eval_proba = (xgb_eval_proba + lgbm_eval_proba) / 2.0
        weighted_sum = sum(oof[name][valid_eval] * weights[name] for name in model_specs)
        ensemble_eval_proba = weighted_sum / sum(weights.values())
        evaluation_series = [
            ("rolling-origin XGB (uncalibrated)", xgb_eval_proba),
            ("rolling-origin LGBM (uncalibrated)", lgbm_eval_proba),
            ("rolling-origin avg(XGB, LGBM)", average_eval_proba),
            ("rolling-origin weighted ensemble", ensemble_eval_proba),
        ]

    evaluation_metrics = {}
    for label, proba in evaluation_series:
        auc = roc_auc_score(y_eval, proba)
        acc = accuracy_score(y_eval, (proba >= 0.5).astype(int))
        brier = brier_score_loss(y_eval, proba)
        ece = expected_calibration_error(y_eval, proba)
        evaluation_metrics[label] = {
            "auc": float(auc), "accuracy": float(acc), "brier": float(brier), "ece": float(ece),
        }
        logger.info(f"{label}: AUC={auc:.4f}  Acc={acc:.4f}  Brier={brier:.4f}  ECE={ece:.4f}")

    frac_pos, mean_pred = calibration_curve(y_eval, average_eval_proba, n_bins=10)
    plt.figure(figsize=(6, 6))
    plt.plot([0, 1], [0, 1], "k--", label="Perfectly calibrated")
    curve_label = "avg(calibrated XGB, LGBM)" if args.mode == "benchmark" else "rolling-origin avg(XGB, LGBM)"
    plt.plot(mean_pred, frac_pos, marker="o", label=curve_label)
    plt.xlabel("Mean predicted probability")
    plt.ylabel("Fraction of actual home wins")
    plt.title(f"Calibration curve — {evaluation_label.lower()}")
    plt.legend()
    plt.tight_layout()
    plot_path = OUTPUTS / (f"calibration_plot_{TEST_YEAR}.png" if args.mode == "benchmark" else "calibration_plot_production.png")
    plt.savefig(plot_path, dpi=150)
    logger.info(f"Saved calibration plot to {plot_path}")

    feature_manifest_path = PROCESSED / "feature_manifest.json"
    feature_manifest = {}
    if feature_manifest_path.exists():
        with open(feature_manifest_path, encoding="utf-8") as handle:
            feature_manifest = json.load(handle)
    model_files = ["train_medians.joblib", "ensemble.joblib", "xgb_calibrated.joblib", "lgbm_calibrated.joblib", "feature_names.json"]
    manifest = {
        "schema_version": 1 if args.mode == "benchmark" else PRODUCTION_MODEL_SCHEMA_VERSION,
        "model_version": model_version,
        "created_at_utc": utc_now_iso(),
        "mode": args.mode,
        "training_years": train_years,
        "calibration_years": calibration_years,
        "evaluation_year": eval_year,
        "validation_scheme": validation_scheme,
        "validation_metrics": evaluation_metrics,
        "optuna_trials": N_OPTUNA_TRIALS,
        "feature_build_id": feature_manifest.get("build_id"),
        "data_version": feature_manifest.get("data_version"),
        "feature_matrix_sha256": feature_manifest.get("feature_matrix_sha256"),
        "production_feature_version": feature_manifest.get("production_feature_version"),
        "artifacts": {name: sha256_file(artifact_dir / name) for name in model_files},
    }
    atomic_write_json(manifest, artifact_dir / "manifest.json")
    if args.mode == "production":
        publish_production_model(MODELS, artifact_dir, model_version)
    logger.info(f"Saved attributable {args.mode} model manifest: {manifest['model_version']}")


if __name__ == "__main__":
    main()
