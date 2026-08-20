# -*- coding: utf-8 -*-
"""Shared probability scoring for backtests and daily predictions.

The model manifest is the source of truth for the promoted probability
artifact. Keeping this logic in one module prevents the live and historical
paths from silently scoring a different model than training selected.
"""
import json
from pathlib import Path

import joblib
import numpy as np


CALIBRATED_AVERAGE = "xgb_lgbm_calibrated_average"
CALIBRATED_ENSEMBLE = "ensemble_calibrated.joblib"
RAW_AVERAGE = "xgb_lgbm_raw_average"
RAW_XGB = "xgb.joblib"
RAW_LGBM = "lgbm.joblib"
RAW_ENSEMBLE = "ensemble.joblib"
PREDICTION_MODEL_TYPES = {
    CALIBRATED_AVERAGE: "calibrated_xgb_lgbm_average",
    CALIBRATED_ENSEMBLE: "calibrated_weighted_ensemble",
    RAW_AVERAGE: "raw_xgb_lgbm_average",
    RAW_XGB: "raw_xgb",
    RAW_LGBM: "raw_lgbm",
    RAW_ENSEMBLE: "raw_weighted_ensemble",
}
SUPPORTED_PREDICTION_ARTIFACTS = set(PREDICTION_MODEL_TYPES)


def load_model_manifest(model_dir: Path) -> dict:
    manifest_path = Path(model_dir) / "manifest.json"
    if not manifest_path.exists():
        return {}
    with open(manifest_path, encoding="utf-8") as handle:
        return json.load(handle)


def selected_prediction_artifact(model_dir: Path, manifest: dict | None = None) -> str:
    """Return the promoted scorer, failing closed for modern production bundles."""
    manifest = load_model_manifest(model_dir) if manifest is None else manifest
    artifact = manifest.get("prediction_artifact")
    if artifact is None:
        if manifest.get("mode") == "production" and int(manifest.get("schema_version", 0)) >= 3:
            raise ValueError("Production model manifest is missing prediction_artifact")
        return CALIBRATED_AVERAGE
    if artifact not in SUPPORTED_PREDICTION_ARTIFACTS:
        raise ValueError(f"Unsupported prediction artifact: {artifact}")
    return artifact


def _positive_probability(model, X, label: str) -> np.ndarray:
    probability = np.asarray(model.predict_proba(X), dtype=float)[:, 1]
    if probability.shape != (len(X),):
        raise ValueError(f"{label} returned an invalid probability shape")
    if not np.isfinite(probability).all() or ((probability < 0) | (probability > 1)).any():
        raise ValueError(f"{label} returned invalid probabilities")
    return probability


def prediction_model_type(artifact: str) -> str:
    if artifact not in PREDICTION_MODEL_TYPES:
        raise ValueError(f"Unsupported prediction artifact: {artifact}")
    return PREDICTION_MODEL_TYPES[artifact]


def score_probability_bundle(model_dir: Path, X, manifest: dict | None = None) -> dict:
    """Score one model bundle and retain component disagreement diagnostics."""
    model_dir = Path(model_dir)
    manifest = load_model_manifest(model_dir) if manifest is None else manifest
    artifact = selected_prediction_artifact(model_dir, manifest)

    xgb_probability = _positive_probability(
        joblib.load(model_dir / "xgb_calibrated.joblib"), X, "calibrated XGBoost"
    )
    lgbm_probability = _positive_probability(
        joblib.load(model_dir / "lgbm_calibrated.joblib"), X, "calibrated LightGBM"
    )

    if artifact in {CALIBRATED_ENSEMBLE, RAW_XGB, RAW_LGBM, RAW_ENSEMBLE}:
        selected_path = model_dir / artifact
        if not selected_path.exists():
            raise FileNotFoundError(f"Selected prediction artifact is missing: {selected_path}")
        primary_probability = _positive_probability(
            joblib.load(selected_path), X, prediction_model_type(artifact)
        )
    elif artifact == RAW_AVERAGE:
        raw_xgb = _positive_probability(joblib.load(model_dir / RAW_XGB), X, "raw XGBoost")
        raw_lgbm = _positive_probability(joblib.load(model_dir / RAW_LGBM), X, "raw LightGBM")
        primary_probability = (raw_xgb + raw_lgbm) / 2.0
    else:
        primary_probability = (xgb_probability + lgbm_probability) / 2.0

    return {
        "home_probability": primary_probability,
        "xgb_home_probability": xgb_probability,
        "lgbm_home_probability": lgbm_probability,
        "model_disagreement": np.abs(xgb_probability - lgbm_probability),
        "prediction_artifact": artifact,
        "prediction_model_type": prediction_model_type(artifact),
    }
