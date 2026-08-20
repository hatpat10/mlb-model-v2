# -*- coding: utf-8 -*-
"""Atomic pointer for versioned production model bundles."""
import json
from pathlib import Path

from artifact_utils import atomic_write_json

REQUIRED_FILES = (
    "manifest.json", "feature_names.json", "train_medians.joblib",
    "xgb.joblib", "lgbm.joblib", "xgb_calibrated.joblib", "lgbm_calibrated.joblib",
    "ensemble.joblib", "ensemble_calibrated.joblib",
)


def resolve_production_model_dir(models_root: Path) -> Path | None:
    production_root = models_root / "production"
    pointer_path = production_root / "current.json"
    if pointer_path.exists():
        with open(pointer_path, encoding="utf-8") as handle:
            pointer = json.load(handle)
        candidate = (production_root / pointer["relative_path"]).resolve()
        if production_root.resolve() not in candidate.parents:
            raise ValueError("Production model pointer escapes models/production")
        if all((candidate / name).exists() for name in REQUIRED_FILES):
            return candidate
    legacy = production_root / "current"
    if all((legacy / name).exists() for name in REQUIRED_FILES):
        return legacy
    return None


def publish_production_model(models_root: Path, artifact_dir: Path, model_version: str) -> None:
    if not all((artifact_dir / name).exists() for name in REQUIRED_FILES):
        missing = [name for name in REQUIRED_FILES if not (artifact_dir / name).exists()]
        raise FileNotFoundError(f"Cannot publish incomplete model bundle: {missing}")
    relative = artifact_dir.resolve().relative_to((models_root / "production").resolve())
    atomic_write_json(
        {"model_version": model_version, "relative_path": relative.as_posix()},
        models_root / "production" / "current.json",
    )
