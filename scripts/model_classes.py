# -*- coding: utf-8 -*-
"""Pickle-stable model wrapper classes shared by 02_train.py, 03_backtest.py,
and 04_predict.py. joblib/pickle need the exact class definition importable
from the same module path at load time, so every script that unpickles a
saved model does `sys.path.insert(0, <this dir>)` then `import model_classes`
rather than redefining these classes inline.
"""
import numpy as np


class WeightedEnsembleClassifier:
    """Combines several already-fitted classifiers via a weighted average
    of their predict_proba outputs. Weights are fixed at construction time
    (determined via inner cross-validation on the training set only — never
    from test-set performance).
    """

    def __init__(self, models: dict, weights: dict):
        self.models = models
        self.weights = weights

    def predict_proba(self, X):
        total_weight = sum(self.weights.get(name, 0.0) for name in self.models)
        proba = np.zeros((len(X), 2))
        for name, model in self.models.items():
            w = self.weights.get(name, 0.0)
            if w == 0.0:
                continue
            proba += w * model.predict_proba(X)
        if total_weight > 0:
            proba /= total_weight
        return proba

    def predict(self, X):
        return (self.predict_proba(X)[:, 1] >= 0.5).astype(int)


class PreFitCalibratedClassifier:
    """Wraps an already-fitted base estimator plus an already-fitted
    isotonic-regression calibrator (fit on out-of-fold predictions), so
    calibration never leaks information the base model was fit on.
    """

    def __init__(self, base_estimator, calibrator):
        self.base_estimator = base_estimator
        self.calibrator = calibrator

    def predict_proba(self, X):
        raw = self.base_estimator.predict_proba(X)[:, 1]
        calibrated = np.clip(self.calibrator.predict(raw), 1e-6, 1 - 1e-6)
        return np.column_stack([1 - calibrated, calibrated])

    def predict(self, X):
        return (self.predict_proba(X)[:, 1] >= 0.5).astype(int)
