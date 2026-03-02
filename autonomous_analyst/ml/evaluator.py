"""Model evaluation helpers."""

from __future__ import annotations

from typing import Any

import numpy as np
from sklearn.metrics import accuracy_score, f1_score, mean_squared_error, r2_score


def evaluate_classification(y_true: Any, y_pred: Any) -> dict[str, float]:
    """Compute standard classification metrics."""
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "f1_score": float(f1_score(y_true, y_pred, average="weighted", zero_division=0)),
    }


def evaluate_regression(y_true: Any, y_pred: Any) -> dict[str, float]:
    """Compute standard regression metrics."""
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    return {
        "rmse": float(rmse),
        "r2": float(r2_score(y_true, y_pred)),
    }
