"""Baseline model training and selection utilities."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.model_selection import KFold, StratifiedKFold, cross_val_score, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from autonomous_analyst.config import RANDOM_STATE
from autonomous_analyst.ml.evaluator import evaluate_classification, evaluate_regression


@dataclass
class TrainResult:
    """Container for model training outputs."""

    problem_type: str
    model_name: str
    metrics: dict[str, float]
    feature_importance: dict[str, float]
    model_comparison: dict[str, dict[str, float]]
    dropped_features: list[str]
    preprocessing_summary: dict[str, Any]


def detect_problem_type(target: pd.Series) -> str:
    """Infer whether target is classification or regression."""
    non_null = target.dropna()
    if non_null.empty:
        return "classification"

    if pd.api.types.is_bool_dtype(non_null) or pd.api.types.is_object_dtype(non_null):
        return "classification"

    if pd.api.types.is_numeric_dtype(non_null):
        unique_count = non_null.nunique()
        threshold = max(20, int(len(non_null) * 0.05))
        if unique_count <= threshold and np.allclose(non_null, non_null.astype(int), atol=1e-8):
            return "classification"
        return "regression"

    return "classification"


def _build_preprocessor(
    X: pd.DataFrame,
    *,
    scale_numeric: bool,
) -> tuple[ColumnTransformer, list[str], list[str]]:
    numeric_features = X.select_dtypes(include=["number", "bool"]).columns.tolist()
    categorical_features = [col for col in X.columns if col not in numeric_features]

    numeric_steps: list[tuple[str, Any]] = [("imputer", SimpleImputer(strategy="median"))]
    if scale_numeric:
        numeric_steps.append(("scaler", StandardScaler()))
    numeric_transformer = Pipeline(steps=numeric_steps)

    categorical_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            (
                "onehot",
                OneHotEncoder(
                    handle_unknown="infrequent_if_exist",
                    min_frequency=0.02,
                ),
            ),
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, numeric_features),
            ("cat", categorical_transformer, categorical_features),
        ]
    )
    return preprocessor, numeric_features, categorical_features


def _is_index_like(series: pd.Series) -> bool:
    """Detect index-like columns that should be excluded from training."""
    non_null = series.dropna()
    if non_null.empty:
        return False
    unique_ratio = float(non_null.nunique()) / float(len(non_null))
    if unique_ratio < 0.98:
        return False
    if not pd.api.types.is_numeric_dtype(non_null):
        return False

    values = non_null.to_numpy(dtype=float)
    diffs = np.diff(values)
    if diffs.size == 0:
        return False
    return bool(np.allclose(diffs, diffs[0], atol=1e-8))


def _select_feature_columns(X: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    """Drop index-like and unnamed artifact columns."""
    drop_cols: list[str] = []
    for col in X.columns:
        lowered = col.strip().lower()
        if lowered.startswith("unnamed"):
            drop_cols.append(col)
            continue
        if _is_index_like(X[col]):
            drop_cols.append(col)
            continue
        if pd.api.types.is_object_dtype(X[col]) or pd.api.types.is_string_dtype(X[col]):
            non_null = X[col].dropna()
            if not non_null.empty:
                unique_ratio = float(non_null.nunique()) / float(len(non_null))
                if unique_ratio > 0.95:
                    drop_cols.append(col)

    kept = [col for col in X.columns if col not in drop_cols]
    if not kept:
        return X.copy(), []
    return X[kept].copy(), drop_cols


def _prepare_target(y: pd.Series, problem_type: str) -> tuple[pd.Series, list[int]]:
    """Clean and normalize target values while tracking dropped row indices."""
    if problem_type == "regression":
        y_num = pd.to_numeric(y, errors="coerce")
        keep_mask = y_num.notna()
        dropped_idx = y_num.index[~keep_mask].tolist()
        return y_num.loc[keep_mask], dropped_idx

    keep_mask = y.notna()
    y_clean = y.loc[keep_mask].astype(str)
    dropped_idx = y.index[~keep_mask].tolist()
    return y_clean, dropped_idx


def _extract_feature_importance(
    pipeline: Pipeline,
    numeric_features: list[str],
    categorical_features: list[str],
    top_n: int = 10,
) -> dict[str, float]:
    """Extract feature importance/coefficient magnitudes when available."""
    model = pipeline.named_steps["model"]
    preprocessor = pipeline.named_steps["preprocessor"]

    transformed_names = numeric_features.copy()
    if categorical_features:
        encoder = (
            preprocessor.named_transformers_["cat"]
            .named_steps["onehot"]
        )
        transformed_names.extend(encoder.get_feature_names_out(categorical_features).tolist())

    values: np.ndarray | None = None
    if hasattr(model, "feature_importances_"):
        values = np.asarray(model.feature_importances_, dtype=float)
    elif hasattr(model, "coef_"):
        coef = np.asarray(model.coef_, dtype=float)
        if coef.ndim > 1:
            coef = np.mean(np.abs(coef), axis=0)
        values = np.abs(coef)

    if values is None:
        return {}

    importance = {
        name: float(score)
        for name, score in zip(transformed_names, values, strict=False)
    }

    sorted_items = sorted(importance.items(), key=lambda kv: kv[1], reverse=True)[:top_n]
    return {k: v for k, v in sorted_items}


def train_best_model(df: pd.DataFrame, target_column: str) -> TrainResult:
    """Train candidate baselines, select best, and return metrics plus metadata."""
    X_raw = df.drop(columns=[target_column])
    y_raw = df[target_column]
    problem_type = detect_problem_type(y_raw)
    y, dropped_target_rows = _prepare_target(y_raw, problem_type=problem_type)
    X_raw = X_raw.loc[y.index]
    X, dropped_features = _select_feature_columns(X_raw)

    if problem_type == "classification":
        candidates: dict[str, Any] = {
            "LogisticRegression": LogisticRegression(max_iter=2000, class_weight="balanced"),
            "RandomForestClassifier": RandomForestClassifier(
                n_estimators=350,
                random_state=RANDOM_STATE,
                n_jobs=-1,
                class_weight="balanced_subsample",
                min_samples_leaf=2,
            ),
        }
        score_key = "f1_score"
    else:
        candidates = {
            "LinearRegression": LinearRegression(),
            "RandomForestRegressor": RandomForestRegressor(
                n_estimators=400,
                random_state=RANDOM_STATE,
                n_jobs=-1,
                min_samples_leaf=2,
            ),
        }
        score_key = "r2"

    if len(y) < 25:
        raise ValueError("Insufficient usable rows after preprocessing. Need at least 25 rows.")

    stratify = y if problem_type == "classification" and y.nunique() > 1 else None
    try:
        X_train, X_test, y_train, y_test = train_test_split(
            X,
            y,
            test_size=0.2,
            random_state=RANDOM_STATE,
            stratify=stratify,
        )
    except ValueError:
        X_train, X_test, y_train, y_test = train_test_split(
            X,
            y,
            test_size=0.2,
            random_state=RANDOM_STATE,
            stratify=None,
        )

    best_name = ""
    best_pipeline: Pipeline | None = None
    best_metrics: dict[str, float] = {}
    best_score = -np.inf
    comparison: dict[str, dict[str, float]] = {}
    cv_splits = 5 if len(y_train) >= 80 else 3
    if problem_type == "classification":
        cv = StratifiedKFold(n_splits=cv_splits, shuffle=True, random_state=RANDOM_STATE)
        cv_metric = "f1_weighted"
    else:
        cv = KFold(n_splits=cv_splits, shuffle=True, random_state=RANDOM_STATE)
        cv_metric = "r2"

    for model_name, estimator in candidates.items():
        scale_numeric = model_name in {"LogisticRegression", "LinearRegression"}
        preprocessor, numeric_features, categorical_features = _build_preprocessor(
            X,
            scale_numeric=scale_numeric,
        )
        pipeline = Pipeline(
            steps=[
                ("preprocessor", preprocessor),
                ("model", estimator),
            ]
        )
        pipeline.fit(X_train, y_train)
        preds = pipeline.predict(X_test)

        if problem_type == "classification":
            metrics = evaluate_classification(y_test, preds)
        else:
            metrics = evaluate_regression(y_test, preds)

        cv_scores = cross_val_score(
            pipeline,
            X_train,
            y_train,
            cv=cv,
            scoring=cv_metric,
            n_jobs=-1,
        )
        metrics["cv_score_mean"] = float(np.mean(cv_scores))
        metrics["cv_score_std"] = float(np.std(cv_scores))
        comparison[model_name] = metrics
        score = (metrics[score_key] * 0.7) + (metrics["cv_score_mean"] * 0.3)
        if score > best_score:
            best_score = score
            best_name = model_name
            best_pipeline = pipeline
            best_metrics = metrics

    if best_pipeline is None:
        raise RuntimeError("No model was successfully trained.")

    importance = _extract_feature_importance(
        best_pipeline, numeric_features=numeric_features, categorical_features=categorical_features
    )

    return TrainResult(
        problem_type=problem_type,
        model_name=best_name,
        metrics=best_metrics,
        feature_importance=importance,
        model_comparison=comparison,
        dropped_features=dropped_features,
        preprocessing_summary={
            "dropped_feature_columns": dropped_features,
            "dropped_target_rows_count": len(dropped_target_rows),
            "final_training_rows": int(len(y)),
            "final_training_features": int(X.shape[1]),
        },
    )
