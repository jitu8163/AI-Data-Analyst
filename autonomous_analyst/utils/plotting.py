"""Plot generation utilities for EDA outputs."""

from __future__ import annotations

from pathlib import Path
import re

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from autonomous_analyst.utils.validation import resolve_output_path

sns.set_theme(style="whitegrid")


def _save_figure(path: Path) -> str:
    """Save active matplotlib figure and return relative output path."""
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    return f"outputs/{path.name}"


def _safe_name(value: str) -> str:
    """Convert column names into filesystem-safe token."""
    return re.sub(r"[^A-Za-z0-9_\\-]+", "_", value).strip("_") or "feature"


def _is_excluded_feature(name: str) -> bool:
    return name.strip().lower().startswith("unnamed")


def _rank_features(df: pd.DataFrame, target_column: str) -> list[str]:
    """Rank candidate features by simple relevance heuristics."""
    candidates = [c for c in df.columns if c != target_column and not _is_excluded_feature(c)]
    if not candidates:
        return []

    target = df[target_column]
    numeric_candidates = [c for c in candidates if pd.api.types.is_numeric_dtype(df[c])]
    ranked: list[str] = []

    if pd.api.types.is_numeric_dtype(target) and numeric_candidates:
        corr_scores = {}
        for col in numeric_candidates:
            pair = df[[col, target_column]].dropna()
            if len(pair) < 3:
                corr_scores[col] = 0.0
            else:
                corr = pair[col].corr(pair[target_column])
                corr_scores[col] = float(abs(corr)) if pd.notna(corr) else 0.0
        ranked.extend(sorted(corr_scores, key=corr_scores.get, reverse=True))
    else:
        ranked.extend(numeric_candidates)

    categorical_candidates = [c for c in candidates if c not in ranked]
    categorical_candidates.sort(key=lambda c: df[c].nunique(dropna=True))
    ranked.extend(categorical_candidates)
    return ranked


def generate_plots(df: pd.DataFrame, target_column: str) -> list[str]:
    """Generate EDA plots and persist them to the outputs directory."""
    saved: list[str] = []

    numeric_cols = df.select_dtypes(include=["number"]).columns.tolist()

    if len(numeric_cols) > 1:
        plt.figure(figsize=(9, 7))
        corr = df[numeric_cols].corr(numeric_only=True)
        sns.heatmap(corr, cmap="coolwarm", annot=False)
        plt.title("Correlation Heatmap")
        saved.append(_save_figure(resolve_output_path("correlation_heatmap.png")))

    plt.figure(figsize=(8, 5))
    if df[target_column].dtype == "object" or str(df[target_column].dtype).startswith("category"):
        order = df[target_column].astype(str).value_counts().index
        sns.countplot(x=df[target_column].astype(str), order=order)
        plt.xticks(rotation=45, ha="right")
    else:
        sns.histplot(df[target_column], kde=True)
    plt.title(f"Target Distribution: {target_column}")
    saved.append(_save_figure(resolve_output_path("target_distribution.png")))

    ranked_features = _rank_features(df, target_column)
    top_features = ranked_features[:3]

    for idx, feature in enumerate(top_features, start=1):
        plt.figure(figsize=(8, 5))
        series = df[feature]
        if pd.api.types.is_numeric_dtype(series):
            sns.histplot(series, kde=True)
        else:
            order = series.astype(str).value_counts().head(20).index
            sns.countplot(x=series.astype(str), order=order)
            plt.xticks(rotation=45, ha="right")
        plt.title(f"Feature Distribution: {feature}")
        feature_token = _safe_name(feature)
        saved.append(_save_figure(resolve_output_path(f"feature_{idx}_{feature_token}.png")))

    return saved
