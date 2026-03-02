"""LangChain tools exposed to the function-calling agent."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import pandas as pd
from langchain_core.tools import StructuredTool
from pydantic import BaseModel, Field

from autonomous_analyst.ml.trainer import TrainResult, train_best_model
from autonomous_analyst.utils.plotting import generate_plots


@dataclass
class AnalysisContext:
    """Shared context passed through tool invocations."""

    dataframe: pd.DataFrame
    target_column: str
    eda_summary: dict[str, Any] = field(default_factory=dict)
    plot_paths: list[str] = field(default_factory=list)
    training_result: TrainResult | None = None
    explanation: str = ""


class AnalyzeDataframeInput(BaseModel):
    """Schema for dataframe analysis tool."""

    include_sample: bool = Field(default=False, description="Include a small head() sample.")


class GeneratePlotsInput(BaseModel):
    """Schema for plot generation tool."""

    enabled: bool = Field(default=True, description="Whether to generate plots.")


class TrainModelInput(BaseModel):
    """Schema for model training tool."""

    compare_models: bool = Field(
        default=True,
        description="If true, compares allowed models and keeps best one.",
    )


class ExplainModelInput(BaseModel):
    """Schema for explanation tool."""

    detail_level: str = Field(default="concise", description="concise or detailed")


def _build_eda_summary(df: pd.DataFrame, include_sample: bool) -> dict[str, Any]:
    numeric_cols = df.select_dtypes(include=["number", "bool"]).columns.tolist()
    categorical_cols = [col for col in df.columns if col not in numeric_cols]

    summary: dict[str, Any] = {
        "shape": {"rows": int(df.shape[0]), "columns": int(df.shape[1])},
        "data_types": {col: str(dtype) for col, dtype in df.dtypes.items()},
        "null_count": {col: int(df[col].isna().sum()) for col in df.columns},
        "basic_statistics": df.describe(include="all").fillna("").to_dict(),
        "numeric_columns": numeric_cols,
        "categorical_columns": categorical_cols,
    }
    if include_sample:
        summary["sample"] = df.head(5).to_dict(orient="records")
    return summary


def _build_explanation(context: AnalysisContext, detail_level: str) -> str:
    if context.training_result is None:
        return "Model training did not run, so no explanation is available."

    result = context.training_result
    metrics = result.metrics

    if result.problem_type == "classification":
        accuracy = metrics.get("accuracy", 0.0)
        f1_score = metrics.get("f1_score", 0.0)
        if f1_score >= 0.9:
            perf_text = "strong"
        elif f1_score >= 0.75:
            perf_text = "moderate"
        else:
            perf_text = "weak"

        overfit_text = (
            "Tree-based models can overfit on small datasets; validate with cross-validation."
            if "RandomForest" in result.model_name
            else "Linear models generally overfit less but may underfit nonlinear patterns."
        )
        metric_line = f"Accuracy={accuracy:.3f}, F1={f1_score:.3f}, indicating {perf_text} predictive quality."
    else:
        rmse = metrics.get("rmse", 0.0)
        r2 = metrics.get("r2", 0.0)
        if r2 >= 0.8:
            perf_text = "strong"
        elif r2 >= 0.5:
            perf_text = "moderate"
        else:
            perf_text = "weak"

        overfit_text = (
            "Random forests can overfit if trees are too deep; evaluate with cross-validation."
            if "RandomForest" in result.model_name
            else "Linear regression has lower overfitting risk but can miss nonlinear structure."
        )
        cv_mean = metrics.get("cv_score_mean")
        if cv_mean is not None:
            metric_line = (
                f"RMSE={rmse:.3f}, R2={r2:.3f}, CV-R2={cv_mean:.3f}, indicating {perf_text} fit quality."
            )
        else:
            metric_line = f"RMSE={rmse:.3f}, R2={r2:.3f}, indicating {perf_text} fit quality."

    if result.feature_importance:
        top_items = list(result.feature_importance.items())[:3]
        top_text = ", ".join(f"{name} ({score:.3f})" for name, score in top_items)
    else:
        top_text = "No model-driven feature importance was available for this estimator."

    dropped_note = ""
    if result.dropped_features:
        dropped_note = f" Excluded likely index/noise features: {', '.join(result.dropped_features[:5])}."
    prep = result.preprocessing_summary
    prep_note = (
        " Preprocessing applied: median/mode imputation, one-hot encoding with rare-category handling"
        f", usable rows={prep.get('final_training_rows', 0)}, features={prep.get('final_training_features', 0)}."
    )
    improvement = "Suggested improvements: feature engineering, outlier checks, and cross-validation monitoring."
    top_models = result.top_models[:2]
    compare_note = ""
    if len(top_models) >= 2:
        first = top_models[0]
        second = top_models[1]
        compare_note = (
            f" Top-2 comparison: {first['model_name']} (score={first['combined_score']:.3f}) vs "
            f"{second['model_name']} (score={second['combined_score']:.3f})."
        )
    warning_note = ""
    if result.leakage_warnings:
        warning_note = " Leakage warnings: " + " ".join(result.leakage_warnings[:3])

    if detail_level == "detailed":
        return (
            f"Selected model: {result.model_name}. {metric_line} "
            f"Overfitting risk: {overfit_text} "
            f"Feature influence: {top_text}.{dropped_note}{prep_note}{compare_note}{warning_note} {improvement}"
        )

    return (
        f"{metric_line} Overfitting risk: {overfit_text} "
        f"Key features: {top_text}.{dropped_note}{prep_note}{compare_note}{warning_note}"
    )


def build_tools(context: AnalysisContext) -> list[StructuredTool]:
    """Create LangChain tools bound to a shared analysis context."""

    def analyze_dataframe(include_sample: bool = False) -> dict[str, Any]:
        context.eda_summary = _build_eda_summary(context.dataframe, include_sample=include_sample)
        return context.eda_summary

    def generate_plots_tool(enabled: bool = True) -> dict[str, Any]:
        if enabled:
            context.plot_paths = generate_plots(context.dataframe, context.target_column)
        return {"plots": context.plot_paths}

    def train_model_tool(compare_models: bool = True) -> dict[str, Any]:
        _ = compare_models
        context.training_result = train_best_model(context.dataframe, context.target_column)
        return {
            "problem_type": context.training_result.problem_type,
            "model_name": context.training_result.model_name,
            "metrics": context.training_result.metrics,
            "feature_importance": context.training_result.feature_importance,
            "model_comparison": context.training_result.model_comparison,
            "top_models": context.training_result.top_models,
            "dropped_features": context.training_result.dropped_features,
            "preprocessing_summary": context.training_result.preprocessing_summary,
            "leakage_warnings": context.training_result.leakage_warnings,
        }

    def explain_model_tool(detail_level: str = "concise") -> dict[str, str]:
        context.explanation = _build_explanation(context, detail_level=detail_level)
        return {"explanation": context.explanation}

    return [
        StructuredTool.from_function(
            func=analyze_dataframe,
            name="analyze_dataframe",
            description=(
                "Analyze the dataset and return shape, dtypes, nulls, statistics, and numeric/categorical split."
            ),
            args_schema=AnalyzeDataframeInput,
        ),
        StructuredTool.from_function(
            func=generate_plots_tool,
            name="generate_plots",
            description=(
                "Generate heatmap, target distribution, and top feature plots, saving images in outputs/."
            ),
            args_schema=GeneratePlotsInput,
        ),
        StructuredTool.from_function(
            func=train_model_tool,
            name="train_model",
            description=(
                "Detect problem type, train allowed baselines, evaluate metrics, and return best model details."
            ),
            args_schema=TrainModelInput,
        ),
        StructuredTool.from_function(
            func=explain_model_tool,
            name="explain_model",
            description=(
                "Generate natural language interpretation of performance, overfitting risk, and feature importance."
            ),
            args_schema=ExplainModelInput,
        ),
    ]
