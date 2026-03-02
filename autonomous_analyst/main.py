"""FastAPI application entrypoint for the Autonomous Data Analyst."""

from __future__ import annotations

from typing import Any

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from pydantic import BaseModel, Field

from autonomous_analyst.agent.agent_builder import run_agent_workflow
from autonomous_analyst.agent.tools import AnalysisContext
from autonomous_analyst.config import OPENAI_API_KEY
from autonomous_analyst.utils.data_loader import load_csv_upload
from autonomous_analyst.utils.validation import ValidationError

app = FastAPI(title="Autonomous Data Analyst", version="1.0.0")


class AnalyzeResponse(BaseModel):
    """Structured response schema for /analyze endpoint."""

    eda_summary: str = Field(..., description="Compact EDA summary stringified JSON.")
    plots: list[str]
    problem_type: str
    model_used: str
    metrics: dict[str, float]
    top_models: list[dict[str, Any]]
    preprocessing_summary: dict[str, Any]
    leakage_warnings: list[str]
    explanation: str


@app.post("/analyze", response_model=AnalyzeResponse)
async def analyze_dataset(
    file: UploadFile = File(...),
    target_column: str | None = Form(default=None),
) -> dict[str, Any]:
    """Analyze uploaded CSV, train a baseline model, and return structured JSON output."""
    try:
        if not OPENAI_API_KEY:
            raise ValidationError(
                "OPENAI_API_KEY is not set. Export it before calling /analyze."
            )

        df = await load_csv_upload(file)

        if target_column:
            if target_column not in df.columns:
                raise ValidationError(f"Provided target column '{target_column}' not found in CSV.")
            selected_target = target_column
        else:
            if "target" not in df.columns:
                raise ValidationError(
                    "Target column not found. Provide target_column parameter or include 'target' column."
                )
            selected_target = "target"

        context = AnalysisContext(dataframe=df, target_column=selected_target)
        result = run_agent_workflow(context)
        return result

    except ValidationError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except HTTPException:
        raise
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=500, detail=f"Analysis failed: {exc}") from exc


@app.get("/health")
def health() -> dict[str, str]:
    """Liveness check endpoint."""
    return {"status": "ok"}
