"""Validation helpers for file handling and analysis constraints."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from autonomous_analyst.config import MAX_FILE_SIZE_MB, MAX_ROWS, OUTPUT_DIR


class ValidationError(ValueError):
    """Raised when request or dataset validation fails."""


def validate_csv_filename(filename: str | None) -> None:
    """Validate uploaded filename and ensure CSV extension."""
    if not filename:
        raise ValidationError("Uploaded file must include a filename.")
    if not filename.lower().endswith(".csv"):
        raise ValidationError("Only CSV files are supported.")


def validate_file_size(size_bytes: int) -> None:
    """Validate uploaded file size."""
    max_size_bytes = MAX_FILE_SIZE_MB * 1024 * 1024
    if size_bytes > max_size_bytes:
        raise ValidationError(
            f"File too large: {size_bytes} bytes. Max allowed is {max_size_bytes} bytes."
        )


def validate_dataframe(df: pd.DataFrame) -> None:
    """Validate basic constraints for dataframe processing."""
    if df.empty:
        raise ValidationError("CSV contains no data.")
    if len(df) > MAX_ROWS:
        raise ValidationError(
            f"Dataset has {len(df)} rows, exceeding MAX_ROWS={MAX_ROWS}."
        )


def resolve_output_path(filename: str) -> Path:
    """Resolve and validate output path inside the outputs directory."""
    candidate = (OUTPUT_DIR / filename).resolve()
    output_root = OUTPUT_DIR.resolve()
    if output_root not in candidate.parents and candidate != output_root:
        raise ValidationError("Attempted write outside the outputs directory.")
    return candidate
