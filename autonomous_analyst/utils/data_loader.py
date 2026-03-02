"""Secure CSV loading utilities."""

from __future__ import annotations

from io import BytesIO

import pandas as pd
from fastapi import UploadFile

from autonomous_analyst.utils.validation import (
    ValidationError,
    validate_csv_filename,
    validate_dataframe,
    validate_file_size,
)


async def load_csv_upload(upload: UploadFile) -> pd.DataFrame:
    """Load CSV content from FastAPI upload object into a pandas DataFrame."""
    validate_csv_filename(upload.filename)
    raw = await upload.read()
    validate_file_size(len(raw))

    try:
        df = pd.read_csv(BytesIO(raw))
    except Exception as exc:  # noqa: BLE001
        raise ValidationError(f"Failed to parse CSV: {exc}") from exc

    validate_dataframe(df)
    return df
