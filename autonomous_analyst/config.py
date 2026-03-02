"""Centralized configuration for the Autonomous Data Analyst service."""

from __future__ import annotations

import os
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = BASE_DIR / "outputs"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
OPENAI_TEMPERATURE = float(os.getenv("OPENAI_TEMPERATURE", "0.0"))
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "").strip()
RANDOM_STATE = int(os.getenv("RANDOM_STATE", "42"))
MAX_FILE_SIZE_MB = int(os.getenv("MAX_FILE_SIZE_MB", "25"))
MAX_ROWS = int(os.getenv("MAX_ROWS", "250000"))
