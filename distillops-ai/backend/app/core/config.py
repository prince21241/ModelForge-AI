"""Runtime settings for the local extraction API.

Paths are resolved from this file so the checkpoint does not depend on the
process working directory. The frozen Phase 4 checkpoint is not modified.
"""

from __future__ import annotations

from pathlib import Path

from src.benchmark import DECODING, FROZEN_STUDENT_DIR

DISTILLOPS_ROOT = Path(__file__).resolve().parents[3]
STUDENT_CHECKPOINT = (DISTILLOPS_ROOT / FROZEN_STUDENT_DIR).resolve()
BASE_MODEL_NAME = "google/flan-t5-small"
MAX_SOURCE_LENGTH = int(DECODING["max_source_length"])
MAX_NEW_TOKENS = int(DECODING["max_new_tokens"])
MAX_DOCUMENT_CHARS = 20_000
ALLOWED_ORIGINS: tuple[str, ...] = (
    "http://127.0.0.1:5173",
    "http://localhost:5173",
)
DEFAULT_REVIEW_WARNING = (
    "The student is not production-validated. Review every extraction before use. "
    "This response has no calibrated confidence score."
)
