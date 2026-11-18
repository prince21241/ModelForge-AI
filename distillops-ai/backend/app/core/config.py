"""Runtime settings for the local extraction API.

Paths are resolved from this file so the checkpoint does not depend on the
process working directory. The frozen Phase 4 checkpoint is not modified.
"""

from __future__ import annotations

import os
from pathlib import Path

from src.benchmark import DECODING, FROZEN_STUDENT_DIR

DISTILLOPS_ROOT = Path(__file__).resolve().parents[3]
try:
    from dotenv import load_dotenv

    load_dotenv(DISTILLOPS_ROOT / ".env", override=False)
except ImportError:
    pass
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
OCR_REVIEW_WARNING = (
    "This document was read with OCR. OCR can misread characters, and the text was not repaired."
)


def _positive_int(name: str, default: int) -> int:
    raw = os.environ.get(name, "").strip()
    if not raw:
        return default
    try:
        value = int(raw)
    except ValueError:
        return default
    return value if value > 0 else default


def _positive_float(name: str, default: float) -> float:
    raw = os.environ.get(name, "").strip()
    if not raw:
        return default
    try:
        value = float(raw)
    except ValueError:
        return default
    return value if value > 0 else default


def max_upload_bytes() -> int:
    """Upload cap. The default is 10 MB."""
    return _positive_int("DISTILLOPS_MAX_UPLOAD_BYTES", 10 * 1024 * 1024)


def max_pdf_pages() -> int:
    """Maximum PDF pages read in one request. The default is 10."""
    return _positive_int("DISTILLOPS_MAX_PDF_PAGES", 10)


def max_image_pixels() -> int:
    """Pixel cap for a decoded image or a rendered PDF page."""
    return _positive_int("DISTILLOPS_MAX_IMAGE_PIXELS", 25_000_000)


def ocr_timeout_seconds() -> float:
    """Budget for OCR on one document. The default is 30 seconds."""
    return _positive_float("DISTILLOPS_OCR_TIMEOUT_SECONDS", 30.0)


def ocr_dpi() -> int:
    """Render resolution for a scanned PDF page, clamped to a safe range."""
    value = _positive_int("DISTILLOPS_OCR_DPI", 150)
    return min(200, max(72, value))
