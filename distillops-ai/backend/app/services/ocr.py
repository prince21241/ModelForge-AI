"""Local Tesseract OCR. No image and no text leave this machine."""

from __future__ import annotations

import logging
import os
import threading

import pytesseract
from PIL import Image, ImageOps

from backend.app.core.errors import CorruptDocumentError, OcrTimeoutError, OcrUnavailableError

logger = logging.getLogger("distillops.api")

_OCR_GATE = threading.Semaphore(1)


def prepare_image(image: Image.Image) -> Image.Image:
    """Correct orientation and improve contrast. The pixels are not replaced with guessed text."""
    oriented = ImageOps.exif_transpose(image)
    if oriented is None:
        oriented = image
    grayscale = ImageOps.grayscale(oriented)
    return ImageOps.autocontrast(grayscale)


def read_image(image: Image.Image, *, timeout: float) -> str:
    """Run one OCR call. Concurrent calls wait so only one Tesseract process is active."""
    if timeout <= 0:
        raise OcrTimeoutError()
    prepared = prepare_image(image)
    command = os.environ.get("TESSERACT_CMD", "").strip()
    if command:
        pytesseract.pytesseract.tesseract_cmd = command
    with _OCR_GATE:
        try:
            text = pytesseract.image_to_string(prepared, lang="eng", timeout=timeout)
        except pytesseract.TesseractNotFoundError as exc:
            raise OcrUnavailableError() from exc
        except RuntimeError as exc:
            if "timeout" in str(exc).lower():
                raise OcrTimeoutError() from exc
            logger.exception("OCR failed.")
            raise CorruptDocumentError() from exc
        except pytesseract.TesseractError as exc:
            if "timeout" in str(exc).lower():
                raise OcrTimeoutError() from exc
            logger.exception("OCR failed.")
            raise CorruptDocumentError() from exc
    if not isinstance(text, str):
        raise CorruptDocumentError()
    return text
