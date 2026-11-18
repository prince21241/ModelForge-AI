"""Read an uploaded invoice into text. Original files are not kept."""

from __future__ import annotations

import io
import logging
import os
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path

import fitz
from PIL import Image, UnidentifiedImageError

from backend.app.core.config import max_image_pixels, max_pdf_pages, ocr_dpi, ocr_timeout_seconds
from backend.app.core.errors import (
    CorruptDocumentError,
    EmptyExtractedTextError,
    EncryptedPdfError,
    OcrTimeoutError,
    PageLimitError,
    SignatureMismatchError,
    TooManyPixelsError,
    UnsupportedFileError,
)
from backend.app.services import ocr as ocr_service

logger = logging.getLogger("distillops.api")

_PDF = b"%PDF-"
_PNG = b"\x89PNG\r\n\x1a\n"
_JPEG = b"\xff\xd8\xff"
_SUFFIXES = {".pdf": "pdf", ".png": "png", ".jpg": "jpeg", ".jpeg": "jpeg"}
_MIMES = {
    "application/pdf": "pdf",
    "image/png": "png",
    "image/jpeg": "jpeg",
    "image/jpg": "jpeg",
}
_CONTENT_TYPES = {"pdf": "application/pdf", "png": "image/png", "jpeg": "image/jpeg"}
_GENERIC_MIMES = {"", "application/octet-stream", "binary/octet-stream"}
_LOW_TEXT_WARNING = "OCR returned very little text. The image may be low quality."
_SMALL_IMAGE_WARNING = "The image is very small and may be hard to read."


@dataclass(frozen=True)
class ParsedDocument:
    """Text recovered from one upload, plus how it was recovered."""

    filename: str
    content_type: str
    page_count: int
    extraction_method: str
    text: str
    document_latency_ms: float
    ocr_used: bool
    warnings: tuple[str, ...]


def parse_upload(data: bytes, filename: str, content_type: str | None) -> ParsedDocument:
    """Copy the upload to a temporary file, read it, and always delete that file."""
    kind = resolve_kind(filename, content_type, data)
    started = time.perf_counter()
    path = _write_temp(data)
    try:
        payload = path.read_bytes()
        if kind == "pdf":
            parsed = _read_pdf(payload)
        else:
            parsed = _read_image(payload)
    finally:
        _delete_temp(path)
    text = parsed["text"]
    if not text.strip():
        raise EmptyExtractedTextError()
    elapsed_ms = (time.perf_counter() - started) * 1000
    return ParsedDocument(
        filename=safe_filename(filename),
        content_type=_CONTENT_TYPES[kind],
        page_count=parsed["page_count"],
        extraction_method=parsed["method"],
        text=text,
        document_latency_ms=elapsed_ms,
        ocr_used=parsed["ocr_used"],
        warnings=tuple(parsed["warnings"]),
    )


def resolve_kind(filename: str, content_type: str | None, data: bytes) -> str:
    """Accept a file only when its signature agrees with the declared type."""
    detected = _signature_kind(data)
    suffix = Path(filename or "").suffix.lower()
    if suffix and suffix not in _SUFFIXES:
        raise UnsupportedFileError()
    mime = (content_type or "").split(";")[0].strip().lower()
    if mime not in _GENERIC_MIMES and mime not in _MIMES:
        raise UnsupportedFileError()
    claimed = _SUFFIXES.get(suffix) or _MIMES.get(mime)
    if claimed is None or detected is None:
        raise UnsupportedFileError()
    mime_kind = _MIMES.get(mime)
    if claimed != detected or (mime_kind is not None and mime_kind != detected):
        raise SignatureMismatchError()
    return detected


def safe_filename(filename: str | None) -> str:
    """Keep the base name for display. It is never used as a path."""
    raw = Path(filename or "invoice").name
    cleaned = "".join(ch if ch.isprintable() and ch not in '\\/:' else "_" for ch in raw)
    cleaned = cleaned.strip(" .") or "invoice"
    return cleaned[:120]


def _signature_kind(data: bytes) -> str | None:
    if data.startswith(_PDF):
        return "pdf"
    if data.startswith(_PNG):
        return "png"
    if data.startswith(_JPEG):
        return "jpeg"
    return None


def _write_temp(data: bytes) -> Path:
    descriptor, name = tempfile.mkstemp(prefix="distillops-upload-", suffix=".bin")
    path = Path(name)
    try:
        os.write(descriptor, data)
    finally:
        os.close(descriptor)
    return path


def _delete_temp(path: Path) -> None:
    try:
        path.unlink(missing_ok=True)
    except OSError:
        logger.exception("Temporary upload could not be deleted.")


def _read_pdf(data: bytes) -> dict[str, object]:
    try:
        document = fitz.open(stream=data, filetype="pdf")
    except Exception as exc:
        raise CorruptDocumentError() from exc
    try:
        if document.needs_pass:
            raise EncryptedPdfError()
        page_count = document.page_count
        if page_count < 1:
            raise CorruptDocumentError()
        if page_count > max_pdf_pages():
            raise PageLimitError()
        parts: list[str] = []
        methods: list[str] = []
        warnings: list[str] = []
        ocr_started = time.perf_counter()
        for page in document:
            native = page.get_text("text") or ""
            if _selectable(native):
                parts.append(native.strip())
                methods.append("text")
                continue
            remaining = ocr_timeout_seconds() - (time.perf_counter() - ocr_started)
            image = _render_page(page)
            warnings.extend(_image_warnings(image))
            text = ocr_service.read_image(image, timeout=remaining)
            warnings.extend(_ocr_warnings(text))
            parts.append(text.strip())
            methods.append("ocr")
    finally:
        document.close()
    if methods and all(method == "text" for method in methods):
        method = "pdf_text"
    elif methods and all(method == "ocr" for method in methods):
        method = "pdf_ocr"
    else:
        method = "pdf_mixed"
    return {
        "text": "\n\n".join(parts).strip(),
        "page_count": page_count,
        "method": method,
        "ocr_used": "ocr" in methods,
        "warnings": warnings,
    }


def _render_page(page: fitz.Page) -> Image.Image:
    scale = ocr_dpi() / 72
    width = float(page.rect.width) * scale
    height = float(page.rect.height) * scale
    if width < 1 or height < 1 or width * height > max_image_pixels():
        raise TooManyPixelsError()
    pixmap = page.get_pixmap(matrix=fitz.Matrix(scale, scale), alpha=False)
    try:
        if pixmap.width * pixmap.height > max_image_pixels():
            raise TooManyPixelsError()
        if pixmap.n == 1:
            mode = "L"
        elif pixmap.n == 3:
            mode = "RGB"
        elif pixmap.n == 4:
            mode = "RGBA"
        else:
            raise CorruptDocumentError()
        return Image.frombytes(mode, (pixmap.width, pixmap.height), pixmap.samples)
    finally:
        pixmap = None


def _read_image(data: bytes) -> dict[str, object]:
    Image.MAX_IMAGE_PIXELS = max_image_pixels()
    try:
        with Image.open(io.BytesIO(data)) as image:
            image.load()
            if image.width * image.height > max_image_pixels():
                raise TooManyPixelsError()
            copied = image.copy()
    except TooManyPixelsError:
        raise
    except Image.DecompressionBombError as exc:
        raise TooManyPixelsError() from exc
    except (UnidentifiedImageError, OSError, ValueError) as exc:
        raise CorruptDocumentError() from exc
    warnings = _image_warnings(copied)
    text = ocr_service.read_image(copied, timeout=ocr_timeout_seconds())
    warnings.extend(_ocr_warnings(text))
    return {
        "text": text.strip(),
        "page_count": 1,
        "method": "image_ocr",
        "ocr_used": True,
        "warnings": warnings,
    }


def _selectable(text: str) -> bool:
    """A short or empty text layer is not mixed with OCR, so the same words are not repeated."""
    return sum(character.isalnum() for character in text) >= 8


def _image_warnings(image: Image.Image) -> list[str]:
    if image.width < 80 or image.height < 80:
        return [_SMALL_IMAGE_WARNING]
    return []


def _ocr_warnings(text: str) -> list[str]:
    if text.strip() and sum(character.isalnum() for character in text) < 12:
        return [_LOW_TEXT_WARNING]
    return []
