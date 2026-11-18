"""Extract fields from an uploaded PDF or image. The original file is not stored."""

from __future__ import annotations

import time

from fastapi import APIRouter, File, Form, Request, UploadFile

from backend.app.core.config import OCR_REVIEW_WARNING, max_upload_bytes
from backend.app.core.errors import EmptyUploadError, OversizedUploadError, UnsupportedModelError
from backend.app.schemas.invoice import DocumentExtractResponse
from backend.app.services.documents import parse_upload
from backend.app.services.student_service import StudentService

router = APIRouter()


@router.post("/documents/extract", response_model=DocumentExtractResponse)
def extract_document(
    request: Request,
    file: UploadFile = File(...),
    model: str = Form("student"),
) -> DocumentExtractResponse:
    started = time.perf_counter()
    if model != "student":
        raise UnsupportedModelError(model)
    payload = _read_limited(file, max_upload_bytes())
    if not payload:
        raise EmptyUploadError()
    parsed = parse_upload(payload, file.filename or "invoice", file.content_type)
    service: StudentService = request.app.state.student_service
    extraction = service.extract(parsed.text)
    warnings = list(extraction.warnings)
    if parsed.ocr_used:
        warnings.insert(1, OCR_REVIEW_WARNING)
    warnings.extend(parsed.warnings)
    total_ms = (time.perf_counter() - started) * 1000
    return DocumentExtractResponse(
        filename=parsed.filename,
        content_type=parsed.content_type,
        page_count=parsed.page_count,
        extraction_method=parsed.extraction_method,
        extracted_text=parsed.text,
        model=extraction.model,
        fields=extraction.fields,
        latency_ms=extraction.latency_ms,
        document_latency_ms=parsed.document_latency_ms,
        model_latency_ms=extraction.latency_ms,
        total_latency_ms=total_ms,
        requires_review=True,
        warnings=warnings,
    )


def _read_limited(file: UploadFile, limit: int) -> bytes:
    chunks: list[bytes] = []
    total = 0
    try:
        while True:
            block = file.file.read(1024 * 1024)
            if not block:
                break
            total += len(block)
            if total > limit:
                raise OversizedUploadError()
            chunks.append(block)
    finally:
        file.file.close()
    return b"".join(chunks)
