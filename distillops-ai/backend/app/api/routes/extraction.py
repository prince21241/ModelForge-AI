"""Extract five invoice fields with the frozen local student."""

from __future__ import annotations

from fastapi import APIRouter, Request

from backend.app.core.errors import UnsupportedModelError
from backend.app.schemas.invoice import ExtractRequest, ExtractResponse
from backend.app.services.student_service import StudentService

router = APIRouter()


@router.post("/extract", response_model=ExtractResponse)
def extract(body: ExtractRequest, request: Request) -> ExtractResponse:
    if body.model != "student":
        raise UnsupportedModelError(body.model)
    service: StudentService = request.app.state.student_service
    return service.extract(body.document_text)
