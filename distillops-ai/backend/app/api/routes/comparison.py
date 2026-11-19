"""Live teacher-versus-student comparison. Opening the screen does not call OpenAI."""

from __future__ import annotations

from fastapi import APIRouter, File, Form, Request, UploadFile

from src.config import Settings
from src.teacher import TeacherClient

from backend.app.api.routes.documents import _read_limited
from backend.app.core.config import max_upload_bytes
from backend.app.core.errors import EmptyUploadError
from backend.app.schemas.comparison import CompareRequest, CompareResponse, CompareStatus
from backend.app.services.comparison import compare_text, comparison_status, ocr_warning, prepare_teacher_call
from backend.app.services.documents import parse_upload
from backend.app.services.student_service import StudentService
from backend.app.services.teacher_budget import TeacherSpendGate

router = APIRouter()


@router.get("/compare/status", response_model=CompareStatus)
def compare_status() -> CompareStatus:
    return comparison_status()


@router.post("/compare", response_model=CompareResponse)
def compare(body: CompareRequest, request: Request) -> CompareResponse:
    settings = prepare_teacher_call(body.teacher_approved)
    return compare_text(
        body.document_text,
        student=_student(request),
        teacher=_teacher(request, settings),
        gate=_gate(request),
    )


@router.post("/documents/compare", response_model=CompareResponse)
def compare_document(
    request: Request,
    file: UploadFile = File(...),
    teacher_approved: str = Form("false"),
) -> CompareResponse:
    settings = prepare_teacher_call(teacher_approved.strip().lower() == "true")
    payload = _read_limited(file, max_upload_bytes())
    if not payload:
        raise EmptyUploadError()
    parsed = parse_upload(payload, file.filename or "invoice", file.content_type)
    return compare_text(
        parsed.text,
        student=_student(request),
        teacher=_teacher(request, settings),
        gate=_gate(request),
        document_latency_ms=parsed.document_latency_ms,
        filename=parsed.filename,
        content_type=parsed.content_type,
        page_count=parsed.page_count,
        extraction_method=parsed.extraction_method,
        extra_warnings=ocr_warning(parsed.ocr_used),
    )


def _student(request: Request) -> StudentService:
    return request.app.state.student_service


def _gate(request: Request) -> TeacherSpendGate:
    return request.app.state.teacher_spend


def _teacher(request: Request, settings: Settings) -> TeacherClient:
    override = getattr(request.app.state, "teacher_client", None)
    if isinstance(override, TeacherClient):
        return override
    return TeacherClient(settings)
