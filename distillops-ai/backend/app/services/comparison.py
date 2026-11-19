"""Compare one invoice with the frozen student and an explicitly approved teacher call.

The same extracted string is passed to both models. Predictions are copied,
not corrected. Nothing in this module writes the invoice or the teacher response.
"""

from __future__ import annotations

import logging
from decimal import Decimal

from src.config import ConfigError, Settings, configured_pricing, load_settings
from src.schemas import ExtractedInvoice
from src.teacher import TeacherClient, TeacherConfigError, TeacherPrediction

from backend.app.core.config import (
    BASE_MODEL_NAME,
    OCR_REVIEW_WARNING,
    public_deployment,
    student_inference_cost_usd,
    teacher_compare_budget_usd,
    teacher_compare_enabled,
    teacher_compare_max_requests,
    teacher_compare_reserve_usd,
)
from backend.app.core.errors import (
    ApiError,
    EmptyDocumentError,
    InferenceFailedError,
    ModelLoadError,
    OversizedDocumentError,
    TeacherApprovalError,
    TeacherBudgetError,
    TeacherDisabledError,
    TeacherKeyMissingError,
)
from backend.app.schemas.comparison import CompareResponse, CompareStatus, FieldComparison, ModelOutcome
from backend.app.schemas.invoice import InvoiceFields
from backend.app.services.student_service import StudentService
from backend.app.services.teacher_budget import TeacherSpendGate

logger = logging.getLogger("distillops.api")

FIELD_LABELS: tuple[tuple[str, str], ...] = (
    ("vendor", "Vendor"),
    ("invoice_number", "Invoice number"),
    ("date", "Date"),
    ("total", "Total"),
    ("currency", "Currency"),
)
DISAGREEMENT_NOTE = (
    "Field differences are model disagreement. They are not labeled as errors "
    "because this invoice has no ground-truth labels."
)
DISABLED_DETAIL = (
    "Live teacher comparison is off. Set DISTILLOPS_TEACHER_COMPARE_ENABLED=true and restart the API. "
    "Opening this screen does not call the teacher."
)
PUBLIC_DETAIL = (
    "Live teacher comparison is unavailable. This process is marked as a public deployment, "
    "and the spending counter is local to one process rather than a shared budget."
)
READY_DETAIL = (
    "Live teacher comparison is on. Invoice text is sent to OpenAI only after you confirm the request."
)


def comparison_status() -> CompareStatus:
    """Describe the gate. This does not construct an OpenAI client."""
    settings, problem = _settings()
    enabled = teacher_compare_enabled()
    public = public_deployment()
    key_ready = bool(settings and settings.openai_api_key)
    model = settings.teacher_model if settings else None
    pricing = _pricing(settings)
    student_cost = student_inference_cost_usd()
    budget = teacher_compare_budget_usd()
    available = bool(enabled and not public and key_ready and model and problem is None)
    if public:
        detail = PUBLIC_DETAIL
    elif not enabled:
        detail = DISABLED_DETAIL
    elif problem:
        detail = problem
    elif not model:
        detail = "TEACHER_MODEL is not set. No teacher request will be sent."
    elif not key_ready:
        detail = "OPENAI_API_KEY is not set. No teacher request will be sent."
    else:
        detail = READY_DETAIL
    return CompareStatus(
        feature_enabled=enabled,
        available=available,
        api_key_configured=key_ready,
        teacher_model=model,
        pricing_configured=pricing is not None,
        student_pricing_configured=student_cost is not None,
        max_requests=teacher_compare_max_requests(),
        budget_usd=None if budget is None else float(budget),
        public_deployment=public,
        budget_scope="refused" if public else "process-local",
        detail=detail,
    )


def prepare_teacher_call(approved: bool) -> Settings:
    """Reject a teacher call before any model runs. Does not contact OpenAI."""
    if not approved:
        raise TeacherApprovalError()
    status = comparison_status()
    if status.public_deployment:
        raise TeacherDisabledError(PUBLIC_DETAIL)
    if not status.feature_enabled:
        raise TeacherDisabledError(DISABLED_DETAIL)
    if not status.available:
        if status.teacher_model is None and "TEACHER_MODEL" in status.detail:
            raise TeacherKeyMissingError(status.detail)
        if not status.api_key_configured:
            raise TeacherKeyMissingError()
        raise TeacherDisabledError(status.detail)
    settings, problem = _settings()
    if settings is None:
        raise TeacherDisabledError(problem or DISABLED_DETAIL)
    budget = teacher_compare_budget_usd()
    if budget is not None and _pricing(settings) is None:
        raise TeacherBudgetError(
            "A teacher spending limit is set, but token prices are not configured. No teacher request was sent."
        )
    return settings


def compare_text(
    document_text: str,
    *,
    student: StudentService,
    teacher: TeacherClient,
    gate: TeacherSpendGate,
    document_latency_ms: float = 0,
    filename: str | None = None,
    content_type: str | None = None,
    page_count: int | None = None,
    extraction_method: str | None = None,
    extra_warnings: list[str] | None = None,
) -> CompareResponse:
    """Run the student, then the teacher, on one shared string."""
    text = document_text if isinstance(document_text, str) else ""
    if not text.strip():
        raise EmptyDocumentError()
    student_side = _run_student(student, text)
    teacher_side = _run_teacher(teacher, gate, text)
    warnings = [DISAGREEMENT_NOTE]
    if extra_warnings:
        warnings.extend(extra_warnings)
    warnings.extend(student_side.warnings)
    warnings.extend(teacher_side.warnings)
    both = student_side.succeeded and teacher_side.succeeded and student_side.fields and teacher_side.fields
    rows = _field_rows(teacher_side.fields if both else None, student_side.fields if both else None)
    agreement = sum(1 for row in rows if row.status == "agree") if both else None
    return CompareResponse(
        teacher=teacher_side,
        student=student_side,
        fields=rows,
        agreement_count=agreement,
        fields_compared=len(rows) if both else 0,
        document_latency_ms=document_latency_ms,
        extracted_text=text,
        filename=filename,
        content_type=content_type,
        page_count=page_count,
        extraction_method=extraction_method,
        partial=not (student_side.succeeded and teacher_side.succeeded),
        warnings=warnings,
    )


def _run_student(student: StudentService, text: str) -> ModelOutcome:
    cost = student_inference_cost_usd()
    pricing = cost is not None
    try:
        extraction = student.extract(text)
    except OversizedDocumentError:
        raise
    except (InferenceFailedError, ModelLoadError) as exc:
        return _failed("student", BASE_MODEL_NAME, exc.detail, pricing_configured=pricing)
    except ApiError as exc:
        return _failed("student", BASE_MODEL_NAME, exc.detail, pricing_configured=pricing)
    malformed = next((warning for warning in extraction.warnings if "was not repaired" in warning), None)
    if malformed:
        return ModelOutcome(
            role="student",
            model=BASE_MODEL_NAME,
            succeeded=False,
            fields=None,
            latency_ms=extraction.latency_ms,
            estimated_cost_usd=_money(cost),
            pricing_configured=pricing,
            error=malformed,
            warnings=[],
        )
    return ModelOutcome(
        role="student",
        model=BASE_MODEL_NAME,
        succeeded=True,
        fields=extraction.fields,
        latency_ms=extraction.latency_ms,
        estimated_cost_usd=_money(cost),
        pricing_configured=pricing,
        warnings=list(extraction.warnings),
    )


def _run_teacher(teacher: TeacherClient, gate: TeacherSpendGate, text: str) -> ModelOutcome:
    settings = teacher.settings
    pricing = _pricing(settings)
    model_name = settings.teacher_model or "gpt-4o-mini"
    reserved = gate.try_reserve(
        max_requests=teacher_compare_max_requests(),
        budget=teacher_compare_budget_usd(),
        reserve=teacher_compare_reserve_usd(),
    )
    try:
        prediction = teacher.extract(document_id="live-compare", document_text=text)
    except TeacherConfigError as exc:
        gate.settle(reserved, None)
        logger.warning("Teacher comparison was not sent: %s", type(exc).__name__)
        return _failed("teacher", model_name, str(exc), pricing_configured=pricing is not None)
    except Exception:
        gate.settle(reserved, None)
        logger.exception("Teacher comparison failed before a usable response.")
        return _failed("teacher", model_name, "The teacher request failed.", pricing_configured=pricing is not None)
    return _teacher_outcome(prediction, pricing, reserved, gate)


def _teacher_outcome(
    prediction: TeacherPrediction,
    pricing: object,
    reserved: Decimal,
    gate: TeacherSpendGate,
) -> ModelOutcome:
    model_name = prediction.model or "gpt-4o-mini"
    priced = pricing is not None
    if prediction.api_error:
        gate.settle(reserved, None)
        return _failed(
            "teacher",
            model_name,
            "The teacher request failed.",
            pricing_configured=priced,
            latency_ms=prediction.latency_ms,
        )
    tokens_in = prediction.input_tokens
    tokens_out = prediction.output_tokens
    cost = None if pricing is None else pricing.estimate_usd(tokens_in, tokens_out)  # type: ignore[attr-defined]
    gate.settle(reserved, cost)
    if prediction.extraction is None:
        reason = prediction.refusal or prediction.validation_error or "The teacher response did not match the invoice schema."
        return ModelOutcome(
            role="teacher",
            model=model_name,
            succeeded=False,
            fields=None,
            latency_ms=prediction.latency_ms,
            input_tokens=tokens_in,
            output_tokens=tokens_out,
            estimated_cost_usd=_money(cost),
            pricing_configured=priced,
            error=reason,
        )
    return ModelOutcome(
        role="teacher",
        model=model_name,
        succeeded=True,
        fields=_fields_from_teacher(prediction.extraction),
        latency_ms=prediction.latency_ms,
        input_tokens=tokens_in,
        output_tokens=tokens_out,
        estimated_cost_usd=_money(cost),
        pricing_configured=priced,
    )


def _fields_from_teacher(extraction: ExtractedInvoice) -> InvoiceFields:
    """Copy the validated teacher fields. Null becomes an empty string and nothing else is edited."""
    total = "" if extraction.total is None else format(extraction.total, "f")
    invoice_date = "" if extraction.invoice_date is None else extraction.invoice_date.isoformat()
    return InvoiceFields(
        vendor=extraction.vendor or "",
        invoice_number=extraction.invoice_number or "",
        date=invoice_date,
        total=total,
        currency=extraction.currency or "",
    )


def _field_rows(teacher: InvoiceFields | None, student: InvoiceFields | None) -> list[FieldComparison]:
    rows: list[FieldComparison] = []
    for key, label in FIELD_LABELS:
        left = None if teacher is None else getattr(teacher, key)
        right = None if student is None else getattr(student, key)
        if left is None or right is None:
            status = "unavailable"
        elif left == right:
            status = "agree"
        else:
            status = "disagree"
        rows.append(
            FieldComparison(
                field=key,
                label=label,
                teacher_value=left,
                student_value=right,
                status=status,
            )
        )
    return rows


def _failed(
    role: str,
    model: str,
    error: str,
    *,
    pricing_configured: bool,
    latency_ms: float | None = None,
) -> ModelOutcome:
    return ModelOutcome(
        role=role,  # type: ignore[arg-type]
        model=model,
        succeeded=False,
        fields=None,
        latency_ms=latency_ms,
        pricing_configured=pricing_configured,
        error=error,
    )


def _settings() -> tuple[Settings | None, str | None]:
    try:
        return load_settings(use_dotenv=False), None
    except ConfigError as exc:
        message = str(exc)
        if "OPENAI" in message or "sk-" in message.lower():
            message = "Teacher settings could not be read. No teacher request was sent."
        return None, message


def _pricing(settings: Settings | None):
    if settings is None:
        return None
    try:
        return configured_pricing(settings)
    except ConfigError:
        return None


def _money(value: Decimal | None) -> float | None:
    if value is None:
        return None
    return float(value)


def ocr_warning(used: bool) -> list[str]:
    return [OCR_REVIEW_WARNING] if used else []
