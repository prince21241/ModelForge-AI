"""Build a verified student-training set from the teacher.

The teacher is called only for the training split, and only when the caller
sets ``live=True``. Predictions that do not match ground truth are kept in the
checkpoint for audit and left out of the training export. They are not replaced
with the ground-truth answers.
"""

from __future__ import annotations

import json
from collections.abc import Callable, Sequence
from datetime import datetime, timezone
from decimal import Decimal
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

from src.config import Settings, configured_pricing
from src.dataset import read_jsonl, write_jsonl
from src.evaluation import (
    EvaluationReport,
    evaluate_predictions,
    field_values_match,
)
from src.evaluation_pricing import Pricing
from src.prompts import build_student_messages
from src.schemas import EXTRACTION_FIELDS, DatasetRecord, ExtractedInvoice, TeacherPrediction
from src.teacher import TeacherClient

TRAIN_SPLIT = "train"


class HeldOutTrainingError(ValueError):
    """Raised when label generation is aimed at anything other than train."""


class StudentFields(BaseModel):
    """Student target. Missing values are empty strings, not guesses."""

    model_config = ConfigDict(extra="forbid")

    vendor: str = ""
    invoice_number: str = ""
    date: str = ""
    total: str = ""
    currency: str = ""


class ChatTurn(BaseModel):
    model_config = ConfigDict(extra="forbid")

    role: Literal["system", "user", "assistant"]
    content: str


class StudentTrainingExample(BaseModel):
    """One verified example in a chat shape later trainers can adapt."""

    model_config = ConfigDict(extra="forbid")

    document_id: str
    teacher_model: str
    verification_status: Literal["verified"]
    invoice_text: str
    teacher_output: StudentFields
    messages: list[ChatTurn] = Field(min_length=3)


class LabelCheckpoint(BaseModel):
    """One finished teacher attempt. The invoice body stays in the split file."""

    model_config = ConfigDict(extra="forbid")

    document_id: str
    prediction: TeacherPrediction


class LabelRejection(BaseModel):
    model_config = ConfigDict(extra="forbid")

    document_id: str
    reason: Literal["api_failure", "schema_invalid", "mismatch"]
    api_error: str | None = None
    validation_error: str | None = None
    refusal: str | None = None
    mismatched_fields: list[str] = Field(default_factory=list)


class LabelRunReport(BaseModel):
    """Phase 2 summary. Training text is not copied into this report."""

    model_config = ConfigDict(extra="forbid")

    split: Literal["train"] = TRAIN_SPLIT
    live: bool
    teacher_model: str
    documents: int = Field(ge=0)
    checkpointed: int = Field(ge=0)
    skipped_existing: int = Field(ge=0)
    api_calls_this_run: int = Field(ge=0)
    successful_responses: int = Field(ge=0)
    api_failure_count: int = Field(ge=0)
    verified_count: int = Field(ge=0)
    rejected_count: int = Field(ge=0)
    unprocessed_count: int = Field(ge=0)
    exact_match_accuracy: float | None = None
    end_to_end_exact_match_accuracy: float | None = None
    field_metrics: list[dict[str, Any]] = Field(default_factory=list)
    latency_ms_avg: float | None = None
    latency_ms_total: float = 0
    input_tokens: int = Field(ge=0, default=0)
    output_tokens: int = Field(ge=0, default=0)
    pricing_configured: bool = False
    estimated_cost_usd: Decimal | None = None
    rejected: list[LabelRejection] = Field(default_factory=list)
    export_count: int = Field(ge=0, default=0)
    export_path: str | None = None


def require_train_split(split: str) -> None:
    """Teacher-label generation reads the training split and nothing else."""
    normalized = split.strip().lower()
    if normalized == "validation":
        normalized = "val"
    if normalized != TRAIN_SPLIT:
        raise HeldOutTrainingError(
            "Teacher-label generation reads only the training split. "
            "The validation split is for scoring, and the test split stays locked."
        )


def require_serial_calls(concurrency: int) -> None:
    """Keep live labeling at one request at a time so a run cannot fan out."""
    if concurrency != 1:
        raise ValueError(
            "Phase 2 calls the teacher one invoice at a time. "
            "Use --limit to cap spend instead of raising concurrency."
        )


def student_fields_from_extraction(extraction: ExtractedInvoice) -> StudentFields:
    """Map the teacher schema onto the student target.

    The teacher uses null for a missing value. The student file uses an empty
    string for that same missing value. This does not change the prediction.
    """
    total = ""
    if extraction.total is not None:
        total = format(extraction.total, "f")
    invoice_date = ""
    if extraction.invoice_date is not None:
        invoice_date = extraction.invoice_date.isoformat()
    return StudentFields(
        vendor=extraction.vendor or "",
        invoice_number=extraction.invoice_number or "",
        date=invoice_date,
        total=total,
        currency=extraction.currency or "",
    )


def load_checkpoint(path: Path) -> list[LabelCheckpoint]:
    if not path.is_file():
        return []
    rows = read_jsonl(path, LabelCheckpoint)
    latest: dict[str, LabelCheckpoint] = {}
    for row in rows:
        latest[row.document_id] = row
    return list(latest.values())


def save_checkpoint(path: Path, rows: Sequence[LabelCheckpoint]) -> None:
    write_jsonl(path, rows)


def generate_teacher_labels(
    records: Sequence[DatasetRecord],
    settings: Settings,
    *,
    checkpoint_path: Path,
    export_path: Path,
    live: bool,
    limit: int | None = None,
    concurrency: int = 1,
    reuse_checkpoint: bool = True,
    client: Any | None = None,
    sleep_fn: Callable[[float], None] | None = None,
) -> LabelRunReport:
    """Label training invoices, or describe the run when ``live`` is false.

    ``live=False`` does not construct a network client and does not write a
    training file. Existing checkpoint rows are still counted so a dry run
    shows how many calls remain.
    """
    require_serial_calls(concurrency)
    if limit is not None and limit <= 0:
        raise ValueError("--limit must be a positive integer.")
    _reject_non_training_records(records)

    stored = load_checkpoint(checkpoint_path) if reuse_checkpoint else []
    by_id = {row.document_id: row for row in stored}
    skipped_existing = sum(record.document_id in by_id for record in records)
    pending = [record for record in records if record.document_id not in by_id]
    if limit is not None:
        pending = pending[:limit]

    calls_this_run = 0
    if live and pending:
        _require_live_settings(settings, client)
        teacher_kwargs: dict[str, Any] = {"client": client}
        if sleep_fn is not None:
            teacher_kwargs["sleep_fn"] = sleep_fn
        teacher = TeacherClient(settings, **teacher_kwargs)
        for record in pending:
            prediction = teacher.extract(
                document_id=record.document_id,
                document_text=record.document_text,
            )
            by_id[record.document_id] = LabelCheckpoint(
                document_id=record.document_id,
                prediction=prediction,
            )
            calls_this_run += 1
            _write_checkpoint_for_records(checkpoint_path, records, by_id)

    ordered = [by_id[record.document_id] for record in records if record.document_id in by_id]
    verified, rejected = _classify(records, ordered)
    exported = _write_export(export_path, verified) if live else []
    pricing = configured_pricing(settings)
    return _build_report(
        records=records,
        predictions=[row.prediction for row in ordered],
        verified=verified,
        rejected=rejected,
        settings=settings,
        pricing=pricing,
        live=live,
        skipped_existing=skipped_existing if reuse_checkpoint else 0,
        api_calls_this_run=calls_this_run,
        export_path=export_path if live else None,
        export_count=len(exported),
    )


def format_label_report(report: LabelRunReport) -> str:
    mode = "live" if report.live else "dry run"
    cost = "not configured"
    if report.pricing_configured and report.estimated_cost_usd is not None:
        cost = f"{report.estimated_cost_usd:.6f}"
    exact = _percent(report.exact_match_accuracy)
    lines = [
        f"DistillOps teacher labels ({mode})",
        "Split: train",
        f"Model: {report.teacher_model or '(unset)'}",
        f"Training invoices: {report.documents}",
        f"Checkpointed: {report.checkpointed}",
        f"Skipped from checkpoint: {report.skipped_existing}",
        f"API calls this run: {report.api_calls_this_run}",
        f"Successful responses: {report.successful_responses}",
        f"API failures: {report.api_failure_count}",
        f"Verified examples: {report.verified_count}",
        f"Rejected examples: {report.rejected_count}",
        f"Not processed yet: {report.unprocessed_count}",
        f"Exact match (valid outputs): {exact}",
        f"Latency ms: avg {_number(report.latency_ms_avg)}  total {_number(report.latency_ms_total)}",
        f"Tokens: input {report.input_tokens}  output {report.output_tokens}",
        f"Estimated cost USD: {cost}",
    ]
    if report.field_metrics:
        lines.extend(["", "Field accuracy"])
        for metric in report.field_metrics:
            lines.append(
                f"  {metric['field']}: {_percent(metric.get('accuracy'))} "
                f"coverage {_percent(metric.get('coverage'))}"
            )
    if report.rejected:
        lines.extend(["", "Rejected"])
        for item in report.rejected:
            detail = item.reason
            if item.mismatched_fields:
                detail = f"{detail} ({', '.join(item.mismatched_fields)})"
            lines.append(f"  {item.document_id}: {detail}")
    if report.export_path:
        lines.append(f"Export: {report.export_path} ({report.export_count} examples)")
    if not report.live:
        lines.append("No API request was sent. Pass --live to call the teacher.")
    return "\n".join(lines)


def _reject_non_training_records(records: Sequence[DatasetRecord]) -> None:
    """The caller passes training rows only. This checks the id prefix used in tests."""
    for record in records:
        if record.document_id.startswith("test-"):
            raise HeldOutTrainingError(
                "A test-split document id was passed to teacher-label generation."
            )


def _require_live_settings(settings: Settings, client: Any | None) -> None:
    from src.teacher import TeacherConfigError

    if not settings.teacher_model:
        raise TeacherConfigError("TEACHER_MODEL is not set. No teacher request was sent.")
    if client is None and not settings.openai_api_key:
        raise TeacherConfigError(
            "OPENAI_API_KEY is not set. No teacher request was sent."
        )


def _write_checkpoint_for_records(
    path: Path,
    records: Sequence[DatasetRecord],
    by_id: dict[str, LabelCheckpoint],
) -> None:
    ordered = [by_id[record.document_id] for record in records if record.document_id in by_id]
    save_checkpoint(path, ordered)


def _classify(
    records: Sequence[DatasetRecord],
    checkpoints: Sequence[LabelCheckpoint],
) -> tuple[list[StudentTrainingExample], list[LabelRejection]]:
    by_id = {row.document_id: row.prediction for row in checkpoints}
    verified: list[StudentTrainingExample] = []
    rejected: list[LabelRejection] = []
    for record in records:
        prediction = by_id.get(record.document_id)
        if prediction is None:
            continue
        rejection = _rejection_for(record, prediction)
        if rejection is not None:
            rejected.append(rejection)
            continue
        assert prediction.extraction is not None
        output = student_fields_from_extraction(prediction.extraction)
        payload = json.dumps(output.model_dump(), ensure_ascii=False, separators=(",", ":"))
        example = StudentTrainingExample(
            document_id=record.document_id,
            teacher_model=prediction.model,
            verification_status="verified",
            invoice_text=record.document_text,
            teacher_output=output,
            messages=[ChatTurn.model_validate(turn) for turn in build_student_messages(record.document_text, payload)],
        )
        _assert_export_has_no_ground_truth_copy(example, record)
        verified.append(example)
    return verified, rejected


def _rejection_for(record: DatasetRecord, prediction: TeacherPrediction) -> LabelRejection | None:
    if prediction.api_error:
        return LabelRejection(
            document_id=record.document_id,
            reason="api_failure",
            api_error=prediction.api_error,
        )
    if not prediction.schema_valid or prediction.extraction is None:
        return LabelRejection(
            document_id=record.document_id,
            reason="schema_invalid",
            validation_error=prediction.validation_error,
            refusal=prediction.refusal,
        )
    mismatched = [
        field
        for field in EXTRACTION_FIELDS
        if not field_values_match(
            field,
            getattr(prediction.extraction, field),
            getattr(record.ground_truth, field),
        )
    ]
    if mismatched:
        return LabelRejection(
            document_id=record.document_id,
            reason="mismatch",
            mismatched_fields=mismatched,
        )
    return None


def _assert_export_has_no_ground_truth_copy(
    example: StudentTrainingExample,
    record: DatasetRecord,
) -> None:
    """The assistant turn must be the teacher JSON, not a ground-truth dump."""
    assistant = example.messages[-1].content
    if "ground_truth" in assistant:
        raise ValueError("Student export included a ground-truth field.")
    parsed = StudentFields.model_validate_json(assistant)
    if parsed != example.teacher_output:
        raise ValueError("Assistant text does not match the teacher output.")


def _write_export(path: Path, examples: Sequence[StudentTrainingExample]) -> list[StudentTrainingExample]:
    write_jsonl(path, examples)
    return list(examples)


def _build_report(
    *,
    records: Sequence[DatasetRecord],
    predictions: Sequence[TeacherPrediction],
    verified: Sequence[StudentTrainingExample],
    rejected: Sequence[LabelRejection],
    settings: Settings,
    pricing: Pricing | None,
    live: bool,
    skipped_existing: int,
    api_calls_this_run: int,
    export_path: Path | None,
    export_count: int,
) -> LabelRunReport:
    paired_records = [record for record in records if any(prediction.document_id == record.document_id for prediction in predictions)]
    paired_predictions = _predictions_in_record_order(paired_records, predictions)
    evaluation = _evaluation_or_empty(
        paired_records,
        paired_predictions,
        teacher_model=settings.teacher_model or "",
        pricing=pricing,
    )
    latency_total = sum(prediction.latency_ms for prediction in paired_predictions)
    return LabelRunReport(
        live=live,
        teacher_model=settings.teacher_model or "",
        documents=len(records),
        checkpointed=len(paired_predictions),
        skipped_existing=skipped_existing,
        api_calls_this_run=api_calls_this_run,
        successful_responses=evaluation.schema_valid_count,
        api_failure_count=evaluation.api_failure_count,
        verified_count=len(verified),
        rejected_count=len(rejected),
        unprocessed_count=len(records) - len(paired_predictions),
        exact_match_accuracy=evaluation.exact_match_accuracy,
        end_to_end_exact_match_accuracy=evaluation.end_to_end_exact_match_accuracy,
        field_metrics=[metric.model_dump() for metric in evaluation.field_metrics],
        latency_ms_avg=evaluation.latency_ms_avg,
        latency_ms_total=latency_total,
        input_tokens=evaluation.input_tokens,
        output_tokens=evaluation.output_tokens,
        pricing_configured=evaluation.pricing_configured,
        estimated_cost_usd=evaluation.estimated_cost_usd,
        rejected=list(rejected),
        export_count=export_count,
        export_path=str(export_path) if export_path is not None else None,
    )


def _predictions_in_record_order(
    records: Sequence[DatasetRecord],
    predictions: Sequence[TeacherPrediction],
) -> list[TeacherPrediction]:
    by_id = {prediction.document_id: prediction for prediction in predictions}
    return [by_id[record.document_id] for record in records]


def _evaluation_or_empty(
    records: Sequence[DatasetRecord],
    predictions: Sequence[TeacherPrediction],
    *,
    teacher_model: str,
    pricing: Pricing | None,
) -> EvaluationReport:
    if not records:
        return evaluate_predictions(
            [],
            [],
            split=TRAIN_SPLIT,
            teacher_model=teacher_model,
            pricing=pricing,
        )
    return evaluate_predictions(
        records,
        predictions,
        split=TRAIN_SPLIT,
        teacher_model=teacher_model,
        pricing=pricing,
    )


def report_timestamp(now: datetime | None = None) -> str:
    current = now or datetime.now(timezone.utc)
    return current.strftime("%Y%m%dT%H%M%SZ")


def _percent(value: float | None) -> str:
    if value is None:
        return "n/a"
    return f"{value * 100:.1f}%"


def _number(value: float | None) -> str:
    if value is None:
        return "n/a"
    return f"{value:.1f}"
