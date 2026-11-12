"""Score student generations on the five extraction fields.

``document_type`` is not predicted and is not included in accuracy or exact
match. Empty strings and null are the same missing value, using the Phase 1
field comparison.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Sequence
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from src.evaluation import field_values_match, percentile
from src.labeling import StudentFields
from src.schemas import DatasetRecord
from src.student_data import STUDENT_OUTPUT_FIELDS, StudentDataError

EVAL_FIELD_BY_STUDENT: dict[str, str] = {
    "vendor": "vendor",
    "invoice_number": "invoice_number",
    "date": "invoice_date",
    "total": "total",
    "currency": "currency",
}


class StudentGeneration(BaseModel):
    """One student decode. Invalid JSON keeps the raw text and no fields."""

    model_config = ConfigDict(extra="forbid")

    document_id: str
    raw_text: str
    json_valid: bool
    fields: StudentFields | None = None
    validation_error: str | None = None
    latency_ms: float = Field(ge=0)


class StudentFieldScore(BaseModel):
    """Accuracy for one of the five student fields."""

    model_config = ConfigDict(extra="forbid")

    field: str
    correct: int = Field(ge=0)
    scored: int = Field(ge=0)
    accuracy: float | None
    ground_truth_present: int = Field(ge=0)
    coverage: float | None
    present_correct: int = Field(ge=0)
    present_scored: int = Field(ge=0)
    accuracy_when_present: float | None
    missing_predicted: int = Field(ge=0)
    missing_correct: int = Field(ge=0)


class StudentDocumentScore(BaseModel):
    """Per-document student score. The invoice body is not copied here."""

    model_config = ConfigDict(extra="forbid")

    document_id: str
    template_group: str
    json_valid: bool
    exact_match: bool
    field_matches: dict[str, bool]
    validation_error: str | None = None
    latency_ms: float = Field(ge=0)
    raw_text: str


class StudentEvalReport(BaseModel):
    """Validation metrics for a reloaded student. The test split is not here."""

    model_config = ConfigDict(extra="forbid")

    split: str
    model_dir: str
    base_model: str | None = None
    documents: int = Field(ge=0)
    layouts: dict[str, int]
    layout_note: str
    json_valid_count: int = Field(ge=0)
    json_validity_rate: float | None
    exact_match_count: int = Field(ge=0)
    exact_match_accuracy: float | None
    end_to_end_exact_match_accuracy: float | None
    scored_fields: list[str]
    document_type_scored: bool = False
    field_metrics: list[StudentFieldScore]
    latency_ms_avg: float | None
    latency_ms_p50: float | None
    latency_ms_p95: float | None
    results: list[StudentDocumentScore]


def validation_layout_note(records: Sequence[DatasetRecord]) -> str:
    """Describe the layouts actually scored."""
    counts = Counter(record.template_group for record in records)
    rendered = ", ".join(f"{name} ({count})" for name, count in sorted(counts.items()))
    if records and counts.get("account_statement", 0) == len(records):
        return (
            f"All {len(records)} validation documents use the account_statement layout. "
            "They are held-out account statements, not unseen copies of the four "
            "training invoice layouts, and they are not the locked test split."
        )
    return (
        f"Validation layout counts: {rendered or 'none'}. "
        "These documents are not the locked test split."
    )


def score_generations(
    records: Sequence[DatasetRecord],
    generations: Sequence[StudentGeneration],
    *,
    split: str,
    model_dir: str,
    base_model: str | None = None,
) -> StudentEvalReport:
    """Score generations against validation labels on the five student fields."""
    if split != "val":
        raise StudentDataError("Student scores are computed for the validation split only.")
    by_id: dict[str, StudentGeneration] = {}
    for generation in generations:
        if generation.document_id in by_id:
            raise StudentDataError(f"Duplicate generation for {generation.document_id}.")
        by_id[generation.document_id] = generation
    record_ids = {record.document_id for record in records}
    missing = [record.document_id for record in records if record.document_id not in by_id]
    extra = sorted(set(by_id) - record_ids)
    if missing or extra:
        raise StudentDataError("Generations must match the validation documents one-for-one.")

    results: list[StudentDocumentScore] = []
    for record in records:
        results.append(_score_document(record, by_id[record.document_id]))

    documents = len(results)
    valid_rows = [result for result in results if result.json_valid]
    exact_match_count = sum(result.exact_match for result in valid_rows)
    latencies = [result.latency_ms for result in results]
    layouts = dict(sorted(Counter(record.template_group for record in records).items()))
    return StudentEvalReport(
        split=split,
        model_dir=model_dir,
        base_model=base_model,
        documents=documents,
        layouts=layouts,
        layout_note=validation_layout_note(records),
        json_valid_count=len(valid_rows),
        json_validity_rate=_rate(len(valid_rows), documents),
        exact_match_count=exact_match_count,
        exact_match_accuracy=_rate(exact_match_count, len(valid_rows)),
        end_to_end_exact_match_accuracy=_rate(exact_match_count, documents),
        scored_fields=list(STUDENT_OUTPUT_FIELDS),
        document_type_scored=False,
        field_metrics=_field_scores(records, by_id),
        latency_ms_avg=(sum(latencies) / len(latencies)) if latencies else None,
        latency_ms_p50=percentile(latencies, 50),
        latency_ms_p95=percentile(latencies, 95),
        results=results,
    )


def format_student_report(report: StudentEvalReport) -> str:
    """Plain-text validation summary."""
    lines = [
        "DistillOps student validation",
        f"Split: {report.split}",
        f"Model directory: {report.model_dir}",
        f"Base model: {report.base_model or 'unknown'}",
        f"Documents: {report.documents}",
        f"Layouts: {_layouts(report.layouts)}",
        report.layout_note,
        "Scored fields: " + ", ".join(report.scored_fields),
        "document_type scored: no",
        (
            "JSON validity: "
            f"{_percent(report.json_validity_rate)} "
            f"({report.json_valid_count}/{report.documents})"
        ),
        (
            "Exact match (valid JSON): "
            f"{_percent(report.exact_match_accuracy)} "
            f"({report.exact_match_count}/{report.json_valid_count})"
        ),
        (
            "Exact match (end to end): "
            f"{_percent(report.end_to_end_exact_match_accuracy)} "
            f"({report.exact_match_count}/{report.documents})"
        ),
        "",
        "Field metrics",
        _field_table(report),
        "",
        (
            "Latency ms: "
            f"avg {_number(report.latency_ms_avg)}  "
            f"p50 {_number(report.latency_ms_p50)}  "
            f"p95 {_number(report.latency_ms_p95)}"
        ),
    ]
    return "\n".join(lines)


def _score_document(record: DatasetRecord, generation: StudentGeneration) -> StudentDocumentScore:
    matches: dict[str, bool] = {}
    if generation.json_valid and generation.fields is not None:
        for student_field, eval_field in EVAL_FIELD_BY_STUDENT.items():
            matches[student_field] = field_values_match(
                eval_field,
                getattr(generation.fields, student_field),
                getattr(record.ground_truth, eval_field),
            )
    return StudentDocumentScore(
        document_id=record.document_id,
        template_group=record.template_group,
        json_valid=generation.json_valid and generation.fields is not None,
        exact_match=bool(matches) and all(matches.values()),
        field_matches=matches,
        validation_error=generation.validation_error,
        latency_ms=generation.latency_ms,
        raw_text=generation.raw_text,
    )


def _field_scores(
    records: Sequence[DatasetRecord],
    by_id: dict[str, StudentGeneration],
) -> list[StudentFieldScore]:
    documents = len(records)
    valid = [
        record
        for record in records
        if by_id[record.document_id].json_valid and by_id[record.document_id].fields is not None
    ]
    scores: list[StudentFieldScore] = []
    for student_field, eval_field in EVAL_FIELD_BY_STUDENT.items():
        correct = 0
        present_correct = 0
        present_scored = 0
        missing_predicted = 0
        missing_correct = 0
        for record in valid:
            fields = by_id[record.document_id].fields
            assert fields is not None
            predicted = getattr(fields, student_field)
            expected = getattr(record.ground_truth, eval_field)
            matched = field_values_match(eval_field, predicted, expected)
            correct += int(matched)
            truth_missing = _is_missing(expected)
            prediction_missing = _is_missing(predicted)
            if not truth_missing:
                present_scored += 1
                present_correct += int(matched)
            if prediction_missing:
                missing_predicted += 1
                missing_correct += int(truth_missing)
        ground_truth_present = sum(
            not _is_missing(getattr(record.ground_truth, eval_field)) for record in records
        )
        scores.append(
            StudentFieldScore(
                field=student_field,
                correct=correct,
                scored=len(valid),
                accuracy=_rate(correct, len(valid)),
                ground_truth_present=ground_truth_present,
                coverage=_rate(ground_truth_present, documents),
                present_correct=present_correct,
                present_scored=present_scored,
                accuracy_when_present=_rate(present_correct, present_scored),
                missing_predicted=missing_predicted,
                missing_correct=missing_correct,
            )
        )
    return scores


def _is_missing(value: Any) -> bool:
    if value is None:
        return True
    if isinstance(value, str):
        return value.strip() == ""
    return False


def _rate(numerator: int, denominator: int) -> float | None:
    if denominator == 0:
        return None
    return numerator / denominator


def _percent(value: float | None) -> str:
    if value is None:
        return "n/a"
    return f"{value * 100:.1f}%"


def _number(value: float | None) -> str:
    if value is None:
        return "n/a"
    return f"{value:.1f}"


def _layouts(layouts: dict[str, int]) -> str:
    if not layouts:
        return "none"
    return ", ".join(f"{name}={count}" for name, count in layouts.items())


def _field_table(report: StudentEvalReport) -> str:
    if not report.field_metrics:
        return "(no documents)"
    header = (
        f"{'field':<16} {'accuracy':>10} {'when_present':>14} "
        f"{'coverage':>10} {'missing_ok':>12}"
    )
    lines = [header]
    for metric in report.field_metrics:
        missing = f"{metric.missing_correct}/{metric.missing_predicted}"
        lines.append(
            f"{metric.field:<16} {_percent(metric.accuracy):>10} "
            f"{_percent(metric.accuracy_when_present):>14} "
            f"{_percent(metric.coverage):>10} "
            f"{missing:>12}"
        )
    return "\n".join(lines)
