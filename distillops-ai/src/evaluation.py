"""Compare teacher predictions with invoice ground truth.

Null compared with null is a match. Field coverage and accuracy-when-present
are reported beside that score so a run of empty fields cannot look perfect
without the reader seeing how often the field existed.
"""

from __future__ import annotations

import math
import re
from collections.abc import Sequence
from datetime import date, datetime
from decimal import Decimal
from typing import Any

import pandas as pd
from pydantic import BaseModel, ConfigDict, Field

from src.evaluation_pricing import Pricing
from src.schemas import EXTRACTION_FIELDS, DatasetRecord, ExtractedInvoice, TeacherPrediction

_ISO_DATE_RE = re.compile(r"^\d{4}-\d{2}-\d{2}$")


class EvaluationError(ValueError):
    """Raised when predictions cannot be aligned to the labeled documents."""


class FieldMetric(BaseModel):
    """Accuracy and coverage for one schema field."""

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


class DocumentEvaluation(BaseModel):
    """Preserved per-document prediction, errors, and field matches."""

    model_config = ConfigDict(extra="forbid")

    document_id: str
    template_group: str
    schema_valid: bool
    api_error: str | None = None
    validation_error: str | None = None
    refusal: str | None = None
    exact_match: bool | None = None
    field_matches: dict[str, bool]
    prediction: ExtractedInvoice | None = None
    ground_truth: ExtractedInvoice
    latency_ms: float = Field(ge=0)
    input_tokens: int = Field(ge=0)
    output_tokens: int = Field(ge=0)
    model: str


class EvaluationReport(BaseModel):
    """Aggregate baseline metrics and the per-document rows behind them."""

    model_config = ConfigDict(extra="forbid")

    split: str
    teacher_model: str
    documents: int = Field(ge=0)
    limit: int | None = None
    schema_valid_count: int = Field(ge=0)
    schema_validity_rate: float | None
    api_failure_count: int = Field(ge=0)
    api_failure_rate: float | None
    exact_match_count: int = Field(ge=0)
    exact_match_accuracy: float | None
    end_to_end_exact_match_accuracy: float | None
    field_metrics: list[FieldMetric]
    latency_ms_avg: float | None
    latency_ms_p50: float | None
    latency_ms_p95: float | None
    input_tokens: int = Field(ge=0)
    output_tokens: int = Field(ge=0)
    pricing_configured: bool
    estimated_cost_usd: Decimal | None = None
    results: list[DocumentEvaluation]


def normalize_whitespace(value: str | None) -> str | None:
    if value is None:
        return None
    collapsed = " ".join(value.split())
    return collapsed or None


def canonical_date(value: date | datetime | str | None) -> str | None:
    """Normalize an ISO date to ``YYYY-MM-DD``. Other strings are left unchanged."""
    if value is None:
        return None
    if isinstance(value, datetime):
        value = value.date()
    if isinstance(value, date):
        return value.isoformat()
    if isinstance(value, str):
        text = value.strip()
        if not text:
            return None
        if _ISO_DATE_RE.fullmatch(text):
            return date.fromisoformat(text).isoformat()
        return text
    return str(value)


def canonical_amount(value: Decimal | str | int | float | None) -> Decimal | None:
    """Parse an amount and compare it numerically. ``10.00`` equals ``10``."""
    if value is None:
        return None
    if isinstance(value, bool):
        raise TypeError("amount cannot be a boolean")
    if isinstance(value, Decimal):
        amount = value
    elif isinstance(value, int):
        amount = Decimal(value)
    elif isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError("amount must be finite")
        amount = Decimal(str(value))
    elif isinstance(value, str):
        text = value.strip().replace(",", "")
        if not text:
            return None
        amount = Decimal(text)
    else:
        raise TypeError("unsupported amount")
    if not amount.is_finite():
        raise ValueError("amount must be finite")
    return amount


def field_values_match(field: str, predicted: Any, expected: Any) -> bool:
    """Return whether one field matches after the normalization for that field."""
    if field in {"vendor", "invoice_number", "document_type"}:
        return normalize_whitespace(_as_optional_str(predicted)) == normalize_whitespace(
            _as_optional_str(expected)
        )
    if field == "currency":
        left = normalize_whitespace(_as_optional_str(predicted))
        right = normalize_whitespace(_as_optional_str(expected))
        if left is None or right is None:
            return left is None and right is None
        return left.upper() == right.upper()
    if field == "invoice_date":
        return canonical_date(predicted) == canonical_date(expected)
    if field == "total":
        return canonical_amount(predicted) == canonical_amount(expected)
    raise KeyError(field)


def percentile(values: Sequence[float], percent: float) -> float | None:
    """Linear interpolation between the closest ranks. Empty input returns None."""
    if not values:
        return None
    if not 0 <= percent <= 100:
        raise ValueError("percent must be between 0 and 100")
    ordered = sorted(float(value) for value in values)
    if len(ordered) == 1:
        return ordered[0]
    rank = (len(ordered) - 1) * (percent / 100)
    low = math.floor(rank)
    high = math.ceil(rank)
    if low == high:
        return ordered[low]
    weight = rank - low
    return ordered[low] * (1 - weight) + ordered[high] * weight


def evaluate_predictions(
    records: Sequence[DatasetRecord],
    predictions: Sequence[TeacherPrediction],
    *,
    split: str,
    teacher_model: str,
    pricing: Pricing | None = None,
    limit: int | None = None,
) -> EvaluationReport:
    """Score ``predictions`` against ``records``. Ids must match one for one."""
    by_id: dict[str, TeacherPrediction] = {}
    for prediction in predictions:
        if prediction.document_id in by_id:
            raise EvaluationError(f"Duplicate prediction for {prediction.document_id}.")
        by_id[prediction.document_id] = prediction
    record_ids = {record.document_id for record in records}
    missing = [record.document_id for record in records if record.document_id not in by_id]
    extra = sorted(set(by_id) - record_ids)
    if missing or extra:
        raise EvaluationError("Predictions must match the evaluated documents one-for-one.")

    results: list[DocumentEvaluation] = []
    for record in records:
        results.append(_score_document(record, by_id[record.document_id]))

    documents = len(results)
    schema_valid_count = sum(result.schema_valid for result in results)
    api_failure_count = sum(result.api_error is not None for result in results)
    exact_match_count = sum(result.exact_match is True for result in results)
    latencies = [result.latency_ms for result in results]
    input_tokens = sum(result.input_tokens for result in results)
    output_tokens = sum(result.output_tokens for result in results)
    estimated_cost = None if pricing is None else pricing.estimate_usd(input_tokens, output_tokens)
    return EvaluationReport(
        split=split,
        teacher_model=teacher_model,
        documents=documents,
        limit=limit,
        schema_valid_count=schema_valid_count,
        schema_validity_rate=_rate(schema_valid_count, documents),
        api_failure_count=api_failure_count,
        api_failure_rate=_rate(api_failure_count, documents),
        exact_match_count=exact_match_count,
        exact_match_accuracy=_rate(exact_match_count, schema_valid_count),
        end_to_end_exact_match_accuracy=_rate(exact_match_count, documents),
        field_metrics=_field_metrics(results),
        latency_ms_avg=(sum(latencies) / len(latencies)) if latencies else None,
        latency_ms_p50=percentile(latencies, 50),
        latency_ms_p95=percentile(latencies, 95),
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        pricing_configured=pricing is not None,
        estimated_cost_usd=estimated_cost,
        results=results,
    )


def format_report(report: EvaluationReport) -> str:
    """Plain-text summary for the baseline command."""
    lines = [
        "DistillOps teacher baseline",
        f"Split: {report.split}",
        f"Model: {report.teacher_model}",
        f"Documents: {report.documents}",
        (
            "Schema validity: "
            f"{_percent(report.schema_validity_rate)} "
            f"({report.schema_valid_count}/{report.documents})"
        ),
        (
            "API failure rate: "
            f"{_percent(report.api_failure_rate)} "
            f"({report.api_failure_count}/{report.documents})"
        ),
        (
            "Exact match (valid outputs): "
            f"{_percent(report.exact_match_accuracy)} "
            f"({report.exact_match_count}/{report.schema_valid_count})"
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
        f"Tokens: input {report.input_tokens}  output {report.output_tokens}",
        f"Estimated cost USD: {_cost(report)}",
    ]
    return "\n".join(lines)


def _score_document(record: DatasetRecord, prediction: TeacherPrediction) -> DocumentEvaluation:
    scored = prediction.schema_valid and prediction.extraction is not None and prediction.api_error is None
    matches: dict[str, bool] = {}
    exact: bool | None = None
    if scored:
        assert prediction.extraction is not None
        for field in EXTRACTION_FIELDS:
            matches[field] = field_values_match(
                field,
                getattr(prediction.extraction, field),
                getattr(record.ground_truth, field),
            )
        exact = all(matches.values())
    return DocumentEvaluation(
        document_id=record.document_id,
        template_group=record.template_group,
        schema_valid=bool(scored),
        api_error=prediction.api_error,
        validation_error=prediction.validation_error,
        refusal=prediction.refusal,
        exact_match=exact,
        field_matches=matches,
        prediction=prediction.extraction if scored else None,
        ground_truth=record.ground_truth,
        latency_ms=prediction.latency_ms,
        input_tokens=prediction.input_tokens,
        output_tokens=prediction.output_tokens,
        model=prediction.model,
    )


def _field_metrics(results: Sequence[DocumentEvaluation]) -> list[FieldMetric]:
    documents = len(results)
    scored_rows = [result for result in results if result.schema_valid]
    metrics: list[FieldMetric] = []
    for field in EXTRACTION_FIELDS:
        correct = sum(result.field_matches.get(field, False) for result in scored_rows)
        present_rows = [result for result in scored_rows if getattr(result.ground_truth, field) is not None]
        present_correct = sum(result.field_matches.get(field, False) for result in present_rows)
        ground_truth_present = sum(getattr(result.ground_truth, field) is not None for result in results)
        metrics.append(
            FieldMetric(
                field=field,
                correct=correct,
                scored=len(scored_rows),
                accuracy=_rate(correct, len(scored_rows)),
                ground_truth_present=ground_truth_present,
                coverage=_rate(ground_truth_present, documents),
                present_correct=present_correct,
                present_scored=len(present_rows),
                accuracy_when_present=_rate(present_correct, len(present_rows)),
            )
        )
    return metrics


def _field_table(report: EvaluationReport) -> str:
    rows = []
    for metric in report.field_metrics:
        rows.append(
            {
                "field": metric.field,
                "accuracy": _percent(metric.accuracy),
                "coverage": _percent(metric.coverage),
                "accuracy_when_present": _percent(metric.accuracy_when_present),
                "present": f"{metric.present_correct}/{metric.present_scored}",
            }
        )
    if not rows:
        return "(no documents)"
    return pd.DataFrame(rows).to_string(index=False)


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


def _cost(report: EvaluationReport) -> str:
    if not report.pricing_configured or report.estimated_cost_usd is None:
        return "not configured"
    return f"{report.estimated_cost_usd:.6f}"


def _as_optional_str(value: Any) -> str | None:
    if value is None:
        return None
    return str(value)
