"""Normalization, exact-match scoring, coverage, and failure handling."""

from datetime import date
from decimal import Decimal

import pytest

from src.evaluation import (
    EvaluationError,
    canonical_amount,
    canonical_date,
    evaluate_predictions,
    field_values_match,
    format_report,
    normalize_whitespace,
    percentile,
)
from src.evaluation_pricing import Pricing
from src.schemas import DatasetRecord, ExtractedInvoice, TeacherPrediction


def _invoice(**overrides: object) -> ExtractedInvoice:
    payload: dict[str, object] = {"document_type": "invoice"}
    payload.update(overrides)
    return ExtractedInvoice.model_validate(payload)


def _record(document_id: str = "inv-synth-001", **overrides: object) -> DatasetRecord:
    return DatasetRecord(
        document_id=document_id,
        document_text="INVOICE\nSynthetic training document.",
        ground_truth=_invoice(**overrides),
        template_group="classic_header",
    )


def _prediction(
    document_id: str = "inv-synth-001",
    extraction: ExtractedInvoice | None = None,
    **overrides: object,
) -> TeacherPrediction:
    payload: dict[str, object] = {
        "document_id": document_id,
        "model": "teacher-test",
        "extraction": extraction,
        "schema_valid": extraction is not None,
        "latency_ms": 10,
        "input_tokens": 4,
        "output_tokens": 2,
    }
    payload.update(overrides)
    return TeacherPrediction.model_validate(payload)


def test_whitespace_dates_and_amounts_normalize_for_comparison() -> None:
    assert normalize_whitespace("  Acme   Supply \n") == "Acme Supply"
    assert normalize_whitespace("   ") is None
    assert canonical_date("2024-01-02") == "2024-01-02"
    assert field_values_match("invoice_date", "2024-01-02", date(2024, 1, 2))
    assert canonical_amount("1,000.50") == Decimal("1000.50")
    assert canonical_amount("10.00") == canonical_amount("10")
    assert canonical_amount("") is None
    assert field_values_match("vendor", "Acme   Corp", "Acme Corp")
    assert field_values_match("vendor", None, None)
    assert field_values_match("vendor", "", None)
    assert not field_values_match("vendor", None, "Acme")
    assert field_values_match("currency", " usd ", "USD")
    assert field_values_match("total", Decimal("10.00"), "10")
    assert not field_values_match("invoice_date", "January 2, 2024", "2024-01-02")


def test_percentiles_use_linear_interpolation() -> None:
    values = [10, 20, 30, 40, 50]
    assert percentile(values, 50) == 30
    assert percentile(values, 95) == pytest.approx(48)
    assert percentile([7], 95) == 7
    assert percentile([], 95) is None


def test_exact_match_counts_null_as_equal_and_one_miss_as_not_exact() -> None:
    truth = _invoice(vendor="Cedar Pine Analytics", total=Decimal("10.00"), currency="USD")
    match = _prediction(extraction=_invoice(vendor="Cedar  Pine   Analytics", total="10", currency="usd"))
    miss = _prediction(
        document_id="inv-synth-002",
        extraction=_invoice(vendor="Other Vendor", total="10", currency="USD"),
    )
    report = evaluate_predictions(
        [
            _record(vendor="Cedar Pine Analytics", total=Decimal("10.00"), currency="USD"),
            _record("inv-synth-002", vendor="Cedar Pine Analytics", total=Decimal("10.00"), currency="USD"),
        ],
        [match, miss],
        split="val",
        teacher_model="teacher-test",
    )
    assert truth.total == Decimal("10.00")
    assert report.exact_match_count == 1
    assert report.exact_match_accuracy == 0.5
    assert report.end_to_end_exact_match_accuracy == 0.5
    vendor = next(metric for metric in report.field_metrics if metric.field == "vendor")
    assert vendor.accuracy == 0.5
    assert vendor.coverage == 1
    assert report.results[0].exact_match is True
    assert report.results[1].field_matches["vendor"] is False


def test_null_predictions_score_perfect_accuracy_with_zero_coverage() -> None:
    report = evaluate_predictions(
        [_record(total=None)],
        [_prediction(extraction=_invoice(total=None))],
        split="val",
        teacher_model="teacher-test",
    )
    total = next(metric for metric in report.field_metrics if metric.field == "total")
    assert total.accuracy == 1
    assert total.coverage == 0
    assert total.accuracy_when_present is None
    assert report.exact_match_accuracy == 1


def test_api_failure_is_not_scored_as_a_null_match() -> None:
    report = evaluate_predictions(
        [_record(total=None, vendor=None)],
        [
            _prediction(
                extraction=None,
                schema_valid=False,
                api_error="RuntimeError status=500",
                latency_ms=40,
                input_tokens=0,
                output_tokens=0,
            )
        ],
        split="val",
        teacher_model="teacher-test",
    )
    assert report.api_failure_count == 1
    assert report.api_failure_rate == 1
    assert report.schema_valid_count == 0
    assert report.exact_match_accuracy is None
    assert report.end_to_end_exact_match_accuracy == 0
    assert report.results[0].exact_match is None
    assert report.results[0].field_matches == {}
    assert report.results[0].api_error == "RuntimeError status=500"
    total = next(metric for metric in report.field_metrics if metric.field == "total")
    assert total.accuracy is None
    assert total.scored == 0


def test_schema_failure_is_separate_from_an_api_failure() -> None:
    report = evaluate_predictions(
        [_record(vendor="Cedar Pine Analytics")],
        [
            _prediction(
                extraction=None,
                schema_valid=False,
                validation_error="total: value_error",
                input_tokens=15,
                output_tokens=6,
                latency_ms=25,
            )
        ],
        split="val",
        teacher_model="teacher-test",
    )
    assert report.api_failure_count == 0
    assert report.schema_validity_rate == 0
    assert report.input_tokens == 15
    assert report.output_tokens == 6
    assert report.results[0].validation_error == "total: value_error"


def test_latency_tokens_and_optional_cost() -> None:
    predictions = [
        _prediction(latency_ms=value, input_tokens=200_000, output_tokens=100_000)
        for value in (10, 20, 30, 40, 50)
    ]
    records = [_record(f"inv-synth-{index:03d}") for index in range(1, 6)]
    for prediction, record in zip(predictions, records, strict=True):
        prediction_id = record.document_id
        predictions[records.index(record)] = prediction.model_copy(update={"document_id": prediction_id})
    unpriced = evaluate_predictions(records, predictions, split="val", teacher_model="teacher-test")
    assert unpriced.latency_ms_avg == 30
    assert unpriced.latency_ms_p50 == 30
    assert unpriced.latency_ms_p95 == pytest.approx(48)
    assert unpriced.input_tokens == 1_000_000
    assert unpriced.output_tokens == 500_000
    assert unpriced.estimated_cost_usd is None
    assert unpriced.pricing_configured is False
    assert "not configured" in format_report(unpriced)

    priced = evaluate_predictions(
        records,
        predictions,
        split="val",
        teacher_model="teacher-test",
        pricing=Pricing(
            input_usd_per_million_tokens=Decimal("0.15"),
            output_usd_per_million_tokens=Decimal("0.60"),
        ),
    )
    assert priced.estimated_cost_usd == Decimal("0.450000")
    assert "0.450000" in format_report(priced)
    assert "Field metrics" in format_report(priced)


def test_prediction_ids_must_match_the_documents() -> None:
    with pytest.raises(EvaluationError):
        evaluate_predictions(
            [_record("inv-synth-001"), _record("inv-synth-002")],
            [_prediction("inv-synth-001", extraction=_invoice())],
            split="val",
            teacher_model="teacher-test",
        )
    with pytest.raises(EvaluationError):
        evaluate_predictions(
            [_record()],
            [
                _prediction(extraction=_invoice()),
                _prediction(extraction=_invoice()),
            ],
            split="val",
            teacher_model="teacher-test",
        )
