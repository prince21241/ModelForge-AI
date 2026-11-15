"""Phase 4 benchmark metrics for the frozen teacher and student.

Comparisons use the existing normalization rules. A wrong date or amount is
kept as produced. Malformed JSON is counted apart from a wrong field.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
from collections.abc import Sequence
from decimal import Decimal, InvalidOperation
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

from src.evaluation import field_values_match, percentile
from src.evaluation_pricing import Pricing
from src.labeling import StudentFields
from src.prompts import STUDENT_TARGET_INSTRUCTION, SYSTEM_PROMPT
from src.schemas import EXTRACTION_FIELDS, DatasetRecord, ExtractedInvoice, TeacherPrediction
from src.student_data import STUDENT_OUTPUT_FIELDS, parse_student_json

EXPERIMENT_NAME = "DistillOps-Phase4-Benchmark"
FROZEN_STUDENT_DIR = Path("models/student/flan-t5-small-dated12/epoch-30")
FROZEN_TEACHER_MODEL = "gpt-4o-mini"
BENCHMARK_FIELDS: tuple[str, ...] = STUDENT_OUTPUT_FIELDS
EVAL_FIELD: dict[str, str] = {
    "vendor": "vendor",
    "invoice_number": "invoice_number",
    "date": "invoice_date",
    "total": "total",
    "currency": "currency",
}
DECODING = {
    "do_sample": False,
    "num_beams": 1,
    "max_new_tokens": 128,
    "max_source_length": 512,
}
WARMUP_DOCUMENT_ID = "benchmark-warmup"
WARMUP_TEXT = "WARMUP INVOICE\nPayee Example Desk\nTotal due 1.00 USD\n"
STUDENT_COST_ASSUMPTION = (
    "Local GPU inference uses no API tokens. Estimated student cost stays "
    "unset unless STUDENT_USD_PER_1000_INFERENCES is configured. No GPU "
    "electricity price is assumed."
)
NORMALIZATION_RULES: tuple[str, ...] = (
    "Vendor and invoice number comparisons collapse whitespace. An empty string and null match.",
    "Currency comparison is case-insensitive.",
    "An ISO date YYYY-MM-DD matches that calendar date. Any other date string is compared unchanged.",
    "A numeric total matches after commas are removed. A non-numeric total is a mismatch and is not rewritten.",
    "document_type is not part of the five-field exact match.",
    "Malformed JSON and schema failures are not counted as missing, hallucinated, or incorrect fields.",
)
_ID_RE = re.compile(r'"document_id"\s*:\s*"([A-Za-z0-9][A-Za-z0-9_-]*)"')
_GROUP_RE = re.compile(r'"template_group"\s*:\s*"([a-z0-9_]+)"')


class BenchmarkError(ValueError):
    """Raised when a benchmark cannot be scored as requested."""


class LockedSplitError(BenchmarkError):
    """Raised before the locked test file is opened."""


class FieldErrorCount(BaseModel):
    """How often one field was missing, invented, or extracted incorrectly."""

    model_config = ConfigDict(extra="forbid")

    field: str
    correct: int = Field(ge=0)
    scored: int = Field(ge=0)
    accuracy: float | None
    missing: int = Field(ge=0)
    hallucinated: int = Field(ge=0)
    incorrect: int = Field(ge=0)


class BenchmarkRow(BaseModel):
    """One invoice score. The source invoice text is not stored."""

    model_config = ConfigDict(extra="forbid")

    document_id: str
    template_group: str
    model_role: Literal["teacher", "student"]
    model_id: str
    json_valid: bool
    schema_valid: bool
    malformed_json: bool
    exact_match: bool
    field_matches: dict[str, bool]
    missing_fields: list[str]
    hallucinated_fields: list[str]
    incorrect_fields: list[str]
    latency_ms: float = Field(ge=0)
    input_tokens: int = Field(ge=0, default=0)
    output_tokens: int = Field(ge=0, default=0)
    api_error: str | None = None
    validation_error: str | None = None
    prediction: dict[str, str]
    expected: dict[str, str]


class BenchmarkReport(BaseModel):
    """Aggregate scores for one model on one split."""

    model_config = ConfigDict(extra="forbid")

    status: Literal["completed", "dry_run", "cache_incomplete", "not_run"]
    model_role: Literal["teacher", "student"]
    model_id: str
    split: str
    documents_in_split: int = Field(ge=0)
    documents_scored: int = Field(ge=0)
    unscored_document_ids: list[str]
    checkpoint: str | None = None
    decoding: dict[str, Any]
    normalization_rules: list[str]
    prompt_fingerprints: dict[str, str]
    json_valid_count: int = Field(ge=0)
    schema_valid_count: int = Field(ge=0)
    malformed_json_count: int = Field(ge=0)
    exact_match_count: int = Field(ge=0)
    exact_match_accuracy: float | None
    end_to_end_exact_match_accuracy: float | None
    field_metrics: list[FieldErrorCount]
    missing_field_count: int = Field(ge=0)
    hallucinated_field_count: int = Field(ge=0)
    incorrect_field_count: int = Field(ge=0)
    latency_ms_avg: float | None
    latency_ms_median: float | None
    latency_ms_p95: float | None
    warmup_ms: float | None = None
    warmup_excluded: bool
    input_tokens: int = Field(ge=0)
    output_tokens: int = Field(ge=0)
    api_calls: int = Field(ge=0)
    api_failure_count: int = Field(ge=0)
    pricing_configured: bool
    estimated_cost_usd: Decimal | None = None
    cost_assumption: str
    rows: list[BenchmarkRow]


def resolve_split(name: str, *, authorize_locked_test: bool) -> str:
    """Map a CLI split name to a dataset split. Test stays closed without the flag."""
    normalized = name.strip().lower()
    if normalized == "validation":
        normalized = "val"
    if normalized == "test":
        if not authorize_locked_test:
            raise LockedSplitError(
                "The locked test split was not opened. Re-run with "
                "--authorize-locked-test only after the test benchmark is approved."
            )
        return "test"
    if normalized != "val":
        raise BenchmarkError("Split must be validation or test.")
    return "val"


def assert_test_file_closed(path: Path, *, authorize_locked_test: bool) -> None:
    """Refuse the locked filename before the caller opens it."""
    if Path(path).name.lower() == "test.jsonl" and not authorize_locked_test:
        raise LockedSplitError(
            "The locked test split was not opened. Re-run with "
            "--authorize-locked-test only after the test benchmark is approved."
        )


def prompt_fingerprints() -> dict[str, str]:
    """Hash the frozen prompts so a later run can see whether they changed."""
    return {
        "student_instruction_sha256": _sha256_text(STUDENT_TARGET_INSTRUCTION),
        "teacher_system_prompt_sha256": _sha256_text(SYSTEM_PROMPT),
    }


def inspect_json_output(raw_text: str) -> tuple[bool, bool, StudentFields | None, str | None]:
    """Separate valid JSON from a five-field schema. Values are not repaired."""
    try:
        payload = json.loads(raw_text)
    except json.JSONDecodeError:
        return False, False, None, "Output is not valid JSON."
    if not isinstance(payload, dict):
        return True, False, None, "Output JSON must be an object."
    fields, error = parse_student_json(raw_text)
    return True, fields is not None, fields, error


def five_fields_from_truth(truth: ExtractedInvoice) -> dict[str, str]:
    """Copy the five labels. Missing values become empty strings."""
    return {
        "vendor": truth.vendor or "",
        "invoice_number": truth.invoice_number or "",
        "date": "" if truth.invoice_date is None else truth.invoice_date.isoformat(),
        "total": "" if truth.total is None else format(truth.total, "f"),
        "currency": truth.currency or "",
    }


def five_fields_from_student(fields: StudentFields) -> dict[str, str]:
    return {name: getattr(fields, name) for name in BENCHMARK_FIELDS}


def five_fields_from_teacher(extraction: ExtractedInvoice) -> dict[str, str]:
    return five_fields_from_truth(extraction)


def classify_field(field: str, predicted: str, expected: str) -> str:
    """Name one field outcome. The predicted string is not modified."""
    eval_field = EVAL_FIELD[field]
    if _values_match(eval_field, predicted, expected):
        return "match"
    predicted_blank = _blank(predicted)
    expected_blank = _blank(expected)
    if predicted_blank and not expected_blank:
        return "missing"
    if not predicted_blank and expected_blank:
        return "hallucinated"
    return "incorrect"


def score_prediction(
    *,
    document_id: str,
    template_group: str,
    model_role: Literal["teacher", "student"],
    model_id: str,
    expected: dict[str, str],
    prediction: dict[str, str] | None,
    json_valid: bool,
    schema_valid: bool,
    validation_error: str | None,
    latency_ms: float,
    input_tokens: int = 0,
    output_tokens: int = 0,
    api_error: str | None = None,
    malformed_json: bool | None = None,
) -> BenchmarkRow:
    """Score one document. Field errors are recorded only for a valid schema."""
    matches: dict[str, bool] = {}
    missing: list[str] = []
    hallucinated: list[str] = []
    incorrect: list[str] = []
    shown = {name: "" for name in BENCHMARK_FIELDS}
    if schema_valid and prediction is not None:
        shown = {name: prediction[name] for name in BENCHMARK_FIELDS}
        for name in BENCHMARK_FIELDS:
            kind = classify_field(name, shown[name], expected[name])
            matches[name] = kind == "match"
            if kind == "missing":
                missing.append(name)
            elif kind == "hallucinated":
                hallucinated.append(name)
            elif kind == "incorrect":
                incorrect.append(name)
    return BenchmarkRow(
        document_id=document_id,
        template_group=template_group,
        model_role=model_role,
        model_id=model_id,
        json_valid=json_valid,
        schema_valid=schema_valid,
        malformed_json=not json_valid if malformed_json is None else malformed_json,
        exact_match=bool(matches) and all(matches.values()),
        field_matches=matches,
        missing_fields=missing,
        hallucinated_fields=hallucinated,
        incorrect_fields=incorrect,
        latency_ms=latency_ms,
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        api_error=api_error,
        validation_error=validation_error,
        prediction=shown if schema_valid and prediction is not None else {},
        expected=expected,
    )


def build_report(
    rows: Sequence[BenchmarkRow],
    *,
    status: Literal["completed", "dry_run", "cache_incomplete", "not_run"],
    model_role: Literal["teacher", "student"],
    model_id: str,
    split: str,
    documents_in_split: int,
    unscored_document_ids: Sequence[str],
    checkpoint: str | None,
    decoding: dict[str, Any],
    warmup_ms: float | None,
    warmup_excluded: bool,
    api_calls: int,
    pricing: Pricing | None,
    cost_assumption: str,
) -> BenchmarkReport:
    """Aggregate rows. Latency percentiles ignore the warmup pass."""
    scored = list(rows)
    schema_rows = [row for row in scored if row.schema_valid]
    exact = sum(row.exact_match for row in schema_rows)
    latencies = [row.latency_ms for row in scored]
    input_tokens = sum(row.input_tokens for row in scored)
    output_tokens = sum(row.output_tokens for row in scored)
    metrics = _field_metrics(schema_rows)
    cost = None if pricing is None else pricing.estimate_usd(input_tokens, output_tokens)
    return BenchmarkReport(
        status=status,
        model_role=model_role,
        model_id=model_id,
        split=split,
        documents_in_split=documents_in_split,
        documents_scored=len(scored),
        unscored_document_ids=list(unscored_document_ids),
        checkpoint=checkpoint,
        decoding=decoding,
        normalization_rules=list(NORMALIZATION_RULES),
        prompt_fingerprints=prompt_fingerprints(),
        json_valid_count=sum(row.json_valid for row in scored),
        schema_valid_count=len(schema_rows),
        malformed_json_count=sum(row.malformed_json for row in scored),
        exact_match_count=exact,
        exact_match_accuracy=_rate(exact, len(schema_rows)),
        end_to_end_exact_match_accuracy=_rate(exact, documents_in_split),
        field_metrics=metrics,
        missing_field_count=sum(len(row.missing_fields) for row in schema_rows),
        hallucinated_field_count=sum(len(row.hallucinated_fields) for row in schema_rows),
        incorrect_field_count=sum(len(row.incorrect_fields) for row in schema_rows),
        latency_ms_avg=(sum(latencies) / len(latencies)) if latencies else None,
        latency_ms_median=percentile(latencies, 50),
        latency_ms_p95=percentile(latencies, 95),
        warmup_ms=warmup_ms,
        warmup_excluded=warmup_excluded,
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        api_calls=api_calls,
        api_failure_count=sum(row.api_error is not None for row in scored),
        pricing_configured=pricing is not None,
        estimated_cost_usd=cost,
        cost_assumption=cost_assumption,
        rows=scored,
    )


def student_cost_rate() -> tuple[Decimal | None, str]:
    """Read the optional local cost assumption. An unset rate stays unset."""
    raw = os.getenv("STUDENT_USD_PER_1000_INFERENCES", "").strip()
    if not raw:
        return None, STUDENT_COST_ASSUMPTION
    try:
        rate = Decimal(raw)
    except InvalidOperation as exc:
        raise BenchmarkError("STUDENT_USD_PER_1000_INFERENCES must be a decimal number.") from exc
    if rate < 0:
        raise BenchmarkError("STUDENT_USD_PER_1000_INFERENCES must be nonnegative.")
    assumption = (
        f"STUDENT_USD_PER_1000_INFERENCES={rate}. Each scored invoice counts as "
        "one local inference. This is not an API token price."
    )
    return rate, assumption


def student_inference_cost(inferences: int, rate_per_1000: Decimal | None) -> Decimal | None:
    """Price local inferences from an explicit rate. None means the rate is unset."""
    if rate_per_1000 is None:
        return None
    if inferences < 0:
        raise BenchmarkError("Inference count cannot be negative.")
    cost = Decimal(inferences) * rate_per_1000 / Decimal(1000)
    return cost.quantize(Decimal("0.000001"))


def latency_summary(values: Sequence[float]) -> dict[str, float | None]:
    """Average, median, and P95. The caller decides whether warmup is included."""
    latencies = [float(value) for value in values]
    return {
        "avg": (sum(latencies) / len(latencies)) if latencies else None,
        "median": percentile(latencies, 50),
        "p95": percentile(latencies, 95),
    }


def score_student_output(
    record: DatasetRecord,
    *,
    raw_text: str,
    latency_ms: float,
    model_id: str,
) -> BenchmarkRow:
    """Score one student string. Invalid JSON does not become a field error."""
    json_valid, schema_valid, fields, error = inspect_json_output(raw_text)
    prediction = five_fields_from_student(fields) if fields is not None else None
    return score_prediction(
        document_id=record.document_id,
        template_group=record.template_group,
        model_role="student",
        model_id=model_id,
        expected=five_fields_from_truth(record.ground_truth),
        prediction=prediction,
        json_valid=json_valid,
        schema_valid=schema_valid,
        validation_error=error,
        latency_ms=latency_ms,
        malformed_json=not json_valid,
    )


def score_teacher_prediction(record: DatasetRecord, prediction: TeacherPrediction) -> BenchmarkRow:
    """Score one teacher response. An API failure stays a row in the report."""
    schema_valid = bool(
        prediction.schema_valid and prediction.extraction is not None and prediction.api_error is None
    )
    extracted = prediction.extraction if schema_valid else None
    received_payload = prediction.api_error is None and (
        prediction.extraction is not None or prediction.validation_error is not None
    )
    return score_prediction(
        document_id=record.document_id,
        template_group=record.template_group,
        model_role="teacher",
        model_id=prediction.model,
        expected=five_fields_from_truth(record.ground_truth),
        prediction=five_fields_from_teacher(extracted) if extracted is not None else None,
        json_valid=schema_valid or received_payload,
        schema_valid=schema_valid,
        validation_error=prediction.validation_error,
        latency_ms=prediction.latency_ms,
        input_tokens=prediction.input_tokens,
        output_tokens=prediction.output_tokens,
        api_error=prediction.api_error,
        malformed_json=False,
    )


def load_teacher_cache(path: Path) -> list[TeacherPrediction]:
    """Load saved teacher rows from a JSON report or a prediction JSONL file."""
    path = Path(path)
    assert_test_file_closed(path, authorize_locked_test=False)
    if not path.is_file():
        raise BenchmarkError(f"Teacher cache not found: {path.name}")
    if path.suffix.lower() == ".jsonl":
        predictions = []
        for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
            if not line.strip():
                continue
            try:
                predictions.append(TeacherPrediction.model_validate_json(line))
            except ValueError as exc:
                raise BenchmarkError(f"Invalid teacher cache row in {path.name} at line {line_number}.") from exc
        return predictions
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise BenchmarkError(f"{path.name} is not a JSON teacher cache.") from exc
    rows = payload.get("results") if isinstance(payload, dict) else None
    if not isinstance(rows, list):
        raise BenchmarkError(f"{path.name} has no results list.")
    predictions: list[TeacherPrediction] = []
    for row in rows:
        if not isinstance(row, dict) or "document_id" not in row:
            raise BenchmarkError(f"{path.name} has a result without a document id.")
        extraction = row.get("prediction")
        predictions.append(
            TeacherPrediction(
                document_id=row["document_id"],
                model=str(row.get("model") or payload.get("teacher_model") or FROZEN_TEACHER_MODEL),
                extraction=None if extraction is None else ExtractedInvoice.model_validate(extraction),
                schema_valid=bool(row.get("schema_valid")),
                api_error=row.get("api_error"),
                validation_error=row.get("validation_error"),
                refusal=row.get("refusal"),
                latency_ms=float(row.get("latency_ms") or 0),
                input_tokens=int(row.get("input_tokens") or 0),
                output_tokens=int(row.get("output_tokens") or 0),
            )
        )
    return predictions


def locked_split_manifest(path: Path) -> dict[str, Any]:
    """Record ids, groups, and a file hash. Invoice text is not copied out."""
    path = Path(path)
    data = path.read_bytes()
    rows: list[dict[str, str]] = []
    for line in data.decode("utf-8").splitlines():
        if not line.strip():
            continue
        document_ids = _ID_RE.findall(line)
        groups = _GROUP_RE.findall(line)
        if not document_ids or not groups:
            raise BenchmarkError(f"{path.name} has a row without an id and template group.")
        rows.append({"document_id": document_ids[0], "template_group": groups[-1]})
    return {
        "split": "test" if path.name.lower() == "test.jsonl" else path.stem,
        "filename": path.name,
        "documents": len(rows),
        "bytes": len(data),
        "sha256": hashlib.sha256(data).hexdigest(),
        "rows": rows,
        "invoice_text_stored": False,
        "ground_truth_stored": False,
    }


def checkpoint_identity(model_dir: Path) -> dict[str, Any]:
    """Identify a frozen checkpoint without loading its weights."""
    model_dir = Path(model_dir)
    required = ("config.json", "model.safetensors", "tokenizer.json", "training_config.json")
    missing = [name for name in required if not (model_dir / name).is_file()]
    if missing:
        raise BenchmarkError(f"{model_dir} is missing {', '.join(missing)}.")
    weights = model_dir / "model.safetensors"
    config_text = (model_dir / "training_config.json").read_text(encoding="utf-8")
    return {
        "path": str(model_dir),
        "training_config_sha256": _sha256_text(config_text),
        "weights_bytes": weights.stat().st_size,
        "weights_sha256": _sha256_file(weights),
    }


def write_report_files(report: BenchmarkReport, json_path: Path) -> tuple[Path, Path]:
    """Write the JSON report and a CSV of per-invoice rows."""
    json_path = Path(json_path)
    json_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.write_text(report.model_dump_json(indent=2) + "\n", encoding="utf-8")
    csv_path = json_path.with_suffix(".csv")
    csv_path.write_text(_csv(report), encoding="utf-8", newline="\n")
    return json_path, csv_path


def _field_metrics(rows: Sequence[BenchmarkRow]) -> list[FieldErrorCount]:
    metrics: list[FieldErrorCount] = []
    for field in BENCHMARK_FIELDS:
        correct = sum(row.field_matches.get(field, False) for row in rows)
        metrics.append(
            FieldErrorCount(
                field=field,
                correct=correct,
                scored=len(rows),
                accuracy=_rate(correct, len(rows)),
                missing=sum(field in row.missing_fields for row in rows),
                hallucinated=sum(field in row.hallucinated_fields for row in rows),
                incorrect=sum(field in row.incorrect_fields for row in rows),
            )
        )
    return metrics


def _values_match(field: str, predicted: str, expected: str) -> bool:
    try:
        return field_values_match(field, predicted, expected)
    except (ValueError, TypeError, ArithmeticError):
        return False


def _blank(value: object) -> bool:
    if value is None:
        return True
    return str(value).strip() == ""


def _rate(numerator: int, denominator: int) -> float | None:
    if denominator == 0:
        return None
    return numerator / denominator


def _sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _csv(report: BenchmarkReport) -> str:
    header = [
        "document_id",
        "template_group",
        "model_role",
        "model_id",
        "json_valid",
        "schema_valid",
        "malformed_json",
        "exact_match",
        "latency_ms",
        "input_tokens",
        "output_tokens",
        "api_error",
        "missing_fields",
        "hallucinated_fields",
        "incorrect_fields",
    ]
    for field in BENCHMARK_FIELDS:
        header.extend([f"{field}_match", f"{field}_predicted", f"{field}_expected"])
    lines = [",".join(header)]
    for row in report.rows:
        values: list[str] = [
            row.document_id,
            row.template_group,
            row.model_role,
            row.model_id,
            str(row.json_valid).lower(),
            str(row.schema_valid).lower(),
            str(row.malformed_json).lower(),
            str(row.exact_match).lower(),
            f"{row.latency_ms:.3f}",
            str(row.input_tokens),
            str(row.output_tokens),
            row.api_error or "",
            "|".join(row.missing_fields),
            "|".join(row.hallucinated_fields),
            "|".join(row.incorrect_fields),
        ]
        for field in BENCHMARK_FIELDS:
            matched = row.field_matches.get(field)
            values.append("" if matched is None else str(matched).lower())
            values.append(row.prediction.get(field, ""))
            values.append(row.expected.get(field, ""))
        lines.append(",".join(_csv_cell(value) for value in values))
    return "\n".join(lines) + "\n"


def _csv_cell(value: str) -> str:
    if any(character in value for character in ",\"\n"):
        return '"' + value.replace('"', '""') + '"'
    return value


def teacher_schema_fields() -> tuple[str, ...]:
    """The teacher payload still carries document_type. The benchmark scores five fields."""
    return EXTRACTION_FIELDS
