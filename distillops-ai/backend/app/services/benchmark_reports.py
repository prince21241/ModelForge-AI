"""Read the saved Phase 4 reports. This module never scores invoices or writes reports."""

from __future__ import annotations

import json
from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field, ValidationError

from backend.app.core.config import DISTILLOPS_ROOT

SAVED_REPORTS: tuple[tuple[str, str, str], ...] = (
    ("teacher", "test", "phase4_teacher_test.json"),
    ("student", "test", "phase4_student_test.json"),
    ("teacher", "val", "phase4_teacher_val.json"),
    ("student", "val", "phase4_student_val.json"),
)


class SavedFieldMetric(BaseModel):
    model_config = ConfigDict(extra="ignore")

    field: str
    correct: int = Field(ge=0)
    scored: int = Field(ge=0)
    accuracy: float | None
    missing: int = Field(ge=0)
    hallucinated: int = Field(ge=0)
    incorrect: int = Field(ge=0)


class SavedBenchmarkRow(BaseModel):
    model_config = ConfigDict(extra="ignore")

    document_id: str
    template_group: str
    exact_match: bool
    incorrect_fields: list[str]
    missing_fields: list[str]
    hallucinated_fields: list[str]
    latency_ms: float = Field(ge=0)
    prediction: dict[str, str]
    expected: dict[str, str]


class SavedReportBody(BaseModel):
    model_config = ConfigDict(extra="ignore")

    status: str
    model_role: str
    model_id: str
    split: str
    documents_in_split: int = Field(ge=0)
    documents_scored: int = Field(ge=0)
    unscored_document_ids: list[str]
    json_valid_count: int = Field(ge=0)
    schema_valid_count: int = Field(ge=0)
    malformed_json_count: int = Field(ge=0)
    exact_match_count: int = Field(ge=0)
    exact_match_accuracy: float | None
    end_to_end_exact_match_accuracy: float | None
    field_metrics: list[SavedFieldMetric]
    missing_field_count: int = Field(ge=0)
    hallucinated_field_count: int = Field(ge=0)
    incorrect_field_count: int = Field(ge=0)
    latency_ms_avg: float | None
    latency_ms_median: float | None
    latency_ms_p95: float | None
    warmup_excluded: bool
    input_tokens: int = Field(ge=0)
    output_tokens: int = Field(ge=0)
    api_calls: int = Field(ge=0)
    api_failure_count: int = Field(ge=0)
    pricing_configured: bool
    estimated_cost_usd: float | None
    cost_assumption: str
    rows: list[SavedBenchmarkRow]


class SavedBenchmark(BaseModel):
    model_config = ConfigDict(extra="forbid")

    available: bool
    filename: str
    model_role: str
    split: str
    detail: str | None = None
    report: SavedReportBody | None = None


class BenchmarkLibrary(BaseModel):
    model_config = ConfigDict(extra="forbid")

    source: str
    rescored: bool
    reports: list[SavedBenchmark]


def load_saved_benchmarks(reports_dir: Path | None = None) -> BenchmarkLibrary:
    """Open the known report files in read-only mode."""
    directory = reports_dir or (DISTILLOPS_ROOT / "reports")
    loaded: list[SavedBenchmark] = []
    for role, split, filename in SAVED_REPORTS:
        path = directory / filename
        loaded.append(_read_one(path, role, split, filename))
    return BenchmarkLibrary(source="phase4-saved-reports", rescored=False, reports=loaded)


def _read_one(path: Path, role: str, split: str, filename: str) -> SavedBenchmark:
    if not path.is_file():
        return SavedBenchmark(
            available=False,
            filename=filename,
            model_role=role,
            split=split,
            detail="This saved report is not on this machine.",
        )
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
        body = SavedReportBody.model_validate(payload)
    except (OSError, json.JSONDecodeError, ValidationError):
        return SavedBenchmark(
            available=False,
            filename=filename,
            model_role=role,
            split=split,
            detail="The saved report could not be read.",
        )
    return SavedBenchmark(available=True, filename=filename, model_role=role, split=split, report=body)
