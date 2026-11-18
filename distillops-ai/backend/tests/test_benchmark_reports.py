"""Saved Phase 4 reports are displayed and never rewritten."""

from __future__ import annotations

import json
from pathlib import Path

from fastapi.testclient import TestClient

from backend.app.main import app
from backend.app.services.benchmark_reports import load_saved_benchmarks

MINIMAL = {
    "status": "completed",
    "model_role": "student",
    "model_id": "frozen-student",
    "split": "test",
    "documents_in_split": 1,
    "documents_scored": 1,
    "unscored_document_ids": [],
    "json_valid_count": 1,
    "schema_valid_count": 1,
    "malformed_json_count": 0,
    "exact_match_count": 1,
    "exact_match_accuracy": 1.0,
    "end_to_end_exact_match_accuracy": 1.0,
    "field_metrics": [
        {
            "field": "vendor",
            "correct": 1,
            "scored": 1,
            "accuracy": 1.0,
            "missing": 0,
            "hallucinated": 0,
            "incorrect": 0,
        }
    ],
    "missing_field_count": 0,
    "hallucinated_field_count": 0,
    "incorrect_field_count": 0,
    "latency_ms_avg": 10.0,
    "latency_ms_median": 10.0,
    "latency_ms_p95": 10.0,
    "warmup_excluded": True,
    "input_tokens": 0,
    "output_tokens": 0,
    "api_calls": 0,
    "api_failure_count": 0,
    "pricing_configured": False,
    "estimated_cost_usd": None,
    "cost_assumption": "unset",
    "rows": [
        {
            "document_id": "sample-1",
            "template_group": "memo",
            "exact_match": True,
            "incorrect_fields": [],
            "missing_fields": [],
            "hallucinated_fields": [],
            "latency_ms": 10.0,
            "prediction": {"vendor": "Northwind Desk"},
            "expected": {"vendor": "Northwind Desk"},
            "document_text": "must not be returned",
        }
    ],
    "invoice_text": "must not be returned",
}


def test_missing_reports_are_marked_unavailable(tmp_path: Path) -> None:
    library = load_saved_benchmarks(tmp_path)
    assert library.rescored is False
    assert library.source == "phase4-saved-reports"
    assert len(library.reports) == 4
    assert all(item.available is False for item in library.reports)


def test_a_saved_report_is_read_without_invoice_text(tmp_path: Path) -> None:
    payload = dict(MINIMAL)
    (tmp_path / "phase4_student_test.json").write_text(json.dumps(payload), encoding="utf-8")
    library = load_saved_benchmarks(tmp_path)
    student = next(item for item in library.reports if item.filename == "phase4_student_test.json")
    assert student.available is True
    assert student.report is not None
    assert student.report.exact_match_count == 1
    dumped = student.model_dump()
    assert "document_text" not in json.dumps(dumped)
    assert "invoice_text" not in json.dumps(dumped)
    assert dumped["report"]["rows"][0]["prediction"]["vendor"] == "Northwind Desk"


def test_corrupt_report_does_not_echo_a_parser_error(tmp_path: Path) -> None:
    (tmp_path / "phase4_teacher_test.json").write_text("{", encoding="utf-8")
    library = load_saved_benchmarks(tmp_path)
    teacher = next(item for item in library.reports if item.filename == "phase4_teacher_test.json")
    assert teacher.available is False
    assert teacher.detail == "The saved report could not be read."


def test_api_reads_the_local_reports_without_changing_them() -> None:
    reports = Path(__file__).resolve().parents[2] / "reports"
    paths = [reports / name for _, _, name in (
        ("teacher", "test", "phase4_teacher_test.json"),
        ("student", "test", "phase4_student_test.json"),
    )]
    if not all(path.is_file() for path in paths):
        return
    before = {path: (path.stat().st_mtime_ns, path.stat().st_size) for path in paths}
    with TestClient(app) as client:
        response = client.get("/api/benchmarks")
    assert response.status_code == 200
    body = response.json()
    assert body["rescored"] is False
    student = next(item for item in body["reports"] if item["filename"] == "phase4_student_test.json")
    teacher = next(item for item in body["reports"] if item["filename"] == "phase4_teacher_test.json")
    assert student["report"]["exact_match_count"] == 3
    assert student["report"]["documents_scored"] == 5
    assert teacher["report"]["exact_match_count"] == 5
    assert teacher["report"]["end_to_end_exact_match_accuracy"] == 1.0
    validation = next(item for item in body["reports"] if item["filename"] == "phase4_teacher_val.json")
    if validation["available"]:
        assert validation["report"]["end_to_end_exact_match_accuracy"] is None
    assert "document_text" not in response.text
    after = {path: (path.stat().st_mtime_ns, path.stat().st_size) for path in paths}
    assert before == after
