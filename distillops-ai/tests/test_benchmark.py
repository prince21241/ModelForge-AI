"""Phase 4 benchmark tests. No model download, no OpenAI call, no locked test read."""

import json
from decimal import Decimal
from pathlib import Path

import pytest

from scripts.run_benchmark import main
from src.benchmark import (
    FROZEN_STUDENT_DIR,
    BenchmarkReport,
    LockedSplitError,
    classify_field,
    inspect_json_output,
    latency_summary,
    load_teacher_cache,
    locked_split_manifest,
    resolve_split,
    score_student_output,
    score_teacher_prediction,
    student_inference_cost,
)
from src.benchmark_mlflow import log_benchmark
from src.config import Settings
from src.dataset import write_jsonl
from src.evaluation_pricing import Pricing
from src.schemas import DatasetRecord, ExtractedInvoice, TeacherPrediction
from src.student_eval import StudentGeneration
from src.student_infer import generate_fields


def _truth(**overrides: object) -> ExtractedInvoice:
    payload: dict[str, object] = {
        "document_type": "invoice",
        "vendor": "Acme Supply",
        "invoice_number": "INV-100",
        "invoice_date": "2024-03-01",
        "total": "10.00",
        "currency": "USD",
    }
    payload.update(overrides)
    return ExtractedInvoice.model_validate(payload)


def _record(document_id: str = "inv-synth-901", text: str = "SECRET INVOICE TEXT", **overrides: object) -> DatasetRecord:
    return DatasetRecord(
        document_id=document_id,
        document_text=text,
        ground_truth=_truth(**overrides),
        template_group="classic_header",
    )


class _Cpu:
    type = "cpu"


def _student_json(**overrides: str) -> str:
    payload = {
        "vendor": "Acme Supply",
        "invoice_number": "INV-100",
        "date": "2024-03-01",
        "total": "10.00",
        "currency": "USD",
    }
    payload.update(overrides)
    return json.dumps(payload, separators=(",", ":"))


def test_json_and_schema_are_separate() -> None:
    json_valid, schema_valid, fields, error = inspect_json_output("not json")
    assert json_valid is False
    assert schema_valid is False
    assert fields is None
    assert error == "Output is not valid JSON."
    json_valid, schema_valid, fields, error = inspect_json_output("[1, 2]")
    assert json_valid is True
    assert schema_valid is False
    assert fields is None
    json_valid, schema_valid, fields, _error = inspect_json_output('{"vendor":"Acme"}')
    assert json_valid is True
    assert schema_valid is False
    json_valid, schema_valid, fields, error = inspect_json_output(_student_json())
    assert json_valid is True
    assert schema_valid is True
    assert fields is not None
    assert error is None


def test_wrong_date_and_amount_are_not_rewritten() -> None:
    row = score_student_output(
        _record(),
        raw_text=_student_json(date="18 Jul 2023", total="USD"),
        latency_ms=3,
        model_id="student-test",
    )
    assert row.prediction["date"] == "18 Jul 2023"
    assert row.prediction["total"] == "USD"
    assert row.incorrect_fields == ["date", "total"]
    assert row.missing_fields == []
    assert row.hallucinated_fields == []
    assert row.exact_match is False
    assert classify_field("date", "18 Jul 2023", "2023-07-18") == "incorrect"


def test_missing_and_hallucinated_values_are_separate_from_malformed_json() -> None:
    missing = score_student_output(
        _record(),
        raw_text=_student_json(vendor=""),
        latency_ms=1,
        model_id="student-test",
    )
    assert missing.missing_fields == ["vendor"]
    assert missing.hallucinated_fields == []
    invented = score_student_output(
        _record(vendor=None),
        raw_text=_student_json(vendor="Invented Desk"),
        latency_ms=1,
        model_id="student-test",
    )
    assert invented.hallucinated_fields == ["vendor"]
    assert invented.missing_fields == []
    broken = score_student_output(
        _record(),
        raw_text="vendor Acme",
        latency_ms=1,
        model_id="student-test",
    )
    assert broken.malformed_json is True
    assert broken.schema_valid is False
    assert broken.missing_fields == []
    assert broken.hallucinated_fields == []
    assert broken.incorrect_fields == []
    assert broken.exact_match is False


def test_numeric_total_and_empty_string_still_match() -> None:
    row = score_student_output(
        _record(total="10"),
        raw_text=_student_json(total="10.00"),
        latency_ms=1,
        model_id="student-test",
    )
    assert row.field_matches["total"] is True
    blank = score_student_output(
        _record(invoice_number=None),
        raw_text=_student_json(invoice_number=""),
        latency_ms=1,
        model_id="student-test",
    )
    assert blank.field_matches["invoice_number"] is True
    assert blank.exact_match is True


def test_teacher_api_failure_is_not_a_field_error() -> None:
    prediction = TeacherPrediction(
        document_id="inv-synth-901",
        model="gpt-4o-mini",
        extraction=None,
        schema_valid=False,
        api_error="timeout",
        latency_ms=12,
        input_tokens=0,
        output_tokens=0,
    )
    row = score_teacher_prediction(_record(), prediction)
    assert row.api_error == "timeout"
    assert row.malformed_json is False
    assert row.schema_valid is False
    assert row.missing_fields == []
    assert row.input_tokens == 0


def test_cost_and_latency_statistics() -> None:
    pricing = Pricing(
        input_usd_per_million_tokens=Decimal("0.15"),
        output_usd_per_million_tokens=Decimal("0.60"),
    )
    assert pricing.estimate_usd(1_000_000, 1_000_000) == Decimal("0.750000")
    assert student_inference_cost(5, None) is None
    assert student_inference_cost(5, Decimal("2")) == Decimal("0.010000")
    summary = latency_summary([10, 20, 30, 40])
    assert summary["avg"] == 25
    assert summary["median"] == 25
    assert summary["p95"] == pytest.approx(38.5)


def test_locked_test_split_is_rejected_before_it_is_opened(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    test_file = tmp_path / "test.jsonl"
    test_file.write_text("DO_NOT_READ_TEST_SPLIT\n", encoding="utf-8")

    def fail_open(*_args: object, **_kwargs: object) -> None:
        raise AssertionError("load_split was called")

    monkeypatch.setattr("scripts.run_benchmark.load_split", fail_open)
    with pytest.raises(LockedSplitError):
        resolve_split("test", authorize_locked_test=False)
    code = main(
        [
            "--model",
            "both",
            "--split",
            "test",
            "--splits-dir",
            str(tmp_path),
            "--reports-dir",
            str(tmp_path / "reports"),
            "--no-mlflow",
        ]
    )
    assert code == 2
    assert test_file.read_text(encoding="utf-8") == "DO_NOT_READ_TEST_SPLIT\n"


def test_authorization_flag_can_score_a_temporary_file_without_keeping_its_text(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    secret = "SECRET INVOICE TEXT"
    write_jsonl(tmp_path / "test.jsonl", [_record(text=secret)])

    def fake_load(_path: Path, _device: object) -> tuple[object, object]:
        return object(), object()

    def fake_generate(*_args: object, **kwargs: object) -> StudentGeneration:
        return StudentGeneration(
            document_id=str(kwargs["document_id"]),
            raw_text=_student_json(),
            json_valid=True,
            fields=None,
            latency_ms=1,
        )

    monkeypatch.setattr("scripts.run_benchmark.load_student", fake_load)
    monkeypatch.setattr("scripts.run_benchmark.generate_fields", fake_generate)
    monkeypatch.setattr("scripts.run_benchmark.checkpoint_identity", lambda _path: {"path": "frozen"})
    monkeypatch.setattr("scripts.run_benchmark.select_device", lambda _name: _Cpu())
    code = main(
        [
            "--model",
            "student",
            "--split",
            "test",
            "--authorize-locked-test",
            "--splits-dir",
            str(tmp_path),
            "--reports-dir",
            str(tmp_path / "reports"),
            "--student-dir",
            str(tmp_path / "student"),
            "--no-warmup",
            "--no-mlflow",
        ]
    )
    assert code == 0
    report = (tmp_path / "reports" / "phase4_student_test.json").read_text(encoding="utf-8")
    assert secret not in report
    assert "inv-synth-901" in report


def test_dry_run_does_not_load_the_student_or_call_the_teacher(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    write_jsonl(tmp_path / "val.jsonl", [_record()])

    def fail_load(*_args: object, **_kwargs: object) -> None:
        raise AssertionError("student load was called")

    monkeypatch.setattr("scripts.run_benchmark.load_student", fail_load)
    code = main(
        [
            "--model",
            "both",
            "--split",
            "validation",
            "--dry-run",
            "--splits-dir",
            str(tmp_path),
            "--reports-dir",
            str(tmp_path / "reports"),
            "--no-mlflow",
        ]
    )
    assert code == 0
    plan = json.loads((tmp_path / "reports" / "phase4_val_dry_run.json").read_text(encoding="utf-8"))
    assert plan["inference_executed"] is False
    assert plan["teacher_api_calls"] == 0
    assert plan["document_ids"] == ["inv-synth-901"]


def test_cached_teacher_response_is_scored_without_a_live_call(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    write_jsonl(tmp_path / "val.jsonl", [_record(), _record("inv-synth-902")])
    cache = {
        "teacher_model": "gpt-4o-mini",
        "results": [
            {
                "document_id": "inv-synth-901",
                "model": "gpt-4o-mini-2024-07-18",
                "schema_valid": True,
                "api_error": None,
                "validation_error": None,
                "latency_ms": 15,
                "input_tokens": 40,
                "output_tokens": 12,
                "prediction": _truth().model_dump(mode="json"),
            }
        ],
    }
    cache_path = tmp_path / "cache.json"
    cache_path.write_text(json.dumps(cache), encoding="utf-8")
    monkeypatch.setattr("scripts.run_benchmark.load_settings", lambda: Settings())
    monkeypatch.setattr("scripts.run_benchmark.configured_pricing", lambda _settings: None)

    def fail_client(*_args: object, **_kwargs: object) -> None:
        raise AssertionError("TeacherClient was constructed")

    monkeypatch.setattr("scripts.run_benchmark.TeacherClient", fail_client)
    code = main(
        [
            "--model",
            "teacher",
            "--split",
            "validation",
            "--splits-dir",
            str(tmp_path),
            "--reports-dir",
            str(tmp_path / "reports"),
            "--teacher-cache",
            str(cache_path),
            "--no-mlflow",
        ]
    )
    assert code == 0
    report = BenchmarkReport.model_validate_json(
        (tmp_path / "reports" / "phase4_teacher_val.json").read_text(encoding="utf-8")
    )
    assert report.status == "cache_incomplete"
    assert report.api_calls == 0
    assert report.documents_scored == 1
    assert report.unscored_document_ids == ["inv-synth-902"]
    assert report.exact_match_count == 1
    assert report.end_to_end_exact_match_accuracy is None
    assert report.input_tokens == 40
    assert report.output_tokens == 12
    assert load_teacher_cache(cache_path)[0].document_id == "inv-synth-901"


def test_live_teacher_without_cost_approval_sends_no_request(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    write_jsonl(tmp_path / "val.jsonl", [_record()])

    def fail_client(*_args: object, **_kwargs: object) -> None:
        raise AssertionError("TeacherClient was constructed")

    monkeypatch.setattr("scripts.run_benchmark.TeacherClient", fail_client)
    code = main(
        [
            "--model",
            "teacher",
            "--split",
            "validation",
            "--live",
            "--splits-dir",
            str(tmp_path),
            "--reports-dir",
            str(tmp_path / "reports"),
            "--no-teacher-cache",
            "--no-mlflow",
        ]
    )
    assert code == 2


def test_manifest_keeps_ids_and_drops_invoice_text(tmp_path: Path) -> None:
    secret = "SECRET MEMO BODY"
    write_jsonl(tmp_path / "test.jsonl", [_record(text=secret, document_id="inv-synth-011")])
    manifest = locked_split_manifest(tmp_path / "test.jsonl")
    encoded = json.dumps(manifest)
    assert secret not in encoded
    assert manifest["invoice_text_stored"] is False
    assert manifest["rows"] == [{"document_id": "inv-synth-011", "template_group": "classic_header"}]
    assert len(manifest["sha256"]) == 64


def test_mlflow_logs_the_phase4_experiment(tmp_path: Path) -> None:
    pytest.importorskip("mlflow")
    report = score_student_output(
        _record(),
        raw_text=_student_json(),
        latency_ms=4,
        model_id="student-test",
    )
    from src.benchmark import build_report

    aggregate = build_report(
        [report],
        status="completed",
        model_role="student",
        model_id="student-test",
        split="val",
        documents_in_split=1,
        unscored_document_ids=[],
        checkpoint="frozen",
        decoding={"do_sample": False, "num_beams": 1, "max_new_tokens": 128, "max_source_length": 512},
        warmup_ms=1.5,
        warmup_excluded=True,
        api_calls=0,
        pricing=None,
        cost_assumption="unset",
    )
    artifact = tmp_path / "report.json"
    artifact.write_text(aggregate.model_dump_json(), encoding="utf-8")
    database = (tmp_path / "mlflow.db").resolve().as_posix()
    tracking_uri = f"sqlite:///{database}"
    run_id = log_benchmark(aggregate, tracking_uri=tracking_uri, artifact_path=artifact)
    assert run_id
    import mlflow

    mlflow.set_tracking_uri(tracking_uri)
    experiment = mlflow.get_experiment_by_name("DistillOps-Phase4-Benchmark")
    assert experiment is not None
    stored = mlflow.get_run(run_id)
    assert stored.data.params["model_id"] == "student-test"
    assert stored.data.metrics["exact_match_count"] == 1
    assert "OPENAI_API_KEY" not in stored.data.params


def test_frozen_checkpoint_files_are_present() -> None:
    if not FROZEN_STUDENT_DIR.is_dir():
        pytest.skip("frozen checkpoint is not on this machine")
    for name in ("config.json", "model.safetensors", "tokenizer.json", "training_config.json"):
        assert (FROZEN_STUDENT_DIR / name).is_file()


def test_student_cli_loads_the_requested_checkpoint(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    write_jsonl(tmp_path / "val.jsonl", [_record()])
    seen: dict[str, Path] = {}

    def fake_load(path: Path, _device: object) -> tuple[object, object]:
        seen["path"] = Path(path)
        return object(), object()

    def fake_generate(*_args: object, **kwargs: object) -> StudentGeneration:
        return StudentGeneration(
            document_id=str(kwargs["document_id"]),
            raw_text=_student_json(),
            json_valid=True,
            latency_ms=2,
        )

    monkeypatch.setattr("scripts.run_benchmark.load_student", fake_load)
    monkeypatch.setattr("scripts.run_benchmark.generate_fields", fake_generate)
    monkeypatch.setattr("scripts.run_benchmark.checkpoint_identity", lambda path: {"path": str(path)})
    monkeypatch.setattr("scripts.run_benchmark.select_device", lambda _name: _Cpu())
    code = main(
        [
            "--model",
            "student",
            "--split",
            "validation",
            "--splits-dir",
            str(tmp_path),
            "--reports-dir",
            str(tmp_path / "reports"),
            "--student-dir",
            str(FROZEN_STUDENT_DIR),
            "--no-warmup",
            "--no-mlflow",
        ]
    )
    assert code == 0
    assert seen["path"] == FROZEN_STUDENT_DIR
    assert generate_fields.__name__ == "generate_fields"
