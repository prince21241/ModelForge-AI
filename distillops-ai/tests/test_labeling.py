"""Phase 2 labeling tests. The teacher client is scripted, so no API key is used."""

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from scripts.generate_teacher_labels import main as label_main
from src.config import Settings
from src.dataset import write_jsonl
from src.labeling import (
    HeldOutTrainingError,
    StudentTrainingExample,
    generate_teacher_labels,
    require_train_split,
    student_fields_from_extraction,
)
from src.schemas import DatasetRecord, ExtractedInvoice, TeacherInvoicePayload
from src.teacher import TeacherConfigError


SECRET = "sk-live-key-should-not-leak"
GROUND_TRUTH_VENDOR = "Ground Truth Vendor Only"


class ScriptedCompletions:
    def __init__(self, events: list[object]) -> None:
        self.events = list(events)
        self.calls: list[dict[str, object]] = []

    def parse(self, **kwargs: object) -> object:
        self.calls.append(kwargs)
        event = self.events.pop(0)
        if isinstance(event, Exception):
            raise event
        return event


class ScriptedClient:
    def __init__(self, events: list[object]) -> None:
        self.completions = ScriptedCompletions(events)
        self.chat = SimpleNamespace(completions=self.completions)


class RetryableStatus(Exception):
    def __init__(self, status_code: int) -> None:
        super().__init__("transient")
        self.status_code = status_code


def _settings() -> Settings:
    return Settings.model_validate(
        {
            "openai_api_key": None,
            "teacher_model": "unit-test-model",
            "teacher_temperature": 0,
            "timeout_seconds": 5,
            "max_retries": 2,
        }
    )


def _invoice(**overrides: object) -> ExtractedInvoice:
    payload: dict[str, object] = {"document_type": "invoice"}
    payload.update(overrides)
    return ExtractedInvoice.model_validate(payload)


def _record(
    document_id: str,
    *,
    text: str = "INVOICE\nFrom: Cedar Pine Analytics\nTotal Due: 10.00 USD",
    vendor: str | None = "Cedar Pine Analytics",
    invoice_number: str | None = None,
    invoice_date: str | None = None,
    total: str | None = "10.00",
    currency: str | None = "USD",
) -> DatasetRecord:
    return DatasetRecord(
        document_id=document_id,
        document_text=text,
        ground_truth=_invoice(
            vendor=vendor,
            invoice_number=invoice_number,
            invoice_date=invoice_date,
            total=total,
            currency=currency,
        ),
        template_group="classic_header",
    )


def _completion(payload: TeacherInvoicePayload, *, prompt_tokens: int = 8, completion_tokens: int = 4):
    message = SimpleNamespace(parsed=payload, refusal=None)
    return SimpleNamespace(
        model="response-model",
        choices=[SimpleNamespace(message=message)],
        usage={"prompt_tokens": prompt_tokens, "completion_tokens": completion_tokens},
    )


def _payload_for(record: DatasetRecord, **overrides: object) -> TeacherInvoicePayload:
    truth = record.ground_truth
    payload: dict[str, object] = {
        "document_type": "invoice",
        "vendor": truth.vendor,
        "invoice_number": truth.invoice_number,
        "invoice_date": None if truth.invoice_date is None else truth.invoice_date.isoformat(),
        "total": None if truth.total is None else format(truth.total, "f"),
        "currency": truth.currency,
    }
    payload.update(overrides)
    return TeacherInvoicePayload.model_validate(payload)


def _run(tmp_path, records, events, **kwargs):
    client = ScriptedClient(events)
    report = generate_teacher_labels(
        records,
        _settings(),
        checkpoint_path=tmp_path / "checkpoint.jsonl",
        export_path=tmp_path / "student_train.jsonl",
        live=True,
        client=client,
        sleep_fn=kwargs.pop("sleep_fn", lambda _delay: None),
        **kwargs,
    )
    return client, report


def test_train_split_is_the_only_accepted_target() -> None:
    require_train_split("train")
    with pytest.raises(HeldOutTrainingError):
        require_train_split("test")
    with pytest.raises(HeldOutTrainingError):
        require_train_split("val")


def test_missing_fields_export_as_empty_strings_and_match_null_labels(tmp_path) -> None:
    record = _record(
        "inv-synth-001",
        vendor=None,
        total=None,
        currency=None,
        text="INVOICE\nNo optional fields are written.",
    )
    client, report = _run(tmp_path, [record], [_completion(_payload_for(record))])
    assert report.verified_count == 1
    assert report.exact_match_accuracy == 1
    example = StudentTrainingExample.model_validate_json(
        (tmp_path / "student_train.jsonl").read_text(encoding="utf-8").strip()
    )
    assert example.teacher_output.vendor == ""
    assert example.teacher_output.date == ""
    assert example.teacher_output.total == ""
    assert example.teacher_output.currency == ""
    assert example.verification_status == "verified"
    assert example.teacher_model == "response-model"
    assert example.messages[-1].role == "assistant"
    assert "ground_truth" not in example.messages[-1].content
    assert client.completions.calls[0]["model"] == "unit-test-model"
    fields = student_fields_from_extraction(record.ground_truth)
    assert fields.total == ""


def test_successful_batch_writes_jsonl_and_skips_duplicates(tmp_path) -> None:
    records = [_record("inv-synth-001"), _record("inv-synth-002", total="12.50")]
    events = [_completion(_payload_for(record)) for record in records]
    client, report = _run(tmp_path, records, events)
    assert report.api_calls_this_run == 2
    assert report.verified_count == 2
    assert report.export_count == 2
    lines = (tmp_path / "student_train.jsonl").read_text(encoding="utf-8").splitlines()
    assert len(lines) == 2
    first = json.loads(lines[0])
    assert first["document_id"] == "inv-synth-001"
    assert first["teacher_output"]["currency"] == "USD"
    assert "messages" in first

    again, second = _run(tmp_path, records, [])
    assert again.completions.calls == []
    assert second.api_calls_this_run == 0
    assert second.skipped_existing == 2
    assert second.verified_count == 2


def test_checkpoint_resume_requests_only_the_remaining_invoice(tmp_path) -> None:
    records = [
        _record("inv-synth-011", text="INVOICE\nFIRST_INVOICE\nTotal Due: 10.00 USD"),
        _record("inv-synth-012", text="INVOICE\nSECOND_INVOICE\nTotal Due: 8.00 USD", total="8.00"),
    ]
    first_client, first = _run(
        tmp_path,
        records,
        [_completion(_payload_for(records[0])), _completion(_payload_for(records[1]))],
        limit=1,
    )
    assert first.api_calls_this_run == 1
    assert first.unprocessed_count == 1
    assert len(first_client.completions.calls) == 1

    second_client, second = _run(
        tmp_path,
        records,
        [_completion(_payload_for(records[1]))],
    )
    assert len(second_client.completions.calls) == 1
    sent = json.dumps(second_client.completions.calls[0]["messages"])
    assert "SECOND_INVOICE" in sent
    assert "FIRST_INVOICE" not in sent
    assert "ground_truth" not in sent
    assert second.verified_count == 2
    assert second.unprocessed_count == 0


def test_api_failure_is_retried_then_rejected_without_exporting(tmp_path) -> None:
    record = _record("inv-synth-021")
    sleeps: list[float] = []
    client, report = _run(
        tmp_path,
        [record],
        [RetryableStatus(429), RetryableStatus(500), RetryableStatus(503)],
        sleep_fn=sleeps.append,
    )
    assert sleeps == [0.5, 1.0]
    assert len(client.completions.calls) == 3
    assert report.api_failure_count == 1
    assert report.verified_count == 0
    assert report.rejected[0].reason == "api_failure"
    assert (tmp_path / "student_train.jsonl").read_text(encoding="utf-8") == ""
    checkpoint = (tmp_path / "checkpoint.jsonl").read_text(encoding="utf-8")
    assert SECRET not in checkpoint
    assert record.document_text not in checkpoint


def test_invalid_teacher_payload_is_rejected(tmp_path) -> None:
    record = _record("inv-synth-022")
    bad = _payload_for(record).model_copy(update={"total": "not-a-number"})
    _client, report = _run(tmp_path, [record], [_completion(bad)])
    assert report.successful_responses == 0
    assert report.rejected[0].reason == "schema_invalid"
    assert report.verified_count == 0
    assert "not-a-number" not in (report.rejected[0].validation_error or "")


def test_mismatch_is_not_replaced_with_ground_truth(tmp_path) -> None:
    record = _record(
        "inv-synth-023",
        text="INVOICE\nFrom: Printed Vendor\nTotal Due: 10.00 USD",
        vendor=GROUND_TRUTH_VENDOR,
    )
    wrong = _payload_for(record, vendor="Printed Vendor")
    _client, report = _run(tmp_path, [record], [_completion(wrong)])
    assert report.verified_count == 0
    assert report.rejected[0].reason == "mismatch"
    assert "vendor" in report.rejected[0].mismatched_fields
    exported = (tmp_path / "student_train.jsonl").read_text(encoding="utf-8")
    checkpoint = (tmp_path / "checkpoint.jsonl").read_text(encoding="utf-8")
    assert GROUND_TRUTH_VENDOR not in exported
    assert GROUND_TRUTH_VENDOR not in checkpoint
    assert "Printed Vendor" in checkpoint


def test_dry_run_and_cli_do_not_call_the_api(tmp_path, monkeypatch: pytest.MonkeyPatch) -> None:
    record = _record("inv-synth-031")
    report = generate_teacher_labels(
        [record],
        _settings(),
        checkpoint_path=tmp_path / "checkpoint.jsonl",
        export_path=tmp_path / "student_train.jsonl",
        live=False,
        client=ScriptedClient([_completion(_payload_for(record))]),
    )
    assert report.live is False
    assert report.api_calls_this_run == 0
    assert not (tmp_path / "student_train.jsonl").exists()

    splits = tmp_path / "splits"
    write_jsonl(splits / "train.jsonl", [record])
    (splits / "test.jsonl").write_text("DO_NOT_READ_TEST_SPLIT\n", encoding="utf-8")
    (splits / "val.jsonl").write_text("DO_NOT_READ_VALIDATION\n", encoding="utf-8")
    monkeypatch.setenv("TEACHER_MODEL", "unit-test-model")
    monkeypatch.setenv("OPENAI_API_KEY", SECRET)
    monkeypatch.setattr(
        "src.teacher.build_openai_client",
        lambda _settings: (_ for _ in ()).throw(AssertionError("live client must not be built")),
    )
    code = label_main(
        [
            "--splits-dir",
            str(splits),
            "--output-dir",
            str(tmp_path / "out"),
            "--reports-dir",
            str(tmp_path / "reports"),
            "--env-file",
            str(tmp_path / "missing.env"),
        ]
    )
    assert code == 0
    assert not (tmp_path / "out" / "student_train.jsonl").exists()
    blocked = label_main(
        [
            "--split",
            "test",
            "--live",
            "--splits-dir",
            str(splits),
            "--reports-dir",
            str(tmp_path / "reports"),
            "--env-file",
            str(tmp_path / "missing.env"),
        ]
    )
    assert blocked == 2
    assert (splits / "test.jsonl").read_text(encoding="utf-8") == "DO_NOT_READ_TEST_SPLIT\n"


def test_live_without_credentials_sends_nothing(tmp_path) -> None:
    record = _record("inv-synth-032")
    with pytest.raises(TeacherConfigError) as caught:
        generate_teacher_labels(
            [record],
            _settings(),
            checkpoint_path=tmp_path / "checkpoint.jsonl",
            export_path=tmp_path / "student_train.jsonl",
            live=True,
        )
    assert "No teacher request was sent" in str(caught.value)
    assert not (tmp_path / "checkpoint.jsonl").exists()


def test_test_document_id_is_refused() -> None:
    record = _record("test-held-out")
    with pytest.raises(HeldOutTrainingError):
        generate_teacher_labels(
            [record],
            _settings(),
            checkpoint_path=Path("unused-checkpoint.jsonl"),
            export_path=Path("unused-export.jsonl"),
            live=False,
        )


def test_concurrency_above_one_is_rejected(tmp_path) -> None:
    with pytest.raises(ValueError, match="one invoice at a time"):
        generate_teacher_labels(
            [_record("inv-synth-033")],
            _settings(),
            checkpoint_path=tmp_path / "checkpoint.jsonl",
            export_path=tmp_path / "student_train.jsonl",
            live=False,
            concurrency=4,
        )
