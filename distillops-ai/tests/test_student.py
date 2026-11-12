"""Offline student tests. No model download and no OpenAI call."""

import json
from pathlib import Path

import pytest
import torch

from scripts.evaluate_student import main as evaluate_main
from scripts.train_student import main as train_main
from src.dataset import write_jsonl
from src.labeling import StudentFields, StudentTrainingExample
from src.prompts import STUDENT_TARGET_INSTRUCTION, build_student_messages, build_student_source
from src.schemas import DatasetRecord, ExtractedInvoice
from src.student_data import (
    StudentDataError,
    assert_tokenizer_keeps_text,
    load_training_examples,
    pairs_from_examples,
    parse_student_json,
    require_fits,
    require_validation_split,
    token_ids,
)
from src.student_eval import StudentGeneration, score_generations
from src.student_infer import select_device
from src.student_train import (
    DEFAULT_STUDENT_MODEL,
    EncodedExample,
    StudentTrainConfig,
    collate_encoded,
)


class FaithfulTokenizer:
    pad_token_id = 0

    def __call__(
        self,
        text,
        *,
        truncation=False,
        padding=False,
        add_special_tokens=True,
        return_attention_mask=False,
    ):
        if truncation:
            raise AssertionError("truncation was requested")
        ids = [min(len(word), 9) for word in text.split()] or [1]
        if add_special_tokens:
            ids = ids + [2]
        return {"input_ids": ids}


class TruncatingTokenizer(FaithfulTokenizer):
    def __call__(self, text, **kwargs):
        encoded = super().__call__(text, **kwargs)
        encoded["input_ids"] = encoded["input_ids"][:3]
        return encoded


def _fields(**overrides) -> StudentFields:
    payload = {
        "vendor": "Acme Supply",
        "invoice_number": "INV-100",
        "date": "2024-03-01",
        "total": "10.00",
        "currency": "USD",
    }
    payload.update(overrides)
    return StudentFields.model_validate(payload)


def _example(document_id: str = "inv-synth-001", text: str = "Vendor: Acme Supply", fields: StudentFields | None = None):
    fields = fields or _fields()
    payload = json.dumps(fields.model_dump(), ensure_ascii=False, separators=(",", ":"))
    return StudentTrainingExample.model_validate(
        {
            "document_id": document_id,
            "teacher_model": "gpt-4o-mini-2024-07-18",
            "verification_status": "verified",
            "invoice_text": text,
            "teacher_output": fields.model_dump(),
            "messages": build_student_messages(text, payload),
        }
    )


def _record(
    document_id: str,
    *,
    vendor: str | None = "Acme Supply",
    invoice_number: str | None = "INV-100",
    invoice_date: str | None = "2024-03-01",
    total: str | None = "10.00",
    currency: str | None = "USD",
    layout: str = "account_statement",
) -> DatasetRecord:
    return DatasetRecord(
        document_id=document_id,
        document_text=f"Statement for {document_id}",
        template_group=layout,
        ground_truth=ExtractedInvoice(
            document_type="invoice",
            vendor=vendor,
            invoice_number=invoice_number,
            invoice_date=invoice_date,
            total=total,
            currency=currency,
        ),
    )


def _generation(document_id: str, fields: StudentFields | None, *, raw: str | None = None, error: str | None = None):
    if fields is None:
        return StudentGeneration(
            document_id=document_id,
            raw_text=raw or "not json",
            json_valid=False,
            fields=None,
            validation_error=error or "Output is not valid JSON.",
            latency_ms=12.5,
        )
    raw_text = raw or json.dumps(fields.model_dump(), separators=(",", ":"))
    return StudentGeneration(
        document_id=document_id,
        raw_text=raw_text,
        json_valid=True,
        fields=fields,
        validation_error=None,
        latency_ms=20.0,
    )


def test_load_training_examples_preserves_text_and_empty_fields(tmp_path: Path):
    text = "Invoice body with no issue date"
    example = _example(
        text=text,
        fields=_fields(date="", currency=""),
    )
    path = tmp_path / "student_train.jsonl"
    write_jsonl(path, [example])
    loaded = load_training_examples(path)
    assert loaded[0].invoice_text == text
    assert loaded[0].teacher_output.date == ""
    assert loaded[0].teacher_output.currency == ""
    pair = pairs_from_examples(loaded)[0]
    assert text in pair.source_text
    assert pair.target_text == example.messages[2].content
    assert '"date":""' in pair.target_text
    assert "ground_truth" not in pair.source_text
    assert pair.source_text.startswith(STUDENT_TARGET_INSTRUCTION)


def test_tokenizer_refuses_silent_truncation():
    text = "alpha beta gamma delta epsilon"
    tokenizer = FaithfulTokenizer()
    assert_tokenizer_keeps_text(tokenizer, text)
    ids = token_ids(tokenizer, text)
    assert len(ids) == 6
    with pytest.raises(StudentDataError, match="not truncated"):
        require_fits(ids, 2, "inv-synth-001", "input")
    with pytest.raises(StudentDataError, match="truncat"):
        assert_tokenizer_keeps_text(TruncatingTokenizer(), text)


def test_locked_test_file_is_rejected_before_reading(tmp_path: Path):
    path = tmp_path / "test.jsonl"
    path.write_text("do-not-read\n", encoding="utf-8")
    with pytest.raises(StudentDataError, match="locked test"):
        load_training_examples(path)


def test_test_document_id_and_eval_split_are_rejected():
    example = _example(document_id="test-001")
    with pytest.raises(StudentDataError, match="locked test"):
        pairs_from_examples([example])
    with pytest.raises(StudentDataError, match="locked test"):
        require_validation_split("test")
    assert require_validation_split("validation") == "val"


def test_parse_keeps_empty_strings_and_rejects_extra_keys():
    parsed, error = parse_student_json(
        '{"vendor":"","invoice_number":"INV-1","date":"","total":"","currency":"USD"}'
    )
    assert error is None
    assert parsed is not None
    assert parsed.vendor == ""
    assert parsed.date == ""
    assert parsed.currency == "USD"
    missing, missing_error = parse_student_json('{"vendor":"Acme"}')
    assert missing is None
    assert missing_error is not None
    extra, extra_error = parse_student_json(
        '{"vendor":"","invoice_number":"","date":"","total":"","currency":"","document_type":"invoice"}'
    )
    assert extra is None
    assert extra_error is not None


def test_score_uses_five_fields_and_not_document_type():
    records = [
        _record("inv-val-1"),
        _record("inv-val-2", vendor="Other Vendor"),
        _record("inv-val-3", vendor=None, invoice_number=None, invoice_date=None, total=None, currency=None),
    ]
    generations = [
        _generation("inv-val-1", _fields()),
        _generation("inv-val-2", _fields()),
        _generation("inv-val-3", _fields(vendor="", invoice_number="", date="", total="", currency="")),
    ]
    report = score_generations(records, generations, split="val", model_dir="models/unit")
    assert report.document_type_scored is False
    assert [metric.field for metric in report.field_metrics] == [
        "vendor",
        "invoice_number",
        "date",
        "total",
        "currency",
    ]
    assert "document_type" not in report.scored_fields
    assert report.exact_match_count == 2
    assert report.json_valid_count == 3
    vendor = report.field_metrics[0]
    assert vendor.correct == 2
    assert vendor.missing_correct == 1
    assert "account_statement" in report.layout_note


def test_invalid_json_is_not_an_exact_match():
    records = [_record("inv-val-1"), _record("inv-val-2")]
    generations = [
        _generation("inv-val-1", _fields()),
        _generation("inv-val-2", None, raw="still thinking", error="Output is not valid JSON."),
    ]
    report = score_generations(records, generations, split="val", model_dir="models/unit")
    assert report.json_valid_count == 1
    assert report.exact_match_count == 1
    assert report.end_to_end_exact_match_accuracy == 0.5
    assert report.results[1].exact_match is False
    assert report.results[1].field_matches == {}


def test_collate_pads_labels_without_dropping_tokens():
    examples = [
        EncodedExample(document_id="a", input_ids=[3, 4, 5], label_ids=[7]),
        EncodedExample(document_id="b", input_ids=[8], label_ids=[9, 6]),
    ]
    batch = collate_encoded(examples, pad_token_id=0)
    assert batch["input_ids"].shape == (2, 3)
    assert batch["labels"].tolist()[0] == [7, -100]
    assert batch["labels"].tolist()[1] == [9, 6]
    assert batch["attention_mask"].tolist()[1] == [1, 0, 0]
    assert torch.equal(batch["input_ids"][0], torch.tensor([3, 4, 5]))


def test_train_config_defaults_to_flan_t5_small():
    config = StudentTrainConfig()
    assert config.model_name == DEFAULT_STUDENT_MODEL == "google/flan-t5-small"
    assert config.seed == 42
    assert config.epochs == 1
    assert config.max_source_length == 512
    saved = StudentTrainConfig.model_validate_json(config.model_dump_json())
    assert saved == config


def test_cpu_device_can_be_selected_without_a_download():
    assert select_device("cpu").type == "cpu"


def test_train_cli_rejects_test_file_without_loading_a_model(tmp_path: Path, monkeypatch):
    path = tmp_path / "test.jsonl"
    path.write_text("do-not-read\n", encoding="utf-8")

    def fail_download(*args, **kwargs):
        raise AssertionError("model weights were requested")

    monkeypatch.setattr("src.student_train.load_tokenizer", fail_download)
    monkeypatch.setattr("src.student_train.load_seq2seq", fail_download)
    code = train_main(["--labels", str(path), "--output-dir", str(tmp_path / "out")])
    assert code == 2


def test_eval_cli_rejects_test_split_without_opening_a_file(monkeypatch):
    def fail_open(*args, **kwargs):
        raise AssertionError("a dataset or model file was opened")

    monkeypatch.setattr("scripts.evaluate_student.load_split", fail_open)
    monkeypatch.setattr("scripts.evaluate_student.load_student", fail_open)
    code = evaluate_main(["--split", "test", "--model-dir", "models/does-not-exist"])
    assert code == 2


def test_student_source_uses_plain_markers_and_keeps_the_invoice():
    text = "Unique invoice sentence 4821"
    source = build_student_source(text)
    assert text in source
    assert "DOCUMENT START\n" + text + "\nDOCUMENT END" in source
    assert "<document>" not in source
    assert "</document>" not in source
    assert '"invoice_number"' not in source
    assert "ground_truth" not in source
    stored = build_student_messages(text, '{"vendor":"Acme"}')
    assert "<document>" in stored[1]["content"]
    assert stored[2]["content"] == '{"vendor":"Acme"}'


def test_training_refuses_the_original_checkpoint(tmp_path: Path):
    from src.student_tokenizer import (
        StudentTokenizerError,
        require_base_flan_model,
        require_new_output_dir,
    )

    require_base_flan_model("google/flan-t5-small")
    with pytest.raises(StudentTokenizerError, match="one-epoch checkpoint"):
        require_base_flan_model(str(tmp_path / "flan-t5-small-epoch1"))
    with pytest.raises(StudentTokenizerError, match="overwrite"):
        require_new_output_dir(tmp_path / "flan-t5-small-epoch1")
