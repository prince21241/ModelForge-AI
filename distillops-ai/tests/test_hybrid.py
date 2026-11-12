"""Hybrid expansion tests. No model download, no OpenAI call, no test split."""

import json
from pathlib import Path

import pytest

from scripts.evaluate_hybrid_teacher import main as teacher_main
from src.dataset import DATASET_SEED, generate_records
from src.hybrid_dataset import (
    HYBRID_COUNT,
    HybridDataError,
    generate_hybrid_invoices,
    validate_hybrid_invoice,
    write_hybrid_dataset,
)
from src.hybrid_training import (
    CONVERGENCE_RULE,
    ORIGINAL_TRAIN_IDS,
    VALIDATION_IDS,
    HybridTrainError,
    TrainingCheckpointScore,
    assemble_training_pairs,
    assert_fresh_checkpoint,
    classify_student_failure,
    convergence_reached,
    hybrid_train_config,
    require_paid_teacher_approval,
    save_teacher_predictions,
    score_teacher_subset,
    select_teacher_subset,
    select_training_checkpoint,
    teacher_mistake_counts,
)
from src.labeling import StudentFields, StudentTrainingExample
from src.prompts import STUDENT_TARGET_INSTRUCTION, build_student_messages, build_student_source
from src.schemas import ExtractedInvoice, TeacherPrediction
from src.student_data import STUDENT_OUTPUT_FIELDS, parse_student_json
from src.student_train import EncodedExample, collate_encoded


def _example(document_id: str) -> StudentTrainingExample:
    fields = StudentFields(
        vendor="Acme Supply",
        invoice_number="INV-100",
        date="2024-03-01",
        total="10.00",
        currency="USD",
    )
    text = f"Invoice body {document_id}"
    payload = json.dumps(fields.model_dump(), separators=(",", ":"))
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


def test_hybrid_invoices_are_forty_unique_and_isolated() -> None:
    first = generate_hybrid_invoices()
    second = generate_hybrid_invoices()
    assert [invoice.model_dump() for invoice in first] == [invoice.model_dump() for invoice in second]
    assert len(first) == HYBRID_COUNT == 40
    ids = [invoice.document_id for invoice in first]
    assert len(set(ids)) == 40
    original_ids = {record.document_id for record in generate_records(DATASET_SEED)}
    assert set(ids).isdisjoint(original_ids)
    assert set(ids).isdisjoint(VALIDATION_IDS)
    assert set(ids).isdisjoint(ORIGINAL_TRAIN_IDS)
    assert ids[0] == "inv-synth-101"
    assert ids[-1] == "inv-synth-140"


def test_hybrid_schema_missing_values_and_text_consistency() -> None:
    invoices = generate_hybrid_invoices()
    empty_vendor = empty_number = empty_date = 0
    month_name_dates = dotted = multiline = abbreviated = 0
    for invoice in invoices:
        validate_hybrid_invoice(invoice)
        payload = invoice.ground_truth.model_dump()
        assert set(payload) == set(STUDENT_OUTPUT_FIELDS)
        assert all(isinstance(payload[field], str) for field in STUDENT_OUTPUT_FIELDS)
        parsed, error = parse_student_json(
            json.dumps(payload, ensure_ascii=False, separators=(",", ":"))
        )
        assert error is None and parsed == invoice.ground_truth
        empty_vendor += invoice.ground_truth.vendor == ""
        empty_number += invoice.ground_truth.invoice_number == ""
        empty_date += invoice.ground_truth.date == ""
        if invoice.template_group == "dotted_notice":
            dotted += 1
            assert "Payee " in invoice.document_text or invoice.ground_truth.vendor == ""
        if invoice.template_group == "stacked_vendor" and invoice.ground_truth.vendor:
            head, _, tail = invoice.ground_truth.vendor.rpartition(" ")
            assert head in invoice.document_text.splitlines()
            assert tail in invoice.document_text.splitlines()
            assert "\\" not in invoice.ground_truth.vendor
            multiline += 1
        if "Inv #:" in invoice.document_text and "Dated:" in invoice.document_text:
            abbreviated += 1
        if invoice.ground_truth.date and any(
            token in invoice.document_text for token in ("Jan", "Feb", "Mar", "Apr", "Jun", "Jul")
        ):
            month_name_dates += 1
        assert "STATEMENT OF ACCOUNT" not in invoice.document_text
        assert "INVOICE memorandum" not in invoice.document_text
        source = build_student_source(invoice.document_text)
        assert source.startswith(STUDENT_TARGET_INSTRUCTION)
        assert "DOCUMENT START" in source and "<document>" not in source
    assert empty_vendor >= 1
    assert empty_number >= 1
    assert empty_date >= 1
    assert dotted == 10
    assert multiline >= 1
    assert abbreviated >= 1
    assert month_name_dates >= 1


def test_hybrid_files_stay_out_of_the_existing_splits(tmp_path: Path) -> None:
    invoices = generate_hybrid_invoices()
    written = write_hybrid_dataset(tmp_path / "hybrid", invoices)
    invoice_lines = (written / "invoices.jsonl").read_text(encoding="utf-8").splitlines()
    label_lines = (written / "ground_truth.jsonl").read_text(encoding="utf-8").splitlines()
    assert len(invoice_lines) == len(label_lines) == 40
    assert "ground_truth" not in invoice_lines[0]
    label = json.loads(label_lines[0])
    assert label["label_source"] == "deterministic_ground_truth"
    with pytest.raises(HybridDataError):
        write_hybrid_dataset(tmp_path / "test.jsonl", invoices)


def test_training_rows_keep_provenance_and_prompt(tmp_path: Path) -> None:
    original = [_example(document_id) for document_id in sorted(ORIGINAL_TRAIN_IDS)]
    hybrid = generate_hybrid_invoices()
    rows = assemble_training_pairs(original, hybrid)
    assert len(rows) == 60
    assert sum(row.provenance == "verified_teacher" for row in rows) == 20
    assert sum(row.provenance == "deterministic_ground_truth" for row in rows) == 40
    assert {row.pair.document_id for row in rows}.isdisjoint(VALIDATION_IDS)
    encoded = [
        EncodedExample(document_id=row.pair.document_id, input_ids=[3], label_ids=[32100, 9, 32101, 1])
        for row in rows[:1]
    ]
    encoded.append(
        EncodedExample(document_id=rows[1].pair.document_id, input_ids=[3, 4], label_ids=[32100, 1])
    )
    batch = collate_encoded(encoded, pad_token_id=0)
    assert batch["labels"][1].tolist()[-2:] == [-100, -100]
    manifest = tmp_path / "training_manifest.json"
    manifest.write_text(
        json.dumps([{"document_id": row.pair.document_id, "provenance": row.provenance} for row in rows]),
        encoding="utf-8",
    )
    assert "deterministic_ground_truth" in manifest.read_text(encoding="utf-8")


def test_teacher_subset_is_fixed_and_live_call_requires_approval(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    invoices = generate_hybrid_invoices()
    subset = select_teacher_subset(invoices)
    assert [invoice.document_id for invoice in subset] == [
        invoice.document_id for invoice in select_teacher_subset(invoices)
    ]
    assert len(subset) == 15
    assert {invoice.template_group for invoice in subset} == {
        "dotted_notice",
        "stacked_vendor",
        "reordered_slip",
        "whitespace_items",
    }
    labels_before = [invoice.ground_truth.model_dump() for invoice in subset]
    code = teacher_main(["--reports-dir", str(tmp_path)])
    assert code == 0
    pending = json.loads((tmp_path / "phase3e_teacher_eval.json").read_text(encoding="utf-8"))
    assert pending["status"] == "pending"
    assert pending["document_ids"] == [invoice.document_id for invoice in subset]
    assert [invoice.ground_truth.model_dump() for invoice in subset] == labels_before
    refused = teacher_main(["--live", "--reports-dir", str(tmp_path / "unused")])
    assert refused == 2
    assert "No request was sent" in capsys.readouterr().err
    with pytest.raises(HybridTrainError, match="approve-cost"):
        require_paid_teacher_approval(live=True, approve_cost=False)


def test_mocked_teacher_does_not_replace_deterministic_labels() -> None:
    subset = select_teacher_subset(generate_hybrid_invoices())
    before = [invoice.model_dump() for invoice in subset]

    class FakeClient:
        def __init__(self) -> None:
            self.settings = type("Settings", (), {"teacher_model": "fake-teacher"})()

        def extract(self, *, document_id: str, document_text: str) -> TeacherPrediction:
            invoice = next(item for item in subset if item.document_id == document_id)
            assert document_text == invoice.document_text
            vendor = invoice.ground_truth.vendor or "Invented Vendor"
            return TeacherPrediction(
                document_id=document_id,
                model="fake-teacher",
                schema_valid=True,
                extraction=ExtractedInvoice(
                    document_type="invoice",
                    vendor=vendor,
                    invoice_number=invoice.ground_truth.invoice_number or None,
                    invoice_date=invoice.ground_truth.date or None,
                    total=invoice.ground_truth.total or None,
                    currency=invoice.ground_truth.currency or None,
                ),
                latency_ms=12,
                input_tokens=30,
                output_tokens=8,
            )

    predictions, report = score_teacher_subset(subset, FakeClient())  # type: ignore[arg-type]
    assert [invoice.model_dump() for invoice in subset] == before
    assert report.documents == 15
    assert report.schema_valid_count == 15
    assert report.input_tokens == 450
    assert report.output_tokens == 120
    mistakes = teacher_mistake_counts(report)
    empty_vendor = sum(invoice.ground_truth.vendor == "" for invoice in subset)
    assert mistakes["hallucinated_values"] == empty_vendor
    assert report.exact_match_count == 15 - empty_vendor


def test_checkpoint_selection_uses_training_scores_only(tmp_path: Path) -> None:
    config = hybrid_train_config()
    assert config.model_name == "google/flan-t5-small"
    assert config.seed == 42
    assert config.learning_rate == 1e-4
    assert config.batch_size == 2
    assert config.epochs == 40
    assert config.max_target_length == 128
    assert "60/60" in CONVERGENCE_RULE
    assert "Validation scores are not" in CONVERGENCE_RULE
    weaker = TrainingCheckpointScore(
        epoch=5,
        mean_loss=0.1,
        json_valid=60,
        opening_brace=60,
        closing_brace=60,
        exact_match=40,
        checkpoint="epoch-05",
    )
    stronger = TrainingCheckpointScore(
        epoch=10,
        mean_loss=0.4,
        json_valid=55,
        opening_brace=60,
        closing_brace=55,
        exact_match=50,
        checkpoint="epoch-10",
    )
    assert select_training_checkpoint([weaker, stronger]) == "epoch-10"
    assert convergence_reached(
        stronger.model_copy(update={"json_valid": 60, "exact_match": 60}),
        60,
    )
    with pytest.raises(HybridTrainError):
        assert_fresh_checkpoint(Path("models/student/flan-t5-small-json-epoch3"))
    fresh = tmp_path / "flan-t5-small-hybrid60"
    assert_fresh_checkpoint(fresh)
    fresh.mkdir()
    with pytest.raises(HybridTrainError, match="not overwritten"):
        assert_fresh_checkpoint(fresh)
    with pytest.raises(HybridTrainError):
        save_teacher_predictions(tmp_path / "ground_truth.jsonl", [])


def test_failure_categories_cover_the_validation_patterns() -> None:
    expected = StudentFields(
        vendor="Cedar Pine Analytics",
        invoice_number="",
        date="2023-07-18",
        total="57298",
        currency="JPY",
    )
    predicted = StudentFields(
        vendor="Cedar Pine Analytics",
        invoice_number="INVICE",
        date="2023-08-23",
        total="527298",
        currency="JPY",
    )
    categories = classify_student_failure(
        parsed=predicted,
        expected=expected,
        invoice_text="Vendor Cedar Pine Analytics\nDated 18 Jul 2023\nTotal due 57298 JPY",
        opening_ok=True,
        closing_ok=True,
    )
    assert "hallucinated values" in categories
    assert "incorrect extraction" in categories
    assert "incorrect extraction" in classify_student_failure(
        parsed=StudentFields(
            vendor="A",
            invoice_number="HYB-1",
            date="2024-01-02",
            total="USD",
            currency="USD",
        ),
        expected=StudentFields(
            vendor="A",
            invoice_number="HYB-1",
            date="2024-01-02",
            total="10.00",
            currency="USD",
        ),
        invoice_text="A HYB-1 2024-01-02 10.00 USD",
        opening_ok=True,
        closing_ok=True,
    )
    assert classify_student_failure(
        parsed=None,
        expected=expected,
        invoice_text="invoice",
        opening_ok=True,
        closing_ok=False,
    ) == ["malformed JSON", "formatting error"]
