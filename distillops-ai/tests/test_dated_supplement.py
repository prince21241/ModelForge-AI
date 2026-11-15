"""Dated-line supplement tests. No model download, no OpenAI call, no test split."""

import json
from datetime import date
from pathlib import Path

import pytest

from src.dataset import DATASET_SEED, generate_records
from src.dated_supplement import (
    DATED_COUNT,
    generate_dated_invoices,
    write_dated_dataset,
)
from src.hybrid_dataset import HybridDataError, generate_hybrid_invoices
from src.hybrid_training import (
    ORIGINAL_TRAIN_IDS,
    VALIDATION_IDS,
    append_dated_pairs,
    assemble_training_pairs,
)
from src.invoice_templates import format_date
from src.labeling import StudentFields, StudentTrainingExample
from src.prompts import build_student_messages, build_student_source
from src.student_data import teacher_target_json

_DOTS = "." * 8
_BANNED_NUMBERS = {"INV-Y547DG", "INV-GRNN6J", "INV-RE6GDA", "INV-98P3DL"}
_BANNED_DATES = {"2023-07-18", "2023-05-05", "2025-05-14", "2023-07-11"}
_BANNED_TOTALS = {"57298", "7650.87", "2935.75", "1758.89", "37.60"}


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


def test_dated_invoices_are_twelve_unique_and_isolated() -> None:
    first = generate_dated_invoices()
    second = generate_dated_invoices()
    assert [invoice.model_dump() for invoice in first] == [invoice.model_dump() for invoice in second]
    assert len(first) == DATED_COUNT == 12
    ids = [invoice.document_id for invoice in first]
    assert ids == [f"inv-synth-{number:03d}" for number in range(201, 213)]
    corpus = generate_records(DATASET_SEED)
    hybrid = generate_hybrid_invoices()
    assert set(ids).isdisjoint(record.document_id for record in corpus)
    assert set(ids).isdisjoint(invoice.document_id for invoice in hybrid)
    assert set(ids).isdisjoint(VALIDATION_IDS)
    assert set(ids).isdisjoint(ORIGINAL_TRAIN_IDS)
    texts = {invoice.document_text for invoice in first}
    assert texts.isdisjoint(record.document_text for record in corpus)
    assert texts.isdisjoint(invoice.document_text for invoice in hybrid)


def test_dated_lines_use_abbreviated_months_and_dotted_totals() -> None:
    invoices = generate_dated_invoices()
    empty_vendor = empty_number = empty_date = 0
    for invoice in invoices:
        fields = invoice.ground_truth
        assert "STATEMENT OF ACCOUNT" not in invoice.document_text
        assert "\\" not in teacher_target_json(fields)
        assert fields.invoice_number not in _BANNED_NUMBERS
        assert fields.date not in _BANNED_DATES
        assert fields.total not in _BANNED_TOTALS
        assert f"Total due {_DOTS} {fields.total} {fields.currency}" in invoice.document_text
        source = build_student_source(invoice.document_text)
        assert "DOCUMENT START" in source
        assert "<document>" not in source
        empty_vendor += fields.vendor == ""
        empty_number += fields.invoice_number == ""
        empty_date += fields.date == ""
        if fields.date:
            abbr = format_date(date.fromisoformat(fields.date), "abbr")
            assert f"Dated {_DOTS} {abbr}" in invoice.document_text
        else:
            assert "Dated" not in invoice.document_text
    assert empty_vendor == 2
    assert empty_number == 2
    assert empty_date == 1


def test_dated_rows_extend_the_sixty_without_replacing_them() -> None:
    original = [_example(document_id) for document_id in sorted(ORIGINAL_TRAIN_IDS)]
    base = assemble_training_pairs(original, generate_hybrid_invoices())
    extended = append_dated_pairs(base, generate_dated_invoices())
    assert len(extended) == 72
    assert [row.pair.document_id for row in extended[:60]] == [row.pair.document_id for row in base]
    assert sum(row.provenance == "verified_teacher" for row in extended) == 20
    assert sum(row.provenance == "deterministic_ground_truth" for row in extended) == 52
    assert extended[-1].pair.document_id == "inv-synth-212"


def test_dated_writer_refuses_the_locked_split_name(tmp_path: Path) -> None:
    invoices = generate_dated_invoices()
    with pytest.raises(HybridDataError):
        write_dated_dataset(tmp_path / "test.jsonl", invoices)
    written = write_dated_dataset(tmp_path / "dated", invoices)
    assert (written / "invoices.jsonl").exists()
    assert (written / "ground_truth.jsonl").exists()
