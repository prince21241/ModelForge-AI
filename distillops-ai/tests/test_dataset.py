"""Dataset generation, JSONL loading, and template-group splits."""

import pytest

from scripts.generate_dataset import main as generate_main
from src.dataset import (
    DatasetLoadError,
    DatasetSplitError,
    assert_no_group_leakage,
    build_dataset,
    generate_records,
    load_split,
    omitted_fields,
    read_jsonl,
    split_records,
    write_dataset,
)
from src.schemas import DatasetRecord, ExtractedInvoice


def _record(document_id: str, group: str) -> DatasetRecord:
    return DatasetRecord(
        document_id=document_id,
        document_text="INVOICE\nSynthetic training document.",
        ground_truth=ExtractedInvoice(document_type="invoice"),
        template_group=group,
    )


def test_generation_is_reproducible_and_changes_with_the_seed() -> None:
    first = [record.model_dump_json() for record in generate_records(42)]
    second = [record.model_dump_json() for record in generate_records(42)]
    other = [record.model_dump_json() for record in generate_records(7)]
    assert first == second
    assert first != other
    assert len(first) == 30


def test_omitted_fields_match_the_fixed_schedule() -> None:
    records = generate_records(42)
    optional = ("vendor", "invoice_number", "invoice_date", "total", "currency")
    seen_omitted = set()
    for record in records:
        index = int(record.document_id.rsplit("-", 1)[1])
        omitted = omitted_fields(index)
        seen_omitted.update(omitted)
        for field in optional:
            value = getattr(record.ground_truth, field)
            if field in omitted:
                assert value is None
            else:
                assert value is not None
    assert seen_omitted == set(optional)


def test_splits_have_no_template_leakage() -> None:
    bundle = build_dataset(42)
    assert [len(bundle.splits[name]) for name in ("train", "val", "test")] == [20, 5, 5]
    group_sets = [
        {record.template_group for record in bundle.splits[name]} for name in ("train", "val", "test")
    ]
    assert group_sets[0].isdisjoint(group_sets[1])
    assert group_sets[0].isdisjoint(group_sets[2])
    assert group_sets[1].isdisjoint(group_sets[2])
    assert set().union(*group_sets) == set(bundle.assignment)
    located: dict[str, set[str]] = {}
    for split_name, records in bundle.splits.items():
        for record in records:
            located.setdefault(record.template_group, set()).add(split_name)
    assert all(len(places) == 1 for places in located.values())
    identifiers = [record.document_id for record in bundle.records]
    assert len(identifiers) == len(set(identifiers))


def test_group_larger_than_every_split_is_rejected() -> None:
    records = [_record(f"doc-{index}", "oversized") for index in range(6)]
    with pytest.raises(DatasetSplitError):
        split_records(records, seed=1, targets={"train": 5, "val": 5, "test": 5})


def test_shared_template_group_is_leakage() -> None:
    splits = {
        "train": [_record("doc-1", "classic_header")],
        "val": [_record("doc-2", "classic_header")],
    }
    with pytest.raises(DatasetSplitError):
        assert_no_group_leakage(splits)


def test_jsonl_roundtrip_and_error_does_not_echo_the_line(tmp_path) -> None:
    bundle = build_dataset(42)
    write_dataset(tmp_path, bundle)
    loaded = {
        name: load_split(tmp_path / "splits", name) for name in ("train", "val", "test")
    }
    assert [record.model_dump_json() for record in loaded["val"]] == [
        record.model_dump_json() for record in bundle.splits["val"]
    ]
    summary = (tmp_path / "processed" / "split_summary.csv").read_text(encoding="utf-8")
    assert summary.count("\n") == 31
    manifest = (tmp_path / "raw" / "synthetic_manifest.json").read_text(encoding="utf-8")
    assert '"contains_real_customer_data": false' in manifest
    assert "document_text" not in manifest
    assert "INV-" not in manifest
    raw_split = (tmp_path / "splits" / "train.jsonl").read_bytes()
    assert b"\r" not in raw_split

    invalid = tmp_path / "broken.jsonl"
    invalid.write_text(
        '{"document_text": "SECRET_TEXT_SHOULD_NOT_LEAK"}\n',
        encoding="utf-8",
    )
    with pytest.raises(DatasetLoadError) as caught:
        read_jsonl(invalid, DatasetRecord)
    assert "SECRET_TEXT_SHOULD_NOT_LEAK" not in str(caught.value)


def test_blank_lines_are_skipped(tmp_path) -> None:
    path = tmp_path / "empty.jsonl"
    path.write_text("\n\n", encoding="utf-8")
    assert read_jsonl(path, DatasetRecord) == []


def test_generate_command_writes_splits(tmp_path, capsys) -> None:
    exit_code = generate_main(["--seed", "42", "--data-dir", str(tmp_path)])
    captured = capsys.readouterr()
    assert exit_code == 0
    assert "Generated 30 synthetic invoices" in captured.out
    assert (tmp_path / "splits" / "val.jsonl").is_file()
    assert "INV-" not in captured.out
    assert "Synthetic training document" not in captured.out
