"""Reproducible synthetic invoices and leakage-free splits.

Every vendor, invoice number, date, and amount is generated. The writers never
read an external document, and the manifest records that the corpus is fictional.
"""

from __future__ import annotations

import json
import random
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import date, timedelta
from decimal import Decimal
from pathlib import Path
from typing import TypeVar

import pandas as pd
from pydantic import BaseModel, ValidationError

from src.invoice_templates import (
    MONTH_ABBREVIATIONS,
    MONTH_NAMES,
    TEMPLATE_GROUPS,
    InvoiceContent,
    amount_in_text,
    date_forms,
    format_amount,
    render_invoice,
)
from src.schemas import DatasetRecord, ExtractedInvoice

DATASET_SEED = 42
DOCUMENTS_PER_TEMPLATE = 5
SPLIT_TARGETS: dict[str, int] = {"train": 20, "val": 5, "test": 5}
SPLIT_FILENAMES: dict[str, str] = {
    "train": "train.jsonl",
    "val": "val.jsonl",
    "test": "test.jsonl",
}

VENDORS: tuple[str, ...] = (
    "Brightline Office Supply",
    "Cedar Pine Analytics",
    "Northstar Freight Desk",
    "Lumen Printing Cooperative",
    "Harbor Field Catering",
    "Copperline Industrial",
    "Vale Moss Consulting",
    "Red Lantern Logistics",
    "Kindred Paper Works",
    "Summit Range Utilities",
    "Blue Finch Bookkeeping",
    "Orchard Street Linens",
    "Glasswater Studio",
    "Pebble Road Hardware",
    "Finch Transit Desk",
)
GENERATOR_CURRENCIES: tuple[str, ...] = ("USD", "EUR", "GBP", "CAD", "AUD", "JPY")
_INVOICE_ALPHABET = "ABCDEFGHJKLMNPQRSTUVWXYZ23456789"
_ISO_DATE_IN_TEXT = re.compile(r"\b\d{4}-\d{2}-\d{2}\b")
_INVOICE_NUMBER_IN_TEXT = re.compile(r"\bINV-[A-Z0-9]+\b")
_DECIMAL_IN_TEXT = re.compile(r"\b\d+\.\d{2}\b")

TModel = TypeVar("TModel", bound=BaseModel)


class DatasetError(ValueError):
    """Base error for dataset generation and loading."""


class DatasetSplitError(DatasetError):
    """Raised when groups cannot be placed without leakage or missed targets."""


class DatasetConsistencyError(DatasetError):
    """Raised when rendered text does not support its own labels."""


class DatasetLoadError(DatasetError):
    """Raised when a JSONL file is missing or contains an invalid record."""


@dataclass(frozen=True)
class DatasetBundle:
    """Generated records, their splits, and the group-to-split map."""

    seed: int
    records: list[DatasetRecord]
    splits: dict[str, list[DatasetRecord]]
    assignment: dict[str, str]


def omitted_fields(index: int) -> frozenset[str]:
    """Fields intentionally left off document ``index`` (1-based).

    The schedule is deterministic and independent of the random stream, so a
    change to sampled names does not silently remove the null cases.
    """
    if index < 1:
        raise ValueError("document index starts at 1")
    omitted: set[str] = set()
    if index % 11 == 0:
        omitted.add("vendor")
    if index % 7 == 0:
        omitted.add("invoice_number")
    if index % 5 == 0:
        omitted.add("invoice_date")
    if index % 13 == 0:
        omitted.add("total")
    if index % 9 == 0:
        omitted.add("currency")
    return frozenset(omitted)


def generate_records(seed: int = DATASET_SEED) -> list[DatasetRecord]:
    """Build the fixed-size synthetic corpus for ``seed``."""
    rng = random.Random(seed)
    records: list[DatasetRecord] = []
    index = 0
    for group in TEMPLATE_GROUPS:
        for _ in range(DOCUMENTS_PER_TEMPLATE):
            index += 1
            content = _sample_content(rng, index)
            text = render_invoice(group, content, rng)
            record = DatasetRecord(
                document_id=f"inv-synth-{index:03d}",
                document_text=text,
                ground_truth=ExtractedInvoice(
                    document_type="invoice",
                    vendor=content.vendor,
                    invoice_number=content.invoice_number,
                    invoice_date=content.invoice_date,
                    total=content.total,
                    currency=content.currency,
                ),
                template_group=group,
            )
            validate_document_consistency(record)
            records.append(record)
    expected = len(TEMPLATE_GROUPS) * DOCUMENTS_PER_TEMPLATE
    if len(records) != expected:
        raise DatasetError(f"Expected {expected} documents, built {len(records)}.")
    return records


def assign_template_groups(
    group_sizes: Mapping[str, int],
    *,
    seed: int,
    targets: Mapping[str, int],
) -> dict[str, str]:
    """Assign every template group to exactly one split.

    A group is never divided. Among splits that still have room, the one with
    the most remaining capacity wins. Ties break in train, val, test order.
    """
    remaining = {name: int(capacity) for name, capacity in targets.items()}
    order = sorted(group_sizes)
    random.Random(seed).shuffle(order)
    tie_break = {name: position for position, name in enumerate(("train", "val", "test"))}
    assignment: dict[str, str] = {}
    for group in order:
        size = group_sizes[group]
        candidates = [split for split, capacity in remaining.items() if capacity >= size]
        if not candidates:
            raise DatasetSplitError(
                f"Template group {group!r} has {size} documents and does not fit "
                "in any remaining split without crossing split boundaries."
            )
        candidates.sort(key=lambda split: (-remaining[split], tie_break.get(split, 99), split))
        chosen = candidates[0]
        assignment[group] = chosen
        remaining[chosen] -= size
    leftover = {split: capacity for split, capacity in remaining.items() if capacity != 0}
    if leftover:
        raise DatasetSplitError(
            "Split targets were not filled exactly. "
            f"Remaining capacity: {leftover}."
        )
    return assignment


def split_records(
    records: Sequence[DatasetRecord],
    *,
    seed: int = DATASET_SEED,
    targets: Mapping[str, int] | None = None,
) -> tuple[dict[str, list[DatasetRecord]], dict[str, str]]:
    """Split records by whole template group."""
    split_targets = dict(targets or SPLIT_TARGETS)
    grouped: dict[str, list[DatasetRecord]] = {}
    for record in records:
        grouped.setdefault(record.template_group, []).append(record)
    assignment = assign_template_groups(
        {name: len(items) for name, items in grouped.items()},
        seed=seed,
        targets=split_targets,
    )
    splits: dict[str, list[DatasetRecord]] = {name: [] for name in split_targets}
    for record in records:
        splits[assignment[record.template_group]].append(record)
    assert_no_group_leakage(splits)
    return splits, assignment


def build_dataset(seed: int = DATASET_SEED) -> DatasetBundle:
    records = generate_records(seed)
    splits, assignment = split_records(records, seed=seed)
    return DatasetBundle(seed=seed, records=records, splits=splits, assignment=assignment)


def write_dataset(root: Path, bundle: DatasetBundle) -> None:
    """Write JSONL splits, a full export, a CSV summary, and a manifest."""
    root = Path(root)
    assert_no_group_leakage(bundle.splits)
    write_jsonl(root / "processed" / "all.jsonl", bundle.records)
    for split_name, filename in SPLIT_FILENAMES.items():
        write_jsonl(root / "splits" / filename, bundle.splits[split_name])
    summary = build_summary_frame(bundle.splits)
    processed = root / "processed"
    processed.mkdir(parents=True, exist_ok=True)
    summary.to_csv(processed / "split_summary.csv", index=False)
    manifest = {
        "synthetic": True,
        "contains_real_customer_data": False,
        "seed": bundle.seed,
        "documents": len(bundle.records),
        "splits": {name: len(rows) for name, rows in bundle.splits.items()},
        "template_groups": {
            group: {"split": split, "documents": DOCUMENTS_PER_TEMPLATE}
            for group, split in sorted(bundle.assignment.items())
        },
        "note": (
            "Fictional vendors, invoice numbers, and amounts generated for "
            "DistillOps AI. Not real customer financial documents."
        ),
    }
    raw_dir = root / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)
    (raw_dir / "synthetic_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def load_split(splits_dir: Path, split: str) -> list[DatasetRecord]:
    """Load one split. ``split`` must be ``train``, ``val``, or ``test``."""
    try:
        filename = SPLIT_FILENAMES[split]
    except KeyError as exc:
        raise DatasetLoadError("Unknown split. Expected train, val, or test.") from exc
    return read_jsonl(Path(splits_dir) / filename, DatasetRecord)


def write_jsonl(path: Path, records: Sequence[BaseModel]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for record in records:
            handle.write(record.model_dump_json() + "\n")


def read_jsonl(path: Path, model: type[TModel]) -> list[TModel]:
    """Load JSONL and validate every record. Errors do not echo line contents."""
    path = Path(path)
    if not path.is_file():
        raise DatasetLoadError(f"Dataset file not found: {path.name}")
    records: list[TModel] = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            stripped = line.strip()
            if not stripped:
                continue
            try:
                records.append(model.model_validate_json(stripped))
            except ValidationError as exc:
                raise DatasetLoadError(
                    f"Invalid record in {path.name} at line {line_number}."
                ) from exc
    return records


def build_summary_frame(splits: Mapping[str, Sequence[DatasetRecord]]) -> pd.DataFrame:
    """One row per document, used to audit split sizes and field coverage."""
    rows: list[dict[str, object]] = []
    for split_name, records in splits.items():
        for record in records:
            truth = record.ground_truth
            rows.append(
                {
                    "split": split_name,
                    "document_id": record.document_id,
                    "template_group": record.template_group,
                    "vendor_present": truth.vendor is not None,
                    "invoice_number_present": truth.invoice_number is not None,
                    "invoice_date_present": truth.invoice_date is not None,
                    "total_present": truth.total is not None,
                    "currency_present": truth.currency is not None,
                }
            )
    frame = pd.DataFrame(rows)
    if frame.empty:
        return frame
    return frame.sort_values(["split", "document_id"]).reset_index(drop=True)


def assert_no_group_leakage(splits: Mapping[str, Sequence[DatasetRecord]]) -> None:
    seen: dict[str, str] = {}
    for split_name, records in splits.items():
        for record in records:
            previous = seen.get(record.template_group)
            if previous is not None and previous != split_name:
                raise DatasetSplitError(
                    f"Template group {record.template_group!r} appears in "
                    f"{previous!r} and {split_name!r}."
                )
            seen[record.template_group] = split_name


def validate_document_consistency(record: DatasetRecord) -> None:
    """Fail generation when the text hides, invents, or contradicts a label."""
    text = record.document_text
    truth = record.ground_truth
    if "INVOICE" not in text.upper():
        raise DatasetConsistencyError(f"{record.document_id} does not read as an invoice.")
    _check_vendor(text, truth.vendor)
    _check_invoice_number(text, truth.invoice_number)
    _check_date(text, truth.invoice_date)
    _check_total(text, truth.total, truth.currency)
    _check_currency(text, truth.currency)


def _sample_content(rng: random.Random, index: int) -> InvoiceContent:
    omitted = omitted_fields(index)
    vendor = None if "vendor" in omitted else rng.choice(VENDORS)
    invoice_number = None if "invoice_number" in omitted else _invoice_number(rng)
    invoice_date = None if "invoice_date" in omitted else _random_date(rng)
    currency = None if "currency" in omitted else rng.choice(GENERATOR_CURRENCIES)
    total = None if "total" in omitted else _random_total(rng, currency)
    return InvoiceContent(
        vendor=vendor,
        invoice_number=invoice_number,
        invoice_date=invoice_date,
        total=total,
        currency=currency,
    )


def _invoice_number(rng: random.Random) -> str:
    suffix = "".join(rng.choice(_INVOICE_ALPHABET) for _ in range(6))
    return f"INV-{suffix}"


def _random_date(rng: random.Random) -> date:
    start = date(2023, 1, 1)
    end = date(2025, 12, 31)
    return start + timedelta(days=rng.randint(0, (end - start).days))


def _random_total(rng: random.Random, currency: str | None) -> Decimal:
    if currency == "JPY":
        return Decimal(rng.randint(800, 90_000))
    cents = rng.randint(1_500, 999_999)
    return (Decimal(cents) / Decimal(100)).quantize(Decimal("0.01"))


def _check_vendor(text: str, vendor: str | None) -> None:
    if vendor is None:
        leaked = [name for name in VENDORS if name in text]
        if leaked:
            raise DatasetConsistencyError("Vendor was omitted but a vendor name was rendered.")
        return
    if vendor not in text:
        raise DatasetConsistencyError("Vendor is missing from the document text.")


def _check_invoice_number(text: str, invoice_number: str | None) -> None:
    found = _INVOICE_NUMBER_IN_TEXT.findall(text)
    if invoice_number is None:
        if found:
            raise DatasetConsistencyError("Invoice number was omitted but a number was rendered.")
        return
    if invoice_number not in found:
        raise DatasetConsistencyError("Invoice number is missing from the document text.")


def _check_date(text: str, invoice_date: date | None) -> None:
    if invoice_date is None:
        if _ISO_DATE_IN_TEXT.search(text):
            raise DatasetConsistencyError("Invoice date was omitted but an ISO date was rendered.")
        for month in (*MONTH_NAMES, *MONTH_ABBREVIATIONS):
            if re.search(rf"\b{month}\b", text):
                raise DatasetConsistencyError("Invoice date was omitted but a month name was rendered.")
        return
    if not any(form in text for form in date_forms(invoice_date)):
        raise DatasetConsistencyError("Invoice date is missing from the document text.")


def _check_total(text: str, total: Decimal | None, currency: str | None) -> None:
    if total is None:
        if _DECIMAL_IN_TEXT.search(text):
            raise DatasetConsistencyError("Total was omitted but a decimal amount was rendered.")
        return
    rendered = format_amount(total, currency)
    if not amount_in_text(text, rendered):
        raise DatasetConsistencyError("Total is missing from the document text.")


def _check_currency(text: str, currency: str | None) -> None:
    present = {code for code in GENERATOR_CURRENCIES if re.search(rf"\b{code}\b", text)}
    if currency is None:
        if present:
            raise DatasetConsistencyError("Currency was omitted but a currency code was rendered.")
        return
    if currency not in present:
        raise DatasetConsistencyError("Currency code is missing from the document text.")
