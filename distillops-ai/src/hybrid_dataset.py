"""Forty synthetic training invoices that sit outside the original splits.

The documents are generated from a fixed seed. Their labels are the values
used to render the text, stored as empty strings when a field is absent.
They are not teacher labels, and they are not copies of the validation or
test invoices. This module does not open the locked test split.
"""

from __future__ import annotations

import json
import random
import re
from dataclasses import dataclass
from datetime import date, timedelta
from decimal import Decimal
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from src.dataset import (
    DATASET_SEED,
    GENERATOR_CURRENCIES,
    DatasetConsistencyError,
    DatasetError,
    generate_records,
    write_jsonl,
)
from src.invoice_templates import (
    MONTH_ABBREVIATIONS,
    MONTH_NAMES,
    InvoiceContent,
    format_amount,
    format_date,
)
from src.labeling import StudentFields

HYBRID_SEED = 42
HYBRID_COUNT = 40
HYBRID_ID_START = 101
HYBRID_GROUPS: tuple[str, ...] = (
    "dotted_notice",
    "stacked_vendor",
    "reordered_slip",
    "whitespace_items",
)
DOCUMENTS_PER_GROUP = 10
_ALPHABET = "ABCDEFGHJKLMNPQRSTUVWXYZ23456789"
_FOOTER = "Synthetic training document. Not a real invoice."
_FORBIDDEN_MARKERS: tuple[str, ...] = (
    "STATEMENT OF ACCOUNT",
    "INVOICE memorandum",
    "Total due ",
    "Vendor ...............",
    "Inv # ................",
    "Dated ................",
    "document: INVOICE",
    "*** INVOICE ***",
    "Invoice No:",
    "From: ",
)
_HYBRID_VENDORS: tuple[str, ...] = (
    "Marrow Street Bindery",
    "Quill and Copper Desk",
    "Eastmill Crate Co",
    "Sable Hour Printers",
    "Lowtide Meter Works",
    "Bracken Field Supply",
    "Ivy Kiln Studio",
    "Paperwharf Clerks",
    "Hearthline Cartage",
    "Juniper Ballot Office",
    "Cobalt Stair Goods",
    "Fennel Court Laundry",
    "Ridgeglass Survey",
    "Mothlight Archives",
    "Tinwhistle Provisions",
    "Amberlane Filing",
)
_MULTILINE_VENDORS: tuple[str, ...] = (
    "Marrow Street Bindery",
    "Quill and Copper Desk",
    "Eastmill Crate Co",
    "Sable Hour Printers",
    "Lowtide Meter Works",
    "Bracken Field Supply",
    "Ivy Kiln Studio",
    "Paperwharf Clerks",
)
_ITEM_LABELS = ("Binding boards", "Courier pouch")
_SAFE_OUTPUT_NAMES = frozenset(
    {"test.jsonl", "val.jsonl", "train.jsonl", "student_train.jsonl"}
)


class HybridDataError(ValueError):
    """Raised when a hybrid invoice would leak into an existing split."""


class HybridInvoice(BaseModel):
    """One new training invoice and the deterministic label that rendered it."""

    model_config = ConfigDict(extra="forbid")

    document_id: str
    document_text: str = Field(min_length=1)
    template_group: Literal[
        "dotted_notice",
        "stacked_vendor",
        "reordered_slip",
        "whitespace_items",
    ]
    ground_truth: StudentFields
    label_source: Literal["deterministic_ground_truth"] = "deterministic_ground_truth"


class HybridInvoiceFile(BaseModel):
    """Invoice text written apart from its label."""

    model_config = ConfigDict(extra="forbid")

    document_id: str
    document_text: str
    template_group: str


class HybridGroundTruthFile(BaseModel):
    """Five student fields. Absent values are empty strings."""

    model_config = ConfigDict(extra="forbid")

    document_id: str
    vendor: str = ""
    invoice_number: str = ""
    date: str = ""
    total: str = ""
    currency: str = ""
    label_source: Literal["deterministic_ground_truth"] = "deterministic_ground_truth"


@dataclass(frozen=True)
class _BannedValues:
    ids: frozenset[str]
    texts: frozenset[str]
    invoice_numbers: frozenset[str]
    vendors: frozenset[str]
    dates: frozenset[date]
    amounts: frozenset[tuple[str | None, str]]


def hybrid_document_id(offset: int) -> str:
    """Map a 1-based offset onto inv-synth-101 and the following ids."""
    if offset < 1 or offset > HYBRID_COUNT:
        raise HybridDataError(f"Hybrid offset {offset} is outside 1..{HYBRID_COUNT}.")
    return f"inv-synth-{HYBRID_ID_START + offset - 1:03d}"


def omitted_hybrid_fields(offset: int) -> frozenset[str]:
    """Leave some vendors, numbers, and dates unprinted. The schedule is fixed."""
    if offset < 1:
        raise HybridDataError("Hybrid offset starts at 1.")
    omitted: set[str] = set()
    if offset % 10 == 0:
        omitted.add("vendor")
    if offset % 8 == 0:
        omitted.add("invoice_number")
    if offset % 7 == 0:
        omitted.add("date")
    return frozenset(omitted)


def generate_hybrid_invoices(seed: int = HYBRID_SEED) -> list[HybridInvoice]:
    """Build the 40 invoices. The same seed always returns the same documents."""
    rng = random.Random(seed)
    banned = _banned_from_original_corpus()
    invoices: list[HybridInvoice] = []
    used_text: set[str] = set()
    for offset in range(1, HYBRID_COUNT + 1):
        group = HYBRID_GROUPS[(offset - 1) // DOCUMENTS_PER_GROUP]
        invoice = _sample_invoice(rng, offset, group, banned, used_text)
        validate_hybrid_invoice(invoice, banned)
        used_text.add(invoice.document_text)
        invoices.append(invoice)
    if len(invoices) != HYBRID_COUNT:
        raise HybridDataError(f"Expected {HYBRID_COUNT} invoices, built {len(invoices)}.")
    ids = [invoice.document_id for invoice in invoices]
    if len(set(ids)) != len(ids):
        raise HybridDataError("Hybrid invoice ids are not unique.")
    return invoices


def validate_hybrid_invoice(invoice: HybridInvoice, banned: _BannedValues | None = None) -> None:
    """Fail when the text and the five labels disagree, or copy a held-out document."""
    banned = banned or _banned_from_original_corpus()
    if invoice.document_id in banned.ids:
        raise HybridDataError(f"{invoice.document_id} collides with the original corpus.")
    if invoice.document_text in banned.texts:
        raise HybridDataError(f"{invoice.document_id} copies an original document.")
    for marker in _FORBIDDEN_MARKERS:
        if marker in invoice.document_text:
            raise HybridDataError(f"{invoice.document_id} uses a held-out layout marker.")
    for original in banned.texts:
        if invoice.document_text in original or original in invoice.document_text:
            raise HybridDataError(f"{invoice.document_id} overlaps an original document.")
    fields = invoice.ground_truth
    _check_optional_text(invoice, "vendor", fields.vendor, banned.vendors)
    _check_invoice_number(invoice, fields.invoice_number, banned.invoice_numbers)
    _check_date(invoice, fields.date, banned.dates)
    _check_total(invoice, fields.total, fields.currency, banned.amounts)
    _check_currency(invoice, fields.currency)
    if "INVOICE" not in invoice.document_text.upper():
        raise DatasetConsistencyError(f"{invoice.document_id} does not read as an invoice.")


def write_hybrid_dataset(root: Path, invoices: list[HybridInvoice]) -> Path:
    """Write invoice text and labels in separate files. Existing splits are refused."""
    root = Path(root)
    _refuse_protected_output(root)
    if len(invoices) != HYBRID_COUNT:
        raise HybridDataError(f"Refusing to write {len(invoices)} invoices.")
    invoice_rows = [
        HybridInvoiceFile(
            document_id=invoice.document_id,
            document_text=invoice.document_text,
            template_group=invoice.template_group,
        )
        for invoice in invoices
    ]
    label_rows = [
        HybridGroundTruthFile(
            document_id=invoice.document_id,
            vendor=invoice.ground_truth.vendor,
            invoice_number=invoice.ground_truth.invoice_number,
            date=invoice.ground_truth.date,
            total=invoice.ground_truth.total,
            currency=invoice.ground_truth.currency,
        )
        for invoice in invoices
    ]
    write_jsonl(root / "invoices.jsonl", invoice_rows)
    write_jsonl(root / "ground_truth.jsonl", label_rows)
    manifest = {
        "synthetic": True,
        "contains_real_customer_data": False,
        "seed": HYBRID_SEED,
        "documents": len(invoices),
        "id_start": hybrid_document_id(1),
        "id_end": hybrid_document_id(HYBRID_COUNT),
        "label_source": "deterministic_ground_truth",
        "note": (
            "Fictional expansion invoices for DistillOps. Labels are the values "
            "used to render the text. They are not teacher predictions and they "
            "are not the validation or test splits."
        ),
    }
    (root / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return root


def _refuse_protected_output(root: Path) -> None:
    if root.name.lower() in _SAFE_OUTPUT_NAMES:
        raise HybridDataError(f"Refusing to write hybrid data over {root.name}.")
    for path in (root, root / "invoices.jsonl", root / "ground_truth.jsonl"):
        if path.name.lower() in _SAFE_OUTPUT_NAMES:
            raise HybridDataError(f"Refusing to write {path.name}.")


def _banned_from_original_corpus() -> _BannedValues:
    """Ban values from the in-memory original generator. The test file stays closed."""
    records = generate_records(DATASET_SEED)
    amounts: set[tuple[str | None, str]] = set()
    for record in records:
        truth = record.ground_truth
        if truth.total is not None:
            amounts.add((truth.currency, format_amount(truth.total, truth.currency)))
    return _BannedValues(
        ids=frozenset(record.document_id for record in records),
        texts=frozenset(record.document_text for record in records),
        invoice_numbers=frozenset(
            record.ground_truth.invoice_number
            for record in records
            if record.ground_truth.invoice_number
        ),
        vendors=frozenset(
            record.ground_truth.vendor for record in records if record.ground_truth.vendor
        ),
        dates=frozenset(
            record.ground_truth.invoice_date
            for record in records
            if record.ground_truth.invoice_date is not None
        ),
        amounts=frozenset(amounts),
    )


def _sample_invoice(
    rng: random.Random,
    offset: int,
    group: str,
    banned: _BannedValues,
    used_text: set[str],
) -> HybridInvoice:
    omitted = omitted_hybrid_fields(offset)
    last_error = "no attempt"
    for _ in range(40):
        content = _sample_content(rng, group, omitted)
        try:
            text = _render(group, content, rng)
        except DatasetError as exc:
            last_error = str(exc)
            continue
        invoice = HybridInvoice(
            document_id=hybrid_document_id(offset),
            document_text=text,
            template_group=group,  # type: ignore[arg-type]
            ground_truth=_fields_from_content(content),
        )
        try:
            if text in used_text:
                raise HybridDataError("duplicate text")
            _reject_banned_values(content, banned)
            validate_hybrid_invoice(invoice, banned)
        except (HybridDataError, DatasetConsistencyError) as exc:
            last_error = str(exc)
            continue
        return invoice
    raise HybridDataError(f"Could not sample {hybrid_document_id(offset)}: {last_error}")


def _sample_content(
    rng: random.Random,
    group: str,
    omitted: frozenset[str],
) -> InvoiceContent:
    if "vendor" in omitted:
        vendor = None
    elif group == "stacked_vendor":
        vendor = rng.choice(_MULTILINE_VENDORS)
    else:
        vendor = rng.choice(_HYBRID_VENDORS)
    invoice_number = None if "invoice_number" in omitted else _invoice_number(rng)
    invoice_date = None if "date" in omitted else _random_date(rng)
    currency = rng.choice(GENERATOR_CURRENCIES)
    total = _random_total(rng, currency)
    return InvoiceContent(
        vendor=vendor,
        invoice_number=invoice_number,
        invoice_date=invoice_date,
        total=total,
        currency=currency,
    )


def _invoice_number(rng: random.Random) -> str:
    suffix = "".join(rng.choice(_ALPHABET) for _ in range(6))
    return f"HYB-{suffix}"


def _random_date(rng: random.Random) -> date:
    start = date(2023, 1, 1)
    end = date(2025, 12, 31)
    return start + timedelta(days=rng.randint(0, (end - start).days))


def _random_total(rng: random.Random, currency: str) -> Decimal:
    if currency == "JPY":
        return Decimal(rng.randint(900, 80_000))
    cents = rng.randint(2_500, 850_000)
    return (Decimal(cents) / Decimal(100)).quantize(Decimal("0.01"))


def _fields_from_content(content: InvoiceContent) -> StudentFields:
    total = ""
    if content.total is not None:
        total = format_amount(content.total, content.currency)
    invoice_date = ""
    if content.invoice_date is not None:
        invoice_date = content.invoice_date.isoformat()
    return StudentFields(
        vendor=content.vendor or "",
        invoice_number=content.invoice_number or "",
        date=invoice_date,
        total=total,
        currency=content.currency or "",
    )


def _render(group: str, content: InvoiceContent, rng: random.Random) -> str:
    if group == "dotted_notice":
        return _render_dotted(content, rng)
    if group == "stacked_vendor":
        return _render_stacked(content, rng)
    if group == "reordered_slip":
        return _render_reordered(content, rng)
    if group == "whitespace_items":
        return _render_whitespace(content, rng)
    raise HybridDataError(f"Unknown hybrid layout {group!r}.")


def _render_dotted(content: InvoiceContent, rng: random.Random) -> str:
    dots = "." * rng.randint(6, 12)
    blocks: list[str] = []
    if content.vendor:
        blocks.append(f"Payee {dots} {content.vendor}")
    if content.invoice_number:
        blocks.append(f"Bill no. {dots} {content.invoice_number}")
    if content.invoice_date:
        blocks.append(f"Issued {dots} {format_date(content.invoice_date, 'abbr')}")
    blocks.append(f"Balance due {dots} {_money(content, comma=False)}")
    rng.shuffle(blocks)
    lines = ["ACCOUNTS PAYABLE INVOICE", "", *blocks, "", *_line_items(content, rng), _FOOTER]
    return "\n".join(lines)


def _render_stacked(content: InvoiceContent, rng: random.Random) -> str:
    lines = ["SELLER INVOICE", "Seller"]
    if content.vendor:
        lines.extend(_vendor_lines(content.vendor))
    if content.invoice_number:
        lines.append(f"Reference {content.invoice_number}")
    if content.invoice_date:
        lines.append(f"Issue date {format_date(content.invoice_date, 'long')}")
    lines.extend(_line_items(content, rng))
    lines.append(f"Amount {_money(content, comma=False)}")
    lines.append(_FOOTER)
    return "\n".join(lines)


def _render_reordered(content: InvoiceContent, rng: random.Random) -> str:
    blocks: list[str] = []
    if content.vendor:
        blocks.append(f"Seller: {content.vendor}")
    if content.invoice_number:
        blocks.append(f"Inv #: {content.invoice_number}")
    if content.invoice_date:
        blocks.append(f"Dated: {format_date(content.invoice_date, 'long_dmy')}")
    blocks.append(f"Sum: {_money(content, comma=False)}")
    rng.shuffle(blocks)
    lines = ["DISPATCH INVOICE", *blocks, *_line_items(content, rng), _FOOTER]
    return "\n".join(lines)


def _render_whitespace(content: InvoiceContent, rng: random.Random) -> str:
    chunks = ["INVOICE WORKSHEET", ""]
    if content.vendor:
        chunks.extend([f"  Seller:   {content.vendor}", ""])
    if content.invoice_number:
        chunks.extend([f"Number: {content.invoice_number}", ""])
    if content.invoice_date:
        chunks.extend([f"   When:   {format_date(content.invoice_date, 'iso')}", ""])
    chunks.extend(_line_items(content, rng))
    chunks.extend(["", f"Amount: {_money(content, comma=True)}", "", _FOOTER])
    return "\n".join(chunks)


def _money(content: InvoiceContent, *, comma: bool) -> str:
    if content.total is None or content.currency is None:
        raise HybridDataError("Hybrid invoices always render a total and a currency.")
    rendered = _rendered_total(content.total, content.currency, comma=comma)
    return f"{rendered} {content.currency}"


def _rendered_total(total: Decimal, currency: str, *, comma: bool) -> str:
    plain = format_amount(total, currency)
    if not comma or currency == "JPY":
        return plain
    whole, _, fraction = f"{total.quantize(Decimal('0.01')):.2f}".partition(".")
    return f"{int(whole):,}.{fraction}"


def _line_items(content: InvoiceContent, rng: random.Random) -> list[str]:
    if content.total is None or content.currency is None:
        return []
    total_text = format_amount(content.total, content.currency)
    comma_text = _rendered_total(content.total, content.currency, comma=True)
    lines: list[str] = []
    for label in _ITEM_LABELS:
        for _ in range(8):
            if content.currency == "JPY":
                rendered = str(rng.randint(11, 90))
            else:
                cents = rng.randint(150, 950)
                rendered = f"{(Decimal(cents) / Decimal(100)).quantize(Decimal('0.01')):.2f}"
            if rendered != total_text and rendered not in comma_text and comma_text not in rendered:
                break
        lines.append(f"{label}: {rendered}")
    return lines


def _reject_banned_values(content: InvoiceContent, banned: _BannedValues) -> None:
    if content.vendor:
        flat = " ".join(content.vendor.split())
        if content.vendor in banned.vendors or flat in banned.vendors:
            raise HybridDataError("Vendor copies an original vendor.")
    if content.invoice_number and content.invoice_number in banned.invoice_numbers:
        raise HybridDataError("Invoice number copies an original number.")
    if content.invoice_date and content.invoice_date in banned.dates:
        raise HybridDataError("Date copies an original date.")
    if content.total is not None:
        amount = format_amount(content.total, content.currency)
        if (content.currency, amount) in banned.amounts:
            raise HybridDataError("Total copies an original amount.")


def _vendor_lines(vendor: str) -> list[str]:
    """Show a vendor name on two lines without putting a backslash in the JSON label."""
    head, separator, tail = vendor.rpartition(" ")
    if not separator:
        return [vendor]
    return [head, tail]


def _check_optional_text(
    invoice: HybridInvoice,
    field: str,
    value: str,
    banned: frozenset[str],
) -> None:
    if value:
        collapsed = " ".join(invoice.document_text.split())
        if " ".join(value.split()) not in collapsed:
            raise DatasetConsistencyError(f"{invoice.document_id} {field} is missing from the text.")
        if value in banned:
            raise HybridDataError(f"{invoice.document_id} {field} copies an original value.")
        return
    leaked = [name for name in _HYBRID_VENDORS if name in invoice.document_text]
    leaked.extend(name for name in _MULTILINE_VENDORS if name in invoice.document_text)
    if leaked:
        raise DatasetConsistencyError(
            f"{invoice.document_id} omits {field} but a vendor name was rendered."
        )


def _check_invoice_number(
    invoice: HybridInvoice,
    value: str,
    banned: frozenset[str],
) -> None:
    found = re.findall(r"\bHYB-[A-Z0-9]+\b", invoice.document_text)
    original = re.findall(r"\bINV-[A-Z0-9]+\b", invoice.document_text)
    if original or any(number in banned for number in found):
        raise HybridDataError(f"{invoice.document_id} contains an original invoice number.")
    if value:
        if value not in found:
            raise DatasetConsistencyError(f"{invoice.document_id} invoice number is missing.")
        return
    if found:
        raise DatasetConsistencyError(
            f"{invoice.document_id} omits the invoice number but one was rendered."
        )


def _check_date(invoice: HybridInvoice, value: str, banned: frozenset[date]) -> None:
    if not value:
        if re.search(r"\b\d{4}-\d{2}-\d{2}\b", invoice.document_text):
            raise DatasetConsistencyError(f"{invoice.document_id} omits the date but prints one.")
        for month in (*MONTH_NAMES, *MONTH_ABBREVIATIONS):
            if re.search(rf"\b{month}\b", invoice.document_text):
                raise DatasetConsistencyError(
                    f"{invoice.document_id} omits the date but prints a month."
                )
        return
    parsed = date.fromisoformat(value)
    if parsed in banned:
        raise HybridDataError(f"{invoice.document_id} copies an original date.")
    forms = (
        parsed.isoformat(),
        f"{MONTH_NAMES[parsed.month - 1]} {parsed.day}, {parsed.year}",
        f"{parsed.day} {MONTH_NAMES[parsed.month - 1]} {parsed.year}",
        f"{parsed.day:02d} {MONTH_ABBREVIATIONS[parsed.month - 1]} {parsed.year}",
    )
    if not any(form in invoice.document_text for form in forms):
        raise DatasetConsistencyError(f"{invoice.document_id} date is missing from the text.")


def _check_total(
    invoice: HybridInvoice,
    value: str,
    currency: str,
    banned: frozenset[tuple[str | None, str]],
) -> None:
    if not value:
        raise DatasetConsistencyError(f"{invoice.document_id} is missing its total label.")
    if (currency or None, value) in banned:
        raise HybridDataError(f"{invoice.document_id} copies an original total.")
    comma = _rendered_total(Decimal(value), currency, comma=True)
    if value not in invoice.document_text and comma not in invoice.document_text:
        raise DatasetConsistencyError(f"{invoice.document_id} total is missing from the text.")


def _check_currency(invoice: HybridInvoice, currency: str) -> None:
    if not currency:
        raise DatasetConsistencyError(f"{invoice.document_id} is missing its currency label.")
    if not re.search(rf"\b{currency}\b", invoice.document_text):
        raise DatasetConsistencyError(f"{invoice.document_id} currency is missing from the text.")
