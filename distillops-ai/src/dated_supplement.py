"""Twelve invoices that use a Dated line and a dotted total.

These targets are deterministic ground truth. The values are not copied from
the five validation invoices, and this module does not open the locked test
split.
"""

from __future__ import annotations

import random
import re
from datetime import date, timedelta
from decimal import Decimal
from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field

from src.dataset import DATASET_SEED, GENERATOR_CURRENCIES, VENDORS, generate_records, write_jsonl
from src.hybrid_dataset import (
    HYBRID_SEED,
    HybridDataError,
    HybridGroundTruthFile,
    HybridInvoiceFile,
    generate_hybrid_invoices,
)
from src.invoice_templates import format_amount, format_date
from src.labeling import StudentFields

DATED_COUNT = 12
DATED_ID_START = 201
DATED_SEED = 42
_DOTS = "." * 8
_FOOTER = "Synthetic training document. Not a real invoice."
_VENDORS: tuple[str, ...] = (
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
)
# Field values printed on the held-out account statements. The supplement must
# not reuse them.
_BANNED_NUMBERS = frozenset({"INV-Y547DG", "INV-GRNN6J", "INV-RE6GDA", "INV-98P3DL"})
_BANNED_DATES = frozenset(
    {date(2023, 7, 18), date(2023, 5, 5), date(2025, 5, 14), date(2023, 7, 11)}
)
_BANNED_TOTALS = frozenset({"57298", "7650.87", "2935.75", "1758.89", "37.60"})
_BANNED_VENDORS = frozenset(VENDORS)
_ALPHABET = "ABCDEFGHJKLMNPQRSTUVWXYZ23456789"


class DatedInvoice(BaseModel):
    """One Dated-line invoice and the label used to render it."""

    model_config = ConfigDict(extra="forbid")

    document_id: str
    document_text: str = Field(min_length=1)
    template_group: str = "dated_dotted"
    ground_truth: StudentFields
    label_source: str = "deterministic_ground_truth"


def generate_dated_invoices(seed: int = DATED_SEED) -> list[DatedInvoice]:
    """Build the 12 invoices. The same seed returns the same documents."""
    rng = random.Random(seed)
    held_out = generate_records(DATASET_SEED)
    hybrid = generate_hybrid_invoices(HYBRID_SEED)
    banned_ids = {record.document_id for record in held_out}
    banned_ids.update(invoice.document_id for invoice in hybrid)
    banned_texts = {record.document_text for record in held_out}
    banned_texts.update(invoice.document_text for invoice in hybrid)
    invoices: list[DatedInvoice] = []
    used_text: set[str] = set()
    for offset in range(1, DATED_COUNT + 1):
        invoice = _sample_invoice(rng, offset, banned_ids, banned_texts, used_text)
        validate_dated_invoice(invoice, banned_ids, banned_texts)
        used_text.add(invoice.document_text)
        invoices.append(invoice)
    return invoices


def validate_dated_invoice(
    invoice: DatedInvoice,
    banned_ids: set[str] | None = None,
    banned_texts: set[str] | None = None,
) -> None:
    """Fail when the text, the label, or a held-out value disagree."""
    if invoice.document_id in (banned_ids or set()):
        raise HybridDataError(f"{invoice.document_id} collides with an existing id.")
    if invoice.document_text in (banned_texts or set()):
        raise HybridDataError(f"{invoice.document_id} copies an existing invoice.")
    if "STATEMENT OF ACCOUNT" in invoice.document_text:
        raise HybridDataError(f"{invoice.document_id} copies the validation header.")
    fields = invoice.ground_truth
    if fields.vendor in _BANNED_VENDORS:
        raise HybridDataError(f"{invoice.document_id} reuses a held-out vendor.")
    if fields.invoice_number in _BANNED_NUMBERS or fields.total in _BANNED_TOTALS:
        raise HybridDataError(f"{invoice.document_id} reuses a validation field value.")
    if fields.date and date.fromisoformat(fields.date) in _BANNED_DATES:
        raise HybridDataError(f"{invoice.document_id} reuses a validation date.")
    if fields.vendor and fields.vendor not in invoice.document_text:
        raise HybridDataError(f"{invoice.document_id} vendor is missing from the text.")
    if fields.invoice_number and fields.invoice_number not in invoice.document_text:
        raise HybridDataError(f"{invoice.document_id} invoice number is missing from the text.")
    if not fields.invoice_number and re.search(r"\bHYB-[A-Z0-9]+\b", invoice.document_text):
        raise HybridDataError(f"{invoice.document_id} prints an invoice number that is unlabeled.")
    if fields.date:
        abbr = format_date(date.fromisoformat(fields.date), "abbr")
        if f"Dated {_DOTS} {abbr}" not in invoice.document_text:
            raise HybridDataError(f"{invoice.document_id} does not print the abbreviated Dated line.")
    elif re.search(r"\bDated\b", invoice.document_text):
        raise HybridDataError(f"{invoice.document_id} prints a date that is unlabeled.")
    if f"Total due {_DOTS} {fields.total} {fields.currency}" not in invoice.document_text:
        raise HybridDataError(f"{invoice.document_id} does not print the dotted total.")
    if "\\" in fields.vendor or "\\" in fields.total:
        raise HybridDataError(f"{invoice.document_id} label contains a backslash.")


def write_dated_dataset(root: Path, invoices: list[DatedInvoice]) -> Path:
    """Write the supplement apart from the original splits and the hybrid files."""
    root = Path(root)
    if root.name.lower() in {"test.jsonl", "val.jsonl", "train.jsonl", "student_train.jsonl"}:
        raise HybridDataError(f"Refusing to write dated invoices over {root.name}.")
    write_jsonl(
        root / "invoices.jsonl",
        [
            HybridInvoiceFile(
                document_id=invoice.document_id,
                document_text=invoice.document_text,
                template_group=invoice.template_group,
            )
            for invoice in invoices
        ],
    )
    write_jsonl(
        root / "ground_truth.jsonl",
        [
            HybridGroundTruthFile(
                document_id=invoice.document_id,
                vendor=invoice.ground_truth.vendor,
                invoice_number=invoice.ground_truth.invoice_number,
                date=invoice.ground_truth.date,
                total=invoice.ground_truth.total,
                currency=invoice.ground_truth.currency,
            )
            for invoice in invoices
        ],
    )
    return root


def _sample_invoice(
    rng: random.Random,
    offset: int,
    banned_ids: set[str],
    banned_texts: set[str],
    used_text: set[str],
) -> DatedInvoice:
    omit_vendor = offset % 5 == 0
    omit_number = offset % 6 == 0
    omit_date = offset == DATED_COUNT
    for _ in range(40):
        vendor = None if omit_vendor else rng.choice(_VENDORS)
        number = None if omit_number else _number(rng)
        invoice_date = None if omit_date else _date(rng)
        currency = rng.choice(GENERATOR_CURRENCIES)
        total = _total(rng, currency)
        plain = format_amount(total, currency)
        if plain in _BANNED_TOTALS or (invoice_date in _BANNED_DATES):
            continue
        text = _render(vendor, number, invoice_date, plain, currency, rng)
        document_id = f"inv-synth-{DATED_ID_START + offset - 1:03d}"
        if document_id in banned_ids or text in used_text or text in banned_texts:
            continue
        fields = StudentFields(
            vendor=vendor or "",
            invoice_number=number or "",
            date="" if invoice_date is None else invoice_date.isoformat(),
            total=plain,
            currency=currency,
        )
        invoice = DatedInvoice(document_id=document_id, document_text=text, ground_truth=fields)
        try:
            validate_dated_invoice(invoice, banned_ids, banned_texts)
        except HybridDataError:
            continue
        return invoice
    raise HybridDataError(f"Could not sample dated invoice {offset}.")


def _number(rng: random.Random) -> str:
    suffix = "".join(rng.choice(_ALPHABET) for _ in range(6))
    return f"HYB-{suffix}"


def _date(rng: random.Random) -> date:
    start = date(2023, 1, 1)
    end = date(2025, 12, 31)
    for _ in range(20):
        picked = start + timedelta(days=rng.randint(0, (end - start).days))
        if picked not in _BANNED_DATES:
            return picked
    raise HybridDataError("Could not sample a date outside the validation dates.")


def _total(rng: random.Random, currency: str) -> Decimal:
    for _ in range(20):
        if currency == "JPY":
            amount = Decimal(rng.randint(900, 80_000))
        else:
            cents = rng.randint(2_500, 850_000)
            amount = (Decimal(cents) / Decimal(100)).quantize(Decimal("0.01"))
        if format_amount(amount, currency) not in _BANNED_TOTALS:
            return amount
    raise HybridDataError("Could not sample a total outside the validation totals.")


def _render(
    vendor: str | None,
    number: str | None,
    invoice_date: date | None,
    total: str,
    currency: str,
    rng: random.Random,
) -> str:
    blocks: list[str] = []
    if vendor:
        blocks.append(f"Payee {_DOTS} {vendor}")
    if number:
        blocks.append(f"Inv # {_DOTS} {number}")
    if invoice_date:
        blocks.append(f"Dated {_DOTS} {format_date(invoice_date, 'abbr')}")
    blocks.append(f"Total due {_DOTS} {total} {currency}")
    rng.shuffle(blocks)
    item_lines = _items(rng, total, currency)
    lines = ["REMITTANCE INVOICE", "", *blocks, "", *item_lines, _FOOTER]
    return "\n".join(lines)


def _items(rng: random.Random, total: str, currency: str) -> list[str]:
    lines: list[str] = []
    for label in ("Binding boards", "Courier pouch"):
        for _ in range(8):
            if currency == "JPY":
                rendered = str(rng.randint(11, 90))
            else:
                cents = rng.randint(150, 950)
                rendered = f"{(Decimal(cents) / Decimal(100)).quantize(Decimal('0.01')):.2f}"
            if rendered != total and rendered not in total and rendered not in _BANNED_TOTALS:
                break
        lines.append(f"{label}: {rendered}")
    return lines
