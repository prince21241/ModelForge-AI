"""Distinct synthetic invoice layouts.

Each layout prints a field only when that field has a value. Dates use an
English month table so the text does not depend on the machine locale.
"""

from __future__ import annotations

import random
import re
from dataclasses import dataclass
from datetime import date
from decimal import Decimal
from typing import Callable

MONTH_NAMES: tuple[str, ...] = (
    "January",
    "February",
    "March",
    "April",
    "May",
    "June",
    "July",
    "August",
    "September",
    "October",
    "November",
    "December",
)
MONTH_ABBREVIATIONS: tuple[str, ...] = (
    "Jan",
    "Feb",
    "Mar",
    "Apr",
    "May",
    "Jun",
    "Jul",
    "Aug",
    "Sep",
    "Oct",
    "Nov",
    "Dec",
)

TEMPLATE_GROUPS: tuple[str, ...] = (
    "classic_header",
    "ascii_table",
    "prose_memo",
    "key_value",
    "account_statement",
    "minimal_block",
)

_SYNTHETIC_FOOTER = "Synthetic training document. Not a real invoice."


@dataclass(frozen=True)
class InvoiceContent:
    """Values the renderer may print. None means the line must be omitted."""

    vendor: str | None
    invoice_number: str | None
    invoice_date: date | None
    total: Decimal | None
    currency: str | None


def format_amount(amount: Decimal, currency: str | None) -> str:
    """Render an amount the same way the consistency checks look for it."""
    if currency == "JPY":
        return str(amount.quantize(Decimal("1")))
    return f"{amount.quantize(Decimal('0.01')):.2f}"


def date_forms(value: date) -> tuple[str, ...]:
    """Explicit date strings a layout is allowed to print."""
    month = MONTH_NAMES[value.month - 1]
    abbreviation = MONTH_ABBREVIATIONS[value.month - 1]
    return (
        value.isoformat(),
        f"{month} {value.day}, {value.year}",
        f"{value.day} {month} {value.year}",
        f"{value.day:02d} {abbreviation} {value.year}",
    )


def format_date(value: date, style: str) -> str:
    forms = date_forms(value)
    styles = {"iso": 0, "long": 1, "long_dmy": 2, "abbr": 3}
    try:
        return forms[styles[style]]
    except KeyError as exc:
        raise ValueError(f"Unknown date style {style!r}.") from exc


def amount_in_text(text: str, rendered: str) -> bool:
    """True when ``rendered`` appears and is not part of a longer number."""
    pattern = rf"(?<!\d)(?<!\d\.){re.escape(rendered)}(?!\d)(?!\.\d)"
    return re.search(pattern, text) is not None


def render_invoice(template_group: str, content: InvoiceContent, rng: random.Random) -> str:
    try:
        renderer = _RENDERERS[template_group]
    except KeyError as exc:
        raise ValueError(f"Unknown template group {template_group!r}.") from exc
    return renderer(content, rng)


def _append_money(
    lines: list[str],
    content: InvoiceContent,
    *,
    total_prefix: str,
    currency_prefix: str,
    total_suffix: str = "",
    currency_suffix: str = "",
) -> None:
    """Print the total when it exists, or the currency alone when there is no total."""
    total = _total_suffix(content)
    if total:
        lines.append(f"{total_prefix}{total}{total_suffix}")
    elif content.currency:
        lines.append(f"{currency_prefix}{content.currency}{currency_suffix}")


def _total_suffix(content: InvoiceContent) -> str | None:
    if content.total is None:
        return None
    amount = format_amount(content.total, content.currency)
    if content.currency:
        return f"{amount} {content.currency}"
    return amount


def _decoy_lines(content: InvoiceContent, rng: random.Random) -> list[str]:
    """Small charges that do not add up to the labeled total.

    They exist so a model that sums line items instead of reading the total
    produces the wrong amount. No decoys are drawn when the total itself is
    absent, and this path does not consume random state in that case.
    """
    if content.total is None:
        return []
    total_text = format_amount(content.total, content.currency)
    lines: list[str] = []
    for label in ("Paper reams", "Local delivery"):
        if content.currency == "JPY":
            amount = Decimal(rng.randint(100, 800))
            rendered = format_amount(amount, "JPY")
            if rendered == total_text:
                rendered = format_amount(amount + Decimal(3), "JPY")
        else:
            cents = rng.randint(300, 2500)
            amount = (Decimal(cents) / Decimal(100)).quantize(Decimal("0.01"))
            rendered = format_amount(amount, None)
            if rendered == total_text:
                rendered = format_amount(amount + Decimal("1.00"), None)
        lines.append(f"{label}: {rendered}")
    return lines


def _render_classic(content: InvoiceContent, rng: random.Random) -> str:
    lines = ["INVOICE", "--------------------------------"]
    if content.vendor:
        lines.append(f"From: {content.vendor}")
    if content.invoice_number:
        lines.append(f"Invoice No: {content.invoice_number}")
    if content.invoice_date:
        lines.append(f"Invoice Date: {format_date(content.invoice_date, 'long')}")
    decoys = _decoy_lines(content, rng)
    if decoys:
        lines.append("Charges:")
        lines.extend(f"  {line}" for line in decoys)
    _append_money(lines, content, total_prefix="Total Due: ", currency_prefix="Currency: ")
    lines.extend(["--------------------------------", "Payment is due upon receipt.", _SYNTHETIC_FOOTER])
    return "\n".join(lines)


def _render_table(content: InvoiceContent, rng: random.Random) -> str:
    rows: list[tuple[str, str]] = [("Document", "INVOICE")]
    if content.vendor:
        rows.append(("Seller", content.vendor))
    if content.invoice_number:
        rows.append(("Reference", content.invoice_number))
    if content.invoice_date:
        rows.append(("Issued", format_date(content.invoice_date, "iso")))
    for line in _decoy_lines(content, rng):
        label, amount = line.split(": ", 1)
        rows.append((label, amount))
    total = _total_suffix(content)
    if total:
        rows.append(("Balance", total))
    elif content.currency:
        rows.append(("Currency", content.currency))
    cells = [f"{left}: {right}" for left, right in rows]
    inner = max(len(cell) for cell in cells)
    border = "+" + "-" * (inner + 2) + "+"
    body = [f"| {cell.ljust(inner)} |" for cell in cells]
    return "\n".join([border, *body, border, _SYNTHETIC_FOOTER])


def _render_memo(content: InvoiceContent, _: random.Random) -> str:
    lines = ["INVOICE memorandum"]
    if content.vendor and content.invoice_number:
        lines.append(
            f"This invoice was issued by {content.vendor} under reference {content.invoice_number}."
        )
    elif content.vendor:
        lines.append(f"This invoice was issued by {content.vendor}.")
    elif content.invoice_number:
        lines.append(f"This invoice uses reference {content.invoice_number}.")
    if content.invoice_date:
        lines.append(
            f"The invoice issue date is {format_date(content.invoice_date, 'long_dmy')}."
        )
    _append_money(
        lines,
        content,
        total_prefix="The stated total is ",
        currency_prefix="The currency code is ",
        total_suffix=".",
        currency_suffix=".",
    )
    lines.append("Figures that are not written here are not part of this invoice.")
    lines.append(_SYNTHETIC_FOOTER)
    return "\n".join(lines)


def _render_key_value(content: InvoiceContent, _: random.Random) -> str:
    lines = ["document: INVOICE"]
    if content.vendor:
        lines.append(f"vendor_name: {content.vendor}")
    if content.invoice_number:
        lines.append(f"number: {content.invoice_number}")
    if content.invoice_date:
        lines.append(f"invoice_date: {format_date(content.invoice_date, 'iso')}")
    if content.total is not None:
        lines.append(f"total_amount: {format_amount(content.total, content.currency)}")
    if content.currency:
        lines.append(f"currency_code: {content.currency}")
    lines.append(_SYNTHETIC_FOOTER)
    return "\n".join(lines)


def _render_statement(content: InvoiceContent, _: random.Random) -> str:
    lines = ["STATEMENT OF ACCOUNT", "Type: INVOICE"]
    if content.vendor:
        lines.append(f"Vendor ............... {content.vendor}")
    if content.invoice_number:
        lines.append(f"Inv # ................ {content.invoice_number}")
    if content.invoice_date:
        lines.append(f"Dated ................ {format_date(content.invoice_date, 'abbr')}")
    _append_money(
        lines,
        content,
        total_prefix="Total due ............ ",
        currency_prefix="Currency ............ ",
    )
    lines.append(_SYNTHETIC_FOOTER)
    return "\n".join(lines)


def _render_minimal(content: InvoiceContent, _: random.Random) -> str:
    lines = ["*** INVOICE ***"]
    identity = [part for part in (content.vendor, content.invoice_number) if part]
    if identity:
        lines.append(" | ".join(identity))
    if content.invoice_date:
        lines.append(format_date(content.invoice_date, "long"))
    _append_money(lines, content, total_prefix="", currency_prefix="")
    lines.extend(["End of invoice.", _SYNTHETIC_FOOTER])
    return "\n".join(lines)


_RENDERERS: dict[str, Callable[[InvoiceContent, random.Random], str]] = {
    "classic_header": _render_classic,
    "ascii_table": _render_table,
    "prose_memo": _render_memo,
    "key_value": _render_key_value,
    "account_statement": _render_statement,
    "minimal_block": _render_minimal,
}
