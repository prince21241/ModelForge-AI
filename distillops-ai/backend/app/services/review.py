"""Flag student output for review without changing the extracted strings."""

from __future__ import annotations

import re
from datetime import date
from decimal import Decimal, InvalidOperation

from src.labeling import StudentFields

from backend.app.core.config import DEFAULT_REVIEW_WARNING
from backend.app.schemas.invoice import InvoiceFields

_ISO_DATE = re.compile(r"^\d{4}-\d{2}-\d{2}$")
_PLAIN_TOTAL = re.compile(r"^[0-9]+(?:\.[0-9]{1,6})?$")
_FIELD_NAMES = ("vendor", "invoice_number", "date", "total", "currency")


def fields_from_student(fields: StudentFields | None) -> InvoiceFields:
    """Copy parsed strings. A failed parse becomes five empty fields."""
    if fields is None:
        return InvoiceFields()
    return InvoiceFields(
        vendor=fields.vendor,
        invoice_number=fields.invoice_number,
        date=fields.date,
        total=fields.total,
        currency=fields.currency,
    )


def review_warnings(fields: InvoiceFields, *, validation_error: str | None) -> list[str]:
    """Build review notes. The field values themselves are left unchanged."""
    warnings = [DEFAULT_REVIEW_WARNING]
    if validation_error:
        warnings.append(
            "Model output did not match the five-field schema and was not repaired. "
            f"{validation_error}"
        )
        return warnings
    for name in _FIELD_NAMES:
        if getattr(fields, name).strip() == "":
            warnings.append(f"{name} is empty.")
    if fields.date.strip() and not _iso_date(fields.date):
        warnings.append("date is not an ISO calendar date YYYY-MM-DD.")
    if fields.total.strip() and not _plain_total(fields.total):
        warnings.append("total is not a plain nonnegative decimal string.")
    return warnings


def _iso_date(value: str) -> bool:
    text = value.strip()
    if not _ISO_DATE.fullmatch(text):
        return False
    try:
        date.fromisoformat(text)
    except ValueError:
        return False
    return True


def _plain_total(value: str) -> bool:
    text = value.strip()
    if not _PLAIN_TOTAL.fullmatch(text):
        return False
    try:
        amount = Decimal(text)
    except InvalidOperation:
        return False
    return amount.is_finite() and amount >= 0
