"""Pydantic contracts for invoice extraction and dataset records.

The domain model stores dates as ``datetime.date`` and amounts as ``Decimal``.
The teacher payload uses strings because provider structured-output schemas do
not reliably accept those Python types. ``extracted_invoice_from_payload``
validates a teacher payload into the domain model.
"""

from __future__ import annotations

import re
from datetime import date, datetime
from decimal import Decimal
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator

# Practical active ISO 4217 set used to reject invented currency codes.
# It is not a claim that every historical code is included.
ISO_4217_CODES: frozenset[str] = frozenset(
    {
        "AED", "AFN", "ALL", "AMD", "ANG", "AOA", "ARS", "AUD", "AWG", "AZN",
        "BAM", "BBD", "BDT", "BGN", "BHD", "BIF", "BMD", "BND", "BOB", "BRL",
        "BSD", "BTN", "BWP", "BYN", "BZD", "CAD", "CDF", "CHF", "CLP", "CNY",
        "COP", "CRC", "CUP", "CVE", "CZK", "DJF", "DKK", "DOP", "DZD", "EGP",
        "ERN", "ETB", "EUR", "FJD", "FKP", "GBP", "GEL", "GHS", "GIP", "GMD",
        "GNF", "GTQ", "GYD", "HKD", "HNL", "HTG", "HUF", "IDR", "ILS", "INR",
        "IQD", "IRR", "ISK", "JMD", "JOD", "JPY", "KES", "KGS", "KHR", "KMF",
        "KPW", "KRW", "KWD", "KYD", "KZT", "LAK", "LBP", "LKR", "LRD", "LSL",
        "LYD", "MAD", "MDL", "MGA", "MKD", "MMK", "MNT", "MOP", "MRU", "MUR",
        "MVR", "MWK", "MXN", "MYR", "MZN", "NAD", "NGN", "NIO", "NOK", "NPR",
        "NZD", "OMR", "PAB", "PEN", "PGK", "PHP", "PKR", "PLN", "PYG", "QAR",
        "RON", "RSD", "RUB", "RWF", "SAR", "SBD", "SCR", "SDG", "SEK", "SGD",
        "SHP", "SLE", "SOS", "SRD", "STN", "SYP", "SZL", "THB", "TJS", "TMT",
        "TND", "TOP", "TRY", "TTD", "TWD", "TZS", "UAH", "UGX", "USD", "UYU",
        "UZS", "VES", "VND", "VUV", "WST", "XAF", "XCD", "XOF", "XPF", "YER",
        "ZAR", "ZMW", "ZWL",
    }
)

_AMOUNT_RE = re.compile(r"^[0-9]+(?:\.[0-9]{1,6})?$")
_ISO_DATE_RE = re.compile(r"^\d{4}-\d{2}-\d{2}$")
_SHORT_TEXT_LIMIT = 200

EXTRACTION_FIELDS: tuple[str, ...] = (
    "document_type",
    "vendor",
    "invoice_number",
    "invoice_date",
    "total",
    "currency",
)


def _clean_optional_text(value: Any, field_name: str) -> str | None:
    if value is None:
        return None
    if not isinstance(value, str):
        raise ValueError(f"{field_name} must be a string or null")
    cleaned = " ".join(value.split())
    if not cleaned:
        return None
    if len(cleaned) > _SHORT_TEXT_LIMIT:
        raise ValueError(f"{field_name} exceeds {_SHORT_TEXT_LIMIT} characters")
    return cleaned


def _parse_total(value: Any) -> Decimal | None:
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, (str, int, Decimal, float)):
        raise ValueError("total must be a nonnegative decimal amount or null")
    if isinstance(value, float):
        if value != value or value in (float("inf"), float("-inf")):  # NaN or infinity
            raise ValueError("total must be a finite decimal amount")
        text = format(value, "f")
        amount = Decimal(text)
    elif isinstance(value, Decimal):
        amount = value
    elif isinstance(value, int):
        amount = Decimal(value)
    else:
        cleaned = value.strip().replace(",", "")
        if not cleaned:
            return None
        if not _AMOUNT_RE.fullmatch(cleaned):
            raise ValueError("total must be a nonnegative decimal string or null")
        amount = Decimal(cleaned)
    if not amount.is_finite():
        raise ValueError("total must be a finite decimal amount")
    if amount < 0:
        raise ValueError("total must be nonnegative")
    exponent = amount.as_tuple().exponent
    if isinstance(exponent, int) and exponent < -6:
        raise ValueError("total supports at most 6 decimal places")
    return amount


def _parse_invoice_date(value: Any) -> date | None:
    if value is None:
        return None
    if isinstance(value, datetime):
        return value.date()
    if isinstance(value, date):
        return value
    if isinstance(value, str):
        text = value.strip()
        if not text:
            return None
        if not _ISO_DATE_RE.fullmatch(text):
            raise ValueError("invoice_date must be an ISO date YYYY-MM-DD or null")
        try:
            return date.fromisoformat(text)
        except ValueError as exc:
            raise ValueError("invoice_date must be a real calendar date") from exc
    raise ValueError("invoice_date must be an ISO date YYYY-MM-DD or null")


def _parse_currency(value: Any) -> str | None:
    if value is None:
        return None
    if not isinstance(value, str):
        raise ValueError("currency must be an ISO 4217 code or null")
    code = value.strip().upper()
    if not code:
        return None
    if code not in ISO_4217_CODES:
        raise ValueError("currency must be a known ISO 4217 code or null")
    return code


class ExtractedInvoice(BaseModel):
    """Structured invoice fields. Unknown values stay null."""

    model_config = ConfigDict(extra="forbid")

    document_type: Literal["invoice"]
    vendor: str | None = None
    invoice_number: str | None = None
    invoice_date: date | None = None
    total: Decimal | None = None
    currency: str | None = None

    @field_validator("vendor", mode="before")
    @classmethod
    def clean_vendor(cls, value: Any) -> str | None:
        return _clean_optional_text(value, "vendor")

    @field_validator("invoice_number", mode="before")
    @classmethod
    def clean_invoice_number(cls, value: Any) -> str | None:
        return _clean_optional_text(value, "invoice_number")

    @field_validator("invoice_date", mode="before")
    @classmethod
    def clean_invoice_date(cls, value: Any) -> date | None:
        return _parse_invoice_date(value)

    @field_validator("total", mode="before")
    @classmethod
    def clean_total(cls, value: Any) -> Decimal | None:
        return _parse_total(value)

    @field_validator("currency", mode="before")
    @classmethod
    def clean_currency(cls, value: Any) -> str | None:
        return _parse_currency(value)


class TeacherInvoicePayload(BaseModel):
    """Structured output requested from the teacher model.

    Amounts and dates stay strings so the provider schema remains inside the
    supported JSON-schema subset. Domain validation happens afterwards.
    """

    model_config = ConfigDict(extra="forbid")

    document_type: Literal["invoice"] = Field(
        description="Use 'invoice' when the document is an invoice."
    )
    vendor: str | None = Field(
        description="Supplier name exactly as written. Null when the document does not state it."
    )
    invoice_number: str | None = Field(
        description="Invoice or reference number exactly as written. Null when absent."
    )
    invoice_date: str | None = Field(
        description="Explicit invoice issue date normalized to YYYY-MM-DD. Null when absent."
    )
    total: str | None = Field(
        description="Explicit invoice total as a nonnegative decimal string. Null when absent."
    )
    currency: str | None = Field(
        description="ISO 4217 code explicitly written in the document. Null when absent."
    )


class DatasetRecord(BaseModel):
    """One labeled invoice. The source text is stored separately from the labels."""

    model_config = ConfigDict(extra="forbid")

    document_id: str = Field(
        min_length=1,
        max_length=64,
        pattern=r"^[A-Za-z0-9][A-Za-z0-9_-]*$",
    )
    document_text: str = Field(min_length=1)
    ground_truth: ExtractedInvoice
    template_group: str = Field(min_length=1, max_length=64, pattern=r"^[a-z0-9_]+$")

    @field_validator("document_text")
    @classmethod
    def document_text_is_present(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("document_text must not be blank")
        return value


class TeacherPrediction(BaseModel):
    """One teacher call. Failures are data, not exceptions, so a run can finish."""

    model_config = ConfigDict(extra="forbid")

    document_id: str
    model: str
    extraction: ExtractedInvoice | None = None
    raw_output: dict[str, Any] | None = None
    schema_valid: bool
    api_error: str | None = None
    validation_error: str | None = None
    refusal: str | None = None
    latency_ms: float = Field(ge=0)
    input_tokens: int = Field(ge=0, default=0)
    output_tokens: int = Field(ge=0, default=0)


def extracted_invoice_from_payload(payload: BaseModel) -> ExtractedInvoice:
    """Validate a teacher payload, or a domain model, into ``ExtractedInvoice``."""
    if isinstance(payload, ExtractedInvoice):
        return payload
    return ExtractedInvoice.model_validate(payload.model_dump())
