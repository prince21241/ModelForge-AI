"""Schema validation for invoices, teacher payloads, and dataset records."""

from datetime import date, datetime
from decimal import Decimal

import pytest
from pydantic import ValidationError

from src.schemas import (
    DatasetRecord,
    ExtractedInvoice,
    TeacherInvoicePayload,
    extracted_invoice_from_payload,
)


def test_full_invoice_normalizes_currency_and_amount() -> None:
    invoice = ExtractedInvoice(
        document_type="invoice",
        vendor="  Cedar   Pine Analytics  ",
        invoice_number=" INV-ABC123 ",
        invoice_date="2024-03-01",
        total="1,234.50",
        currency="eur",
    )
    assert invoice.vendor == "Cedar Pine Analytics"
    assert invoice.invoice_number == "INV-ABC123"
    assert invoice.invoice_date == date(2024, 3, 1)
    assert invoice.total == Decimal("1234.50")
    assert invoice.currency == "EUR"


def test_absent_fields_stay_null() -> None:
    invoice = ExtractedInvoice(
        document_type="invoice",
        vendor=" ",
        invoice_number="",
        invoice_date="  ",
        total="",
        currency=" ",
    )
    assert invoice.vendor is None
    assert invoice.invoice_number is None
    assert invoice.invoice_date is None
    assert invoice.total is None
    assert invoice.currency is None


def test_datetime_invoice_date_keeps_the_calendar_day() -> None:
    invoice = ExtractedInvoice(
        document_type="invoice",
        invoice_date=datetime(2024, 5, 6, 15, 30),
    )
    assert invoice.invoice_date == date(2024, 5, 6)


@pytest.mark.parametrize(
    "value",
    ["-1", "-0.01", "$10.00", "10.00 USD", "not-a-number", True, "10.1234567"],
)
def test_invalid_totals_are_rejected(value: object) -> None:
    with pytest.raises(ValidationError):
        ExtractedInvoice(document_type="invoice", total=value)


@pytest.mark.parametrize("value", ["2024-02-31", "03/01/2024", "March 1, 2024", "USD"])
def test_invalid_dates_are_rejected(value: str) -> None:
    with pytest.raises(ValidationError):
        ExtractedInvoice(document_type="invoice", invoice_date=value)


def test_unknown_currency_and_document_type_are_rejected() -> None:
    with pytest.raises(ValidationError):
        ExtractedInvoice(document_type="invoice", currency="US")
    with pytest.raises(ValidationError):
        ExtractedInvoice(document_type="receipt")


def test_extra_fields_are_rejected() -> None:
    with pytest.raises(ValidationError):
        ExtractedInvoice.model_validate({"document_type": "invoice", "notes": "invented"})


def test_zero_total_is_allowed() -> None:
    assert ExtractedInvoice(document_type="invoice", total="0").total == Decimal("0")
    assert ExtractedInvoice(document_type="invoice", total=Decimal("0.00")).total == Decimal("0.00")


def test_long_vendor_is_rejected() -> None:
    with pytest.raises(ValidationError):
        ExtractedInvoice(document_type="invoice", vendor="A" * 201)


def test_json_roundtrip_preserves_decimal_and_date() -> None:
    invoice = ExtractedInvoice(
        document_type="invoice",
        vendor="Kindred Paper Works",
        invoice_number="INV-Q7K2LM",
        invoice_date=date(2024, 1, 2),
        total=Decimal("10.50"),
        currency="USD",
    )
    restored = ExtractedInvoice.model_validate_json(invoice.model_dump_json())
    assert restored == invoice
    assert restored.total == Decimal("10.50")
    assert restored.invoice_date == date(2024, 1, 2)


def test_dataset_record_preserves_original_text() -> None:
    text = "  INVOICE\n  keep   the source spacing\n"
    record = DatasetRecord(
        document_id="inv-synth-001",
        document_text=text,
        ground_truth=ExtractedInvoice(document_type="invoice"),
        template_group="classic_header",
    )
    assert record.document_text == text
    restored = DatasetRecord.model_validate_json(record.model_dump_json())
    assert restored.document_text == text
    assert restored.ground_truth.document_type == "invoice"


def test_blank_document_text_is_rejected() -> None:
    with pytest.raises(ValidationError):
        DatasetRecord(
            document_id="inv-synth-001",
            document_text="   ",
            ground_truth=ExtractedInvoice(document_type="invoice"),
            template_group="classic_header",
        )


def test_teacher_payload_converts_into_the_domain_model() -> None:
    payload = TeacherInvoicePayload(
        document_type="invoice",
        vendor="  ",
        invoice_number=None,
        invoice_date="2024-03-01",
        total="1,234.50",
        currency="gbp",
    )
    extracted = extracted_invoice_from_payload(payload)
    assert extracted.vendor is None
    assert extracted.invoice_date == date(2024, 3, 1)
    assert extracted.total == Decimal("1234.50")
    assert extracted.currency == "GBP"


def test_invalid_teacher_payload_raises_validation_error() -> None:
    payload = TeacherInvoicePayload(
        document_type="invoice",
        vendor=None,
        invoice_number=None,
        invoice_date=None,
        total="nope",
        currency=None,
    )
    with pytest.raises(ValidationError):
        extracted_invoice_from_payload(payload)
