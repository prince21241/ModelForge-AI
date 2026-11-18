"""Upload, PDF, and OCR tests. Inference and OCR are mocked except one PDF smoke test."""

from __future__ import annotations

import io
import tempfile
import threading
import time
from pathlib import Path

import fitz
import pytest
import pytesseract
from fastapi.testclient import TestClient
from PIL import Image

from backend.app.core.config import STUDENT_CHECKPOINT
from backend.app.core.errors import CorruptDocumentError
from backend.app.main import app
from backend.app.services.documents import parse_upload
from backend.app.services.ocr import read_image
from backend.app.services.student_service import StudentService
from src.labeling import StudentFields
from src.student_data import StudentDataError
from src.student_eval import StudentGeneration

INVOICE = "INVOICE\nVendor: Northwind Desk\nInvoice number: INV-100\nDate: 2024-01-15\nTotal: 42.00 USD\n"


def _generation() -> StudentGeneration:
    return StudentGeneration.model_validate(
        {
            "document_id": "api-request",
            "raw_text": "{}",
            "json_valid": True,
            "fields": StudentFields(
                vendor="Northwind Desk",
                invoice_number="INV-100",
                date="2024-01-15",
                total="42.00",
                currency="USD",
            ),
            "validation_error": None,
            "latency_ms": 15.5,
        }
    )


def _service(generator) -> StudentService:
    def loader(_path: Path, _device: object) -> tuple[object, object]:
        return object(), object()

    return StudentService(checkpoint=STUDENT_CHECKPOINT, generator=generator, loader=loader)


def _text_pdf(text: str = INVOICE, pages: int = 1) -> bytes:
    document = fitz.open()
    for index in range(pages):
        page = document.new_page()
        page.insert_textbox(fitz.Rect(72, 72, 540, 700), text if index == 0 else f"Continuation page {index}")
    payload = document.tobytes()
    document.close()
    return payload


def _image_pdf() -> bytes:
    buffer = io.BytesIO()
    Image.new("RGB", (240, 140), "white").save(buffer, format="PNG")
    document = fitz.open()
    page = document.new_page(width=240, height=140)
    page.insert_image(page.rect, stream=buffer.getvalue())
    payload = document.tobytes()
    document.close()
    return payload


def _mixed_pdf() -> bytes:
    document = fitz.open()
    first = document.new_page()
    first.insert_text((72, 72), "PAGEONEVENDOR Northwind")
    buffer = io.BytesIO()
    Image.new("RGB", (240, 140), "white").save(buffer, format="PNG")
    second = document.new_page(width=240, height=140)
    second.insert_image(second.rect, stream=buffer.getvalue())
    payload = document.tobytes()
    document.close()
    return payload


def _encrypted_pdf() -> bytes:
    document = fitz.open()
    page = document.new_page()
    page.insert_text((72, 72), "Vendor: Hidden Desk")
    payload = document.tobytes(encryption=fitz.PDF_ENCRYPT_AES_256, user_pw="secret", owner_pw="owner")
    document.close()
    return payload


def _png(width: int = 120, height: int = 80) -> bytes:
    buffer = io.BytesIO()
    Image.new("RGB", (width, height), "white").save(buffer, format="PNG")
    return buffer.getvalue()


def _jpeg() -> bytes:
    buffer = io.BytesIO()
    Image.new("RGB", (120, 80), "white").save(buffer, format="JPEG")
    return buffer.getvalue()


def _leftovers() -> set[Path]:
    return set(Path(tempfile.gettempdir()).glob("distillops-upload-*"))


@pytest.fixture
def client():
    with TestClient(app) as test_client:
        yield test_client


def _post(client: TestClient, name: str, payload: bytes, mime: str, model: str | None = "student"):
    files = {"file": (name, payload, mime)}
    data = None if model is None else {"model": model}
    return client.post("/api/documents/extract", files=files, data=data)


def test_text_pdf_uses_native_text_and_the_student(client: TestClient, monkeypatch: pytest.MonkeyPatch) -> None:
    seen: dict[str, str] = {}

    def generator(_model, _tokenizer, text: str, **_kwargs: object) -> StudentGeneration:
        seen["text"] = text
        return _generation()

    def fail_ocr(*_args: object, **_kwargs: object) -> str:
        raise AssertionError("ocr ran")

    monkeypatch.setattr("backend.app.services.documents.ocr_service.read_image", fail_ocr)
    client.app.state.student_service = _service(generator)
    response = _post(client, "invoice.pdf", _text_pdf(), "application/pdf", model=None)
    assert response.status_code == 200
    body = response.json()
    assert body["filename"] == "invoice.pdf"
    assert body["content_type"] == "application/pdf"
    assert body["page_count"] == 1
    assert body["extraction_method"] == "pdf_text"
    assert "Northwind Desk" in body["extracted_text"]
    assert "Northwind Desk" in seen["text"]
    assert body["fields"]["vendor"] == "Northwind Desk"
    assert body["model"] == "student"
    assert body["latency_ms"] == 15.5
    assert body["model_latency_ms"] == 15.5
    assert body["document_latency_ms"] >= 0
    assert body["total_latency_ms"] >= body["document_latency_ms"]
    assert body["requires_review"] is True
    assert "confidence" not in body
    assert not any("read with OCR" in warning for warning in body["warnings"])
    assert "Traceback" not in response.text


def test_scanned_pdf_uses_mocked_ocr_once(client: TestClient, monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[str] = []

    def fake_ocr(_image: Image.Image, *, timeout: float) -> str:
        calls.append(str(timeout))
        return "PAGETWOVENDOR scanned"

    def generator(_model, _tokenizer, text: str, **_kwargs: object) -> StudentGeneration:
        calls.append(text)
        return _generation()

    monkeypatch.setattr("backend.app.services.documents.ocr_service.read_image", fake_ocr)
    client.app.state.student_service = _service(generator)
    response = _post(client, "scan.pdf", _image_pdf(), "application/pdf")
    assert response.status_code == 200
    body = response.json()
    assert body["extraction_method"] == "pdf_ocr"
    assert body["page_count"] == 1
    assert calls[-1] == "PAGETWOVENDOR scanned"
    assert any("read with OCR" in warning for warning in body["warnings"])


def test_mixed_pdf_keeps_page_order_without_duplicate_ocr(client: TestClient, monkeypatch: pytest.MonkeyPatch) -> None:
    calls = {"ocr": 0}

    def fake_ocr(_image: Image.Image, *, timeout: float) -> str:
        calls["ocr"] += 1
        return "PAGETWOVENDOR"

    def generator(_model, _tokenizer, text: str, **_kwargs: object) -> StudentGeneration:
        calls["text"] = text
        return _generation()

    monkeypatch.setattr("backend.app.services.documents.ocr_service.read_image", fake_ocr)
    client.app.state.student_service = _service(generator)
    response = _post(client, "mixed.pdf", _mixed_pdf(), "application/pdf")
    assert response.status_code == 200
    body = response.json()
    assert body["extraction_method"] == "pdf_mixed"
    assert body["page_count"] == 2
    assert calls["ocr"] == 1
    text = calls["text"]
    assert text.index("PAGEONEVENDOR") < text.index("PAGETWOVENDOR")
    assert text.count("PAGEONEVENDOR") == 1


def test_png_and_jpeg_use_ocr(client: TestClient, monkeypatch: pytest.MonkeyPatch) -> None:
    def fake_ocr(_image: Image.Image, *, timeout: float) -> str:
        return INVOICE

    monkeypatch.setattr("backend.app.services.documents.ocr_service.read_image", fake_ocr)
    client.app.state.student_service = _service(lambda *_args, **_kwargs: _generation())
    png = _post(client, "invoice.png", _png(), "image/png")
    jpeg = _post(client, "invoice.jpg", _jpeg(), "image/jpeg")
    assert png.status_code == 200
    assert jpeg.status_code == 200
    assert png.json()["extraction_method"] == "image_ocr"
    assert png.json()["content_type"] == "image/png"
    assert jpeg.json()["content_type"] == "image/jpeg"
    assert jpeg.json()["page_count"] == 1


def test_unsupported_extension_is_rejected(client: TestClient) -> None:
    response = _post(client, "notes.txt", b"hello", "text/plain")
    assert response.status_code == 415
    assert "Unsupported file type" in response.json()["detail"]


def test_signature_mismatch_is_rejected(client: TestClient) -> None:
    response = _post(client, "photo.png", _text_pdf(), "image/png")
    assert response.status_code == 415
    assert response.json()["detail"] == "The file signature does not match its type."


def test_empty_file_is_rejected(client: TestClient) -> None:
    response = _post(client, "invoice.pdf", b"", "application/pdf")
    assert response.status_code == 422
    assert response.json()["detail"] == "The uploaded file is empty."


def test_oversized_upload_is_rejected(client: TestClient, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("backend.app.api.routes.documents.max_upload_bytes", lambda: 32)

    def generator(*_args: object, **_kwargs: object) -> StudentGeneration:
        raise AssertionError("inference ran")

    client.app.state.student_service = _service(generator)
    response = _post(client, "invoice.pdf", _text_pdf(), "application/pdf")
    assert response.status_code == 413
    assert "size limit" in response.json()["detail"]


def test_encrypted_pdf_is_rejected(client: TestClient) -> None:
    response = _post(client, "secret.pdf", _encrypted_pdf(), "application/pdf")
    assert response.status_code == 422
    assert response.json()["detail"] == "This PDF is encrypted and was not opened."
    assert "secret" not in response.text


def test_corrupt_pdf_and_image_are_rejected(client: TestClient) -> None:
    pdf = _post(client, "broken.pdf", b"%PDF-1.7\nnot a pdf", "application/pdf")
    png = _post(client, "broken.png", b"\x89PNG\r\n\x1a\nnot-an-image", "image/png")
    assert pdf.status_code == 422
    assert png.status_code == 422
    assert pdf.json()["detail"] == "The file is corrupt or is not a readable PDF or image."
    assert "Traceback" not in pdf.text
    assert "distillops-upload-" not in pdf.text


def test_page_limit_is_enforced(client: TestClient) -> None:
    response = _post(client, "long.pdf", _text_pdf(pages=11), "application/pdf")
    assert response.status_code == 413
    assert "page limit" in response.json()["detail"]


def test_missing_tesseract_is_reported(client: TestClient, monkeypatch: pytest.MonkeyPatch) -> None:
    def missing(*_args: object, **_kwargs: object) -> str:
        raise pytesseract.TesseractNotFoundError()

    monkeypatch.setattr("backend.app.services.ocr.pytesseract.image_to_string", missing)
    response = _post(client, "invoice.png", _png(), "image/png")
    assert response.status_code == 503
    assert "Tesseract" in response.json()["detail"]


def test_ocr_timeout_is_reported(client: TestClient, monkeypatch: pytest.MonkeyPatch) -> None:
    def timed_out(*_args: object, **_kwargs: object) -> str:
        raise RuntimeError("Tesseract process timeout")

    monkeypatch.setattr("backend.app.services.ocr.pytesseract.image_to_string", timed_out)
    response = _post(client, "invoice.png", _png(), "image/png")
    assert response.status_code == 408
    assert response.json()["detail"] == "OCR timed out before the document was read."


def test_empty_ocr_text_does_not_call_the_model(client: TestClient, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        "backend.app.services.documents.ocr_service.read_image",
        lambda *_args, **_kwargs: " \n",
    )

    def generator(*_args: object, **_kwargs: object) -> StudentGeneration:
        raise AssertionError("inference ran")

    client.app.state.student_service = _service(generator)
    response = _post(client, "invoice.png", _png(), "image/png")
    assert response.status_code == 422
    assert response.json()["detail"] == "No readable text was found in the document."


def test_overlong_extracted_text_is_not_truncated(client: TestClient) -> None:
    document = fitz.open()
    line = "INV-100 Northwind Desk "
    for _page in range(3):
        page = document.new_page(width=1200, height=4000)
        for line_number in range(320):
            page.insert_text((36, 20 + line_number * 12), line)
    payload = document.tobytes()
    document.close()

    def generator(*_args: object, **_kwargs: object) -> StudentGeneration:
        raise AssertionError("inference ran")

    client.app.state.student_service = _service(generator)
    response = _post(client, "huge.pdf", payload, "application/pdf")
    assert response.status_code == 413
    assert "not truncated" in response.json()["detail"]


def test_token_limit_is_rejected(client: TestClient, monkeypatch: pytest.MonkeyPatch) -> None:
    def fail_ocr(*_args: object, **_kwargs: object) -> str:
        raise AssertionError("ocr ran")

    monkeypatch.setattr("backend.app.services.documents.ocr_service.read_image", fail_ocr)

    def generator(*_args: object, **_kwargs: object) -> StudentGeneration:
        raise StudentDataError("api-request source is 600 tokens, above the limit of 512.")

    client.app.state.student_service = _service(generator)
    response = _post(client, "invoice.pdf", _text_pdf("Vendor: Short Desk"), "application/pdf")
    assert response.status_code == 413
    assert "512 source tokens" in response.json()["detail"]


def test_decompression_bomb_is_rejected(client: TestClient, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("backend.app.services.documents.max_image_pixels", lambda: 100)
    response = _post(client, "bomb.png", _png(200, 200), "image/png")
    assert response.status_code == 413
    assert "too large" in response.json()["detail"]


def test_temporary_files_are_removed(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        "backend.app.services.documents.ocr_service.read_image",
        lambda *_args, **_kwargs: "Vendor: Sample Desk",
    )
    before = _leftovers()
    parsed = parse_upload(_image_pdf(), r"..\secret\scan.pdf", "application/pdf")
    assert parsed.filename == "scan.pdf"
    with pytest.raises(CorruptDocumentError):
        parse_upload(b"%PDF-1.4\nbad", "broken.pdf", "application/pdf")
    assert _leftovers() <= before


def test_ocr_calls_do_not_overlap() -> None:
    active = 0
    peak = 0
    guard = threading.Lock()

    def slow(_image: object, **_kwargs: object) -> str:
        nonlocal active, peak
        with guard:
            active += 1
            peak = max(peak, active)
        time.sleep(0.15)
        with guard:
            active -= 1
        return "INVOICE 100"

    image = Image.new("RGB", (90, 90), "white")
    original = pytesseract.image_to_string
    pytesseract.image_to_string = slow
    try:
        threads = [threading.Thread(target=read_image, args=(image,), kwargs={"timeout": 5}) for _ in range(2)]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()
    finally:
        pytesseract.image_to_string = original
    assert peak == 1


def test_unsupported_model_does_not_parse(client: TestClient, monkeypatch: pytest.MonkeyPatch) -> None:
    def fail_parse(*_args: object, **_kwargs: object) -> None:
        raise AssertionError("parser ran")

    monkeypatch.setattr("backend.app.api.routes.documents.parse_upload", fail_parse)
    response = _post(client, "invoice.pdf", _text_pdf(), "application/pdf", model="teacher")
    assert response.status_code == 400


def test_logs_do_not_include_invoice_text(client: TestClient, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture) -> None:
    def fail_ocr(*_args: object, **_kwargs: object) -> str:
        raise AssertionError("ocr ran")

    monkeypatch.setattr("backend.app.services.documents.ocr_service.read_image", fail_ocr)
    client.app.state.student_service = _service(lambda *_args, **_kwargs: _generation())
    with caplog.at_level("DEBUG"):
        response = _post(client, "invoice.pdf", _text_pdf(), "application/pdf")
    assert response.status_code == 200
    assert "Northwind" not in caplog.text


def test_low_quality_ocr_adds_a_warning(client: TestClient, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("backend.app.services.documents.ocr_service.read_image", lambda *_a, **_k: "AB")
    client.app.state.student_service = _service(lambda *_args, **_kwargs: _generation())
    response = _post(client, "tiny.png", _png(40, 40), "image/png")
    assert response.status_code == 200
    warnings = response.json()["warnings"]
    assert any("very little text" in warning for warning in warnings)
    assert any("very small" in warning for warning in warnings)
    assert response.json()["fields"]["total"] == "42.00"


@pytest.mark.skipif(not STUDENT_CHECKPOINT.exists(), reason="frozen student checkpoint is not present")
def test_real_pdf_upload_smoke(client: TestClient) -> None:
    response = _post(client, "sample.pdf", _text_pdf(), "application/pdf")
    assert response.status_code == 200
    body = response.json()
    assert body["extraction_method"] == "pdf_text"
    assert "Northwind Desk" in body["extracted_text"]
    assert body["model"] == "student"
    assert body["requires_review"] is True
    assert body["latency_ms"] > 0
    for name in ("vendor", "invoice_number", "date", "total", "currency"):
        assert isinstance(body["fields"][name], str)
    assert "confidence" not in body
