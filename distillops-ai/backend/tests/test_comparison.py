"""Live comparison tests. The OpenAI client is never constructed."""

from __future__ import annotations

import io
import threading
from decimal import Decimal
from pathlib import Path
from types import SimpleNamespace

import fitz
import pytest
from fastapi.testclient import TestClient
from PIL import Image

from backend.app.core.config import STUDENT_CHECKPOINT
from backend.app.main import app
from backend.app.services.student_service import StudentService
from backend.app.services.teacher_budget import TeacherSpendGate
from src.config import load_settings
from src.labeling import StudentFields
from src.schemas import TeacherInvoicePayload
from src.student_eval import StudentGeneration
from src.student_infer import StudentInferenceError
from src.teacher import TeacherClient

INVOICE = "INVOICE\nVendor: Northwind Desk\nInvoice number: INV-100\nDate: 2024-01-15\nTotal: 42.00 USD\n"
SECRET = "sk-test-secret"
ENV_NAMES = (
    "DISTILLOPS_TEACHER_COMPARE_ENABLED",
    "DISTILLOPS_PUBLIC_DEPLOYMENT",
    "DISTILLOPS_TEACHER_COMPARE_MAX_REQUESTS",
    "DISTILLOPS_TEACHER_COMPARE_BUDGET_USD",
    "DISTILLOPS_TEACHER_COMPARE_RESERVE_USD",
    "OPENAI_API_KEY",
    "TEACHER_MODEL",
    "TEACHER_TEMPERATURE",
    "TEACHER_INPUT_USD_PER_MILLION_TOKENS",
    "TEACHER_OUTPUT_USD_PER_MILLION_TOKENS",
    "STUDENT_USD_PER_1000_INFERENCES",
)


class Scripted:
    def __init__(self, responses: list[object]) -> None:
        self.responses = list(responses)
        self.calls: list[dict[str, object]] = []
        self.chat = SimpleNamespace(completions=self)

    def parse(self, **kwargs: object) -> object:
        self.calls.append(kwargs)
        item = self.responses.pop(0)
        if isinstance(item, Exception):
            raise item
        return item


@pytest.fixture
def client():
    with TestClient(app) as test_client:
        yield test_client


@pytest.fixture(autouse=True)
def _isolated(monkeypatch: pytest.MonkeyPatch) -> None:
    for name in ENV_NAMES:
        monkeypatch.delenv(name, raising=False)

    def boom(_settings: object) -> None:
        raise AssertionError("A test constructed the OpenAI client.")

    monkeypatch.setattr("src.teacher.build_openai_client", boom)


def _enable(monkeypatch: pytest.MonkeyPatch, **extra: str) -> None:
    monkeypatch.setenv("DISTILLOPS_TEACHER_COMPARE_ENABLED", "true")
    monkeypatch.setenv("OPENAI_API_KEY", SECRET)
    monkeypatch.setenv("TEACHER_MODEL", "gpt-4o-mini")
    monkeypatch.setenv("TEACHER_TEMPERATURE", "0")
    for key, value in extra.items():
        monkeypatch.setenv(key, value)


def _generation(**overrides: object) -> StudentGeneration:
    payload: dict[str, object] = {
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
    payload.update(overrides)
    return StudentGeneration.model_validate(payload)


def _student(generator) -> StudentService:
    def loader(_path: Path, _device: object) -> tuple[object, object]:
        return object(), object()

    return StudentService(checkpoint=STUDENT_CHECKPOINT, generator=generator, loader=loader)


def _payload(**overrides: object) -> TeacherInvoicePayload:
    body: dict[str, object] = {
        "document_type": "invoice",
        "vendor": "Northwind Desk",
        "invoice_number": "INV-100",
        "invoice_date": "2024-01-15",
        "total": "42.00",
        "currency": "USD",
    }
    body.update(overrides)
    return TeacherInvoicePayload.model_validate(body)


def _completion(payload: TeacherInvoicePayload, *, prompt_tokens: int = 1000, completion_tokens: int = 40):
    message = SimpleNamespace(parsed=payload, refusal=None)
    return SimpleNamespace(
        model="gpt-4o-mini-2024-07-18",
        choices=[SimpleNamespace(message=message)],
        usage={"prompt_tokens": prompt_tokens, "completion_tokens": completion_tokens},
    )


def _attach(client: TestClient, generator, script: Scripted) -> Scripted:
    client.app.state.student_service = _student(generator)
    client.app.state.teacher_client = TeacherClient(load_settings(use_dotenv=False), client=script)
    return script


def _post(client: TestClient, text: str = INVOICE, approved: bool = True):
    return client.post("/api/compare", json={"document_text": text, "teacher_approved": approved})


def test_status_is_off_by_default_and_does_not_call_openai(client: TestClient) -> None:
    response = client.get("/api/compare/status")
    assert response.status_code == 200
    body = response.json()
    assert body["feature_enabled"] is False
    assert body["available"] is False
    assert body["api_key_configured"] is False
    assert "does not call the teacher" in body["detail"]
    assert SECRET not in response.text


def test_missing_approval_does_not_call_the_teacher(client: TestClient, monkeypatch: pytest.MonkeyPatch) -> None:
    _enable(monkeypatch)
    script = _attach(client, lambda *_args, **_kwargs: _generation(), Scripted([_completion(_payload())]))
    response = _post(client, approved=False)
    assert response.status_code == 403
    assert "No teacher request was sent" in response.json()["detail"]
    assert script.calls == []


def test_disabled_feature_rejects_an_approved_request(client: TestClient, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", SECRET)
    monkeypatch.setenv("TEACHER_MODEL", "gpt-4o-mini")
    script = _attach(client, lambda *_args, **_kwargs: _generation(), Scripted([_completion(_payload())]))
    response = _post(client)
    assert response.status_code == 403
    assert "Live teacher comparison is off" in response.json()["detail"]
    assert script.calls == []


def test_missing_api_key_rejects_before_a_teacher_call(client: TestClient, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("DISTILLOPS_TEACHER_COMPARE_ENABLED", "true")
    monkeypatch.setenv("TEACHER_MODEL", "gpt-4o-mini")
    script = _attach(client, lambda *_args, **_kwargs: _generation(), Scripted([_completion(_payload())]))
    response = _post(client)
    assert response.status_code == 503
    assert response.json()["detail"] == "OPENAI_API_KEY is not set. No teacher request was sent."
    assert script.calls == []


def test_public_deployment_fails_closed(client: TestClient, monkeypatch: pytest.MonkeyPatch) -> None:
    _enable(monkeypatch, DISTILLOPS_PUBLIC_DEPLOYMENT="true")
    script = _attach(client, lambda *_args, **_kwargs: _generation(), Scripted([_completion(_payload())]))
    response = _post(client)
    assert response.status_code == 403
    assert "public deployment" in response.json()["detail"]
    assert script.calls == []


def test_request_limit_blocks_the_second_call(client: TestClient, monkeypatch: pytest.MonkeyPatch) -> None:
    _enable(monkeypatch, DISTILLOPS_TEACHER_COMPARE_MAX_REQUESTS="1")
    script = _attach(client, lambda *_args, **_kwargs: _generation(), Scripted([_completion(_payload()), _completion(_payload())]))
    assert _post(client).status_code == 200
    blocked = _post(client)
    assert blocked.status_code == 429
    assert "request limit" in blocked.json()["detail"]
    assert len(script.calls) == 1


def test_spending_limit_blocks_a_call_when_the_reserve_does_not_fit(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    _enable(
        monkeypatch,
        DISTILLOPS_TEACHER_COMPARE_BUDGET_USD="0.01",
        DISTILLOPS_TEACHER_COMPARE_RESERVE_USD="0.02",
        TEACHER_INPUT_USD_PER_MILLION_TOKENS="3",
        TEACHER_OUTPUT_USD_PER_MILLION_TOKENS="6",
    )
    script = _attach(client, lambda *_args, **_kwargs: _generation(), Scripted([_completion(_payload())]))
    response = _post(client)
    assert response.status_code == 429
    assert "spending limit" in response.json()["detail"]
    assert script.calls == []


def test_a_budget_without_prices_fails_closed(client: TestClient, monkeypatch: pytest.MonkeyPatch) -> None:
    _enable(monkeypatch, DISTILLOPS_TEACHER_COMPARE_BUDGET_USD="1")
    script = _attach(client, lambda *_args, **_kwargs: _generation(), Scripted([_completion(_payload())]))
    response = _post(client)
    assert response.status_code == 429
    assert "token prices are not configured" in response.json()["detail"]
    assert script.calls == []


def test_concurrent_reservations_cannot_share_one_slot() -> None:
    gate = TeacherSpendGate()
    results: list[str] = []
    lock = threading.Lock()

    def attempt() -> None:
        try:
            gate.try_reserve(max_requests=1, budget=Decimal("0.05"), reserve=Decimal("0.05"))
            outcome = "ok"
        except Exception:
            outcome = "no"
        with lock:
            results.append(outcome)

    threads = [threading.Thread(target=attempt) for _ in range(8)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    assert results.count("ok") == 1
    assert results.count("no") == 7


def test_both_models_read_the_same_text_and_report_agreement(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    _enable(
        monkeypatch,
        TEACHER_INPUT_USD_PER_MILLION_TOKENS="3",
        TEACHER_OUTPUT_USD_PER_MILLION_TOKENS="6",
        STUDENT_USD_PER_1000_INFERENCES="2",
    )
    seen: dict[str, str] = {}

    def generator(_model, _tokenizer, text: str, **_kwargs: object) -> StudentGeneration:
        seen["student"] = text
        return _generation()

    script = _attach(client, generator, Scripted([_completion(_payload(vendor="Cedar Pine"), prompt_tokens=1000, completion_tokens=500)]))
    response = _post(client)
    assert response.status_code == 200
    body = response.json()
    assert seen["student"] == INVOICE
    user_message = script.calls[0]["messages"][1]["content"]
    assert seen["student"] in user_message
    assert script.calls[0]["temperature"] == 0
    assert script.calls[0]["model"] == "gpt-4o-mini"
    assert body["teacher"]["model"] == "gpt-4o-mini-2024-07-18"
    assert body["student"]["model"] == "google/flan-t5-small"
    assert body["teacher"]["fields"]["vendor"] == "Cedar Pine"
    assert body["student"]["fields"]["vendor"] == "Northwind Desk"
    vendor = next(row for row in body["fields"] if row["field"] == "vendor")
    assert vendor["status"] == "disagree"
    assert next(row["status"] for row in body["fields"] if row["field"] == "date") == "agree"
    assert body["agreement_count"] == 4
    assert body["fields_compared"] == 5
    assert body["partial"] is False
    assert body["teacher"]["input_tokens"] == 1000
    assert body["teacher"]["output_tokens"] == 500
    assert body["teacher"]["estimated_cost_usd"] == pytest.approx(0.006)
    assert body["student"]["input_tokens"] is None
    assert body["student"]["estimated_cost_usd"] == pytest.approx(0.002)
    assert "model disagreement" in body["warnings"][0]
    assert SECRET not in response.text


def test_missing_prices_leave_cost_null(client: TestClient, monkeypatch: pytest.MonkeyPatch) -> None:
    _enable(monkeypatch)
    _attach(client, lambda *_args, **_kwargs: _generation(), Scripted([_completion(_payload())]))
    body = _post(client).json()
    assert body["teacher"]["pricing_configured"] is False
    assert body["teacher"]["estimated_cost_usd"] is None
    assert body["student"]["pricing_configured"] is False
    assert body["student"]["estimated_cost_usd"] is None
    assert body["teacher"]["input_tokens"] == 1000


def test_teacher_failure_still_returns_the_student(client: TestClient, monkeypatch: pytest.MonkeyPatch) -> None:
    _enable(monkeypatch)
    script = _attach(client, lambda *_args, **_kwargs: _generation(), Scripted([RuntimeError("provider exploded")]))
    response = _post(client)
    assert response.status_code == 200
    body = response.json()
    assert body["partial"] is True
    assert body["teacher"]["succeeded"] is False
    assert body["teacher"]["fields"] is None
    assert body["teacher"]["input_tokens"] is None
    assert body["student"]["fields"]["vendor"] == "Northwind Desk"
    assert body["agreement_count"] is None
    assert "provider exploded" not in response.text
    assert len(script.calls) == 1


def test_student_failure_still_returns_the_teacher(client: TestClient, monkeypatch: pytest.MonkeyPatch) -> None:
    _enable(monkeypatch)

    def generator(*_args, **_kwargs):
        raise StudentInferenceError("checkpoint stack trace")

    response_script = _attach(client, generator, Scripted([_completion(_payload())]))
    response = _post(client)
    assert response.status_code == 200
    body = response.json()
    assert body["partial"] is True
    assert body["student"]["succeeded"] is False
    assert body["student"]["error"] == "Student inference failed."
    assert body["teacher"]["fields"]["currency"] == "USD"
    assert "checkpoint stack" not in response.text
    assert len(response_script.calls) == 1


def test_malformed_teacher_json_is_not_repaired(client: TestClient, monkeypatch: pytest.MonkeyPatch) -> None:
    _enable(monkeypatch)
    _attach(client, lambda *_args, **_kwargs: _generation(), Scripted([_completion(_payload(currency="BOGUS"))]))
    body = _post(client).json()
    assert body["teacher"]["succeeded"] is False
    assert body["teacher"]["fields"] is None
    assert body["teacher"]["error"]
    assert body["student"]["fields"]["currency"] == "USD"
    assert body["partial"] is True
    assert body["agreement_count"] is None


def test_blank_and_oversized_text_do_not_call_the_teacher(client: TestClient, monkeypatch: pytest.MonkeyPatch) -> None:
    _enable(monkeypatch)
    script = _attach(client, lambda *_args, **_kwargs: _generation(), Scripted([_completion(_payload())]))
    blank = _post(client, "   ")
    huge = _post(client, "A" * 20_001)
    assert blank.status_code == 422
    assert huge.status_code == 413
    assert script.calls == []


def test_text_pdf_and_png_send_one_shared_string(client: TestClient, monkeypatch: pytest.MonkeyPatch) -> None:
    _enable(monkeypatch)
    seen: list[str] = []

    def generator(_model, _tokenizer, text: str, **_kwargs: object) -> StudentGeneration:
        seen.append(text)
        return _generation()

    def fake_ocr(_image: Image.Image, *, timeout: float) -> str:
        return INVOICE

    monkeypatch.setattr("backend.app.services.documents.ocr_service.read_image", fake_ocr)
    script = _attach(client, generator, Scripted([_completion(_payload()), _completion(_payload())]))
    pdf = client.post(
        "/api/documents/compare",
        files={"file": ("invoice.pdf", _text_pdf(), "application/pdf")},
        data={"teacher_approved": "true"},
    )
    png = client.post(
        "/api/documents/compare",
        files={"file": ("invoice.png", _png(), "image/png")},
        data={"teacher_approved": "true"},
    )
    assert pdf.status_code == 200
    assert png.status_code == 200
    assert pdf.json()["extraction_method"] == "pdf_text"
    assert png.json()["extraction_method"] == "image_ocr"
    assert seen[0] in script.calls[0]["messages"][1]["content"]
    assert seen[1] in script.calls[1]["messages"][1]["content"]
    assert seen[1] == INVOICE.strip()
    assert "read with OCR" in " ".join(png.json()["warnings"])


def test_invalid_upload_does_not_call_the_teacher(client: TestClient, monkeypatch: pytest.MonkeyPatch) -> None:
    _enable(monkeypatch)
    script = _attach(client, lambda *_args, **_kwargs: _generation(), Scripted([_completion(_payload())]))
    response = client.post(
        "/api/documents/compare",
        files={"file": ("notes.txt", b"hello", "text/plain")},
        data={"teacher_approved": "true"},
    )
    assert response.status_code == 415
    assert script.calls == []


def _text_pdf() -> bytes:
    document = fitz.open()
    page = document.new_page()
    page.insert_textbox(fitz.Rect(72, 72, 540, 700), INVOICE)
    payload = document.tobytes()
    document.close()
    return payload


def _png() -> bytes:
    buffer = io.BytesIO()
    Image.new("RGB", (80, 40), "white").save(buffer, format="PNG")
    return buffer.getvalue()
