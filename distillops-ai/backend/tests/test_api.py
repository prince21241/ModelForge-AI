"""API tests. Mocked calls do not load weights. One smoke test uses the frozen checkpoint."""

from __future__ import annotations

import threading
import time
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from backend.app.core.config import STUDENT_CHECKPOINT
from backend.app.main import app
from backend.app.services.student_service import StudentService
from src.labeling import StudentFields
from src.student_data import StudentDataError
from src.student_eval import StudentGeneration
from src.student_infer import StudentInferenceError

SAMPLE = "INVOICE\nVendor: Northwind Desk\nInvoice number: INV-100\nDate: 2024-01-15\nTotal: 42.00 USD\n"


def _generation(**overrides: object) -> StudentGeneration:
    fields = StudentFields(
        vendor="Northwind Desk",
        invoice_number="INV-100",
        date="2024-01-15",
        total="42.00",
        currency="USD",
    )
    payload: dict[str, object] = {
        "document_id": "api-request",
        "raw_text": '{"vendor":"Northwind Desk","invoice_number":"INV-100","date":"2024-01-15","total":"42.00","currency":"USD"}',
        "json_valid": True,
        "fields": fields,
        "validation_error": None,
        "latency_ms": 15.5,
    }
    payload.update(overrides)
    return StudentGeneration.model_validate(payload)


def _service(generator, loader=None) -> StudentService:
    def default_loader(_path: Path, _device: object) -> tuple[object, object]:
        return object(), object()

    return StudentService(checkpoint=STUDENT_CHECKPOINT, generator=generator, loader=loader or default_loader)


@pytest.fixture
def client():
    with TestClient(app) as test_client:
        yield test_client


def test_health_does_not_claim_the_model_is_loaded(client: TestClient) -> None:
    response = client.get("/api/health")
    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "ok"
    assert body["service"] == "distillops"
    assert body["student_loaded"] is False
    assert "epoch-30" in body["student_checkpoint"]


def test_models_lists_the_student_and_reports_loaded_false(client: TestClient) -> None:
    response = client.get("/api/models")
    assert response.status_code == 200
    models = response.json()["models"]
    assert len(models) == 1
    assert models[0]["name"] == "student"
    assert models[0]["identifier"] == "google/flan-t5-small"
    assert models[0]["available"] is True
    assert models[0]["loaded"] is False


def test_extract_returns_the_schema_from_mocked_inference(client: TestClient) -> None:
    seen: dict[str, object] = {}

    def generator(_model, _tokenizer, text: str, **kwargs: object) -> StudentGeneration:
        seen["text"] = text
        seen["max_new_tokens"] = kwargs["max_new_tokens"]
        seen["max_source_length"] = kwargs["max_source_length"]
        return _generation()

    client.app.state.student_service = _service(generator)
    response = client.post("/api/extract", json={"document_text": SAMPLE, "model": "student"})
    assert response.status_code == 200
    body = response.json()
    assert body["model"] == "student"
    assert body["fields"] == {
        "vendor": "Northwind Desk",
        "invoice_number": "INV-100",
        "date": "2024-01-15",
        "total": "42.00",
        "currency": "USD",
    }
    assert body["latency_ms"] == 15.5
    assert body["requires_review"] is True
    assert "confidence" not in body
    assert any("not production-validated" in warning for warning in body["warnings"])
    assert seen["text"] == SAMPLE
    assert seen["max_new_tokens"] == 128
    assert seen["max_source_length"] == 512
    assert client.get("/api/models").json()["models"][0]["loaded"] is True


def test_empty_input_is_rejected_before_inference(client: TestClient) -> None:
    def generator(*_args: object, **_kwargs: object) -> StudentGeneration:
        raise AssertionError("inference ran")

    client.app.state.student_service = _service(generator)
    response = client.post("/api/extract", json={"document_text": "  \n", "model": "student"})
    assert response.status_code == 422
    assert response.json()["detail"] == "document_text must not be blank."
    assert "Traceback" not in response.text


def test_oversized_input_is_rejected(client: TestClient) -> None:
    def generator(*_args: object, **_kwargs: object) -> StudentGeneration:
        raise AssertionError("inference ran")

    client.app.state.student_service = _service(generator)
    response = client.post("/api/extract", json={"document_text": "A" * 20_001, "model": "student"})
    assert response.status_code == 413
    assert "not truncated" in response.json()["detail"]


def test_token_overflow_is_not_truncated(client: TestClient) -> None:
    def generator(*_args: object, **_kwargs: object) -> StudentGeneration:
        raise StudentDataError("api-request input is 600 tokens, above the limit of 512. The text was not truncated.")

    client.app.state.student_service = _service(generator)
    response = client.post("/api/extract", json={"document_text": SAMPLE, "model": "student"})
    assert response.status_code == 413
    assert "512" in response.json()["detail"]


def test_unsupported_model_does_not_call_the_student(client: TestClient) -> None:
    def generator(*_args: object, **_kwargs: object) -> StudentGeneration:
        raise AssertionError("inference ran")

    client.app.state.student_service = _service(generator)
    response = client.post("/api/extract", json={"document_text": SAMPLE, "model": "teacher"})
    assert response.status_code == 400
    assert "Unsupported model" in response.json()["detail"]


def test_malformed_model_output_is_not_repaired(client: TestClient) -> None:
    def generator(*_args: object, **_kwargs: object) -> StudentGeneration:
        return _generation(
            raw_text="not json",
            json_valid=False,
            fields=None,
            validation_error="Output is not valid JSON.",
            latency_ms=9,
        )

    client.app.state.student_service = _service(generator)
    response = client.post("/api/extract", json={"document_text": SAMPLE, "model": "student"})
    assert response.status_code == 200
    body = response.json()
    assert body["fields"] == {
        "vendor": "",
        "invoice_number": "",
        "date": "",
        "total": "",
        "currency": "",
    }
    assert body["requires_review"] is True
    assert any("not repaired" in warning for warning in body["warnings"])
    assert "Northwind" not in response.text


def test_model_loading_failure_returns_503(client: TestClient) -> None:
    def loader(_path: Path, _device: object) -> tuple[object, object]:
        raise RuntimeError("weights missing")

    client.app.state.student_service = _service(lambda *_a, **_k: _generation(), loader=loader)
    response = client.post("/api/extract", json={"document_text": SAMPLE, "model": "student"})
    assert response.status_code == 503
    assert response.json()["detail"] == "The student model could not be loaded."
    assert "weights missing" not in response.text
    assert "Traceback" not in response.text


def test_inference_failure_hides_the_exception(client: TestClient) -> None:
    def generator(*_args: object, **_kwargs: object) -> StudentGeneration:
        raise StudentInferenceError("decoder exploded sk-secret")

    client.app.state.student_service = _service(generator)
    response = client.post("/api/extract", json={"document_text": SAMPLE, "model": "student"})
    assert response.status_code == 500
    assert response.json()["detail"] == "Student inference failed."
    assert "sk-secret" not in response.text
    assert "Traceback" not in response.text


def test_suspicious_date_and_total_stay_unchanged(client: TestClient) -> None:
    fields = StudentFields(
        vendor="Northwind Desk",
        invoice_number="",
        date="18 Jul 2023",
        total="USD",
        currency="USD",
    )

    def generator(*_args: object, **_kwargs: object) -> StudentGeneration:
        return _generation(fields=fields, latency_ms=3)

    client.app.state.student_service = _service(generator)
    body = client.post("/api/extract", json={"document_text": SAMPLE, "model": "student"}).json()
    assert body["fields"]["date"] == "18 Jul 2023"
    assert body["fields"]["total"] == "USD"
    assert body["fields"]["invoice_number"] == ""
    assert any("ISO" in warning for warning in body["warnings"])
    assert any("total" in warning for warning in body["warnings"])
    assert any("invoice_number is empty" in warning for warning in body["warnings"])


def test_concurrent_extract_calls_are_serialized() -> None:
    active = {"current": 0, "max": 0}

    def generator(*_args: object, **_kwargs: object) -> StudentGeneration:
        active["current"] += 1
        active["max"] = max(active["max"], active["current"])
        time.sleep(0.05)
        active["current"] -= 1
        return _generation()

    service = _service(generator)
    errors: list[BaseException] = []

    def call() -> None:
        try:
            service.extract(SAMPLE)
        except BaseException as exc:  # pragma: no cover - surfaced below
            errors.append(exc)

    threads = [threading.Thread(target=call) for _ in range(4)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    assert errors == []
    assert active["max"] == 1


def test_cors_allowlist_includes_the_vite_dev_server(client: TestClient) -> None:
    allowed = client.get("/api/health", headers={"Origin": "http://127.0.0.1:5173"})
    assert allowed.headers["access-control-allow-origin"] == "http://127.0.0.1:5173"
    blocked = client.get("/api/health", headers={"Origin": "https://example.invalid"})
    assert blocked.headers.get("access-control-allow-origin") is None


def test_real_checkpoint_smoke() -> None:
    if not STUDENT_CHECKPOINT.is_dir():
        pytest.skip("frozen checkpoint is not on this machine")
    with TestClient(app) as client:
        started = client.get("/api/health")
        assert started.json()["student_loaded"] is False
        response = client.post("/api/extract", json={"document_text": SAMPLE, "model": "student"})
        assert response.status_code == 200
        body = response.json()
        assert body["model"] == "student"
        assert body["latency_ms"] > 0
        assert body["requires_review"] is True
        assert set(body["fields"]) == {"vendor", "invoice_number", "date", "total", "currency"}
        assert all(isinstance(value, str) for value in body["fields"].values())
        assert client.get("/api/models").json()["models"][0]["loaded"] is True
