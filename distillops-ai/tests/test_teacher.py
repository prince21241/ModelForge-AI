"""Teacher client behavior with a scripted API client. No network calls."""

import logging
from datetime import datetime, timezone
from decimal import Decimal
from types import SimpleNamespace

import pytest

from scripts.run_teacher_baseline import main as baseline_main
from scripts.run_teacher_baseline import run_baseline
from src.config import ConfigError, Settings, configured_pricing, load_settings
from src.dataset import build_dataset, write_dataset
from src.prompts import SYSTEM_PROMPT, build_messages
from src.schemas import TeacherInvoicePayload
from src.teacher import TeacherClient, TeacherConfigError, is_retryable, public_error

SECRET_KEY = "sk-super-secret-value"
SENTINEL = "SENTINEL_DOCUMENT_TEXT_should_not_log"


class ScriptedCompletions:
    def __init__(self, events: list[object]) -> None:
        self.events = list(events)
        self.calls: list[dict[object, object]] = []

    def parse(self, **kwargs: object) -> object:
        self.calls.append(kwargs)
        if not self.events:
            raise AssertionError("scripted client ran out of responses")
        event = self.events.pop(0)
        if isinstance(event, Exception):
            raise event
        return event


class ScriptedClient:
    def __init__(self, events: list[object]) -> None:
        self.completions = ScriptedCompletions(events)
        self.chat = SimpleNamespace(completions=self.completions)


class RetryableStatus(Exception):
    def __init__(self, message: str, status_code: int) -> None:
        super().__init__(message)
        self.status_code = status_code


class APITimeoutError(Exception):
    """Name matches a retryable provider timeout without importing the SDK."""


def _completion(payload: TeacherInvoicePayload, *, prompt_tokens: int = 11, completion_tokens: int = 7):
    message = SimpleNamespace(parsed=payload, refusal=None)
    return SimpleNamespace(
        model="response-model",
        choices=[SimpleNamespace(message=message)],
        usage={"prompt_tokens": prompt_tokens, "completion_tokens": completion_tokens},
    )


def _payload() -> TeacherInvoicePayload:
    return TeacherInvoicePayload(
        document_type="invoice",
        vendor="Cedar Pine Analytics",
        invoice_number="INV-Q7K2LM",
        invoice_date="2024-03-01",
        total="19.25",
        currency="USD",
    )


def _settings(**overrides: object) -> Settings:
    payload: dict[str, object] = {
        "openai_api_key": None,
        "teacher_model": "unit-test-model",
        "timeout_seconds": 12,
        "max_retries": 2,
        "retry_backoff_seconds": 0.5,
        "teacher_temperature": 0,
    }
    payload.update(overrides)
    return Settings.model_validate(payload)


def _isolate(monkeypatch: pytest.MonkeyPatch) -> None:
    for name in (
        "OPENAI_API_KEY",
        "TEACHER_MODEL",
        "TEACHER_TIMEOUT_SECONDS",
        "TEACHER_MAX_RETRIES",
        "TEACHER_RETRY_BACKOFF_SECONDS",
        "TEACHER_TEMPERATURE",
        "TEACHER_INPUT_USD_PER_MILLION_TOKENS",
        "TEACHER_OUTPUT_USD_PER_MILLION_TOKENS",
    ):
        monkeypatch.delenv(name, raising=False)


def test_prompt_forbids_inference_and_keeps_the_document_in_the_user_turn() -> None:
    lowered = SYSTEM_PROMPT.lower()
    assert "explicitly" in lowered
    assert "do not infer" in lowered
    assert "null" in lowered
    messages = build_messages(SENTINEL)
    assert messages[0]["role"] == "system"
    assert SENTINEL not in messages[0]["content"]
    assert SENTINEL in messages[1]["content"]


def test_successful_call_records_model_tokens_and_latency(caplog: pytest.LogCaptureFixture) -> None:
    client_double = ScriptedClient([_completion(_payload(), prompt_tokens=20, completion_tokens=5)])
    sleeps: list[float] = []
    times = iter([10.0, 10.25])
    teacher = TeacherClient(
        _settings(),
        client=client_double,
        sleep_fn=sleeps.append,
        clock=lambda: next(times),
    )
    with caplog.at_level(logging.INFO, logger="distillops.teacher"):
        prediction = teacher.extract(document_id="inv-synth-001", document_text=SENTINEL)
    call = client_double.completions.calls[0]
    assert call["model"] == "unit-test-model"
    assert call["timeout"] == 12
    assert call["temperature"] == 0
    assert call["response_format"] is TeacherInvoicePayload
    assert prediction.schema_valid is True
    assert prediction.model == "response-model"
    assert prediction.extraction is not None
    assert prediction.extraction.total is not None
    assert prediction.input_tokens == 20
    assert prediction.output_tokens == 5
    assert prediction.latency_ms == 250
    assert sleeps == []
    assert SENTINEL not in caplog.text
    assert SECRET_KEY not in caplog.text


def test_temperature_is_omitted_when_unset() -> None:
    client_double = ScriptedClient([_completion(_payload())])
    teacher = TeacherClient(_settings(teacher_temperature=None), client=client_double)
    teacher.extract(document_id="inv-synth-001", document_text="INVOICE")
    assert "temperature" not in client_double.completions.calls[0]


def test_retryable_failures_are_bounded_and_do_not_leak(caplog: pytest.LogCaptureFixture) -> None:
    secret_failure = RetryableStatus(f"{SENTINEL} {SECRET_KEY}", status_code=500)
    client_double = ScriptedClient([secret_failure, APITimeoutError("slow"), _completion(_payload())])
    sleeps: list[float] = []
    teacher = TeacherClient(_settings(openai_api_key=SECRET_KEY), client=client_double, sleep_fn=sleeps.append)
    with caplog.at_level(logging.INFO, logger="distillops.teacher"):
        prediction = teacher.extract(document_id="inv-synth-004", document_text=SENTINEL)
    assert prediction.schema_valid is True
    assert sleeps == [0.5, 1.0]
    assert len(client_double.completions.calls) == 3
    assert SENTINEL not in caplog.text
    assert SECRET_KEY not in caplog.text


def test_exhausted_retries_return_a_safe_api_error(caplog: pytest.LogCaptureFixture) -> None:
    failure = RetryableStatus(f"upstream echoed {SENTINEL} and {SECRET_KEY}", status_code=429)
    client_double = ScriptedClient([failure, failure, failure])
    sleeps: list[float] = []
    teacher = TeacherClient(_settings(openai_api_key=SECRET_KEY), client=client_double, sleep_fn=sleeps.append)
    with caplog.at_level(logging.WARNING, logger="distillops.teacher"):
        prediction = teacher.extract(document_id="inv-synth-005", document_text=SENTINEL)
    assert prediction.schema_valid is False
    assert prediction.extraction is None
    assert prediction.api_error == "RetryableStatus status=429"
    assert SENTINEL not in prediction.api_error
    assert SECRET_KEY not in prediction.api_error
    assert sleeps == [0.5, 1.0]
    assert len(client_double.completions.calls) == 3
    assert SENTINEL not in caplog.text
    assert SECRET_KEY not in caplog.text


def test_non_retryable_error_is_not_retried() -> None:
    client_double = ScriptedClient([RetryableStatus("bad request", status_code=400)])
    sleeps: list[float] = []
    teacher = TeacherClient(_settings(), client=client_double, sleep_fn=sleeps.append)
    prediction = teacher.extract(document_id="inv-synth-006", document_text=SENTINEL)
    assert prediction.api_error == "RetryableStatus status=400"
    assert sleeps == []
    assert len(client_double.completions.calls) == 1
    assert is_retryable(APITimeoutError("slow"))
    assert public_error(RetryableStatus(SENTINEL, status_code=429)) == "RetryableStatus status=429"


def test_invalid_model_output_is_a_schema_error_not_an_exception() -> None:
    payload = _payload().model_copy(update={"total": "not-a-number", "vendor": "V" * 250})
    client_double = ScriptedClient([_completion(payload)])
    teacher = TeacherClient(_settings(), client=client_double)
    prediction = teacher.extract(document_id="inv-synth-007", document_text=SENTINEL)
    assert prediction.schema_valid is False
    assert prediction.api_error is None
    assert prediction.validation_error is not None
    assert "not-a-number" not in prediction.validation_error
    assert prediction.raw_output is not None
    assert prediction.raw_output["total"] == "not-a-number"
    assert len(prediction.raw_output["vendor"]) == 200
    assert prediction.input_tokens == 11


def test_refusal_keeps_token_usage_and_is_not_an_api_failure() -> None:
    message = SimpleNamespace(parsed=None, refusal="cannot extract this document")
    completion = SimpleNamespace(
        model="response-model",
        choices=[SimpleNamespace(message=message)],
        usage=SimpleNamespace(prompt_tokens=9, completion_tokens=3),
    )
    teacher = TeacherClient(_settings(), client=ScriptedClient([completion]))
    prediction = teacher.extract(document_id="inv-synth-008", document_text=SENTINEL)
    assert prediction.schema_valid is False
    assert prediction.api_error is None
    assert prediction.refusal == "cannot extract this document"
    assert prediction.input_tokens == 9
    assert prediction.output_tokens == 3


def test_missing_credentials_raise_before_a_client_is_built() -> None:
    teacher = TeacherClient(_settings(teacher_model=None, openai_api_key=SECRET_KEY))
    with pytest.raises(TeacherConfigError) as missing_model:
        teacher.extract(document_id="inv-synth-009", document_text=SENTINEL)
    assert "TEACHER_MODEL" in str(missing_model.value)
    assert SECRET_KEY not in str(missing_model.value)
    assert "No teacher request was sent" in str(missing_model.value)

    unconfigured = TeacherClient(_settings(openai_api_key=None))
    with pytest.raises(TeacherConfigError) as missing_key:
        unconfigured.extract(document_id="inv-synth-010", document_text=SENTINEL)
    assert "OPENAI_API_KEY" in str(missing_key.value)
    assert SECRET_KEY not in str(missing_key.value)


def test_settings_hide_the_key_and_require_both_prices(monkeypatch: pytest.MonkeyPatch) -> None:
    _isolate(monkeypatch)
    monkeypatch.setenv("OPENAI_API_KEY", SECRET_KEY)
    monkeypatch.setenv("TEACHER_MODEL", "custom-teacher")
    monkeypatch.setenv("TEACHER_TEMPERATURE", "")
    monkeypatch.setenv("TEACHER_INPUT_USD_PER_MILLION_TOKENS", "0.15")
    monkeypatch.setenv("TEACHER_OUTPUT_USD_PER_MILLION_TOKENS", "0.60")
    settings = load_settings(use_dotenv=False)
    assert settings.teacher_model == "custom-teacher"
    assert settings.teacher_temperature is None
    assert SECRET_KEY not in repr(settings)
    pricing = configured_pricing(settings)
    assert pricing is not None
    assert pricing.estimate_usd(1_000_000, 500_000) == Decimal("0.450000")

    _isolate(monkeypatch)
    monkeypatch.setenv("TEACHER_INPUT_USD_PER_MILLION_TOKENS", "0.15")
    partial = load_settings(use_dotenv=False)
    with pytest.raises(ConfigError):
        configured_pricing(partial)


def test_baseline_refuses_the_test_split_and_limits_api_work(tmp_path, monkeypatch: pytest.MonkeyPatch) -> None:
    exit_code = baseline_main(["--split", "test"])
    assert exit_code == 2

    bundle = build_dataset(42)
    data_dir = tmp_path / "data"
    write_dataset(data_dir, bundle)
    test_split = data_dir / "splits" / "test.jsonl"
    test_split.write_text("DO_NOT_READ_TEST_SPLIT\n", encoding="utf-8")
    validation = bundle.splits["val"][:1]
    truth = validation[0].ground_truth
    payload = TeacherInvoicePayload(
        document_type="invoice",
        vendor=truth.vendor,
        invoice_number=truth.invoice_number,
        invoice_date=None if truth.invoice_date is None else truth.invoice_date.isoformat(),
        total=None if truth.total is None else format(truth.total, "f"),
        currency=truth.currency,
    )
    client_double = ScriptedClient([_completion(payload)])
    fixed_now = lambda: datetime(2026, 11, 3, 12, 0, tzinfo=timezone.utc)
    report, path = run_baseline(
        split="validation",
        splits_dir=data_dir / "splits",
        reports_dir=tmp_path / "reports",
        settings=_settings(),
        limit=1,
        client=client_double,
        now=fixed_now,
    )
    assert path.name == "teacher_baseline_val_20261103T120000Z.json"
    body = path.read_text(encoding="utf-8")
    assert "Synthetic training document" not in body
    assert "DO_NOT_READ_TEST_SPLIT" not in body
    assert len(client_double.completions.calls) == 1
    assert report.documents == 1
    assert report.schema_valid_count == 1
    assert report.exact_match_count == 1
    assert report.limit == 1

    _isolate(monkeypatch)
    monkeypatch.setenv("TEACHER_MODEL", "unit-test-model")
    monkeypatch.setenv("OPENAI_API_KEY", "")
    monkeypatch.setattr(
        "src.teacher.build_openai_client",
        lambda settings: (_ for _ in ()).throw(AssertionError("live client must not be built")),
    )
    missing_key = baseline_main(
        [
            "--split",
            "val",
            "--limit",
            "1",
            "--splits-dir",
            str(data_dir / "splits"),
            "--reports-dir",
            str(tmp_path / "unused-reports"),
            "--env-file",
            str(tmp_path / "missing.env"),
        ]
    )
    assert missing_key == 2
    assert not (tmp_path / "unused-reports").exists()
