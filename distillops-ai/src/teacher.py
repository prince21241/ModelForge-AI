"""OpenAI teacher client for invoice extraction.

Live calls are optional. Construct the client with a fake ``client`` in tests,
or call it without ``OPENAI_API_KEY`` and it raises before any network I/O.
Logs include document ids, model names, timings, and token counts. They do not
include the API key or the document text.
"""

from __future__ import annotations

import logging
import time
from collections.abc import Callable, Mapping
from typing import Any

from pydantic import BaseModel, ValidationError

from src.config import Settings
from src.prompts import build_messages
from src.schemas import (
    TeacherInvoicePayload,
    TeacherPrediction,
    extracted_invoice_from_payload,
)

logger = logging.getLogger("distillops.teacher")

_RETRYABLE_STATUS_CODES = {408, 409, 429, 500, 502, 503, 504}
_RETRYABLE_TYPE_NAMES = {
    "APITimeoutError",
    "APIConnectionError",
    "RateLimitError",
    "InternalServerError",
    "ConnectError",
    "ReadTimeout",
    "WriteTimeout",
    "PoolTimeout",
}
_RAW_TEXT_LIMIT = 200


class TeacherConfigError(RuntimeError):
    """Raised when a live call is requested without the required settings."""


class TeacherClient:
    """Extract invoice fields with a configurable teacher model.

    The OpenAI SDK retry setting is forced to zero. This class owns a bounded
    retry loop so the attempt count is the same for the real client and for tests.
    """

    def __init__(
        self,
        settings: Settings,
        *,
        client: Any | None = None,
        sleep_fn: Callable[[float], None] = time.sleep,
        clock: Callable[[], float] = time.perf_counter,
    ) -> None:
        self._settings = settings
        self._client = client
        self._sleep = sleep_fn
        self._clock = clock

    @property
    def settings(self) -> Settings:
        return self._settings

    def extract(self, *, document_id: str, document_text: str) -> TeacherPrediction:
        """Extract one document. API failures return a prediction instead of raising."""
        self._validate_ready()
        model_name = self._settings.teacher_model or ""
        started = self._clock()
        try:
            completion = self._call_with_retries(document_id=document_id, document_text=document_text)
        except TeacherConfigError:
            raise
        except Exception as exc:
            latency_ms = self._elapsed_ms(started)
            logger.warning(
                "Teacher request failed document_id=%s model=%s error_type=%s latency_ms=%.1f",
                document_id,
                model_name,
                type(exc).__name__,
                latency_ms,
            )
            return TeacherPrediction(
                document_id=document_id,
                model=model_name,
                extraction=None,
                schema_valid=False,
                api_error=public_error(exc),
                latency_ms=latency_ms,
            )
        return self._prediction_from_completion(
            document_id=document_id,
            completion=completion,
            latency_ms=self._elapsed_ms(started),
            fallback_model=model_name,
        )

    def _validate_ready(self) -> None:
        if not self._settings.teacher_model:
            raise TeacherConfigError("TEACHER_MODEL is not set. No teacher request was sent.")
        if self._client is None and not self._settings.openai_api_key:
            raise TeacherConfigError(
                "OPENAI_API_KEY is not set. No teacher request was sent. "
                "Teacher calls are optional and unit tests do not need a key."
            )

    def _call_with_retries(self, *, document_id: str, document_text: str) -> Any:
        client = self._require_client()
        attempts = self._settings.max_retries + 1
        for attempt in range(attempts):
            try:
                kwargs: dict[str, Any] = {
                    "model": self._settings.teacher_model,
                    "messages": build_messages(document_text),
                    "response_format": TeacherInvoicePayload,
                    "timeout": self._settings.timeout_seconds,
                }
                if self._settings.teacher_temperature is not None:
                    kwargs["temperature"] = self._settings.teacher_temperature
                return client.chat.completions.parse(**kwargs)
            except TeacherConfigError:
                raise
            except Exception as exc:
                if attempt >= self._settings.max_retries or not is_retryable(exc):
                    raise
                delay = min(8.0, self._settings.retry_backoff_seconds * (2**attempt))
                logger.warning(
                    "Retrying teacher request document_id=%s attempt=%s error_type=%s",
                    document_id,
                    attempt + 1,
                    type(exc).__name__,
                )
                self._sleep(delay)
        raise RuntimeError("Teacher retry loop exited without a result.")

    def _require_client(self) -> Any:
        if self._client is not None:
            return self._client
        self._client = build_openai_client(self._settings)
        return self._client

    def _elapsed_ms(self, started: float) -> float:
        return max(0.0, (self._clock() - started) * 1000)

    def _prediction_from_completion(
        self,
        *,
        document_id: str,
        completion: Any,
        latency_ms: float,
        fallback_model: str,
    ) -> TeacherPrediction:
        model_name = getattr(completion, "model", None) or fallback_model
        input_tokens, output_tokens = read_usage(getattr(completion, "usage", None))
        choices = getattr(completion, "choices", None) or []
        base = TeacherPrediction(
            document_id=document_id,
            model=model_name,
            extraction=None,
            schema_valid=False,
            latency_ms=latency_ms,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
        )
        if not choices:
            logger.warning(
                "Teacher response had no choices document_id=%s model=%s",
                document_id,
                model_name,
            )
            return base.model_copy(
                update={"validation_error": "Teacher response did not include a choice."}
            )
        message = choices[0].message
        refusal = getattr(message, "refusal", None)
        if refusal:
            logger.info(
                "Teacher refused document_id=%s model=%s latency_ms=%.1f",
                document_id,
                model_name,
                latency_ms,
            )
            return base.model_copy(update={"refusal": _truncate(str(refusal))})
        parsed = getattr(message, "parsed", None)
        if parsed is None:
            return base.model_copy(
                update={"validation_error": "Teacher response did not include a parsed schema."}
            )
        raw_output = _bounded_raw(parsed)
        try:
            extraction = extracted_invoice_from_payload(parsed)
        except ValidationError as exc:
            logger.info(
                "Teacher output failed schema validation document_id=%s model=%s",
                document_id,
                model_name,
            )
            return base.model_copy(
                update={
                    "raw_output": raw_output,
                    "validation_error": summarize_validation(exc),
                }
            )
        logger.info(
            "Teacher extraction succeeded document_id=%s model=%s latency_ms=%.1f "
            "input_tokens=%s output_tokens=%s",
            document_id,
            model_name,
            latency_ms,
            input_tokens,
            output_tokens,
        )
        return base.model_copy(
            update={
                "extraction": extraction,
                "raw_output": raw_output,
                "schema_valid": True,
            }
        )


def build_openai_client(settings: Settings) -> Any:
    """Construct the official OpenAI client. Imported lazily so tests can skip it."""
    if not settings.openai_api_key:
        raise TeacherConfigError("OPENAI_API_KEY is not set. No teacher request was sent.")
    from openai import OpenAI

    return OpenAI(
        api_key=settings.openai_api_key,
        timeout=settings.timeout_seconds,
        max_retries=0,
    )


def is_retryable(exc: Exception) -> bool:
    status = getattr(exc, "status_code", None)
    if isinstance(status, int) and status in _RETRYABLE_STATUS_CODES:
        return True
    return type(exc).__name__ in _RETRYABLE_TYPE_NAMES


def public_error(exc: Exception) -> str:
    """Exception type and status only, so keys and document text cannot leak."""
    status = getattr(exc, "status_code", None)
    parts = [type(exc).__name__]
    if isinstance(status, int):
        parts.append(f"status={status}")
    return " ".join(parts)


def read_usage(usage: Any) -> tuple[int, int]:
    if usage is None:
        return 0, 0
    input_tokens = _usage_value(usage, "prompt_tokens", "input_tokens")
    output_tokens = _usage_value(usage, "completion_tokens", "output_tokens")
    return input_tokens, output_tokens


def _usage_value(usage: Any, *names: str) -> int:
    for name in names:
        value = usage.get(name) if isinstance(usage, Mapping) else getattr(usage, name, None)
        if value is not None:
            return int(value)
    return 0


def summarize_validation(exc: ValidationError) -> str:
    """Field locations and error types, without echoing the rejected values."""
    parts: list[str] = []
    for error in exc.errors():
        location = ".".join(str(item) for item in error.get("loc", ()))
        error_type = str(error.get("type", "value_error"))
        parts.append(f"{location}: {error_type}" if location else error_type)
    text = "; ".join(parts) or "schema validation failed"
    return _truncate(text)


def _bounded_raw(parsed: Any) -> dict[str, Any] | None:
    if not isinstance(parsed, BaseModel):
        return None
    bounded: dict[str, Any] = {}
    for key, value in parsed.model_dump().items():
        if isinstance(value, str):
            bounded[key] = _truncate(value)
        else:
            bounded[key] = value
    return bounded


def _truncate(value: str, limit: int = _RAW_TEXT_LIMIT) -> str:
    if len(value) <= limit:
        return value
    return value[:limit]
