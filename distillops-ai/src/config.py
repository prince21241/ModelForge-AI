"""Runtime settings loaded from the environment.

The teacher model name and both cost rates are configuration, not constants
inside the client. Cost stays unset unless both rate variables are present.
"""

from __future__ import annotations

import os
from decimal import Decimal, InvalidOperation
from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field, field_validator

from src.evaluation_pricing import Pricing


class ConfigError(ValueError):
    """Raised when environment settings are missing or inconsistent."""


class Settings(BaseModel):
    """Process settings for dataset tools and the teacher client."""

    model_config = ConfigDict(extra="forbid")

    openai_api_key: str | None = Field(default=None, repr=False)
    teacher_model: str | None = None
    timeout_seconds: float = Field(default=30, gt=0, le=180)
    max_retries: int = Field(default=2, ge=0, le=5)
    retry_backoff_seconds: float = Field(default=0.5, ge=0, le=30)
    teacher_temperature: float | None = 0
    input_usd_per_million_tokens: Decimal | None = None
    output_usd_per_million_tokens: Decimal | None = None

    @field_validator("openai_api_key", "teacher_model", mode="before")
    @classmethod
    def blank_string_is_missing(cls, value: object) -> object:
        if isinstance(value, str):
            stripped = value.strip()
            return stripped or None
        return value

    @field_validator("teacher_temperature")
    @classmethod
    def temperature_in_range(cls, value: float | None) -> float | None:
        if value is None:
            return None
        if not 0 <= value <= 2:
            raise ValueError("teacher_temperature must be between 0 and 2")
        return value

    @field_validator("input_usd_per_million_tokens", "output_usd_per_million_tokens")
    @classmethod
    def price_is_nonnegative(cls, value: Decimal | None) -> Decimal | None:
        if value is not None and value < 0:
            raise ValueError("token prices must be nonnegative")
        return value


def load_settings(
    env_file: str | Path | None = None,
    *,
    use_dotenv: bool = True,
) -> Settings:
    """Load settings from the process environment.

    ``use_dotenv`` reads ``env_file`` when it exists. Existing process variables
    win, matching python-dotenv's default. Pass ``use_dotenv=False`` in tests
    that want a fully isolated environment.
    """
    if use_dotenv:
        from dotenv import load_dotenv

        load_dotenv(env_file or ".env", override=False)
    try:
        return Settings(
            openai_api_key=_optional_str("OPENAI_API_KEY"),
            teacher_model=_optional_str("TEACHER_MODEL"),
            timeout_seconds=_float_env("TEACHER_TIMEOUT_SECONDS", 30),
            max_retries=_int_env("TEACHER_MAX_RETRIES", 2),
            retry_backoff_seconds=_float_env("TEACHER_RETRY_BACKOFF_SECONDS", 0.5),
            teacher_temperature=_temperature_env(),
            input_usd_per_million_tokens=_decimal_env("TEACHER_INPUT_USD_PER_MILLION_TOKENS"),
            output_usd_per_million_tokens=_decimal_env("TEACHER_OUTPUT_USD_PER_MILLION_TOKENS"),
        )
    except (ConfigError, ValueError) as exc:
        if isinstance(exc, ConfigError):
            raise
        raise ConfigError(str(exc)) from exc


def configured_pricing(settings: Settings) -> Pricing | None:
    """Return a price card only when both rates were configured."""
    input_price = settings.input_usd_per_million_tokens
    output_price = settings.output_usd_per_million_tokens
    if input_price is None and output_price is None:
        return None
    if input_price is None or output_price is None:
        raise ConfigError(
            "Set both TEACHER_INPUT_USD_PER_MILLION_TOKENS and "
            "TEACHER_OUTPUT_USD_PER_MILLION_TOKENS, or leave both unset. "
            "DistillOps does not assume a provider price."
        )
    return Pricing(
        input_usd_per_million_tokens=input_price,
        output_usd_per_million_tokens=output_price,
    )


def _optional_str(name: str) -> str | None:
    raw = os.getenv(name)
    if raw is None:
        return None
    stripped = raw.strip()
    return stripped or None


def _float_env(name: str, default: float) -> float:
    raw = os.getenv(name)
    if raw is None or raw.strip() == "":
        return default
    try:
        return float(raw)
    except ValueError as exc:
        raise ConfigError(f"{name} must be a number.") from exc


def _int_env(name: str, default: int) -> int:
    raw = os.getenv(name)
    if raw is None or raw.strip() == "":
        return default
    try:
        return int(raw)
    except ValueError as exc:
        raise ConfigError(f"{name} must be an integer.") from exc


def _decimal_env(name: str) -> Decimal | None:
    raw = os.getenv(name)
    if raw is None or raw.strip() == "":
        return None
    try:
        return Decimal(raw.strip())
    except InvalidOperation as exc:
        raise ConfigError(f"{name} must be a decimal number.") from exc


def _temperature_env() -> float | None:
    if "TEACHER_TEMPERATURE" not in os.environ:
        return 0.0
    raw = os.environ["TEACHER_TEMPERATURE"].strip()
    if raw == "":
        return None
    try:
        return float(raw)
    except ValueError as exc:
        raise ConfigError("TEACHER_TEMPERATURE must be a number or blank.") from exc
