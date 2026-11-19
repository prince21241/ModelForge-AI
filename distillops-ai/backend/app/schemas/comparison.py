"""Live teacher-versus-student comparison. Predictions are not repaired."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from backend.app.schemas.invoice import InvoiceFields


class CompareRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    document_text: str
    teacher_approved: bool = False


class CompareStatus(BaseModel):
    model_config = ConfigDict(extra="forbid")

    feature_enabled: bool
    available: bool
    api_key_configured: bool
    teacher_model: str | None
    pricing_configured: bool
    student_pricing_configured: bool
    max_requests: int = Field(ge=1)
    budget_usd: float | None
    public_deployment: bool
    budget_scope: str
    detail: str


class ModelOutcome(BaseModel):
    model_config = ConfigDict(extra="forbid")

    role: Literal["teacher", "student"]
    model: str
    succeeded: bool
    fields: InvoiceFields | None
    latency_ms: float | None = Field(default=None, ge=0)
    input_tokens: int | None = Field(default=None, ge=0)
    output_tokens: int | None = Field(default=None, ge=0)
    estimated_cost_usd: float | None = None
    pricing_configured: bool
    error: str | None = None
    warnings: list[str] = Field(default_factory=list)


class FieldComparison(BaseModel):
    model_config = ConfigDict(extra="forbid")

    field: str
    label: str
    teacher_value: str | None
    student_value: str | None
    status: Literal["agree", "disagree", "unavailable"]


class CompareResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    teacher: ModelOutcome
    student: ModelOutcome
    fields: list[FieldComparison]
    agreement_count: int | None = Field(ge=0)
    fields_compared: int = Field(ge=0)
    document_latency_ms: float = Field(ge=0)
    extracted_text: str
    filename: str | None = None
    content_type: str | None = None
    page_count: int | None = Field(default=None, ge=1)
    extraction_method: str | None = None
    partial: bool
    warnings: list[str]
