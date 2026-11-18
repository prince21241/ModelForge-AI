"""Request and response models for local invoice extraction."""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field


class InvoiceFields(BaseModel):
    """Five student fields. Empty strings mean the value was not extracted."""

    model_config = ConfigDict(extra="forbid")

    vendor: str = ""
    invoice_number: str = ""
    date: str = ""
    total: str = ""
    currency: str = ""


class ExtractRequest(BaseModel):
    """One invoice body and the model name requested by the client."""

    model_config = ConfigDict(extra="forbid")

    document_text: str
    model: str = "student"


class ExtractResponse(BaseModel):
    """Inference result. Field text is the model output, not a repaired value."""

    model_config = ConfigDict(extra="forbid")

    model: str
    fields: InvoiceFields
    latency_ms: float = Field(ge=0)
    requires_review: bool
    warnings: list[str]


class DocumentExtractResponse(BaseModel):
    """One uploaded invoice. latency_ms is the student generation time."""

    model_config = ConfigDict(extra="forbid")

    filename: str
    content_type: str
    page_count: int = Field(ge=1)
    extraction_method: str
    extracted_text: str
    model: str
    fields: InvoiceFields
    latency_ms: float = Field(ge=0)
    document_latency_ms: float = Field(ge=0)
    model_latency_ms: float = Field(ge=0)
    total_latency_ms: float = Field(ge=0)
    requires_review: bool
    warnings: list[str]


class HealthResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    status: str
    service: str
    student_checkpoint: str
    student_loaded: bool


class ModelInfo(BaseModel):
    model_config = ConfigDict(extra="forbid")

    name: str
    identifier: str
    checkpoint: str
    available: bool
    loaded: bool


class ModelsResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    models: list[ModelInfo]


class ErrorBody(BaseModel):
    model_config = ConfigDict(extra="forbid")

    detail: str
