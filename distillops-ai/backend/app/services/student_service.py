"""Load the frozen student once and extract invoices with the Phase 4 decoder.

Generation is serialized so concurrent requests do not share one GPU step.
The checkpoint files are opened read-only through the existing loader.
"""

from __future__ import annotations

import logging
import threading
from typing import Any, Protocol

from src.student_data import StudentDataError
from src.student_eval import StudentGeneration
from src.student_infer import StudentInferenceError, generate_fields, load_student, select_device

from backend.app.core.config import MAX_DOCUMENT_CHARS, MAX_NEW_TOKENS, MAX_SOURCE_LENGTH, STUDENT_CHECKPOINT
from backend.app.core.errors import (
    EmptyDocumentError,
    InferenceFailedError,
    ModelLoadError,
    OversizedDocumentError,
)
from backend.app.schemas.invoice import ExtractResponse
from backend.app.services.review import fields_from_student, review_warnings

logger = logging.getLogger("distillops.api")


class FieldGenerator(Protocol):
    """The small surface tests replace when they do not load weights."""

    def __call__(self, model: Any, tokenizer: Any, invoice_text: str, **kwargs: Any) -> StudentGeneration:
        """Decode one document."""


class StudentService:
    """Process-wide student. The first extraction loads the checkpoint."""

    def __init__(
        self,
        *,
        checkpoint: Any = STUDENT_CHECKPOINT,
        generator: FieldGenerator = generate_fields,
        loader: Any = load_student,
    ) -> None:
        self._checkpoint = checkpoint
        self._generator = generator
        self._loader = loader
        self._lock = threading.Lock()
        self._model: Any = None
        self._tokenizer: Any = None
        self._device: Any = None
        self._loaded = False

    @property
    def loaded(self) -> bool:
        return self._loaded

    @property
    def checkpoint(self) -> str:
        return str(self._checkpoint)

    def extract(self, document_text: str) -> ExtractResponse:
        """Run one greedy decode. Blank and oversized text never reach the model."""
        text = document_text if isinstance(document_text, str) else ""
        if not text.strip():
            raise EmptyDocumentError()
        if len(text) > MAX_DOCUMENT_CHARS:
            raise OversizedDocumentError()
        self._ensure_loaded()
        try:
            with self._lock:
                generation = self._generator(
                    self._model,
                    self._tokenizer,
                    text,
                    document_id="api-request",
                    max_source_length=MAX_SOURCE_LENGTH,
                    max_new_tokens=MAX_NEW_TOKENS,
                    device=self._device,
                )
        except StudentDataError as exc:
            if "above the limit" in str(exc):
                raise OversizedDocumentError(
                    "document_text exceeds 512 source tokens and was not truncated."
                ) from exc
            logger.exception("Student input was rejected.")
            raise InferenceFailedError() from exc
        except StudentInferenceError as exc:
            logger.exception("Student inference failed.")
            raise InferenceFailedError() from exc
        except Exception as exc:
            logger.exception("Student inference failed.")
            raise InferenceFailedError() from exc
        parsed = fields_from_student(generation.fields)
        return ExtractResponse(
            model="student",
            fields=parsed,
            latency_ms=generation.latency_ms,
            requires_review=True,
            warnings=review_warnings(parsed, validation_error=generation.validation_error),
        )

    def _ensure_loaded(self) -> None:
        if self._loaded:
            return
        with self._lock:
            if self._loaded:
                return
            try:
                device = select_device("auto")
                model, tokenizer = self._loader(self._checkpoint, device)
            except Exception as exc:
                logger.exception("Student checkpoint could not be loaded.")
                raise ModelLoadError() from exc
            self._device = device
            self._model = model
            self._tokenizer = tokenizer
            self._loaded = True
