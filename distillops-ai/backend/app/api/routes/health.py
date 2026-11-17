"""Status and model inventory. Loaded is true only after a successful load."""

from __future__ import annotations

from fastapi import APIRouter, Request

from backend.app.core.config import BASE_MODEL_NAME, STUDENT_CHECKPOINT
from backend.app.schemas.invoice import HealthResponse, ModelInfo, ModelsResponse
from backend.app.services.student_service import StudentService

router = APIRouter()


def _service(request: Request) -> StudentService:
    return request.app.state.student_service


@router.get("/health", response_model=HealthResponse)
def health(request: Request) -> HealthResponse:
    service = _service(request)
    return HealthResponse(
        status="ok",
        service="distillops",
        student_checkpoint=str(STUDENT_CHECKPOINT),
        student_loaded=service.loaded,
    )


@router.get("/models", response_model=ModelsResponse)
def models(request: Request) -> ModelsResponse:
    service = _service(request)
    return ModelsResponse(
        models=[
            ModelInfo(
                name="student",
                identifier=BASE_MODEL_NAME,
                checkpoint=service.checkpoint,
                available=True,
                loaded=service.loaded,
            )
        ]
    )
