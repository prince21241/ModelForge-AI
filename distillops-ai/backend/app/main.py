"""DistillOps local extraction API.

    python -m uvicorn backend.app.main:app --host 127.0.0.1 --port 8000
"""

from __future__ import annotations

from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from backend.app.api.routes import benchmarks, comparison, documents, extraction, health
from backend.app.core.config import ALLOWED_ORIGINS
from backend.app.core.errors import ApiError
from backend.app.services.student_service import StudentService
from backend.app.services.teacher_budget import TeacherSpendGate


@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.student_service = StudentService()
    app.state.teacher_spend = TeacherSpendGate()
    app.state.teacher_client = None
    yield


app = FastAPI(title="DistillOps AI", version="0.5.3", lifespan=lifespan)
app.add_middleware(
    CORSMiddleware,
    allow_origins=list(ALLOWED_ORIGINS),
    allow_credentials=False,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["Accept", "Content-Type"],
)
app.include_router(health.router, prefix="/api")
app.include_router(extraction.router, prefix="/api")
app.include_router(documents.router, prefix="/api")
app.include_router(benchmarks.router, prefix="/api")
app.include_router(comparison.router, prefix="/api")


@app.exception_handler(ApiError)
async def api_error_handler(_request: Request, exc: ApiError) -> JSONResponse:
    return JSONResponse(status_code=exc.status_code, content={"detail": exc.detail})
