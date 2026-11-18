"""Serve the frozen Phase 4 reports. Nothing is scored or written here."""

from __future__ import annotations

from fastapi import APIRouter

from backend.app.services.benchmark_reports import BenchmarkLibrary, load_saved_benchmarks

router = APIRouter()


@router.get("/benchmarks", response_model=BenchmarkLibrary)
def benchmarks() -> BenchmarkLibrary:
    return load_saved_benchmarks()
