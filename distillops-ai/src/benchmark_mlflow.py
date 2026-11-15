"""Local MLflow tracking for one Phase 4 benchmark report.

The tracker records explicit parameters and metrics only. It does not log the
process environment, so an API key in the environment is not written.
"""

from __future__ import annotations

from pathlib import Path

from src.benchmark import EXPERIMENT_NAME, BenchmarkReport


def log_benchmark(
    report: BenchmarkReport,
    *,
    tracking_uri: str,
    artifact_path: Path | None = None,
) -> str:
    """Start one run in the Phase 4 experiment and return its id."""
    import mlflow

    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment(EXPERIMENT_NAME)
    with mlflow.start_run(run_name=f"{report.model_role}-{report.split}-{report.status}"):
        mlflow.log_params(_params(report))
        metrics = _metrics(report)
        if metrics:
            mlflow.log_metrics(metrics)
        if artifact_path is not None and Path(artifact_path).is_file():
            mlflow.log_artifact(str(artifact_path))
        return str(mlflow.active_run().info.run_id)


def _params(report: BenchmarkReport) -> dict[str, str]:
    decoding = report.decoding
    return {
        "model_role": report.model_role,
        "model_id": report.model_id,
        "status": report.status,
        "split": report.split,
        "checkpoint": report.checkpoint or "",
        "documents_in_split": str(report.documents_in_split),
        "documents_scored": str(report.documents_scored),
        "do_sample": str(decoding.get("do_sample", "")),
        "num_beams": str(decoding.get("num_beams", "")),
        "max_new_tokens": str(decoding.get("max_new_tokens", "")),
        "max_source_length": str(decoding.get("max_source_length", "")),
        "warmup_excluded": str(report.warmup_excluded),
        "pricing_configured": str(report.pricing_configured),
        "cost_assumption": report.cost_assumption[:500],
        "student_instruction_sha256": report.prompt_fingerprints.get("student_instruction_sha256", ""),
        "teacher_system_prompt_sha256": report.prompt_fingerprints.get("teacher_system_prompt_sha256", ""),
        "api_calls": str(report.api_calls),
    }


def _metrics(report: BenchmarkReport) -> dict[str, float]:
    metrics: dict[str, float] = {
        "json_valid_count": float(report.json_valid_count),
        "schema_valid_count": float(report.schema_valid_count),
        "malformed_json_count": float(report.malformed_json_count),
        "exact_match_count": float(report.exact_match_count),
        "missing_field_count": float(report.missing_field_count),
        "hallucinated_field_count": float(report.hallucinated_field_count),
        "incorrect_field_count": float(report.incorrect_field_count),
        "input_tokens": float(report.input_tokens),
        "output_tokens": float(report.output_tokens),
        "api_failure_count": float(report.api_failure_count),
    }
    optional = {
        "json_validity": _rate(report.json_valid_count, report.documents_scored),
        "exact_match_accuracy": report.exact_match_accuracy,
        "end_to_end_exact_match_accuracy": report.end_to_end_exact_match_accuracy,
        "latency_ms_avg": report.latency_ms_avg,
        "latency_ms_median": report.latency_ms_median,
        "latency_ms_p95": report.latency_ms_p95,
        "warmup_ms": report.warmup_ms,
        "estimated_cost_usd": None if report.estimated_cost_usd is None else float(report.estimated_cost_usd),
    }
    for name, value in optional.items():
        if value is not None:
            metrics[name] = float(value)
    for metric in report.field_metrics:
        if metric.accuracy is not None:
            metrics[f"field_accuracy_{metric.field}"] = float(metric.accuracy)
    return metrics


def _rate(numerator: int, denominator: int) -> float | None:
    if denominator == 0:
        return None
    return numerator / denominator
