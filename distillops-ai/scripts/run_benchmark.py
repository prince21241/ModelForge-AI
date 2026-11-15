"""Compare the frozen teacher and student on one split.

The locked test split stays closed unless ``--authorize-locked-test`` is
present. A teacher request is sent only when both ``--live`` and
``--approve-cost`` are present. Cached teacher rows are reused otherwise.

    python -m scripts.run_benchmark --model student --split validation
    python -m scripts.run_benchmark --model both --split validation --dry-run
    python -m scripts.run_benchmark --prepare-locked-test
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

import torch

from src.benchmark import (
    DECODING,
    FROZEN_STUDENT_DIR,
    FROZEN_TEACHER_MODEL,
    WARMUP_DOCUMENT_ID,
    WARMUP_TEXT,
    BenchmarkError,
    BenchmarkReport,
    LockedSplitError,
    assert_test_file_closed,
    build_report,
    checkpoint_identity,
    load_teacher_cache,
    locked_split_manifest,
    prompt_fingerprints,
    resolve_split,
    score_student_output,
    score_teacher_prediction,
    student_cost_rate,
    student_inference_cost,
    write_report_files,
)
from src.benchmark_mlflow import log_benchmark
from src.config import ConfigError, configured_pricing, load_settings
from src.dataset import DatasetLoadError, load_split
from src.hybrid_training import HybridTrainError, require_paid_teacher_approval
from src.schemas import DatasetRecord
from src.student_infer import StudentInferenceError, generate_fields, load_student, select_device
from src.teacher import TeacherClient, TeacherConfigError


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Benchmark the frozen teacher and student.")
    parser.add_argument("--model", choices=["teacher", "student", "both"], default="both")
    parser.add_argument("--split", choices=["validation", "test"], default="validation")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--authorize-locked-test", action="store_true")
    parser.add_argument("--prepare-locked-test", action="store_true")
    parser.add_argument("--live", action="store_true")
    parser.add_argument("--approve-cost", action="store_true")
    parser.add_argument("--student-dir", type=Path, default=FROZEN_STUDENT_DIR)
    parser.add_argument("--splits-dir", type=Path, default=Path("data/splits"))
    parser.add_argument("--reports-dir", type=Path, default=Path("reports"))
    parser.add_argument(
        "--teacher-cache",
        type=Path,
        default=Path("reports/teacher_baseline_val_20261106T201057Z.json"),
    )
    parser.add_argument("--no-teacher-cache", action="store_true")
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    parser.add_argument("--no-warmup", action="store_true")
    parser.add_argument("--mlflow-uri", default="sqlite:///mlflow.db")
    parser.add_argument("--no-mlflow", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        if args.prepare_locked_test:
            return _prepare_locked_test(args)
        return _run(args)
    except LockedSplitError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    except (
        BenchmarkError,
        DatasetLoadError,
        StudentInferenceError,
        TeacherConfigError,
        ConfigError,
        HybridTrainError,
    ) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    except RuntimeError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1


def _run(args: argparse.Namespace) -> int:
    split = resolve_split(args.split, authorize_locked_test=args.authorize_locked_test)
    split_path = Path(args.splits_dir) / ("test.jsonl" if split == "test" else "val.jsonl")
    assert_test_file_closed(split_path, authorize_locked_test=args.authorize_locked_test)
    records = load_split(args.splits_dir, split)
    if args.dry_run:
        plan = _dry_plan(args, split, records)
        path = _write_json(Path(args.reports_dir) / f"phase4_{split}_dry_run.json", plan)
        print(json.dumps(plan, indent=2))
        print(f"Dry run: {path}")
        return 0
    exit_code = 0
    if args.model in {"student", "both"}:
        report, path = _run_student(args, split, records)
        print(_summary(report))
        print(f"Report: {path}")
        exit_code = _log_or_warn(args, report, path) or exit_code
    if args.model in {"teacher", "both"}:
        report, path = _run_teacher(args, split, records)
        print(_summary(report))
        print(f"Report: {path}")
        exit_code = _log_or_warn(args, report, path) or exit_code
    return exit_code


def _run_student(
    args: argparse.Namespace,
    split: str,
    records: list[DatasetRecord],
) -> tuple[BenchmarkReport, Path]:
    identity = checkpoint_identity(args.student_dir)
    device = select_device(args.device)
    model, tokenizer = load_student(args.student_dir, device)
    warmup_ms = None
    if not args.no_warmup:
        warmup_ms = _warmup(model, tokenizer, device)
    rows = []
    for record in records:
        started = time.perf_counter()
        if device.type == "cuda":
            torch.cuda.synchronize()
        generation = generate_fields(
            model,
            tokenizer,
            record.document_text,
            document_id=record.document_id,
            max_source_length=int(DECODING["max_source_length"]),
            max_new_tokens=int(DECODING["max_new_tokens"]),
            device=device,
        )
        if device.type == "cuda":
            torch.cuda.synchronize()
        latency_ms = (time.perf_counter() - started) * 1000
        rows.append(
            score_student_output(
                record,
                raw_text=generation.raw_text,
                latency_ms=latency_ms,
                model_id=str(args.student_dir),
            )
        )
    rate, assumption = student_cost_rate()
    report = build_report(
        rows,
        status="completed",
        model_role="student",
        model_id=str(args.student_dir),
        split=split,
        documents_in_split=len(records),
        unscored_document_ids=[],
        checkpoint=str(args.student_dir),
        decoding={**DECODING, "checkpoint_identity": identity},
        warmup_ms=warmup_ms,
        warmup_excluded=not args.no_warmup,
        api_calls=0,
        pricing=None,
        cost_assumption=assumption,
    )
    report = report.model_copy(
        update={
            "pricing_configured": rate is not None,
            "estimated_cost_usd": student_inference_cost(report.documents_scored, rate),
        }
    )
    json_path, _csv_path = write_report_files(
        report,
        Path(args.reports_dir) / f"phase4_student_{split}.json",
    )
    return report, json_path


def _run_teacher(
    args: argparse.Namespace,
    split: str,
    records: list[DatasetRecord],
) -> tuple[BenchmarkReport, Path]:
    pricing = configured_pricing(load_settings())
    assumption = (
        "Teacher cost uses TEACHER_INPUT_USD_PER_MILLION_TOKENS and "
        "TEACHER_OUTPUT_USD_PER_MILLION_TOKENS when both are set."
    )
    if args.live:
        require_paid_teacher_approval(live=True, approve_cost=args.approve_cost)
        predictions = []
        client = TeacherClient(load_settings())
        for record in records:
            predictions.append(client.extract(document_id=record.document_id, document_text=record.document_text))
        by_id = {prediction.document_id: prediction for prediction in predictions}
        rows = [score_teacher_prediction(record, by_id[record.document_id]) for record in records]
        status = "completed"
        api_calls = len(records)
        model_id = predictions[0].model if predictions else FROZEN_TEACHER_MODEL
        unscored: list[str] = []
    else:
        cached = [] if args.no_teacher_cache else _safe_cache(args.teacher_cache)
        by_id = {}
        for prediction in cached:
            if prediction.document_id in by_id:
                raise BenchmarkError(f"Duplicate cached prediction for {prediction.document_id}.")
            by_id[prediction.document_id] = prediction
        rows = [
            score_teacher_prediction(record, by_id[record.document_id])
            for record in records
            if record.document_id in by_id
        ]
        unscored = [record.document_id for record in records if record.document_id not in by_id]
        if not cached:
            status = "not_run"
        elif unscored:
            status = "cache_incomplete"
        else:
            status = "completed"
        api_calls = 0
        model_id = rows[0].model_id if rows else FROZEN_TEACHER_MODEL
        assumption = assumption + " No new API call was sent."
    report = build_report(
        rows,
        status=status,  # type: ignore[arg-type]
        model_role="teacher",
        model_id=model_id,
        split=split,
        documents_in_split=len(records),
        unscored_document_ids=unscored,
        checkpoint=None,
        decoding={"temperature": 0, "response_format": "TeacherInvoicePayload"},
        warmup_ms=None,
        warmup_excluded=False,
        api_calls=api_calls,
        pricing=pricing,
        cost_assumption=assumption,
    )
    if status != "completed":
        report = report.model_copy(update={"end_to_end_exact_match_accuracy": None})
    json_path, _csv_path = write_report_files(
        report,
        Path(args.reports_dir) / f"phase4_teacher_{split}.json",
    )
    return report, json_path


def _prepare_locked_test(args: argparse.Namespace) -> int:
    """Write the test manifest and the future command. Do not score invoices."""
    path = Path(args.splits_dir) / "test.jsonl"
    if not path.is_file():
        raise BenchmarkError(f"Test split file not found: {path.name}")
    manifest = locked_split_manifest(path)
    plan = {
        "status": "prepared",
        "split": "test",
        "documents": manifest["documents"],
        "invoice_text_stored": False,
        "manifest": manifest,
        "frozen_student": str(args.student_dir),
        "teacher_model": FROZEN_TEACHER_MODEL,
        "decoding": DECODING,
        "prompt_fingerprints": prompt_fingerprints(),
        "expected_teacher_api_calls": manifest["documents"],
        "max_teacher_attempts_per_invoice": "1 + TEACHER_MAX_RETRIES",
        "command": (
            "python -m scripts.run_benchmark --model both --split test "
            "--authorize-locked-test --live --approve-cost"
        ),
        "note": "This plan does not authorize the command. The test benchmark has not been run.",
    }
    written = _write_json(Path(args.reports_dir) / "phase4_test_plan.json", plan)
    print(json.dumps({key: value for key, value in plan.items() if key != "manifest"}, indent=2))
    print(f"Test plan: {written}")
    print(f"Documents: {manifest['documents']}")
    print("Invoice text stored: false")
    return 0


def _warmup(model: Any, tokenizer: Any, device: torch.device) -> float:
    if device.type == "cuda":
        torch.cuda.synchronize()
    started = time.perf_counter()
    generate_fields(
        model,
        tokenizer,
        WARMUP_TEXT,
        document_id=WARMUP_DOCUMENT_ID,
        max_source_length=int(DECODING["max_source_length"]),
        max_new_tokens=int(DECODING["max_new_tokens"]),
        device=device,
    )
    if device.type == "cuda":
        torch.cuda.synchronize()
    return (time.perf_counter() - started) * 1000


def _dry_plan(args: argparse.Namespace, split: str, records: list[DatasetRecord]) -> dict[str, Any]:
    return {
        "status": "dry_run",
        "models": ["teacher", "student"] if args.model == "both" else [args.model],
        "split": split,
        "documents": len(records),
        "document_ids": [record.document_id for record in records],
        "student_checkpoint": str(args.student_dir),
        "teacher_model": FROZEN_TEACHER_MODEL,
        "decoding": DECODING,
        "prompt_fingerprints": prompt_fingerprints(),
        "teacher_api_calls": 0,
        "inference_executed": False,
        "warmup_planned": not args.no_warmup and args.model in {"student", "both"},
    }


def _safe_cache(path: Path) -> list:
    if not Path(path).is_file():
        return []
    return load_teacher_cache(path)


def _log_or_warn(args: argparse.Namespace, report: BenchmarkReport, artifact: Path) -> int:
    if args.no_mlflow or report.status in {"not_run", "dry_run"}:
        return 0
    try:
        run_id = log_benchmark(report, tracking_uri=args.mlflow_uri, artifact_path=artifact)
    except Exception as exc:
        print(f"warning: MLflow log failed: {type(exc).__name__}", file=sys.stderr)
        return 1
    print(f"MLflow run: {run_id}")
    return 0


def _summary(report: BenchmarkReport) -> str:
    return (
        f"{report.model_role} {report.status} split={report.split} "
        f"json {report.json_valid_count}/{report.documents_scored} "
        f"schema {report.schema_valid_count}/{report.documents_scored} "
        f"exact {report.exact_match_count}/{report.documents_in_split} "
        f"malformed {report.malformed_json_count} "
        f"missing {report.missing_field_count} "
        f"hallucinated {report.hallucinated_field_count} "
        f"incorrect {report.incorrect_field_count} "
        f"api_calls {report.api_calls}"
    )


def _write_json(path: Path, payload: dict[str, Any]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return path


if __name__ == "__main__":
    sys.exit(main())
