"""Prepare the 15-invoice teacher evaluation.

A live call is sent only when both --live and --approve-cost are present.
The default writes a pending report and does not contact OpenAI.

    python -m scripts.evaluate_hybrid_teacher
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from src.config import configured_pricing, load_settings
from src.hybrid_dataset import generate_hybrid_invoices
from src.hybrid_training import (
    HybridTrainError,
    pending_teacher_report,
    require_paid_teacher_approval,
    save_teacher_predictions,
    score_teacher_subset,
    select_teacher_subset,
    teacher_mistake_counts,
)
from src.teacher import TeacherClient


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Score the teacher on 15 new invoices.")
    parser.add_argument("--reports-dir", type=Path, default=Path("reports"))
    parser.add_argument(
        "--predictions",
        type=Path,
        default=Path("data/processed/hybrid/teacher_predictions.jsonl"),
    )
    parser.add_argument("--live", action="store_true")
    parser.add_argument("--approve-cost", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        return _run(args)
    except HybridTrainError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2


def _run(args: argparse.Namespace) -> int:
    invoices = generate_hybrid_invoices()
    reports_dir = Path(args.reports_dir)
    reports_dir.mkdir(parents=True, exist_ok=True)
    report_path = reports_dir / "phase3e_teacher_eval.json"
    if not args.live:
        payload = pending_teacher_report(invoices)
        report_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
        print(json.dumps(payload, indent=2))
        print(f"Report: {report_path}")
        return 0
    require_paid_teacher_approval(live=True, approve_cost=args.approve_cost)
    settings = load_settings()
    client = TeacherClient(settings)
    subset = select_teacher_subset(invoices)
    predictions, scored = score_teacher_subset(
        subset,
        client,
        pricing=configured_pricing(settings),
    )
    save_teacher_predictions(args.predictions, predictions)
    mistakes = teacher_mistake_counts(scored)
    payload = {
        "status": "completed",
        "documents": scored.documents,
        "document_ids": [invoice.document_id for invoice in subset],
        "json_validity": scored.schema_validity_rate,
        "json_valid_count": scored.schema_valid_count,
        "exact_match_accuracy": scored.end_to_end_exact_match_accuracy,
        "exact_match_count": scored.exact_match_count,
        "field_accuracy": [
            {"field": metric.field, "accuracy": metric.accuracy, "correct": metric.correct, "scored": metric.scored}
            for metric in scored.field_metrics
        ],
        "hallucinated_values": mistakes["hallucinated_values"],
        "missing_fields": mistakes["missing_fields"],
        "latency_ms_avg": scored.latency_ms_avg,
        "input_tokens": scored.input_tokens,
        "output_tokens": scored.output_tokens,
        "estimated_cost_usd": None
        if scored.estimated_cost_usd is None
        else str(scored.estimated_cost_usd),
        "predictions_path": str(args.predictions),
    }
    report_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2))
    print(f"Report: {report_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
