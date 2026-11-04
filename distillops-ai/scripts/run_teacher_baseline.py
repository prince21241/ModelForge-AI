"""Run the teacher extraction baseline on train or validation data.

The test split is refused. It is held out for a later, one-time comparison and
is not available for prompt tuning.

    python -m scripts.run_teacher_baseline --split val
    python -m scripts.run_teacher_baseline --split val --limit 2
"""

from __future__ import annotations

import argparse
import logging
import sys
from collections.abc import Callable
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from src.config import ConfigError, configured_pricing, load_settings
from src.dataset import DatasetLoadError, load_split
from src.evaluation import EvaluationReport, evaluate_predictions, format_report
from src.teacher import TeacherClient, TeacherConfigError


class HeldOutSplitError(ValueError):
    """Raised when a command would read the held-out test split."""


class EmptySplitError(DatasetLoadError):
    """Raised when the requested split has no documents to score."""


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Score the teacher model on a train or validation split. The test split is held out."
    )
    parser.add_argument(
        "--split",
        default="val",
        choices=["train", "val", "validation", "test"],
        help="Split to score. 'test' is rejected. Default: val.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Score only the first N documents. Use this to cap API spend.",
    )
    parser.add_argument("--splits-dir", type=Path, default=Path("data/splits"))
    parser.add_argument("--reports-dir", type=Path, default=Path("reports"))
    parser.add_argument("--env-file", type=Path, default=Path(".env"))
    return parser


def resolve_eval_split(split: str) -> str:
    """Map CLI split names to files. Never returns the test split."""
    normalized = split.strip().lower()
    if normalized == "validation":
        normalized = "val"
    if normalized == "test":
        raise HeldOutSplitError(
            "The test split is held out. run_teacher_baseline will not read it. "
            "Use --split val. Do not use the test split for prompt tuning."
        )
    if normalized not in {"train", "val"}:
        raise HeldOutSplitError(f"Unsupported split {split!r}. Use train or val.")
    return normalized


def run_baseline(
    *,
    split: str,
    splits_dir: Path,
    reports_dir: Path,
    settings: Any,
    limit: int | None = None,
    client: Any | None = None,
    sleep_fn: Callable[[float], None] | None = None,
    now: Callable[[], datetime] | None = None,
) -> tuple[EvaluationReport, Path]:
    """Load one split, call the teacher, score the outputs, and write a JSON report."""
    resolved = resolve_eval_split(split)
    if limit is not None and limit <= 0:
        raise ValueError("--limit must be a positive integer.")
    records = load_split(splits_dir, resolved)
    if limit is not None:
        records = records[:limit]
    if not records:
        raise EmptySplitError(f"No documents to evaluate in the {resolved} split.")

    teacher_kwargs: dict[str, Any] = {"client": client}
    if sleep_fn is not None:
        teacher_kwargs["sleep_fn"] = sleep_fn
    teacher = TeacherClient(settings, **teacher_kwargs)
    predictions = [
        teacher.extract(document_id=record.document_id, document_text=record.document_text)
        for record in records
    ]
    report = evaluate_predictions(
        records,
        predictions,
        split=resolved,
        teacher_model=settings.teacher_model or "",
        pricing=configured_pricing(settings),
        limit=limit,
    )
    reports_dir.mkdir(parents=True, exist_ok=True)
    current = now() if now is not None else datetime.now(timezone.utc)
    stamp = current.strftime("%Y%m%dT%H%M%SZ")
    path = reports_dir / f"teacher_baseline_{resolved}_{stamp}.json"
    path.write_text(report.model_dump_json(indent=2) + "\n", encoding="utf-8")
    return report, path


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    try:
        resolve_eval_split(args.split)
        if args.limit is not None and args.limit <= 0:
            raise ValueError("--limit must be a positive integer.")
        settings = load_settings(args.env_file)
        report, path = run_baseline(
            split=args.split,
            splits_dir=args.splits_dir,
            reports_dir=args.reports_dir,
            settings=settings,
            limit=args.limit,
        )
    except EmptySplitError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 3
    except (HeldOutSplitError, TeacherConfigError, ConfigError, DatasetLoadError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    print(format_report(report))
    print(f"Report: {path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
