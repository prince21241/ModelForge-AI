"""Create verified student-training rows from the teacher.

Live calls are off unless ``--live`` is present. The command reads the training
split only. Validation is for scoring, and the test split stays locked.

    python -m scripts.generate_teacher_labels
    python -m scripts.generate_teacher_labels --dry-run
    python -m scripts.generate_teacher_labels --live --limit 2
    python -m scripts.generate_teacher_labels --live
"""

from __future__ import annotations

import argparse
import logging
import sys
from datetime import datetime, timezone
from pathlib import Path

from src.config import ConfigError, load_settings
from src.dataset import DatasetLoadError, load_split
from src.labeling import (
    HeldOutTrainingError,
    format_label_report,
    generate_teacher_labels,
    report_timestamp,
    require_train_split,
)
from src.teacher import TeacherConfigError


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Label the training invoices with the teacher. "
            "No API call is made unless --live is set."
        )
    )
    parser.add_argument(
        "--split",
        default="train",
        choices=["train", "val", "validation", "test"],
        help="Only 'train' is accepted. Default: train.",
    )
    parser.add_argument(
        "--live",
        action="store_true",
        help="Send invoices to the teacher. Omit this for a dry run.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Describe the run and do not call the API. This is the default.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Maximum number of new teacher requests on a live run.",
    )
    parser.add_argument(
        "--concurrency",
        type=int,
        default=1,
        help="Must stay 1. Use --limit to control spend.",
    )
    parser.add_argument(
        "--fresh",
        action="store_true",
        help="Ignore the checkpoint and label from scratch on a live run.",
    )
    parser.add_argument("--splits-dir", type=Path, default=Path("data/splits"))
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/processed/teacher_labels"),
        help="Checkpoint and student JSONL directory.",
    )
    parser.add_argument("--reports-dir", type=Path, default=Path("reports"))
    parser.add_argument("--env-file", type=Path, default=Path(".env"))
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    live = bool(args.live) and not bool(args.dry_run)
    try:
        require_train_split(args.split)
        if args.limit is not None and args.limit <= 0:
            raise ValueError("--limit must be a positive integer.")
        settings = load_settings(args.env_file)
        records = load_split(args.splits_dir, "train")
        if not records:
            print("error: the training split has no documents.", file=sys.stderr)
            return 3
        report = generate_teacher_labels(
            records,
            settings,
            checkpoint_path=args.output_dir / "checkpoint.jsonl",
            export_path=args.output_dir / "student_train.jsonl",
            live=live,
            limit=args.limit,
            concurrency=args.concurrency,
            reuse_checkpoint=not args.fresh,
        )
    except (HeldOutTrainingError, TeacherConfigError, ConfigError, DatasetLoadError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    text = format_label_report(report)
    print(text)
    reports_dir = Path(args.reports_dir)
    reports_dir.mkdir(parents=True, exist_ok=True)
    stamp = report_timestamp(datetime.now(timezone.utc))
    json_path = reports_dir / f"phase2_teacher_labels_{stamp}.json"
    text_path = reports_dir / f"phase2_teacher_labels_{stamp}.txt"
    json_path.write_text(report.model_dump_json(indent=2) + "\n", encoding="utf-8")
    text_path.write_text(text + "\n", encoding="utf-8")
    print(f"Report: {json_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
