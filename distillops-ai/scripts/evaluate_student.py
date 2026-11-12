"""Score a saved student on the validation split.

The locked test split is rejected before any dataset file is opened. Scoring
covers the five student fields only. ``document_type`` is not filled in and
is not part of the accuracy.

    python -m scripts.evaluate_student --model-dir models/student/flan-t5-small-epoch1 --split val
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

from src.dataset import DatasetLoadError, load_split
from src.labeling import report_timestamp
from src.student_data import StudentDataError, require_validation_split
from src.student_eval import format_student_report, score_generations
from src.student_infer import StudentInferenceError, generate_fields, load_student, select_device
from src.student_train import StudentTrainConfig


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Evaluate a saved student on the validation split.")
    parser.add_argument(
        "--split",
        default="val",
        choices=["train", "val", "validation", "test"],
        help="Only 'val' is accepted. Default: val.",
    )
    parser.add_argument(
        "--model-dir",
        type=Path,
        default=Path("models/student/flan-t5-small-json-epoch1"),
    )
    parser.add_argument("--splits-dir", type=Path, default=Path("data/splits"))
    parser.add_argument("--reports-dir", type=Path, default=Path("reports"))
    parser.add_argument("--max-source-length", type=int, default=None)
    parser.add_argument("--max-new-tokens", type=int, default=None)
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        split = require_validation_split(args.split)
        config = _saved_config(args.model_dir)
        max_source_length = args.max_source_length or (
            config.max_source_length if config is not None else 512
        )
        max_new_tokens = args.max_new_tokens or (
            config.max_target_length if config is not None else 128
        )
        records = load_split(args.splits_dir, split)
        if not records:
            print("error: the validation split has no documents.", file=sys.stderr)
            return 3
        device = select_device(args.device)
        model, tokenizer = load_student(args.model_dir, device)
        generations = [
            generate_fields(
                model,
                tokenizer,
                record.document_text,
                document_id=record.document_id,
                max_source_length=max_source_length,
                max_new_tokens=max_new_tokens,
                device=device,
            )
            for record in records
        ]
        report = score_generations(
            records,
            generations,
            split=split,
            model_dir=str(args.model_dir),
            base_model=config.model_name if config is not None else None,
        )
    except (StudentDataError, StudentInferenceError, DatasetLoadError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    except RuntimeError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    text = format_student_report(report)
    print(text)
    reports_dir = Path(args.reports_dir)
    reports_dir.mkdir(parents=True, exist_ok=True)
    stamp = report_timestamp(datetime.now(timezone.utc))
    json_path = reports_dir / f"phase3_student_val_{stamp}.json"
    text_path = reports_dir / f"phase3_student_val_{stamp}.txt"
    json_path.write_text(report.model_dump_json(indent=2) + "\n", encoding="utf-8")
    text_path.write_text(text + "\n", encoding="utf-8")
    print(f"Report: {json_path}")
    return 0


def _saved_config(model_dir: Path) -> StudentTrainConfig | None:
    path = Path(model_dir) / "training_config.json"
    if not path.is_file():
        return None
    return StudentTrainConfig.model_validate(json.loads(path.read_text(encoding="utf-8")))


if __name__ == "__main__":
    sys.exit(main())
