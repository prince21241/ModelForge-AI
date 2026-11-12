"""Fine-tune the student on verified teacher labels.

The command reads the Phase 2 training file only. It does not open the
validation split or the locked test split, and it does not call OpenAI.

    python -m scripts.train_student --epochs 1 --output-dir models/student/flan-t5-small-json-epoch1
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from src.dataset import DatasetLoadError
from src.student_data import StudentDataError, load_training_examples, pairs_from_examples
from src.student_infer import StudentInferenceError
from src.student_tokenizer import StudentTokenizerError
from src.student_train import (
    DEFAULT_STUDENT_MODEL,
    StudentTrainConfig,
    StudentTrainError,
    train_student,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Fine-tune a small student on the verified teacher JSONL file."
    )
    parser.add_argument(
        "--labels",
        type=Path,
        default=Path("data/processed/teacher_labels/student_train.jsonl"),
        help="Verified teacher-label JSONL. The test split is rejected.",
    )
    parser.add_argument("--model", default=DEFAULT_STUDENT_MODEL)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("models/student/flan-t5-small-json-epoch1"),
    )
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--max-source-length", type=int, default=512)
    parser.add_argument("--max-target-length", type=int, default=128)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        if args.epochs < 1 or args.batch_size < 1:
            raise StudentTrainError("Epochs and batch size must be positive.")
        examples = load_training_examples(args.labels)
        pairs = pairs_from_examples(examples)
        config = StudentTrainConfig(
            model_name=args.model,
            seed=args.seed,
            epochs=args.epochs,
            batch_size=args.batch_size,
            learning_rate=args.learning_rate,
            weight_decay=args.weight_decay,
            max_source_length=args.max_source_length,
            max_target_length=args.max_target_length,
            device=args.device,
        )
        metrics = train_student(pairs, config, args.output_dir)
    except (
        StudentDataError,
        StudentTrainError,
        StudentInferenceError,
        StudentTokenizerError,
        DatasetLoadError,
        ValueError,
    ) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    except RuntimeError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    print(f"Examples: {metrics.examples}")
    print(f"Steps: {metrics.steps}")
    print(f"Mean loss: {metrics.mean_loss:.4f}")
    print("Epoch mean loss: " + ", ".join(f"{value:.4f}" for value in metrics.epoch_mean_loss))
    print(f"Device: {metrics.device}")
    print(f"Parameters: {metrics.parameter_count}")
    print(f"GPU memory after load bytes: {metrics.gpu_memory_after_load_bytes}")
    print(f"Peak GPU allocated bytes: {metrics.peak_gpu_memory_allocated_bytes}")
    print(f"Peak GPU reserved bytes: {metrics.peak_gpu_memory_reserved_bytes}")
    print(f"Reload ok: {metrics.reload_ok}")
    print(f"Truncated inputs: {metrics.truncated_inputs}")
    print(f"Saved: {args.output_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
