"""Train a fresh FLAN-T5-small student on 72 examples.

The 60 hybrid rows stay as they are. Twelve new invoices add a Dated line
with an abbreviated month and a dotted total. The run starts from
google/flan-t5-small, chooses the checkpoint on the training set, then scores
the five validation invoices once.

    python -m scripts.train_dated_supplement
"""

from __future__ import annotations

from pathlib import Path

from scripts.train_hybrid_student import run_training
from src.dated_supplement import generate_dated_invoices, write_dated_dataset
from src.hybrid_training import hybrid_train_config

OUTPUT_DIR = Path("models/student/flan-t5-small-dated12")
REPORT_PATH = Path("reports/phase3f_dated12.json")
PREVIOUS = Path("models/student/flan-t5-small-hybrid60/epoch-30")
CONVERGENCE_RULE = (
    "Stop before 40 epochs only when greedy JSON validity and exact field "
    "match on all 72 training examples are both 72/72. The kept checkpoint "
    "is the one with the highest training exact match, then JSON validity, "
    "then closing-brace count, then opening-brace count, then the lower "
    "epoch loss. Validation scores are not an input to that choice."
)


def main() -> None:
    config = hybrid_train_config()
    dated = generate_dated_invoices(config.seed)
    write_dated_dataset(Path("data/processed/dated_supplement"), dated)
    run_training(
        output_dir=OUTPUT_DIR,
        report_path=REPORT_PATH,
        extra_invoices=dated,
        previous_checkpoint=PREVIOUS,
        previous_key="hybrid_epoch30_validation",
        convergence_rule=CONVERGENCE_RULE,
        provenance={
            "verified_teacher": 20,
            "deterministic_ground_truth": 52,
            "dated_supplement": 12,
            "teacher_subset": "evaluation_only_not_used_as_targets",
        },
    )


if __name__ == "__main__":
    main()
