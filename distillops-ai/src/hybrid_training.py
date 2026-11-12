"""Combine verified teacher rows with deterministic hybrid labels.

The new 40 targets are the rendered ground truth, not teacher predictions.
A teacher subset can be scored separately and must not replace those labels.
Training checkpoint choice uses training scores only.
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from src.dataset import write_jsonl
from src.evaluation import EvaluationReport, evaluate_predictions, field_values_match
from src.evaluation_pricing import Pricing
from src.hybrid_dataset import HYBRID_COUNT, HYBRID_GROUPS, HybridInvoice
from src.labeling import StudentFields, StudentTrainingExample
from src.prompts import build_student_source
from src.schemas import EXTRACTION_FIELDS, DatasetRecord, ExtractedInvoice, TeacherPrediction
from src.student_data import StudentPair, pairs_from_examples, teacher_target_json
from src.student_train import StudentTrainConfig
from src.teacher import TeacherClient

ORIGINAL_TRAIN_IDS: frozenset[str] = frozenset(
    {
        "inv-synth-001",
        "inv-synth-002",
        "inv-synth-003",
        "inv-synth-004",
        "inv-synth-005",
        "inv-synth-006",
        "inv-synth-007",
        "inv-synth-008",
        "inv-synth-009",
        "inv-synth-010",
        "inv-synth-016",
        "inv-synth-017",
        "inv-synth-018",
        "inv-synth-019",
        "inv-synth-020",
        "inv-synth-026",
        "inv-synth-027",
        "inv-synth-028",
        "inv-synth-029",
        "inv-synth-030",
    }
)
VALIDATION_IDS: frozenset[str] = frozenset(
    {
        "inv-synth-021",
        "inv-synth-022",
        "inv-synth-023",
        "inv-synth-024",
        "inv-synth-025",
    }
)
HYBRID_OUTPUT_DIR = Path("models/student/flan-t5-small-hybrid60")
PROTECTED_CHECKPOINTS: frozenset[str] = frozenset(
    {
        "flan-t5-small-epoch1",
        "flan-t5-small-json-epoch1",
        "flan-t5-small-json-epoch3",
        "flan-t5-small-overfit5",
        "flan-t5-small-train20-memorize",
    }
)
CONVERGENCE_RULE = (
    "Stop before 40 epochs only when greedy JSON validity and exact field "
    "match on all 60 training examples are both 60/60. The kept checkpoint "
    "is the one with the highest training exact match, then JSON validity, "
    "then closing-brace count, then opening-brace count, then the lower "
    "epoch loss. Validation scores are not an input to that choice."
)
TEACHER_SAMPLE_SIZE = 15


class HybridTrainError(ValueError):
    """Raised when the 60-example training set is not safe to use."""


class ProvenancePair(BaseModel):
    """One student pair plus where its target came from."""

    model_config = ConfigDict(extra="forbid")

    pair: StudentPair
    provenance: Literal["verified_teacher", "deterministic_ground_truth"]


class TrainingCheckpointScore(BaseModel):
    """Training-split scores for one saved epoch. Validation is not stored here."""

    model_config = ConfigDict(extra="forbid")

    epoch: int = Field(ge=1)
    mean_loss: float
    json_valid: int = Field(ge=0)
    opening_brace: int = Field(ge=0)
    closing_brace: int = Field(ge=0)
    exact_match: int = Field(ge=0)
    checkpoint: str


def hybrid_train_config() -> StudentTrainConfig:
    """Fresh FLAN-T5-small run. It does not resume from epoch 25."""
    return StudentTrainConfig(
        model_name="google/flan-t5-small",
        seed=42,
        epochs=40,
        batch_size=2,
        learning_rate=1e-4,
        max_source_length=512,
        max_target_length=128,
        device="cuda",
    )


def assert_fresh_checkpoint(path: Path) -> None:
    """Refuse an output path that would replace an existing student checkpoint."""
    path = Path(path)
    if path.name in PROTECTED_CHECKPOINTS or any(part in PROTECTED_CHECKPOINTS for part in path.parts):
        raise HybridTrainError(f"Refusing to write over {path}.")
    if path.exists():
        raise HybridTrainError(f"{path} already exists and was not overwritten.")


def assemble_training_pairs(
    original: Sequence[StudentTrainingExample],
    hybrid: Sequence[HybridInvoice],
    *,
    enforce_canonical_ids: bool = True,
) -> list[ProvenancePair]:
    """Stack the 20 verified rows in front of the 40 deterministic rows."""
    if enforce_canonical_ids:
        found = {example.document_id for example in original}
        if len(original) != 20 or found != ORIGINAL_TRAIN_IDS:
            raise HybridTrainError("The original training file is not the verified 20 invoices.")
    if len(hybrid) != HYBRID_COUNT:
        raise HybridTrainError(f"Expected {HYBRID_COUNT} hybrid invoices, found {len(hybrid)}.")
    original_ids = {example.document_id for example in original}
    rows: list[ProvenancePair] = [
        ProvenancePair(pair=pair, provenance="verified_teacher")
        for pair in pairs_from_examples(original)
    ]
    for invoice in hybrid:
        if invoice.document_id in original_ids or invoice.document_id in VALIDATION_IDS:
            raise HybridTrainError(f"{invoice.document_id} is not a new training invoice.")
        if invoice.label_source != "deterministic_ground_truth":
            raise HybridTrainError(f"{invoice.document_id} is not a deterministic label.")
        source = build_student_source(invoice.document_text)
        if invoice.document_text not in source or "DOCUMENT START" not in source:
            raise HybridTrainError(f"{invoice.document_id} prompt is not the student source.")
        if "<document>" in source or "ground_truth" in source:
            raise HybridTrainError(f"{invoice.document_id} prompt includes held-out text.")
        rows.append(
            ProvenancePair(
                pair=StudentPair(
                    document_id=invoice.document_id,
                    source_text=source,
                    target_text=teacher_target_json(invoice.ground_truth),
                    teacher_output=invoice.ground_truth,
                ),
                provenance="deterministic_ground_truth",
            )
        )
    if len(rows) != 60:
        raise HybridTrainError(f"Expected 60 training rows, built {len(rows)}.")
    verified = sum(row.provenance == "verified_teacher" for row in rows)
    deterministic = sum(row.provenance == "deterministic_ground_truth" for row in rows)
    if verified != 20 or deterministic != 40:
        raise HybridTrainError(
            f"Unexpected label provenance counts: {verified} verified, {deterministic} deterministic."
        )
    return rows


def select_teacher_subset(invoices: Sequence[HybridInvoice]) -> list[HybridInvoice]:
    """Take 15 invoices with fixed offsets inside each layout. No random draw."""
    if len(invoices) != HYBRID_COUNT:
        raise HybridTrainError("Teacher sampling expects the 40 hybrid invoices.")
    ordered = sorted(invoices, key=lambda invoice: invoice.document_id)
    chosen: list[HybridInvoice] = []
    for group_index, group in enumerate(HYBRID_GROUPS):
        block = [invoice for invoice in ordered if invoice.template_group == group]
        if len(block) != 10:
            raise HybridTrainError(f"{group} does not have 10 invoices.")
        offsets = [0, 4, 9] if group_index == len(HYBRID_GROUPS) - 1 else [0, 2, 4, 9]
        chosen.extend(block[offset] for offset in offsets)
    if len(chosen) != TEACHER_SAMPLE_SIZE:
        raise HybridTrainError(f"Teacher subset has {len(chosen)} invoices, not 15.")
    if len({invoice.document_id for invoice in chosen}) != TEACHER_SAMPLE_SIZE:
        raise HybridTrainError("Teacher subset ids are not unique.")
    return chosen


def require_paid_teacher_approval(*, live: bool, approve_cost: bool) -> None:
    """Real OpenAI calls wait for an explicit cost approval."""
    if not live:
        return
    if not approve_cost:
        raise HybridTrainError(
            "Paid teacher calls need --live and --approve-cost. No request was sent."
        )


def hybrid_record(invoice: HybridInvoice) -> DatasetRecord:
    """View a deterministic label as the schema the teacher scorer already uses."""
    fields = invoice.ground_truth
    return DatasetRecord(
        document_id=invoice.document_id,
        document_text=invoice.document_text,
        template_group=invoice.template_group,
        ground_truth=ExtractedInvoice(
            document_type="invoice",
            vendor=fields.vendor or None,
            invoice_number=fields.invoice_number or None,
            invoice_date=fields.date or None,
            total=fields.total or None,
            currency=fields.currency or None,
        ),
    )


def score_teacher_subset(
    invoices: Sequence[HybridInvoice],
    client: TeacherClient,
    *,
    pricing: Pricing | None = None,
) -> tuple[list[TeacherPrediction], EvaluationReport]:
    """Score the teacher. The deterministic labels on ``invoices`` are not edited."""
    before = [invoice.ground_truth.model_dump() for invoice in invoices]
    predictions = [
        client.extract(document_id=invoice.document_id, document_text=invoice.document_text)
        for invoice in invoices
    ]
    after = [invoice.ground_truth.model_dump() for invoice in invoices]
    if before != after:
        raise HybridTrainError("Teacher scoring changed a deterministic label.")
    report = evaluate_predictions(
        [hybrid_record(invoice) for invoice in invoices],
        predictions,
        split="hybrid_teacher_subset",
        teacher_model=client.settings.teacher_model or "",
        pricing=pricing,
        limit=len(invoices),
    )
    return predictions, report


def save_teacher_predictions(path: Path, predictions: Sequence[TeacherPrediction]) -> None:
    """Write teacher output beside the deterministic labels, not over them."""
    path = Path(path)
    if path.name.lower() in {"ground_truth.jsonl", "student_train.jsonl", "test.jsonl"}:
        raise HybridTrainError(f"Refusing to write teacher predictions to {path.name}.")
    write_jsonl(path, predictions)


def teacher_mistake_counts(report: EvaluationReport) -> dict[str, int]:
    """Count missing and hallucinated fields on schema-valid teacher rows."""
    hallucinated = 0
    missing = 0
    for row in report.results:
        if not row.schema_valid or row.prediction is None:
            continue
        for field in EXTRACTION_FIELDS:
            if field == "document_type":
                continue
            if row.field_matches.get(field, False):
                continue
            predicted = getattr(row.prediction, field)
            expected = getattr(row.ground_truth, field)
            if _blank(predicted) and not _blank(expected):
                missing += 1
            elif not _blank(predicted) and _blank(expected):
                hallucinated += 1
    return {"hallucinated_values": hallucinated, "missing_fields": missing}


def select_training_checkpoint(scores: Sequence[TrainingCheckpointScore]) -> str:
    """Pick a checkpoint from training scores. Validation fields are not accepted."""
    if not scores:
        raise HybridTrainError("There is no training checkpoint to select.")
    ranked = sorted(
        scores,
        key=lambda score: (
            score.exact_match,
            score.json_valid,
            score.closing_brace,
            score.opening_brace,
            -score.mean_loss,
        ),
    )
    return ranked[-1].checkpoint


def convergence_reached(score: TrainingCheckpointScore, training_rows: int) -> bool:
    """True when the predefined training-set criterion is met."""
    return score.json_valid == training_rows and score.exact_match == training_rows


def pending_teacher_report(invoices: Sequence[HybridInvoice]) -> dict[str, object]:
    """Describe the 15-invoice teacher run without sending it."""
    subset = select_teacher_subset(invoices)
    return {
        "status": "pending",
        "reason": (
            "No OpenAI request was sent. A live teacher evaluation needs "
            "--live and --approve-cost."
        ),
        "documents": len(subset),
        "document_ids": [invoice.document_id for invoice in subset],
        "label_source_unchanged": "deterministic_ground_truth",
        "json_validity": None,
        "exact_match_accuracy": None,
        "field_accuracy": None,
        "hallucinated_values": None,
        "missing_fields": None,
        "latency_ms_avg": None,
        "input_tokens": None,
        "output_tokens": None,
        "estimated_cost_usd": None,
    }


def classify_student_failure(
    *,
    parsed: StudentFields | None,
    expected: StudentFields,
    invoice_text: str,
    opening_ok: bool,
    closing_ok: bool,
) -> list[str]:
    """Name the failure kinds for one greedy decode. A match returns an empty list."""
    found: list[str] = []
    if parsed is None:
        found.append("malformed JSON")
    if not opening_ok or not closing_ok or parsed is None:
        found.append("formatting error")
    if parsed is None:
        return found
    for student_field, eval_field in _EVAL_FIELD.items():
        predicted = getattr(parsed, student_field)
        truth = getattr(expected, student_field)
        if safe_field_match(eval_field, predicted, truth):
            continue
        if _blank(predicted) and not _blank(truth):
            found.append("missing fields")
        elif not _blank(predicted) and _blank(truth):
            found.append("hallucinated values")
        elif not _blank(predicted) and predicted not in invoice_text:
            found.append("hallucinated values")
            found.append("incorrect extraction")
        else:
            found.append("incorrect extraction")
    return list(dict.fromkeys(found))


_EVAL_FIELD = {
    "vendor": "vendor",
    "invoice_number": "invoice_number",
    "date": "invoice_date",
    "total": "total",
    "currency": "currency",
}


def safe_field_match(eval_field: str, predicted: object, expected: object) -> bool:
    """Return false when a parsed string is not a date or amount the scorer can read."""
    try:
        return field_values_match(eval_field, predicted, expected)
    except (ValueError, TypeError, ArithmeticError):
        return False


def _blank(value: object) -> bool:
    if value is None:
        return True
    return str(value).strip() == ""
