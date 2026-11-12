"""Load verified teacher labels and turn them into student training pairs.

The locked test split is refused by file name before it is opened. Validation
rows are not read here. Ground truth is not copied into the model input.
"""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, Protocol

from pydantic import BaseModel, ConfigDict, Field

from src.dataset import DatasetLoadError, read_jsonl
from src.labeling import StudentFields, StudentTrainingExample
from src.prompts import STUDENT_TARGET_INSTRUCTION, build_student_source

STUDENT_OUTPUT_FIELDS: tuple[str, ...] = (
    "vendor",
    "invoice_number",
    "date",
    "total",
    "currency",
)
LOCKED_TEST_FILENAMES: frozenset[str] = frozenset({"test.jsonl"})


class StudentDataError(ValueError):
    """Raised when a student example cannot be used without changing its text."""


class TokenEncoder(Protocol):
    """The small tokenizer surface the length checks need."""

    def __call__(
        self,
        text: str,
        *,
        truncation: bool = False,
        padding: bool | str = False,
        add_special_tokens: bool = True,
        return_attention_mask: bool = False,
    ) -> Mapping[str, Any]:
        """Encode one string. Truncation must stay off."""


class StudentPair(BaseModel):
    """One supervised pair. The target is the teacher JSON, not the answer key."""

    model_config = ConfigDict(extra="forbid")

    document_id: str
    source_text: str = Field(min_length=1)
    target_text: str = Field(min_length=1)
    teacher_output: StudentFields


def refuse_locked_test_path(path: Path) -> None:
    """Reject the held-out test file before any open call."""
    if Path(path).name.lower() in LOCKED_TEST_FILENAMES:
        raise StudentDataError(
            "The locked test split cannot be opened for student training or evaluation."
        )


def require_validation_split(split: str) -> str:
    """Student scoring reads the validation split and nothing else."""
    normalized = split.strip().lower()
    if normalized == "validation":
        normalized = "val"
    if normalized == "test":
        raise StudentDataError("Student evaluation does not read the locked test split.")
    if normalized != "val":
        raise StudentDataError("Student evaluation reads only the validation split.")
    return "val"


def teacher_target_json(fields: StudentFields) -> str:
    """Compact JSON with empty strings kept for missing fields."""
    payload = {field: getattr(fields, field) for field in STUDENT_OUTPUT_FIELDS}
    return json.dumps(payload, ensure_ascii=False, separators=(",", ":"))


def parse_student_json(text: str) -> tuple[StudentFields | None, str | None]:
    """Parse a student generation. Missing keys are not filled in."""
    try:
        payload = json.loads(text)
    except json.JSONDecodeError:
        return None, "Output is not valid JSON."
    if not isinstance(payload, dict):
        return None, "Output JSON must be an object."
    if set(payload) != set(STUDENT_OUTPUT_FIELDS):
        return None, "Output JSON keys do not match the five student fields."
    for field in STUDENT_OUTPUT_FIELDS:
        if not isinstance(payload[field], str):
            return None, f"{field} must be a string. Use an empty string when it is missing."
    return StudentFields.model_validate(payload), None


def validate_training_example(example: StudentTrainingExample) -> None:
    """Reject a row that drops text, invents a field, or carries ground truth."""
    if example.document_id.lower().startswith("test-"):
        raise StudentDataError(
            f"{example.document_id} is reserved for the locked test split."
        )
    if not example.invoice_text.strip():
        raise StudentDataError(f"{example.document_id} has blank invoice text.")
    roles = [turn.role for turn in example.messages[:3]]
    if roles != ["system", "user", "assistant"]:
        raise StudentDataError(
            f"{example.document_id} is not a system, user, assistant example."
        )
    system, user, assistant = (turn.content for turn in example.messages[:3])
    if system != STUDENT_TARGET_INSTRUCTION:
        raise StudentDataError(
            f"{example.document_id} does not use the student instruction."
        )
    if example.invoice_text not in user:
        raise StudentDataError(
            f"{example.document_id} does not preserve the original invoice text."
        )
    if "ground_truth" in user or "ground_truth" in assistant:
        raise StudentDataError(f"{example.document_id} contains ground truth.")
    parsed, error = parse_student_json(assistant)
    if parsed is None:
        raise StudentDataError(error or f"{example.document_id} has invalid teacher JSON.")
    if parsed != example.teacher_output:
        raise StudentDataError(
            f"{example.document_id} assistant text does not match the teacher output."
        )


def load_training_examples(path: Path) -> list[StudentTrainingExample]:
    """Load verified teacher rows. A test filename is rejected before reading."""
    path = Path(path)
    refuse_locked_test_path(path)
    try:
        examples = read_jsonl(path, StudentTrainingExample)
    except DatasetLoadError:
        raise
    if not examples:
        raise StudentDataError(f"{path.name} has no training examples.")
    seen: set[str] = set()
    for example in examples:
        if example.document_id in seen:
            raise StudentDataError(f"Duplicate training document {example.document_id}.")
        seen.add(example.document_id)
        validate_training_example(example)
    return examples


def pairs_from_examples(examples: Sequence[StudentTrainingExample]) -> list[StudentPair]:
    """Build encoder text and the teacher target. Invoice text is not shortened."""
    pairs: list[StudentPair] = []
    for example in examples:
        validate_training_example(example)
        source = build_student_source(example.invoice_text)
        if example.invoice_text not in source:
            raise StudentDataError(
                f"{example.document_id} source dropped the invoice text."
            )
        if "ground_truth" in source:
            raise StudentDataError(f"{example.document_id} source contains ground truth.")
        target = example.messages[2].content
        pairs.append(
            StudentPair(
                document_id=example.document_id,
                source_text=source,
                target_text=target,
                teacher_output=example.teacher_output,
            )
        )
    return pairs


def token_ids(
    tokenizer: TokenEncoder,
    text: str,
    *,
    add_special_tokens: bool = True,
) -> list[int]:
    """Encode one string with truncation left off."""
    encoded = tokenizer(
        text,
        truncation=False,
        padding=False,
        add_special_tokens=add_special_tokens,
        return_attention_mask=False,
    )
    ids = encoded["input_ids"]
    if isinstance(ids, Mapping):
        raise StudentDataError("Tokenizer returned an unexpected input id shape.")
    if ids and isinstance(ids[0], list):
        ids = ids[0]
    return [int(token_id) for token_id in ids]


def assert_tokenizer_keeps_text(tokenizer: TokenEncoder, text: str) -> None:
    """Fail when a tokenizer cuts text instead of returning every token.

    A marker is appended and encoded again. Equal lengths mean the extra
    characters were dropped, which is the silent-truncation case this project
    refuses.
    """
    marker = "\nEND_OF_DOCUMENT_MARKER"
    without_marker = token_ids(tokenizer, text, add_special_tokens=False)
    with_marker = token_ids(tokenizer, text + marker, add_special_tokens=False)
    if len(with_marker) <= len(without_marker):
        raise StudentDataError(
            "The tokenizer did not keep the full text. "
            "Training stopped instead of truncating the invoice."
        )


def require_fits(ids: Sequence[int], limit: int, document_id: str, side: str) -> None:
    """Raise when a sequence is longer than the configured maximum."""
    if limit < 1:
        raise StudentDataError("Sequence limits must be positive.")
    if len(ids) > limit:
        raise StudentDataError(
            f"{document_id} {side} is {len(ids)} tokens, above the limit of {limit}. "
            "The text was not truncated."
        )
