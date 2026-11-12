"""Give FLAN-T5 real tokens for JSON braces.

The published vocabulary has no ``{`` or ``}``. Both were encoded as ``<unk>``,
so the decoder could not spell a JSON object. These helpers add the two
characters with the tokenizer API and check that a teacher target still
decodes with its braces intact.
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from pathlib import Path

from transformers import AddedToken

from src.student_data import StudentPair, token_ids

BASE_STUDENT_MODEL = "google/flan-t5-small"
ORIGINAL_CHECKPOINT_NAME = "flan-t5-small-epoch1"
JSON_BRACE_TOKENS: tuple[str, str] = ("{", "}")
JSON_ROUNDTRIP_SAMPLE = (
    '{"vendor":"Acme","invoice_number":"INV-1","date":"","total":"10.00","currency":"USD"}'
)


class StudentTokenizerError(ValueError):
    """Raised when brace tokens or the source wrapper are not usable."""


def require_base_flan_model(model_name: str) -> None:
    """Training starts from the published weights, not the failed checkpoint."""
    if model_name != BASE_STUDENT_MODEL:
        raise StudentTokenizerError(
            "Student training starts from google/flan-t5-small. "
            "The original one-epoch checkpoint is not a valid starting point."
        )


def require_new_output_dir(output_dir: Path) -> None:
    """Keep the original one-epoch checkpoint where it is."""
    if Path(output_dir).name == ORIGINAL_CHECKPOINT_NAME:
        raise StudentTokenizerError("Refusing to overwrite the original one-epoch checkpoint.")


def add_json_brace_tokens(tokenizer) -> tuple[int, int]:
    """Add ``{`` and ``}`` when they are still unknown. Returns their ids."""
    existing = _brace_ids(tokenizer)
    if existing is not None:
        return existing
    added = tokenizer.add_tokens(
        [
            AddedToken("{", normalized=False, special=False),
            AddedToken("}", normalized=False, special=False),
        ],
        special_tokens=False,
    )
    if added != 2:
        raise StudentTokenizerError("The tokenizer did not add distinct tokens for { and }.")
    checked = _brace_ids(tokenizer)
    if checked is None:
        raise StudentTokenizerError("{ and } are still not distinct known tokens.")
    return checked


def verify_json_target(tokenizer, target_text: str) -> str:
    """Decode one JSON target and require the same object, including braces."""
    unknown_id = tokenizer.unk_token_id
    brace_ids = _brace_ids(tokenizer)
    if brace_ids is None:
        raise StudentTokenizerError("{ and } must be distinct tokens before training.")
    token_id_list = token_ids(tokenizer, target_text, add_special_tokens=False)
    if unknown_id in token_id_list:
        raise StudentTokenizerError("A JSON target still contains <unk>.")
    if brace_ids[0] not in token_id_list or brace_ids[1] not in token_id_list:
        raise StudentTokenizerError("A JSON target lost a brace token.")
    decoded = tokenizer.decode(token_id_list, skip_special_tokens=True)
    if "{" not in decoded or "}" not in decoded:
        raise StudentTokenizerError("Decoding removed the JSON braces.")
    if json.loads(decoded) != json.loads(target_text):
        raise StudentTokenizerError("Decoding changed the JSON target.")
    return decoded


def verify_training_texts(tokenizer, pairs: Sequence[StudentPair]) -> str:
    """Check the sample target, every teacher target, and every source."""
    if not pairs:
        raise StudentTokenizerError("There are no training pairs to verify.")
    sample_decoded = verify_json_target(tokenizer, JSON_ROUNDTRIP_SAMPLE)
    unknown_id = tokenizer.unk_token_id
    for pair in pairs:
        verify_json_target(tokenizer, pair.target_text)
        if "<document>" in pair.source_text or "</document>" in pair.source_text:
            raise StudentTokenizerError(f"{pair.document_id} still uses an angle-bracket wrapper.")
        if "DOCUMENT START" not in pair.source_text or "DOCUMENT END" not in pair.source_text:
            raise StudentTokenizerError(f"{pair.document_id} is missing the document markers.")
        if pair.source_text.count("DOCUMENT START") != 1 or pair.source_text.count("DOCUMENT END") != 1:
            raise StudentTokenizerError(f"{pair.document_id} has repeated document markers.")
        source_ids = token_ids(tokenizer, pair.source_text, add_special_tokens=False)
        if unknown_id in source_ids:
            raise StudentTokenizerError(
                f"{pair.document_id} source still contains an unknown token."
            )
    return sample_decoded


def resize_student_embeddings(model, tokenizer) -> tuple[int, int]:
    """Grow the input and output matrices to the expanded vocabulary.

    The published FLAN-T5 checkpoint stores the input embedding and the output
    head as different tensors. Resizing must keep them separate. Tying them
    replaces the pretrained output head and the loss stops being meaningful.
    """
    vocabulary_size = len(tokenizer)
    model.config.tie_word_embeddings = False
    model.resize_token_embeddings(vocabulary_size)
    input_embedding = model.get_input_embeddings()
    output_embedding = model.get_output_embeddings()
    input_rows = int(input_embedding.weight.shape[0])
    output_rows = input_rows if output_embedding is None else int(output_embedding.weight.shape[0])
    if input_rows != vocabulary_size or output_rows != vocabulary_size:
        raise StudentTokenizerError(
            "The resized model vocabulary does not match the tokenizer "
            f"({input_rows} input rows, {output_rows} output rows, {vocabulary_size} tokens)."
        )
    if output_embedding is not None and input_embedding.weight.data_ptr() == output_embedding.weight.data_ptr():
        raise StudentTokenizerError(
            "Resizing tied the output head to the input embedding. The pretrained head was not kept."
        )
    return input_rows, output_rows


def _brace_ids(tokenizer) -> tuple[int, int] | None:
    unknown_id = tokenizer.unk_token_id
    left = tokenizer.convert_tokens_to_ids("{")
    right = tokenizer.convert_tokens_to_ids("}")
    if left is None or right is None:
        return None
    if left == unknown_id or right == unknown_id or left == right:
        return None
    if tokenizer.convert_ids_to_tokens(left) != "{" or tokenizer.convert_ids_to_tokens(right) != "}":
        return None
    return int(left), int(right)
