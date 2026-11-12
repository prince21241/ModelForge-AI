"""Load a saved student and extract the five invoice fields.

Generation does not call OpenAI. Inputs that do not fit the configured length
raise an error instead of being truncated. ``document_type`` is not added to
the model output.
"""

from __future__ import annotations

import inspect
import time
from pathlib import Path
from typing import Any

import torch

from src.prompts import build_student_source
from src.student_data import (
    assert_tokenizer_keeps_text,
    parse_student_json,
    require_fits,
    token_ids,
)
from src.student_eval import StudentGeneration


class StudentInferenceError(RuntimeError):
    """Raised when a saved student cannot be loaded or run."""


def select_device(requested: str = "auto") -> torch.device:
    """Pick CUDA when it is available, unless the caller asks for CPU."""
    choice = requested.strip().lower()
    if choice not in {"auto", "cpu", "cuda"}:
        raise StudentInferenceError("Device must be auto, cpu, or cuda.")
    if choice == "cpu":
        return torch.device("cpu")
    if choice == "cuda" and not torch.cuda.is_available():
        raise StudentInferenceError("CUDA was requested but PyTorch cannot see a GPU.")
    if choice == "cuda" or torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def load_seq2seq(model_name_or_path: str) -> Any:
    """Load a sequence-to-sequence model in float32 without replacing PyTorch."""
    from transformers import AutoModelForSeq2SeqLM

    loader = AutoModelForSeq2SeqLM.from_pretrained
    try:
        parameter_names = set(inspect.signature(loader).parameters)
    except (TypeError, ValueError):
        parameter_names = set()
    if "dtype" in parameter_names or not parameter_names:
        try:
            return loader(model_name_or_path, dtype=torch.float32)
        except TypeError:
            pass
    if "torch_dtype" in parameter_names:
        return loader(model_name_or_path, torch_dtype=torch.float32)
    return loader(model_name_or_path)


def load_tokenizer(model_name_or_path: str) -> Any:
    from transformers import AutoTokenizer

    return AutoTokenizer.from_pretrained(model_name_or_path)


def load_student(model_dir: Path, device: torch.device) -> tuple[Any, Any]:
    """Reload weights and tokenizer from a directory written by training."""
    model_dir = Path(model_dir)
    if not model_dir.is_dir():
        raise StudentInferenceError(f"Student directory not found: {model_dir}")
    tokenizer = load_tokenizer(str(model_dir))
    model = load_seq2seq(str(model_dir))
    if getattr(tokenizer, "pad_token_id", None) is None:
        raise StudentInferenceError("The student tokenizer has no pad token.")
    model.to(device)
    model.eval()
    return model, tokenizer


def generate_fields(
    model: Any,
    tokenizer: Any,
    invoice_text: str,
    *,
    document_id: str,
    max_source_length: int,
    max_new_tokens: int,
    device: torch.device,
) -> StudentGeneration:
    """Greedy-decode one invoice. The raw string is parsed and not repaired."""
    if not invoice_text.strip():
        raise StudentInferenceError(f"{document_id} has blank invoice text.")
    source = build_student_source(invoice_text)
    if invoice_text not in source:
        raise StudentInferenceError(f"{document_id} prompt dropped the invoice text.")
    assert_tokenizer_keeps_text(tokenizer, source)
    input_ids = token_ids(tokenizer, source, add_special_tokens=True)
    require_fits(input_ids, max_source_length, document_id, "input")
    tensor = torch.tensor([input_ids], dtype=torch.long, device=device)
    attention = torch.ones_like(tensor)
    start = time.perf_counter()
    with torch.inference_mode():
        output_ids = model.generate(
            input_ids=tensor,
            attention_mask=attention,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            num_beams=1,
        )
    latency_ms = (time.perf_counter() - start) * 1000
    raw_text = tokenizer.decode(output_ids[0], skip_special_tokens=True).strip()
    fields, error = parse_student_json(raw_text)
    return StudentGeneration(
        document_id=document_id,
        raw_text=raw_text,
        json_valid=fields is not None,
        fields=fields,
        validation_error=error,
        latency_ms=latency_ms,
    )
