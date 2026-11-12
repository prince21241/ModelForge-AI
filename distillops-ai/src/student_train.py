"""Fine-tune a small sequence-to-sequence student on verified teacher JSON.

The loop is single-process. Invoice text that does not fit the configured
length stops the run. Weights are saved with the tokenizer and the training
config so a later command can reload them without OpenAI.
"""

from __future__ import annotations

import random
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import torch
from pydantic import BaseModel, ConfigDict, Field

from src.student_data import (
    StudentPair,
    assert_tokenizer_keeps_text,
    require_fits,
    token_ids,
)
from src.student_infer import load_seq2seq, load_student, load_tokenizer, select_device
from src.student_tokenizer import (
    BASE_STUDENT_MODEL,
    add_json_brace_tokens,
    require_base_flan_model,
    require_new_output_dir,
    resize_student_embeddings,
    verify_json_target,
    verify_training_texts,
)

DEFAULT_STUDENT_MODEL = BASE_STUDENT_MODEL


class StudentTrainError(RuntimeError):
    """Raised when training cannot finish without changing the data."""


class StudentTrainConfig(BaseModel):
    """Knobs for one local fine-tune. Distributed training is not included."""

    model_config = ConfigDict(extra="forbid")

    model_name: str = DEFAULT_STUDENT_MODEL
    seed: int = 42
    epochs: int = Field(ge=1, default=1)
    batch_size: int = Field(ge=1, default=2)
    learning_rate: float = Field(gt=0, default=1e-4)
    weight_decay: float = Field(ge=0, default=0.0)
    max_grad_norm: float = Field(gt=0, default=1.0)
    max_source_length: int = Field(ge=1, default=512)
    max_target_length: int = Field(ge=1, default=128)
    device: str = "auto"


class EncodedExample(BaseModel):
    """Token ids for one invoice. Padding is applied later, per batch."""

    model_config = ConfigDict(extra="forbid")

    document_id: str
    input_ids: list[int]
    label_ids: list[int]


class StudentTrainMetrics(BaseModel):
    """Loss and memory for a finished run. Validation is not computed here."""

    model_config = ConfigDict(extra="forbid")

    model_name: str
    examples: int = Field(ge=0)
    epochs: int = Field(ge=1)
    steps: int = Field(ge=0)
    mean_loss: float
    epoch_mean_loss: list[float]
    step_loss: list[float]
    device: str
    parameter_count: int = Field(ge=0)
    gpu_memory_after_load_bytes: int | None = None
    peak_gpu_memory_allocated_bytes: int | None = None
    peak_gpu_memory_reserved_bytes: int | None = None
    reload_ok: bool
    truncated_inputs: int = 0


def set_seed(seed: int) -> None:
    """Seed Python and PyTorch. GPU kernels are still not fully deterministic."""
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def encode_pairs(
    tokenizer: Any,
    pairs: Sequence[StudentPair],
    *,
    max_source_length: int,
    max_target_length: int,
) -> list[EncodedExample]:
    """Tokenize every pair. Over-long invoices raise instead of being cut."""
    encoded: list[EncodedExample] = []
    for pair in pairs:
        assert_tokenizer_keeps_text(tokenizer, pair.source_text)
        assert_tokenizer_keeps_text(tokenizer, pair.target_text)
        source_ids = token_ids(tokenizer, pair.source_text, add_special_tokens=True)
        target_ids = token_ids(tokenizer, pair.target_text, add_special_tokens=True)
        require_fits(source_ids, max_source_length, pair.document_id, "input")
        require_fits(target_ids, max_target_length, pair.document_id, "output")
        encoded.append(
            EncodedExample(
                document_id=pair.document_id,
                input_ids=source_ids,
                label_ids=target_ids,
            )
        )
    return encoded


def collate_encoded(examples: Sequence[EncodedExample], pad_token_id: int) -> dict[str, torch.Tensor]:
    """Pad a batch. Label padding is -100 so it does not contribute to the loss."""
    if not examples:
        raise StudentTrainError("Cannot collate an empty batch.")
    width = max(len(example.input_ids) for example in examples)
    label_width = max(len(example.label_ids) for example in examples)
    input_rows: list[list[int]] = []
    mask_rows: list[list[int]] = []
    label_rows: list[list[int]] = []
    for example in examples:
        source_pad = width - len(example.input_ids)
        input_rows.append(list(example.input_ids) + [pad_token_id] * source_pad)
        mask_rows.append([1] * len(example.input_ids) + [0] * source_pad)
        target_pad = label_width - len(example.label_ids)
        label_rows.append(list(example.label_ids) + [-100] * target_pad)
    return {
        "input_ids": torch.tensor(input_rows, dtype=torch.long),
        "attention_mask": torch.tensor(mask_rows, dtype=torch.long),
        "labels": torch.tensor(label_rows, dtype=torch.long),
    }


def train_student(
    pairs: Sequence[StudentPair],
    config: StudentTrainConfig,
    output_dir: Path,
) -> StudentTrainMetrics:
    """Fine-tune, save, and reload once to prove the checkpoint opens."""
    if not pairs:
        raise StudentTrainError("There are no training pairs.")
    output_dir = Path(output_dir)
    require_base_flan_model(config.model_name)
    require_new_output_dir(output_dir)
    set_seed(config.seed)
    device = select_device(config.device)
    tokenizer = load_tokenizer(config.model_name)
    if getattr(tokenizer, "pad_token_id", None) is None:
        raise StudentTrainError("The student tokenizer has no pad token.")
    left_id, right_id = add_json_brace_tokens(tokenizer)
    sample_decoded = verify_training_texts(tokenizer, pairs)
    print(f"json brace ids: left={left_id} right={right_id}")
    print(f"json sample decoded: {sample_decoded}")
    encoded = encode_pairs(
        tokenizer,
        pairs,
        max_source_length=config.max_source_length,
        max_target_length=config.max_target_length,
    )
    model = load_seq2seq(config.model_name)
    input_rows, output_rows = resize_student_embeddings(model, tokenizer)
    print(f"resized embeddings: input {input_rows} output {output_rows}")
    model.to(device)
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()
    memory_after_load = _gpu_allocated(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )
    step_loss: list[float] = []
    epoch_mean_loss: list[float] = []
    for epoch in range(config.epochs):
        order = list(range(len(encoded)))
        random.Random(config.seed + epoch).shuffle(order)
        epoch_losses: list[float] = []
        model.train()
        for start in range(0, len(order), config.batch_size):
            batch_examples = [encoded[index] for index in order[start : start + config.batch_size]]
            batch = collate_encoded(batch_examples, int(tokenizer.pad_token_id))
            batch = {key: value.to(device) for key, value in batch.items()}
            optimizer.zero_grad(set_to_none=True)
            outputs = model(**batch)
            loss = outputs.loss
            if loss is None or not torch.isfinite(loss):
                raise StudentTrainError("Training loss was not finite.")
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), config.max_grad_norm)
            optimizer.step()
            value = float(loss.detach().cpu())
            step_loss.append(value)
            epoch_losses.append(value)
        epoch_mean = sum(epoch_losses) / len(epoch_losses)
        epoch_mean_loss.append(epoch_mean)
        print(f"epoch {epoch + 1}/{config.epochs} mean_loss {epoch_mean:.4f}")

    if device.type == "cuda":
        torch.cuda.synchronize()
    peak_allocated = _gpu_peak_allocated(device)
    peak_reserved = _gpu_peak_reserved(device)
    parameter_count = sum(parameter.numel() for parameter in model.parameters())
    model.eval()
    save_student(model, tokenizer, output_dir, config)
    del model
    if device.type == "cuda":
        torch.cuda.empty_cache()
    reloaded, reloaded_tokenizer = load_student(output_dir, device)
    reloaded_count = sum(parameter.numel() for parameter in reloaded.parameters())
    if reloaded_count != parameter_count:
        raise StudentTrainError("Reloaded student parameter count does not match training.")
    probe = reloaded_tokenizer("invoice", truncation=False, padding=False)
    if "input_ids" not in probe or not list(probe["input_ids"]):
        raise StudentTrainError("Reloaded tokenizer did not encode text.")
    verify_json_target(reloaded_tokenizer, pairs[0].target_text)
    del reloaded
    metrics = StudentTrainMetrics(
        model_name=config.model_name,
        examples=len(encoded),
        epochs=config.epochs,
        steps=len(step_loss),
        mean_loss=sum(step_loss) / len(step_loss),
        epoch_mean_loss=epoch_mean_loss,
        step_loss=step_loss,
        device=str(device),
        parameter_count=parameter_count,
        gpu_memory_after_load_bytes=memory_after_load,
        peak_gpu_memory_allocated_bytes=peak_allocated,
        peak_gpu_memory_reserved_bytes=peak_reserved,
        reload_ok=True,
        truncated_inputs=0,
    )
    (output_dir / "training_metrics.json").write_text(
        metrics.model_dump_json(indent=2) + "\n",
        encoding="utf-8",
    )
    return metrics


def save_student(model: Any, tokenizer: Any, output_dir: Path, config: StudentTrainConfig) -> None:
    """Write weights, tokenizer, and the config needed to repeat the run."""
    output_dir.mkdir(parents=True, exist_ok=True)
    # The published checkpoint stores the input embedding and output head
    # separately even though the config says they are tied. Record that the
    # saved student keeps those tensors separate. This does not change values.
    if getattr(getattr(model, "config", None), "tie_word_embeddings", False):
        model.config.tie_word_embeddings = False
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    (output_dir / "training_config.json").write_text(
        config.model_dump_json(indent=2) + "\n",
        encoding="utf-8",
    )


def _gpu_allocated(device: torch.device) -> int | None:
    if device.type != "cuda":
        return None
    return int(torch.cuda.memory_allocated())


def _gpu_peak_allocated(device: torch.device) -> int | None:
    if device.type != "cuda":
        return None
    return int(torch.cuda.max_memory_allocated())


def _gpu_peak_reserved(device: torch.device) -> int | None:
    if device.type != "cuda":
        return None
    return int(torch.cuda.max_memory_reserved())
