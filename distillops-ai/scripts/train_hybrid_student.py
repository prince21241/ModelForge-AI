"""Train a fresh FLAN-T5-small student on 60 hybrid examples.

The run starts from google/flan-t5-small. It does not resume from epoch 25
and it does not read the locked test split. Validation is scored once, after
the training checkpoint has already been chosen.

    python -m scripts.train_hybrid_student
"""

from __future__ import annotations

import json
import random
import time
from pathlib import Path

import torch

from src.dataset import load_split
from src.hybrid_dataset import generate_hybrid_invoices
from src.hybrid_training import (
    CONVERGENCE_RULE,
    HYBRID_OUTPUT_DIR,
    VALIDATION_IDS,
    TrainingCheckpointScore,
    assemble_training_pairs,
    assert_fresh_checkpoint,
    classify_student_failure,
    convergence_reached,
    hybrid_train_config,
    safe_field_match,
    select_training_checkpoint,
)
from src.labeling import student_fields_from_extraction
from src.prompts import build_student_source
from src.student_data import (
    STUDENT_OUTPUT_FIELDS,
    StudentPair,
    assert_tokenizer_keeps_text,
    load_training_examples,
    parse_student_json,
    require_fits,
    require_validation_split,
    teacher_target_json,
    token_ids,
)
from src.student_infer import load_seq2seq, load_student, load_tokenizer, select_device
from src.student_tokenizer import (
    add_json_brace_tokens,
    require_base_flan_model,
    resize_student_embeddings,
    verify_training_texts,
)
from src.student_train import collate_encoded, encode_pairs, save_student, set_seed

LABELS = Path("data/processed/teacher_labels/student_train.jsonl")
REPORT_PATH = Path("reports/phase3e_hybrid60.json")
EPOCH25_DIR = Path("models/student/flan-t5-small-train20-memorize/epoch-25")
EXPECTED_LEFT = 32100
EXPECTED_RIGHT = 32101
EVAL_FIELD = {
    "vendor": "vendor",
    "invoice_number": "invoice_number",
    "date": "invoice_date",
    "total": "total",
    "currency": "currency",
}


def content_and_eos(token_list: list[int], decoder_start: int, eos_id: int) -> tuple[list[int], bool]:
    """Drop the decoder-start id. The last body id is EOS when generation stopped."""
    body = list(token_list)
    if body and body[0] == decoder_start:
        body = body[1:]
    ended = bool(body) and body[-1] == eos_id
    content = body[:-1] if ended else body
    return content, ended


def score_pairs(model, tokenizer, rows, texts: dict[str, str], device) -> dict:
    """Greedy-decode training or validation rows. The model is not updated."""
    model.eval()
    decoder_start = int(model.config.decoder_start_token_id)
    eos_id = int(tokenizer.eos_token_id)
    config = hybrid_train_config()
    field_hits = {field: 0 for field in STUDENT_OUTPUT_FIELDS}
    parseable = 0
    json_valid = 0
    opening = 0
    closing = 0
    exact = 0
    predictions = []
    with torch.inference_mode():
        for row in rows:
            pair = row["pair"]
            expected = row["expected"]
            invoice_text = texts[pair.document_id]
            assert_tokenizer_keeps_text(tokenizer, pair.source_text)
            input_ids = token_ids(tokenizer, pair.source_text, add_special_tokens=True)
            require_fits(input_ids, config.max_source_length, pair.document_id, "input")
            tensor = torch.tensor([input_ids], dtype=torch.long, device=device)
            started = time.perf_counter()
            generated = model.generate(
                input_ids=tensor,
                attention_mask=torch.ones_like(tensor),
                max_new_tokens=config.max_target_length,
                do_sample=False,
                num_beams=1,
            )[0].tolist()
            latency_ms = (time.perf_counter() - started) * 1000
            content, ended = content_and_eos(generated, decoder_start, eos_id)
            opening_ok = bool(content) and content[0] == EXPECTED_LEFT
            closing_ok = ended and bool(content) and content[-1] == EXPECTED_RIGHT
            raw_text = tokenizer.decode(generated, skip_special_tokens=True).strip()
            fields, error = parse_student_json(raw_text)
            field_ok = {}
            if fields is not None:
                parseable += 1
                json_valid += 1
                for student_field, eval_field in EVAL_FIELD.items():
                    matched = safe_field_match(
                        eval_field,
                        getattr(fields, student_field),
                        getattr(expected, student_field),
                    )
                    field_ok[student_field] = matched
                    field_hits[student_field] += int(matched)
                exact_ok = all(field_ok.values())
            else:
                exact_ok = False
            exact += int(exact_ok)
            opening += int(opening_ok)
            closing += int(closing_ok)
            record = {
                "document_id": pair.document_id,
                "json_valid": fields is not None,
                "opening_brace": opening_ok,
                "closing_brace": closing_ok,
                "final_content_token": content[-1] if content else None,
                "exact_match": exact_ok,
                "field_matches": field_ok,
                "validation_error": error,
                "latency_ms": round(latency_ms, 1),
                "raw_text": raw_text,
                "expected": {field: getattr(expected, field) for field in STUDENT_OUTPUT_FIELDS},
                "predicted": None
                if fields is None
                else {field: getattr(fields, field) for field in STUDENT_OUTPUT_FIELDS},
                "failure_categories": classify_student_failure(
                    parsed=fields,
                    expected=expected,
                    invoice_text=invoice_text,
                    opening_ok=opening_ok,
                    closing_ok=closing_ok,
                ),
            }
            if not exact_ok:
                record["token_ids"] = generated
            predictions.append(record)
    field_accuracy = {
        field: (field_hits[field] / parseable if parseable else None) for field in STUDENT_OUTPUT_FIELDS
    }
    latencies = [row["latency_ms"] for row in predictions]
    return {
        "examples": len(predictions),
        "json_valid": json_valid,
        "opening_brace": opening,
        "closing_brace": closing,
        "exact_match": exact,
        "parseable": parseable,
        "field_accuracy": field_accuracy,
        "latency_ms_avg": round(sum(latencies) / len(latencies), 1) if latencies else None,
        "predictions": predictions,
    }


def summarize(score: dict) -> str:
    total = score["examples"]
    return (
        f"valid {score['json_valid']}/{total} "
        f"open {score['opening_brace']}/{total} "
        f"close {score['closing_brace']}/{total} "
        f"exact {score['exact_match']}/{total}"
    )


def write_report(payload: dict) -> None:
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def validation_rows() -> tuple[list[dict], dict[str, str]]:
    """Load the five validation invoices. The test split is not requested."""
    require_validation_split("val")
    records = load_split(Path("data/splits"), "val")
    ids = [record.document_id for record in records]
    if ids != [
        "inv-synth-021",
        "inv-synth-022",
        "inv-synth-023",
        "inv-synth-024",
        "inv-synth-025",
    ]:
        raise SystemExit(f"Validation ids were {ids}.")
    if set(ids) != VALIDATION_IDS:
        raise SystemExit("Validation id set changed.")
    rows = []
    texts = {}
    for record in records:
        expected = student_fields_from_extraction(record.ground_truth)
        rows.append(
            {
                "pair": StudentPair(
                    document_id=record.document_id,
                    source_text=build_student_source(record.document_text),
                    target_text=teacher_target_json(expected),
                    teacher_output=expected,
                ),
                "expected": expected,
            }
        )
        texts[record.document_id] = record.document_text
    return rows, texts


def main() -> None:
    config = hybrid_train_config()
    require_base_flan_model(config.model_name)
    assert_fresh_checkpoint(HYBRID_OUTPUT_DIR)
    original = load_training_examples(LABELS)
    hybrid = generate_hybrid_invoices(config.seed)
    prepared = assemble_training_pairs(original, hybrid)
    texts = {example.document_id: example.invoice_text for example in original}
    texts.update({invoice.document_id: invoice.document_text for invoice in hybrid})
    rows = [
        {"pair": item.pair, "expected": item.pair.teacher_output, "provenance": item.provenance}
        for item in prepared
    ]
    device = select_device(config.device)
    started = time.perf_counter()
    report: dict = {
        "convergence_rule": CONVERGENCE_RULE,
        "label_provenance": {
            "verified_teacher": 20,
            "deterministic_ground_truth": 40,
            "teacher_subset": "evaluation_only_not_used_as_targets",
        },
        "epoch_mean_loss": [],
        "evaluations": [],
        "stopped_early": False,
        "best_checkpoint": None,
    }
    set_seed(config.seed)
    tokenizer = load_tokenizer(config.model_name)
    left_id, right_id = add_json_brace_tokens(tokenizer)
    if (left_id, right_id) != (EXPECTED_LEFT, EXPECTED_RIGHT):
        raise SystemExit(f"Brace ids were {left_id}, {right_id}.")
    pairs = [row["pair"] for row in rows]
    print(verify_training_texts(tokenizer, pairs), flush=True)
    encoded = encode_pairs(
        tokenizer,
        pairs,
        max_source_length=config.max_source_length,
        max_target_length=config.max_target_length,
    )
    eos_id = int(tokenizer.eos_token_id)
    for example in encoded:
        labels = example.label_ids
        if labels[0] != EXPECTED_LEFT or labels[-2] != EXPECTED_RIGHT or labels[-1] != eos_id:
            raise SystemExit(f"{example.document_id} target braces were not encoded.")
        if tokenizer.unk_token_id in labels:
            raise SystemExit(f"{example.document_id} target contains an unknown token.")
    probe = collate_encoded(encoded, int(tokenizer.pad_token_id))
    for index, example in enumerate(encoded):
        label_row = probe["labels"][index].tolist()
        if label_row[: len(example.label_ids)] != example.label_ids:
            raise SystemExit(f"{example.document_id} labels changed while batching.")
        if any(token != -100 for token in label_row[len(example.label_ids) :]):
            raise SystemExit(f"{example.document_id} padding is not excluded from the loss.")
    model = load_seq2seq(config.model_name)
    input_rows, output_rows = resize_student_embeddings(model, tokenizer)
    print(f"resized embeddings: input {input_rows} output {output_rows}", flush=True)
    model.to(device)
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats()
        memory_after_load = int(torch.cuda.memory_allocated())
    else:
        memory_after_load = None
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate, weight_decay=0.0)
    train_started = time.perf_counter()
    scores = []
    completed = 0
    for epoch in range(1, config.epochs + 1):
        order = list(range(len(encoded)))
        random.Random(config.seed + epoch).shuffle(order)
        losses = []
        model.train()
        for start in range(0, len(order), config.batch_size):
            batch_examples = [encoded[index] for index in order[start : start + config.batch_size]]
            batch = collate_encoded(batch_examples, int(tokenizer.pad_token_id))
            batch = {key: value.to(device) for key, value in batch.items()}
            optimizer.zero_grad(set_to_none=True)
            outputs = model(**batch)
            loss = outputs.loss
            if loss is None or not torch.isfinite(loss):
                raise SystemExit("Training loss was not finite.")
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), config.max_grad_norm)
            optimizer.step()
            losses.append(float(loss.detach().cpu()))
        mean_loss = sum(losses) / len(losses)
        report["epoch_mean_loss"].append(round(mean_loss, 6))
        print(f"epoch {epoch}/{config.epochs} mean_loss {mean_loss:.4f}", flush=True)
        completed = epoch
        if epoch % 5 != 0:
            write_report(report)
            continue
        score = score_pairs(model, tokenizer, rows, texts, device)
        checkpoint_dir = HYBRID_OUTPUT_DIR / f"epoch-{epoch:02d}"
        save_student(model, tokenizer, checkpoint_dir, config.model_copy(update={"epochs": epoch}))
        checkpoint_score = {
            "epoch": epoch,
            "mean_loss": round(mean_loss, 6),
            "json_valid": score["json_valid"],
            "opening_brace": score["opening_brace"],
            "closing_brace": score["closing_brace"],
            "exact_match": score["exact_match"],
            "field_accuracy": score["field_accuracy"],
            "checkpoint": str(checkpoint_dir),
            "failures": [
                {
                    "document_id": row["document_id"],
                    "raw_text": row["raw_text"],
                    "token_ids": row.get("token_ids"),
                    "final_content_token": row["final_content_token"],
                    "failure_categories": row["failure_categories"],
                }
                for row in score["predictions"]
                if not row["exact_match"]
            ],
        }
        report["evaluations"].append(checkpoint_score)
        scores.append(
            TrainingCheckpointScore(
                epoch=epoch,
                mean_loss=round(mean_loss, 6),
                json_valid=score["json_valid"],
                opening_brace=score["opening_brace"],
                closing_brace=score["closing_brace"],
                exact_match=score["exact_match"],
                checkpoint=str(checkpoint_dir),
            )
        )
        print(f"epoch {epoch} {summarize(score)}", flush=True)
        write_report(report)
        if convergence_reached(scores[-1], len(rows)):
            report["stopped_early"] = epoch < config.epochs
            report["stop_reason"] = CONVERGENCE_RULE
            print("STOP early: training JSON validity and exact match are 60/60", flush=True)
            break
    if device.type == "cuda":
        torch.cuda.synchronize()
        peak_allocated = int(torch.cuda.max_memory_allocated())
        peak_reserved = int(torch.cuda.max_memory_reserved())
    else:
        peak_allocated = None
        peak_reserved = None
    report["epochs_completed"] = completed
    report["final_epoch_loss"] = report["epoch_mean_loss"][-1]
    report["train_seconds"] = round(time.perf_counter() - train_started, 1)
    report["wall_seconds"] = round(time.perf_counter() - started, 1)
    report["gpu_memory_after_load_bytes"] = memory_after_load
    report["peak_gpu_memory_allocated_bytes"] = peak_allocated
    report["peak_gpu_memory_reserved_bytes"] = peak_reserved
    report["best_checkpoint"] = select_training_checkpoint(scores)
    del model
    if device.type == "cuda":
        torch.cuda.empty_cache()
    reloaded, reloaded_tokenizer = load_student(report["best_checkpoint"], device)
    confirm = score_pairs(reloaded, reloaded_tokenizer, rows, texts, device)
    report["reload_check"] = {
        "checkpoint": report["best_checkpoint"],
        "json_valid": confirm["json_valid"],
        "opening_brace": confirm["opening_brace"],
        "closing_brace": confirm["closing_brace"],
        "exact_match": confirm["exact_match"],
        "field_accuracy": confirm["field_accuracy"],
    }
    print(f"RELOAD {summarize(confirm)}", flush=True)
    val_rows, val_texts = validation_rows()
    validation = score_pairs(reloaded, reloaded_tokenizer, val_rows, val_texts, device)
    report["validation"] = _public_score(validation)
    print(f"VALIDATION {summarize(validation)}", flush=True)
    del reloaded
    if device.type == "cuda":
        torch.cuda.empty_cache()
    previous_model, previous_tokenizer = load_student(EPOCH25_DIR, device)
    previous = score_pairs(previous_model, previous_tokenizer, val_rows, val_texts, device)
    report["epoch25_validation"] = _public_score(previous)
    print(f"EPOCH25 {summarize(previous)}", flush=True)
    write_report(report)
    print(
        f"DONE best {report['best_checkpoint']} loss {report['final_epoch_loss']} "
        f"seconds {report['train_seconds']}",
        flush=True,
    )


def _public_score(score: dict) -> dict:
    failures = [row for row in score["predictions"] if not row["exact_match"]]
    return {
        "json_valid": score["json_valid"],
        "opening_brace": score["opening_brace"],
        "closing_brace": score["closing_brace"],
        "exact_match": score["exact_match"],
        "field_accuracy": score["field_accuracy"],
        "latency_ms_avg": score["latency_ms_avg"],
        "predictions": [
            {key: value for key, value in row.items() if key != "token_ids"}
            for row in score["predictions"]
        ],
        "failure_count": len(failures),
    }


if __name__ == "__main__":
    main()
