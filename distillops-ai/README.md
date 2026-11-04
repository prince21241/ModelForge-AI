# DistillOps AI

DistillOps AI is a portfolio platform for distilling a large teacher language model into a smaller student model. The task is structured extraction from invoices: vendor, invoice number, issue date, total, and currency.

Phase 1 builds the parts you need before any fine-tuning:

- a Pydantic schema that keeps unknown fields null
- 30 fictional invoices with a fixed seed and no template leakage across splits
- an OpenAI teacher client that returns schema-constrained JSON
- an evaluation report for accuracy, latency, token use, and optional cost
- a validation baseline command that cannot read the held-out test split

Fine-tuning, a React dashboard, and vLLM serving are later phases. They are not in this repository yet.

The live teacher API was not called while this phase was built. Unit tests use a fake client. A baseline run sends requests only after you add your own API key.

## What model distillation is

A teacher model is a large model that already performs a task well. Distillation transfers that behavior to a smaller student model. In this project the teacher does not train the student by sharing its weights. It writes structured labels. Those labels become the student's training set.

The student is useful when it keeps enough extraction accuracy while answering faster and at a lower token cost. Phase 1 creates the teacher, the labeled documents, and the scoreboard. Later phases train the student and compare the two on the same documents.

## Why invoice extraction

Invoice fields are a good distillation task for four reasons:

- The output is a small schema, so exact match is unambiguous.
- Missing values must stay null. Guessing a currency from `$` or a date from "today" is a real error, and the metrics count it.
- Amounts and dates need normalization. `10.00` and `10` are the same total. `2024-01-02` is the same day whether it is stored as a date or an ISO string.
- Accuracy, latency, and cost can be compared on the same request later, which is the point of distilling a student.

The documents in this repo are generated. They use fictional vendors and amounts. Do not replace them with real customer invoices.

## Architecture

```text
synthetic invoices
        |
        v
data/splits/{train,val,test}.jsonl
        |
        +---- train, val ----> TeacherClient (OpenAI structured output)
        |                              |
        |                              v
        |                      TeacherPrediction
        |                              |
        +---- ground truth ----> evaluation.evaluate_predictions
                                       |
                                       v
                                 reports/*.json

test.jsonl stays unread by the baseline command
```

Library code lives in `src/` and does not start a server or read a `.env` on import. The scripts are thin command lines. A later FastAPI app can import `schemas`, `teacher`, and `evaluation` without going through the CLI.

| Module | Role |
| --- | --- |
| `src/schemas.py` | Invoice schema, dataset record, teacher prediction |
| `src/dataset.py` | Seeded generation, group-wise splits, JSONL read/write |
| `src/invoice_templates.py` | Six invoice layouts |
| `src/prompts.py` | Teacher instructions |
| `src/teacher.py` | Configurable OpenAI client, timeout, bounded retries |
| `src/evaluation.py` | Field accuracy, exact match, latency, tokens, optional cost |
| `src/config.py` | Environment settings |
| `src/evaluation_pricing.py` | Price card used only when both rates are set |

## Project layout

```text
distillops-ai/
  src/                  library modules
  scripts/              generate_dataset and run_teacher_baseline
  data/raw/             synthetic manifest (generated, not committed)
  data/processed/       full JSONL export and split summary
  data/splits/          train.jsonl, val.jsonl, test.jsonl
  reports/              baseline JSON reports
  tests/                pytest, with the provider mocked
  models/               reserved for future student checkpoints
  requirements.txt
  .env.example
```

Generated splits, reports, `.env`, and model weights are gitignored.

## Setup on Windows PowerShell

From the `distillops-ai` directory:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
Copy-Item .env.example .env
```

If activation is blocked by execution policy, allow it for the current window and try again:

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.\.venv\Scripts\Activate.ps1
```

Python 3.11 or newer is required. This phase was exercised with Python 3.12.

Edit `.env` before a live baseline. Tests do not need a key.

## Environment variables

| Variable | Required for a live run | Purpose |
| --- | --- | --- |
| `OPENAI_API_KEY` | Yes | Sent only to the OpenAI client. It is excluded from logs and from `Settings` repr. |
| `TEACHER_MODEL` | Yes | Structured-output model name. The client does not hardcode one. |
| `TEACHER_TIMEOUT_SECONDS` | No | Per-attempt timeout. Default 30. Maximum 180. |
| `TEACHER_MAX_RETRIES` | No | Extra attempts after the first. Default 2, maximum 5. |
| `TEACHER_RETRY_BACKOFF_SECONDS` | No | Base delay. Wait is `min(8, base * 2^attempt)`. |
| `TEACHER_TEMPERATURE` | No | Default `0`. Leave the value blank to omit the parameter. |
| `TEACHER_INPUT_USD_PER_MILLION_TOKENS` | No | Your input token rate. |
| `TEACHER_OUTPUT_USD_PER_MILLION_TOKENS` | No | Your output token rate. |

Set both price variables or neither. One price without the other is a configuration error. DistillOps does not assume an OpenAI list price.

The SDK client's own retries are turned off. `TeacherClient` owns the retry loop so the bound is the same in production and in tests. Status codes 408, 409, 429, 500, 502, 503, and 504 are retried, along with timeout and connection errors. Other failures are returned in the report and are not retried.

Logs record the document id, model, latency, token counts, and exception type. They do not record the API key or the invoice text.

## Dataset generation

```powershell
python -m scripts.generate_dataset
python -m scripts.generate_dataset --seed 42 --data-dir data
```

The command writes:

- `data/splits/train.jsonl` — 20 documents
- `data/splits/val.jsonl` — 5 documents
- `data/splits/test.jsonl` — 5 documents
- `data/processed/all.jsonl`
- `data/processed/split_summary.csv`
- `data/raw/synthetic_manifest.json`

Each JSONL row has `document_id`, `document_text`, `ground_truth`, and `template_group`. The original text is stored separately from the labels. Every row is validated with Pydantic before it is written.

Six layouts are used: classic header, ASCII table, prose memo, key/value, account statement, and a minimal block. Each layout contributes 5 invoices. A whole layout goes to one split, so a template cannot leak from train into validation or test. With groups of five, validation and test each receive one layout that training never sees. The assignment is shuffled with the same seed as the text, and the targets are filled exactly (20 / 5 / 5).

Some fields are deliberately missing. The schedule is fixed by document index, not by the random draw, so nulls do not disappear if a vendor name changes. The renderer omits a field that is null. It does not print `N/A` or a guessed currency.

Two layouts include small line items that do not add up to the labeled total. The teacher prompt tells the model to read the explicit total and not to sum those lines.

Running the command twice with the same seed rewrites the same JSONL.

## Baseline execution

Generate the dataset first, then:

```powershell
python -m scripts.run_teacher_baseline --split val
python -m scripts.run_teacher_baseline --split val --limit 2
```

`--limit` sends only the first N validation documents. Use it while checking a key or a model so a full split is not billed by accident.

`--split train` is allowed if you want a training-set diagnostic. `--split test` exits with an error and does not open `test.jsonl`. The test split stays unused until a later phase freezes the prompt and the student, and then scores it once.

The command writes `reports/teacher_baseline_<split>_<utc timestamp>.json` and prints a summary. The JSON keeps each document's prediction, field matches, API error, schema error, and refusal. It does not copy `document_text`.

Exit codes:

| Code | Meaning |
| --- | --- |
| 0 | The report was written. Individual API failures are inside the report. |
| 2 | Bad arguments, held-out test split, missing settings, or an unreadable split file. |
| 3 | The split exists but has no documents. |

A missing key or model raises before any request. The error says that no teacher request was sent.

## Evaluation metrics

Scores are joined on `document_id`. Text fields are compared after whitespace is collapsed. ISO dates are compared as `YYYY-MM-DD`. Amounts use `Decimal`, so `10.00` equals `10`. Currency codes are compared case-insensitively. Null against null is correct.

| Metric | Definition |
| --- | --- |
| Per-field accuracy | Matches divided by schema-valid documents. Null against null counts as a match. |
| Coverage | Share of all evaluated documents whose ground truth has a non-null value for that field. |
| Accuracy when present | Matches divided by schema-valid documents whose ground truth is non-null. This is the check against a high score caused by empty fields. |
| Exact match (valid outputs) | Share of schema-valid documents where every field matches. |
| Exact match (end to end) | Same count divided by every document. API failures and invalid JSON count as misses. |
| Schema validity | Share of documents that parsed into the invoice schema. |
| API failure rate | Share of documents whose call failed. A refusal or a schema error is not an API failure. |
| Latency | Average, p50, and p95 of elapsed milliseconds, including failed attempts. Percentiles are linear interpolation between ranks. |
| Tokens | Sum of input tokens and sum of output tokens reported by the provider, including calls that failed validation after a response. |
| Estimated cost | Computed only when both price variables are set. Otherwise the report says `not configured`. |

Cost formula, using the rates you supply:

```text
estimated_usd =
    input_tokens  / 1_000_000 * TEACHER_INPUT_USD_PER_MILLION_TOKENS
  + output_tokens / 1_000_000 * TEACHER_OUTPUT_USD_PER_MILLION_TOKENS
```

The result is quantized to 6 decimal places. No sample rate in `.env.example` is applied automatically.

## Tests

```powershell
pytest
```

The tests cover schema validation, JSONL loading, split sizes, template leakage, numeric and date normalization, exact match, null-versus-null scoring, coverage, API failures, retry bounds, and the held-out test split. External calls are scripted. `pytest` does not need `OPENAI_API_KEY`.

## What Phase 1 does not do

- It does not fine-tune a student model.
- It does not serve a model with vLLM or FastAPI.
- It does not ship a dashboard.
- It does not prove that a live OpenAI call succeeds. That depends on your key, model access, and network.

## Roadmap

**Phase 2 — Distillation dataset.** Freeze the Phase 1 prompt. Run the teacher on the train split only and store those labels next to the ground truth. Export a supervised fine-tuning file for the student. Do not tune the prompt on the test split.

**Phase 3 — Student fine-tuning.** Fine-tune a smaller open model, for example with LoRA, from the Phase 2 file. Write checkpoints under `models/`. Record the base model, seed, and training config. Checkpoints stay gitignored.

**Phase 4 — Teacher versus student.** Score both models on the validation split with this evaluation harness. Compare exact match, accuracy when present, schema validity, latency, tokens, and cost. After those choices are frozen, run the test split once.

**Phase 5 — Deployment.** Add a FastAPI service and serve the student with vLLM or a similar runtime. Measure latency and cost under a small fixed load. A review UI can come after the service is measurable. The Phase 1 package layout is meant to be imported by that service rather than rewritten.
