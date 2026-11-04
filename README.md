# ModelForge AI

DistillOps AI is a portfolio project for distilling a large teacher language model into a smaller student model. The task is structured extraction from invoices: vendor, invoice number, date, total, and currency.

A large model can read those fields well, but it is slow and expensive to run on every document. The goal is to have the teacher write clean labels, train a smaller student on them, and keep the student only if it stays accurate enough while costing less and answering faster.

## Phase 1

This repository currently contains Phase 1: the measurement setup, not the trained student.

- Pydantic schema that leaves unknown fields null instead of guessing
- 30 fictional invoices, generated from a fixed seed
- Train, validation, and held-out test splits with no shared invoice layout
- Configurable OpenAI teacher client with timeouts and bounded retries
- Scorecard for field accuracy, exact match, schema validity, latency, tokens, and optional cost

Fine-tuning, FastAPI, and vLLM are later phases. The live teacher API was not called while Phase 1 was built. Tests use a fake client.

Full setup, metrics, and the Phase 2–5 roadmap are in [distillops-ai/README.md](distillops-ai/README.md).

## Run it

From this directory, on Windows PowerShell:

```powershell
cd distillops-ai
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
Copy-Item .env.example .env
python -m scripts.generate_dataset
pytest
python -m scripts.run_teacher_baseline --split val --limit 2
```

Put `OPENAI_API_KEY` and `TEACHER_MODEL` in `.env` before a live baseline. `--split test` is rejected so the held-out split is not used for prompt tuning. `--limit` caps how many documents are sent to the API.

The invoices are synthetic. They are not real customer financial documents.
