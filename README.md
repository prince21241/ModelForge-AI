# ModelForge AI

DistillOps AI is a portfolio project for distilling a large teacher language model into a smaller student model. The task is structured extraction from invoices: vendor, invoice number, date, total, and currency.

A large model can read those fields well, but it is slow and expensive to run on every document. The goal is to have the teacher write clean labels, train a smaller student on them, and keep the student only if it stays accurate enough while costing less and answering faster.

## Phase 1 through Phase 3

Phase 1 is the measurement setup: schema, fake invoices, teacher client, and scorecard. Phase 2 labeled all 20 training invoices with `gpt-4o-mini`. All 20 labels matched the ground truth. Phase 3 then trained `google/flan-t5-small` on those labels. A three-epoch run with real `{` and `}` tokens finished on 12 November 2026. The checkpoint reloads, but none of the five validation account statements came back as valid JSON. The locked test split has not been scored.

- Pydantic schema that leaves unknown fields null instead of guessing
- 30 fictional invoices, generated from a fixed seed
- Train, validation, and held-out test splits with no shared invoice layout
- Configurable OpenAI teacher client with timeouts and bounded retries
- Scorecard for field accuracy, exact match, schema validity, latency, tokens, and optional cost

Fine-tuning has a one-epoch smoke test. FastAPI and vLLM are still later phases. Tests use a fake client or local fixtures and do not call OpenAI. A live labeling run happens only when you pass `--live`.

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
python -m scripts.generate_teacher_labels --dry-run
```

Put `OPENAI_API_KEY` and `TEACHER_MODEL` in `.env` before a live baseline. `--split test` is rejected so the held-out split is not used for prompt tuning. `--limit` caps how many documents are sent to the API.

The invoices are synthetic. They are not real customer financial documents.
