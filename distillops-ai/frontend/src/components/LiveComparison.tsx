import { useEffect, useState } from "react";
import { AlertTriangle, Check, Loader2, Minus } from "lucide-react";
import { ApiRequestError, compareDocument, compareInvoice, fetchCompareStatus } from "../api";
import type { ComparisonPoint } from "../benchmarkMetrics";
import { SAMPLE_INVOICE_TEXT } from "../sampleInvoice";
import type { CompareResponse, CompareStatus, FieldComparison, InputMode, ModelOutcome } from "../types";
import { ComparisonChart } from "./BenchmarkCharts";
import { DocumentPreview } from "./DocumentPreview";
import { UploadPanel } from "./UploadPanel";

function money(outcome: ModelOutcome): string {
  if (!outcome.pricing_configured || outcome.estimated_cost_usd === null) {
    return "Not configured";
  }
  return `$${outcome.estimated_cost_usd.toFixed(6)}`;
}

function latencyLabel(value: number | null): string {
  return value === null ? "Not available" : `${Math.round(value)} ms`;
}

function tokenLabel(outcome: ModelOutcome): string {
  if (outcome.input_tokens === null || outcome.output_tokens === null) {
    return "Not available";
  }
  return `${outcome.input_tokens} in / ${outcome.output_tokens} out`;
}

function shownValue(value: string | null): string {
  if (value === null) {
    return "Not available";
  }
  return value === "" ? "(empty)" : value;
}

function FieldRow({ row }: { row: FieldComparison }) {
  const agree = row.status === "agree";
  const disagree = row.status === "disagree";
  const label = agree ? "Fields agree" : disagree ? "Model disagreement" : "Not available";
  const Icon = agree ? Check : disagree ? AlertTriangle : Minus;
  const tone = agree
    ? "border-emerald-200 bg-emerald-50"
    : disagree
      ? "border-amber-200 bg-amber-50"
      : "border-slate-200 bg-slate-50";
  return (
    <article className={`rounded-xl border p-3 ${tone}`} data-testid={`live-field-${row.field}`}>
      <div className="flex flex-wrap items-center gap-2 text-sm font-medium text-slate-900">
        <Icon className="h-4 w-4" aria-hidden="true" />
        <h4>{row.label}</h4>
        <span>{label}</span>
      </div>
      <dl className="mt-2 grid gap-2 text-sm sm:grid-cols-2">
        <div>
          <dt className="text-slate-500">Teacher</dt>
          <dd className="font-medium text-slate-900">{shownValue(row.teacher_value)}</dd>
        </div>
        <div>
          <dt className="text-slate-500">Student</dt>
          <dd className="font-medium text-slate-900">{shownValue(row.student_value)}</dd>
        </div>
      </dl>
    </article>
  );
}

export function LiveComparison() {
  const [status, setStatus] = useState<CompareStatus | null>(null);
  const [statusError, setStatusError] = useState<string | null>(null);
  const [mode, setMode] = useState<InputMode>("text");
  const [documentText, setDocumentText] = useState("");
  const [file, setFile] = useState<File | null>(null);
  const [confirming, setConfirming] = useState(false);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [result, setResult] = useState<CompareResponse | null>(null);

  useEffect(() => {
    let cancelled = false;
    fetchCompareStatus()
      .then((next) => {
        if (!cancelled) {
          setStatus(next);
        }
      })
      .catch((caught: unknown) => {
        if (!cancelled) {
          setStatusError(caught instanceof ApiRequestError ? caught.message : "Backend unavailable.");
        }
      });
    return () => {
      cancelled = true;
    };
  }, []);

  const available = status?.available === true;

  function ask() {
    setError(null);
    if (mode === "upload" && !file) {
      setError("Choose an invoice file before comparing.");
      return;
    }
    if (mode === "text" && !documentText.trim()) {
      setError("Enter invoice text before comparing.");
      return;
    }
    setConfirming(true);
  }

  async function confirm() {
    setConfirming(false);
    setLoading(true);
    setError(null);
    try {
      const next = mode === "upload" && file ? await compareDocument(file) : await compareInvoice(documentText);
      setResult(next);
    } catch (caught) {
      setResult(null);
      setError(caught instanceof ApiRequestError ? caught.message : "Backend unavailable.");
    } finally {
      setLoading(false);
    }
  }

  const latency: ComparisonPoint[] = result
    ? [
        {
          key: "latency",
          label: "This invoice",
          teacher: result.teacher.latency_ms,
          student: result.student.latency_ms,
          teacherLabel: latencyLabel(result.teacher.latency_ms),
          studentLabel: latencyLabel(result.student.latency_ms),
        },
      ]
    : [];

  return (
    <section className="space-y-4" data-testid="live-comparison">
      <header className="rounded-2xl border border-slate-200 bg-white p-5 shadow-sm">
        <h2 className="text-lg font-semibold text-slate-900">Live comparison</h2>
        <p className="mt-2 max-w-3xl text-sm leading-6 text-slate-600">
          The frozen student runs locally. The teacher runs only after you confirm, and both models read the same
          text. A difference is model disagreement, not a scored error. This page has no ground-truth labels.
        </p>
      </header>
      {statusError ? (
        <p className="rounded-lg bg-rose-50 px-3 py-2 text-sm text-rose-800" role="alert">
          {statusError}
        </p>
      ) : null}
      {status ? (
        <p className="rounded-2xl border border-slate-200 bg-white p-5 text-sm leading-6 text-slate-700 shadow-sm" role="status">
          {status.detail}
        </p>
      ) : null}
      {status === null && statusError === null ? <p className="text-sm text-slate-500">Checking live comparison.</p> : null}
      <div className="rounded-2xl border border-slate-200 bg-white p-5 shadow-sm">
        <div className="flex flex-wrap gap-2" role="tablist" aria-label="Live comparison input">
          <button
            type="button"
            role="tab"
            aria-selected={mode === "text"}
            onClick={() => setMode("text")}
            className={`rounded-lg px-3 py-2 text-sm font-medium ${mode === "text" ? "bg-navy text-white" : "border border-slate-300 bg-white text-slate-700"}`}
          >
            Paste invoice text
          </button>
          <button
            type="button"
            role="tab"
            aria-selected={mode === "upload"}
            onClick={() => setMode("upload")}
            className={`rounded-lg px-3 py-2 text-sm font-medium ${mode === "upload" ? "bg-navy text-white" : "border border-slate-300 bg-white text-slate-700"}`}
          >
            Upload invoice file
          </button>
        </div>
        {mode === "text" ? (
          <div className="mt-4">
            <label htmlFor="live-invoice-text" className="text-sm font-medium text-slate-800">
              Invoice text
            </label>
            <textarea
              id="live-invoice-text"
              value={documentText}
              onChange={(event) => setDocumentText(event.target.value)}
              rows={8}
              className="mt-2 w-full rounded-xl border border-slate-300 px-3 py-2 text-sm"
              placeholder="Paste invoice text"
            />
            <button
              type="button"
              onClick={() => setDocumentText(SAMPLE_INVOICE_TEXT)}
              className="mt-2 text-sm font-medium text-blue-700"
            >
              Sample invoice
            </button>
          </div>
        ) : (
          <div className="mt-4">
            <UploadPanel
              file={file}
              onFile={(next, problem) => {
                setFile(next);
                setError(problem ?? null);
              }}
            />
            <DocumentPreview file={file} />
          </div>
        )}
        {error ? (
          <p className="mt-3 rounded-lg bg-rose-50 px-3 py-2 text-sm text-rose-800" role="alert">
            {error}
          </p>
        ) : null}
        {loading ? (
          <p className="mt-3 flex items-center gap-2 text-sm text-slate-600" role="status">
            <Loader2 className="h-4 w-4 animate-spin" aria-hidden="true" />
            Comparing models.
          </p>
        ) : null}
        <button
          type="button"
          onClick={ask}
          disabled={!available || loading}
          className="mt-4 rounded-lg bg-blue-700 px-4 py-2 text-sm font-medium text-white disabled:cursor-not-allowed disabled:bg-slate-300"
        >
          Compare models
        </button>
      </div>
      {confirming ? (
        <div className="fixed inset-0 z-20 flex items-center justify-center bg-slate-900/40 p-4" role="presentation">
          <div
            role="dialog"
            aria-modal="true"
            aria-labelledby="teacher-consent-title"
            className="max-w-lg rounded-2xl bg-white p-5 shadow-xl"
          >
            <h3 id="teacher-consent-title" className="text-lg font-semibold text-slate-900">
              Send this invoice to the teacher?
            </h3>
            <p className="mt-3 text-sm leading-6 text-slate-700">
              The invoice text leaves this computer and is sent to OpenAI. API usage may incur charges. The student
              still runs on this machine. DistillOps does not store the invoice or the teacher response.
            </p>
            <div className="mt-4 flex flex-wrap gap-2">
              <button type="button" onClick={() => void confirm()} className="rounded-lg bg-blue-700 px-4 py-2 text-sm font-medium text-white">
                Send to the teacher
              </button>
              <button
                type="button"
                onClick={() => setConfirming(false)}
                className="rounded-lg border border-slate-300 px-4 py-2 text-sm font-medium text-slate-700"
              >
                Cancel
              </button>
            </div>
          </div>
        </div>
      ) : null}
      {result ? (
        <div className="space-y-4" data-testid="live-result">
          {result.partial ? (
            <p className="rounded-lg bg-amber-50 px-3 py-2 text-sm text-amber-900" role="status">
              One model did not return fields. The other prediction is shown unchanged.
            </p>
          ) : null}
          <div className="grid grid-cols-1 gap-4 sm:grid-cols-2 xl:grid-cols-4" data-testid="live-metrics">
            <article className="rounded-2xl border border-slate-200 bg-white p-5 shadow-sm">
              <h3 className="text-xs font-semibold uppercase tracking-wide text-slate-500">Field agreement</h3>
              <p className="mt-3 text-2xl font-semibold text-slate-900">
                {result.agreement_count === null ? "Not available" : `${result.agreement_count} of ${result.fields_compared}`}
              </p>
              <p className="mt-2 text-sm text-slate-600">Fields that match each other. This is not accuracy.</p>
            </article>
            <article className="rounded-2xl border border-slate-200 bg-white p-5 shadow-sm" data-testid="live-teacher-latency">
              <h3 className="text-xs font-semibold uppercase tracking-wide text-slate-500">Teacher latency</h3>
              <p className="mt-3 text-2xl font-semibold text-blue-700">{latencyLabel(result.teacher.latency_ms)}</p>
              <p className="mt-2 text-sm text-slate-600">{result.teacher.model}</p>
            </article>
            <article className="rounded-2xl border border-slate-200 bg-white p-5 shadow-sm" data-testid="live-student-latency">
              <h3 className="text-xs font-semibold uppercase tracking-wide text-slate-500">Student latency</h3>
              <p className="mt-3 text-2xl font-semibold text-teal-700">{latencyLabel(result.student.latency_ms)}</p>
              <p className="mt-2 text-sm text-slate-600">{result.student.model}</p>
            </article>
            <article className="rounded-2xl border border-slate-200 bg-white p-5 shadow-sm" data-testid="live-tokens">
              <h3 className="text-xs font-semibold uppercase tracking-wide text-slate-500">Teacher tokens</h3>
              <p className="mt-3 text-2xl font-semibold text-slate-900">{tokenLabel(result.teacher)}</p>
              <p className="mt-2 text-sm text-slate-600">
                Student tokens: {tokenLabel(result.student)}. Local inference does not report API tokens. A token count
                of zero is not a local cost.
              </p>
            </article>
          </div>
          <div className="grid grid-cols-1 gap-4 sm:grid-cols-2">
            <article className="rounded-2xl border border-slate-200 bg-white p-5 shadow-sm" data-testid="live-teacher-cost">
              <h3 className="text-xs font-semibold uppercase tracking-wide text-slate-500">Estimated teacher cost</h3>
              <p className="mt-3 text-2xl font-semibold text-slate-900">{money(result.teacher)}</p>
            </article>
            <article className="rounded-2xl border border-slate-200 bg-white p-5 shadow-sm" data-testid="live-student-cost">
              <h3 className="text-xs font-semibold uppercase tracking-wide text-slate-500">Estimated student cost</h3>
              <p className="mt-3 text-2xl font-semibold text-slate-900">{money(result.student)}</p>
            </article>
          </div>
          <ComparisonChart
            title="Latency on this invoice"
            description="Each bar is that model's own latency. It is not an accuracy comparison."
            data={latency}
            unit=" ms"
            testId="live-latency"
          />
          <div className="grid gap-4 lg:grid-cols-2">
            <section className="rounded-2xl border border-slate-200 bg-white p-5 shadow-sm">
              <h3 className="text-base font-semibold text-blue-700">Teacher</h3>
              {result.teacher.error ? <p className="mt-2 text-sm text-rose-800">{result.teacher.error}</p> : null}
            </section>
            <section className="rounded-2xl border border-slate-200 bg-white p-5 shadow-sm">
              <h3 className="text-base font-semibold text-teal-700">Student</h3>
              {result.student.error ? <p className="mt-2 text-sm text-rose-800">{result.student.error}</p> : null}
            </section>
          </div>
          <div className="space-y-3">
            {result.fields.map((row) => (
              <FieldRow key={row.field} row={row} />
            ))}
          </div>
          <p className="text-sm leading-6 text-slate-600">{result.warnings[0]}</p>
          {result.extraction_method ? (
            <p className="text-sm text-slate-600">
              Document read: {result.extraction_method}. Both models received this text.
            </p>
          ) : (
            <p className="text-sm text-slate-600">Both models received the pasted text.</p>
          )}
        </div>
      ) : null}
    </section>
  );
}
