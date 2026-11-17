import { useEffect, useState } from "react";
import { ApiRequestError, extractInvoice, fetchHealth, fetchModels } from "./api";
import { ComingSoon } from "./components/ComingSoon";
import { ExtractWorkspace } from "./components/ExtractWorkspace";
import { Sidebar } from "./components/Sidebar";
import { TopBar } from "./components/TopBar";
import { EMPTY_FIELDS, type ExtractResponse, type InvoiceFields, type ViewId } from "./types";

export default function App() {
  const [view, setView] = useState<ViewId>("dashboard");
  const [connected, setConnected] = useState(false);
  const [modelLabel, setModelLabel] = useState("student");
  const [documentText, setDocumentText] = useState("");
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [result, setResult] = useState<ExtractResponse | null>(null);
  const [edited, setEdited] = useState<InvoiceFields>(EMPTY_FIELDS);

  useEffect(() => {
    let cancelled = false;
    async function loadStatus() {
      try {
        const [health, models] = await Promise.all([fetchHealth(), fetchModels()]);
        if (cancelled) {
          return;
        }
        setConnected(health.status === "ok");
        const student = models.models.find((model) => model.name === "student");
        setModelLabel(student ? `${student.name} · ${student.identifier}` : "student");
      } catch {
        if (!cancelled) {
          setConnected(false);
          setModelLabel("student");
        }
      }
    }
    void loadStatus();
    return () => {
      cancelled = true;
    };
  }, [result]);

  function editField(key: keyof InvoiceFields, value: string) {
    setEdited((current) => ({ ...current, [key]: value }));
  }

  async function onExtract() {
    if (!documentText.trim()) {
      setError("Enter invoice text before extracting.");
      return;
    }
    setLoading(true);
    setError(null);
    try {
      const next = await extractInvoice(documentText);
      setResult(next);
      setEdited({ ...next.fields });
    } catch (caught) {
      const message = caught instanceof ApiRequestError ? caught.message : "Backend unavailable.";
      setError(message);
    } finally {
      setLoading(false);
    }
  }

  function onClear() {
    setDocumentText("");
    setResult(null);
    setEdited(EMPTY_FIELDS);
    setError(null);
  }

  return (
    <div className="min-h-screen bg-slate-100 text-slate-900 md:grid md:grid-cols-[16rem_1fr]">
      <Sidebar view={view} onChange={setView} />
      <div className="min-w-0">
        <TopBar connected={connected} modelLabel={modelLabel} />
        <main className="p-4 md:p-8">
          {view === "comparison" ? (
            <ComingSoon
              title="Model comparison"
              body="Teacher and student comparison stays in the Phase 4 benchmark. This screen will not invent a new score."
            />
          ) : null}
          {view === "benchmarks" ? (
            <ComingSoon
              title="Benchmark results"
              body="Locked-test scores stay in the saved Phase 4 reports. This screen is not a second evaluation."
            />
          ) : null}
          {view === "dashboard" || view === "extract" ? (
            <div className="space-y-6">
              {view === "dashboard" ? (
                <section className="rounded-2xl border border-slate-200 bg-white p-5 shadow-sm">
                  <h2 className="text-lg font-semibold">Local student extraction</h2>
                  <p className="mt-2 max-w-3xl text-sm leading-6 text-slate-600">
                    DistillOps reads one invoice with the frozen FLAN-T5-small student. Review every result. The
                    response does not include a confidence score.
                  </p>
                </section>
              ) : null}
              <ExtractWorkspace
                documentText={documentText}
                onDocumentText={setDocumentText}
                loading={loading}
                error={error}
                result={result}
                edited={edited}
                onEdit={editField}
                onExtract={() => void onExtract()}
                onClear={onClear}
              />
            </div>
          ) : null}
        </main>
      </div>
    </div>
  );
}
