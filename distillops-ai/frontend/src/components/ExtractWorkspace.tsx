import { Loader2 } from "lucide-react";
import { SAMPLE_INVOICE_LABEL, SAMPLE_INVOICE_TEXT } from "../sampleInvoice";
import { isDocumentResult, type ExtractResponse, type InputMode, type InvoiceFields } from "../types";
import { InferenceDetails } from "./InferenceDetails";
import { ResultsPanel } from "./ResultsPanel";
import { UploadPanel } from "./UploadPanel";

type ExtractWorkspaceProps = {
  mode: InputMode;
  onMode: (mode: InputMode) => void;
  documentText: string;
  onDocumentText: (value: string) => void;
  file: File | null;
  onFile: (file: File | null, error?: string) => void;
  loading: boolean;
  error: string | null;
  result: ExtractResponse | null;
  edited: InvoiceFields;
  onEdit: (key: keyof InvoiceFields, value: string) => void;
  onExtract: () => void;
  onClear: () => void;
};

export function ExtractWorkspace({
  mode,
  onMode,
  documentText,
  onDocumentText,
  file,
  onFile,
  loading,
  error,
  result,
  edited,
  onEdit,
  onExtract,
  onClear,
}: ExtractWorkspaceProps) {
  const documentResult = isDocumentResult(result) ? result : null;
  return (
    <div className="space-y-6">
      <div className="flex flex-wrap gap-2" role="tablist" aria-label="Invoice input">
        <button
          type="button"
          role="tab"
          aria-selected={mode === "text"}
          onClick={() => onMode("text")}
          className={`rounded-lg px-3 py-2 text-sm font-medium ${
            mode === "text" ? "bg-navy text-white" : "border border-slate-300 bg-white text-slate-700"
          }`}
        >
          Paste invoice text
        </button>
        <button
          type="button"
          role="tab"
          aria-selected={mode === "upload"}
          onClick={() => onMode("upload")}
          className={`rounded-lg px-3 py-2 text-sm font-medium ${
            mode === "upload" ? "bg-navy text-white" : "border border-slate-300 bg-white text-slate-700"
          }`}
        >
          Upload invoice file
        </button>
      </div>
      <div className="grid gap-6 lg:grid-cols-2">
        <section className="rounded-2xl border border-slate-200 bg-white p-5 shadow-sm">
          {mode === "text" ? (
            <>
              <div className="flex flex-wrap items-start justify-between gap-3">
                <div>
                  <h2 className="text-lg font-semibold text-slate-900">Invoice text</h2>
                  <p className="mt-1 text-sm text-slate-500">Paste one invoice. The student reads this text as written.</p>
                </div>
                <button
                  type="button"
                  onClick={() => onDocumentText(SAMPLE_INVOICE_TEXT)}
                  className="rounded-lg border border-slate-300 px-3 py-2 text-sm font-medium text-slate-700 hover:bg-slate-50"
                >
                  {SAMPLE_INVOICE_LABEL}
                </button>
              </div>
              <p className="mt-3 text-xs text-slate-500">
                The sample button only fills this box. Extraction still comes from the student API.
              </p>
              <textarea
                value={documentText}
                onChange={(event) => onDocumentText(event.target.value)}
                aria-label="Invoice text"
                className="mt-3 min-h-72 w-full rounded-xl border border-slate-300 p-3 text-sm leading-6 outline-none ring-blue-600 focus:ring-2"
                placeholder="Paste invoice text"
              />
            </>
          ) : (
            <>
              <h2 className="text-lg font-semibold text-slate-900">Invoice file</h2>
              <p className="mt-1 text-sm text-slate-500">
                The file stays in this browser. The server reads it once and does not keep the original.
              </p>
              <div className="mt-3">
                <UploadPanel file={file} onFile={onFile} />
              </div>
            </>
          )}
          <div className="mt-4 flex flex-wrap gap-3">
            <button
              type="button"
              onClick={onExtract}
              disabled={loading}
              className="inline-flex items-center gap-2 rounded-lg bg-blue-700 px-4 py-2 text-sm font-semibold text-white hover:bg-blue-800 disabled:cursor-wait disabled:bg-blue-400"
            >
              {loading ? <Loader2 className="h-4 w-4 animate-spin" aria-hidden="true" /> : null}
              {loading ? "Extracting fields" : "Extract Fields"}
            </button>
            <button
              type="button"
              onClick={onClear}
              className="rounded-lg border border-slate-300 px-4 py-2 text-sm font-medium text-slate-700 hover:bg-slate-50"
            >
              Clear
            </button>
          </div>
          {loading ? (
            <p className="mt-3 text-sm text-blue-800" role="status">
              {mode === "upload"
                ? "Reading the document. OCR and the first model request can take longer."
                : "Running the student. The first request loads the checkpoint and can take longer."}
            </p>
          ) : null}
          {error ? (
            <p className="mt-3 rounded-lg bg-rose-50 px-3 py-2 text-sm text-rose-800" role="alert">
              {error}
            </p>
          ) : null}
        </section>
        <ResultsPanel raw={result?.fields ?? null} edited={edited} onEdit={onEdit} />
      </div>
      <InferenceDetails result={result} />
      {documentResult ? (
        <details className="rounded-2xl border border-slate-200 bg-white p-5 shadow-sm">
          <summary className="cursor-pointer text-sm font-semibold text-slate-900">Extracted text</summary>
          <pre className="mt-3 max-h-80 overflow-auto whitespace-pre-wrap text-sm leading-6 text-slate-700">
            {documentResult.extracted_text}
          </pre>
        </details>
      ) : null}
    </div>
  );
}
