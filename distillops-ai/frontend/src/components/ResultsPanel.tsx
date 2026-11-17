import { FIELD_LABELS, type InvoiceFields } from "../types";

type ResultsPanelProps = {
  raw: InvoiceFields | null;
  edited: InvoiceFields;
  onEdit: (key: keyof InvoiceFields, value: string) => void;
};

export function ResultsPanel({ raw, edited, onEdit }: ResultsPanelProps) {
  return (
    <section className="rounded-2xl border border-slate-200 bg-white p-5 shadow-sm" aria-labelledby="results-heading">
      <h2 id="results-heading" className="text-lg font-semibold text-slate-900">
        Extraction results
      </h2>
      <p className="mt-1 text-sm text-slate-500">Edits stay in this form. The model prediction is kept beside them.</p>
      <div className="mt-4 space-y-4">
        {FIELD_LABELS.map((field) => {
          const predicted = raw?.[field.key] ?? "";
          const value = edited[field.key];
          const changed = raw !== null && value !== predicted;
          return (
            <div key={field.key}>
              <label htmlFor={`field-${field.key}`} className="text-sm font-medium text-slate-700">
                {field.label}
              </label>
              <input
                id={`field-${field.key}`}
                value={value}
                onChange={(event) => onEdit(field.key, event.target.value)}
                disabled={raw === null}
                className="mt-1 w-full rounded-lg border border-slate-300 px-3 py-2 text-sm outline-none ring-blue-600 focus:ring-2 disabled:bg-slate-50"
              />
              <p className="mt-1 text-xs text-slate-500">
                {raw === null
                  ? "Waiting for a model prediction."
                  : changed
                    ? `Edited. Model predicted “${predicted || "empty"}”.`
                    : "Model prediction"}
              </p>
            </div>
          );
        })}
      </div>
    </section>
  );
}
