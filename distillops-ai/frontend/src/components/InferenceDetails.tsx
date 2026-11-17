import type { ExtractResponse } from "../types";

type InferenceDetailsProps = {
  result: ExtractResponse | null;
};

export function InferenceDetails({ result }: InferenceDetailsProps) {
  return (
    <section className="rounded-2xl border border-slate-200 bg-white p-5 shadow-sm" aria-labelledby="details-heading">
      <h2 id="details-heading" className="text-lg font-semibold text-slate-900">
        Inference details
      </h2>
      {result === null ? (
        <p className="mt-3 text-sm text-slate-500">Run an extraction to see the model, latency, and review notes.</p>
      ) : (
        <dl className="mt-4 grid gap-3 text-sm sm:grid-cols-3">
          <div>
            <dt className="text-slate-500">Model</dt>
            <dd className="font-medium text-slate-900">{result.model}</dd>
          </div>
          <div>
            <dt className="text-slate-500">Latency</dt>
            <dd className="font-medium text-slate-900">{Math.round(result.latency_ms)} ms</dd>
          </div>
          <div>
            <dt className="text-slate-500">Review status</dt>
            <dd className="font-medium text-amber-800">{result.requires_review ? "Needs review" : "No review flag"}</dd>
          </div>
          <div className="sm:col-span-3">
            <dt className="text-slate-500">Validation warnings</dt>
            <dd>
              {result.warnings.length === 0 ? (
                <p className="mt-1 text-slate-700">None.</p>
              ) : (
                <ul className="mt-2 list-disc space-y-1 pl-5 text-slate-700">
                  {result.warnings.map((warning) => (
                    <li key={warning}>{warning}</li>
                  ))}
                </ul>
              )}
            </dd>
          </div>
        </dl>
      )}
    </section>
  );
}
