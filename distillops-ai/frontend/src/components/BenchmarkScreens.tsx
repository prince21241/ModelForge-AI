import { useEffect, useState } from "react";
import { ApiRequestError, fetchBenchmarks } from "../api";
import type { BenchmarkLibrary, SavedBenchmark, SavedReportBody } from "../types";

function countLabel(count: number, total: number, rate: number | null): string {
  if (rate === null) {
    return `${count} scored. End-to-end accuracy was not reported.`;
  }
  return `${count} / ${total}`;
}

function money(report: SavedReportBody): string {
  if (!report.pricing_configured || report.estimated_cost_usd === null) {
    return "Not configured";
  }
  return `$${report.estimated_cost_usd.toFixed(4)}`;
}

function ReportCard({ item }: { item: SavedBenchmark }) {
  const report = item.report;
  return (
    <article className="rounded-2xl border border-slate-200 bg-white p-5 shadow-sm">
      <p className="text-xs font-semibold uppercase tracking-wide text-blue-700">
        {item.model_role} · {item.split}
      </p>
      <h3 className="mt-1 text-lg font-semibold text-slate-900">{item.filename}</h3>
      {report === null ? (
        <p className="mt-3 text-sm text-slate-600">{item.detail}</p>
      ) : (
        <>
          <dl className="mt-4 grid gap-3 text-sm sm:grid-cols-3">
            <div>
              <dt className="text-slate-500">Status</dt>
              <dd className="font-medium text-slate-900">{report.status}</dd>
            </div>
            <div>
              <dt className="text-slate-500">Exact match</dt>
              <dd className="font-medium text-slate-900">
                {countLabel(report.exact_match_count, report.documents_in_split, report.end_to_end_exact_match_accuracy)}
              </dd>
            </div>
            <div>
              <dt className="text-slate-500">JSON valid</dt>
              <dd className="font-medium text-slate-900">
                {report.json_valid_count} / {report.documents_scored}
              </dd>
            </div>
            <div>
              <dt className="text-slate-500">Median latency</dt>
              <dd className="font-medium text-slate-900">
                {report.latency_ms_median === null ? "Not reported" : `${Math.round(report.latency_ms_median)} ms`}
              </dd>
            </div>
            <div>
              <dt className="text-slate-500">Field errors</dt>
              <dd className="font-medium text-slate-900">
                {report.missing_field_count} missing, {report.hallucinated_field_count} invented,{" "}
                {report.incorrect_field_count} incorrect
              </dd>
            </div>
            <div>
              <dt className="text-slate-500">Estimated cost</dt>
              <dd className="font-medium text-slate-900">{money(report)}</dd>
            </div>
          </dl>
          {report.unscored_document_ids.length > 0 ? (
            <p className="mt-3 text-sm text-slate-600">
              Unscored documents: {report.unscored_document_ids.join(", ")}.
            </p>
          ) : null}
          <div className="mt-4 overflow-x-auto">
            <table className="w-full text-left text-sm">
              <thead className="text-slate-500">
                <tr>
                  <th className="py-2 pr-3 font-medium">Document</th>
                  <th className="py-2 pr-3 font-medium">Exact</th>
                  <th className="py-2 pr-3 font-medium">Incorrect fields</th>
                  <th className="py-2 font-medium">Latency</th>
                </tr>
              </thead>
              <tbody>
                {report.rows.map((row) => (
                  <tr key={row.document_id} className="border-t border-slate-100">
                    <td className="py-2 pr-3 text-slate-800">{row.document_id}</td>
                    <td className="py-2 pr-3">{row.exact_match ? "Yes" : "No"}</td>
                    <td className="py-2 pr-3">{row.incorrect_fields.join(", ") || "None"}</td>
                    <td className="py-2">{Math.round(row.latency_ms)} ms</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </>
      )}
    </article>
  );
}

function useLibrary() {
  const [library, setLibrary] = useState<BenchmarkLibrary | null>(null);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    let cancelled = false;
    fetchBenchmarks()
      .then((next) => {
        if (!cancelled) {
          setLibrary(next);
        }
      })
      .catch((caught: unknown) => {
        if (!cancelled) {
          setError(caught instanceof ApiRequestError ? caught.message : "Backend unavailable.");
        }
      });
    return () => {
      cancelled = true;
    };
  }, []);

  return { library, error };
}

export function BenchmarkView() {
  const { library, error } = useLibrary();
  return (
    <section className="space-y-4">
      <div className="rounded-2xl border border-slate-200 bg-white p-5 shadow-sm">
        <h2 className="text-lg font-semibold text-slate-900">Benchmark results</h2>
        <p className="mt-2 max-w-3xl text-sm leading-6 text-slate-600">
          These are the saved Phase 4 reports. Opening this screen does not score invoices again.
        </p>
      </div>
      {error ? (
        <p className="rounded-lg bg-rose-50 px-3 py-2 text-sm text-rose-800" role="alert">
          {error}
        </p>
      ) : null}
      {library === null && error === null ? <p className="text-sm text-slate-500">Loading saved reports.</p> : null}
      {library?.reports.map((item) => (
        <ReportCard key={item.filename} item={item} />
      ))}
    </section>
  );
}

export function ComparisonView() {
  const { library, error } = useLibrary();
  const teacher = library?.reports.find((item) => item.model_role === "teacher" && item.split === "test");
  const student = library?.reports.find((item) => item.model_role === "student" && item.split === "test");
  const ready = teacher?.report && student?.report ? { teacher: teacher.report, student: student.report } : null;
  const fields = ready?.student.field_metrics.map((metric) => metric.field) ?? [];

  return (
    <section className="space-y-4">
      <div className="rounded-2xl border border-slate-200 bg-white p-5 shadow-sm">
        <h2 className="text-lg font-semibold text-slate-900">Model comparison</h2>
        <p className="mt-2 max-w-3xl text-sm leading-6 text-slate-600">
          Teacher and student numbers below are the completed Phase 4 test reports. This screen does not call the
          teacher and does not rescore the locked test.
        </p>
      </div>
      {error ? (
        <p className="rounded-lg bg-rose-50 px-3 py-2 text-sm text-rose-800" role="alert">
          {error}
        </p>
      ) : null}
      {library && !ready ? (
        <p className="rounded-2xl border border-slate-200 bg-white p-5 text-sm text-slate-600 shadow-sm">
          The completed test reports are not both available on this machine.
        </p>
      ) : null}
      {ready ? (
        <div className="rounded-2xl border border-slate-200 bg-white p-5 shadow-sm">
          <div className="overflow-x-auto">
            <table className="w-full text-left text-sm">
              <thead className="text-slate-500">
                <tr>
                  <th className="py-2 pr-3 font-medium">Measure</th>
                  <th className="py-2 pr-3 font-medium">Teacher</th>
                  <th className="py-2 font-medium">Student</th>
                </tr>
              </thead>
              <tbody>
                <tr className="border-t border-slate-100">
                  <td className="py-2 pr-3">Exact match</td>
                  <td className="py-2 pr-3">
                    {countLabel(
                      ready.teacher.exact_match_count,
                      ready.teacher.documents_in_split,
                      ready.teacher.end_to_end_exact_match_accuracy,
                    )}
                  </td>
                  <td className="py-2">
                    {countLabel(
                      ready.student.exact_match_count,
                      ready.student.documents_in_split,
                      ready.student.end_to_end_exact_match_accuracy,
                    )}
                  </td>
                </tr>
                {fields.map((field) => {
                  const left = ready.teacher.field_metrics.find((metric) => metric.field === field);
                  const right = ready.student.field_metrics.find((metric) => metric.field === field);
                  return (
                    <tr key={field} className="border-t border-slate-100">
                      <td className="py-2 pr-3">{field}</td>
                      <td className="py-2 pr-3">
                        {left ? `${left.correct} / ${left.scored}` : "Not reported"}
                      </td>
                      <td className="py-2">{right ? `${right.correct} / ${right.scored}` : "Not reported"}</td>
                    </tr>
                  );
                })}
                <tr className="border-t border-slate-100">
                  <td className="py-2 pr-3">Median latency</td>
                  <td className="py-2 pr-3">
                    {ready.teacher.latency_ms_median === null
                      ? "Not reported"
                      : `${Math.round(ready.teacher.latency_ms_median)} ms`}
                  </td>
                  <td className="py-2">
                    {ready.student.latency_ms_median === null
                      ? "Not reported"
                      : `${Math.round(ready.student.latency_ms_median)} ms`}
                  </td>
                </tr>
                <tr className="border-t border-slate-100">
                  <td className="py-2 pr-3">Estimated cost</td>
                  <td className="py-2 pr-3">{money(ready.teacher)}</td>
                  <td className="py-2">{money(ready.student)}</td>
                </tr>
              </tbody>
            </table>
          </div>
          <p className="mt-4 text-xs leading-5 text-slate-500">{ready.student.cost_assumption}</p>
        </div>
      ) : null}
    </section>
  );
}
