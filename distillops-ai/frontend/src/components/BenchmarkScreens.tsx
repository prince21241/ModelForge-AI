import { useEffect, useState, type ReactNode } from "react";
import { LiveComparison } from "./LiveComparison";
import { ChevronDown } from "lucide-react";
import { ApiRequestError, fetchBenchmarks } from "../api";
import {
  displayModelName,
  errorComparison,
  errorSeries,
  exactMatchDetail,
  exactMatchRate,
  fieldComparison,
  fieldSeries,
  formatCost,
  jsonValidityDetail,
  jsonValidityRate,
  latencyComparison,
  latencySeries,
  lockedTestLabel,
  percentText,
} from "../benchmarkMetrics";
import type { BenchmarkLibrary, SavedBenchmark, SavedReportBody } from "../types";
import { ComparisonChart, SingleChart } from "./BenchmarkCharts";

const STUDY_LABELS = ["Historical Phase 4 benchmark", "Teacher: GPT-4o-mini", "Student: FLAN-T5-small"] as const;

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

function testReport(library: BenchmarkLibrary | null, role: string): SavedReportBody | null {
  return library?.reports.find((item) => item.model_role === role && item.split === "test")?.report ?? null;
}

function StudyBanner({
  library,
  title,
  note,
}: {
  library: BenchmarkLibrary | null;
  title: string;
  note: string;
}) {
  const locked = lockedTestLabel([testReport(library, "teacher"), testReport(library, "student")]);
  return (
    <header className="rounded-2xl border border-slate-200 bg-white p-5 shadow-sm">
      <p className="text-xs font-semibold uppercase tracking-wide text-blue-700">Historical Phase 4 benchmark</p>
      <h2 className="mt-2 text-lg font-semibold text-slate-900">{title}</h2>
      <p className="mt-2 max-w-3xl text-sm leading-6 text-slate-600">{note}</p>
      {library ? (
        <ul className="mt-4 flex flex-wrap gap-2">
          {[...STUDY_LABELS, locked].map((label) => (
            <li key={label} className="rounded-full border border-slate-200 bg-slate-50 px-3 py-1 text-xs font-medium text-slate-700">
              {label}
            </li>
          ))}
        </ul>
      ) : null}
      <p className="mt-4 max-w-3xl text-sm leading-6 text-slate-600">
        This sample is too small for reliable production accuracy claims.
      </p>
    </header>
  );
}

function KpiCard({
  testId,
  label,
  value,
  detail,
  tone,
}: {
  testId: string;
  label: string;
  value: string;
  detail: string;
  tone: "blue" | "teal";
}) {
  return (
    <article className="rounded-2xl border border-slate-200 bg-white p-5 shadow-sm" data-testid={testId}>
      <h3 className="text-xs font-semibold uppercase tracking-wide text-slate-500">{label}</h3>
      <p className={`mt-3 text-3xl font-semibold tracking-tight ${tone === "blue" ? "text-blue-700" : "text-teal-700"}`}>
        {value}
      </p>
      <p className="mt-2 text-sm text-slate-600">{detail}</p>
    </article>
  );
}

function InvoiceTable({ report }: { report: SavedReportBody }) {
  if (report.rows.length === 0) {
    return <p className="text-sm text-slate-600">No scored invoice rows in this report.</p>;
  }
  return (
    <div className="max-w-full overflow-x-auto">
      <table className="w-full min-w-[20rem] text-left text-sm">
        <thead className="text-slate-500">
          <tr>
            <th className="py-2 pr-3 font-medium" scope="col">
              Document
            </th>
            <th className="py-2 pr-3 font-medium" scope="col">
              Exact
            </th>
            <th className="py-2 pr-3 font-medium" scope="col">
              Incorrect fields
            </th>
            <th className="py-2 font-medium" scope="col">
              Latency
            </th>
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
  );
}

function InvoiceDetails({ label, children }: { label: string; children: ReactNode }) {
  return (
    <details className="group rounded-2xl border border-slate-200 bg-white shadow-sm">
      <summary className="flex cursor-pointer list-none items-center justify-between px-5 py-4 text-sm font-semibold text-slate-900 [&::-webkit-details-marker]:hidden">
        {label}
        <ChevronDown className="h-4 w-4 text-slate-500 transition group-open:rotate-180" aria-hidden="true" />
      </summary>
      <div className="border-t border-slate-100 px-5 py-4">{children}</div>
    </details>
  );
}

function ReportPanel({ item }: { item: SavedBenchmark }) {
  const report = item.report;
  const tone = item.model_role === "teacher" ? "#2563EB" : "#0D9488";
  return (
    <article className="min-w-0 space-y-4 rounded-2xl border border-slate-200 bg-white p-5 shadow-sm" data-testid={`report-${item.model_role}-${item.split}`}>
      <div>
        <p className="text-xs font-semibold uppercase tracking-wide text-blue-700">
          {item.model_role} · {item.split}
        </p>
        <h3 className="mt-1 text-lg font-semibold text-slate-900">{item.filename}</h3>
        <p className="mt-1 text-sm text-slate-600">
          {item.split === "test"
            ? "Locked test split."
            : "Validation split. End-to-end accuracy is shown only when the saved report includes it."}
        </p>
      </div>
      {report === null ? (
        <p className="text-sm text-slate-600">{item.detail ?? "Not available"}</p>
      ) : (
        <>
          <dl className="grid gap-3 sm:grid-cols-2 xl:grid-cols-4">
            <div>
              <dt className="text-xs uppercase tracking-wide text-slate-500">Model</dt>
              <dd className="mt-1 text-sm font-medium text-slate-900">{displayModelName(report)}</dd>
              <dd className="text-xs text-slate-500">{report.model_id}</dd>
            </div>
            <div>
              <dt className="text-xs uppercase tracking-wide text-slate-500">Status</dt>
              <dd className="mt-1 text-sm font-medium text-slate-900">{report.status}</dd>
            </div>
            <div>
              <dt className="text-xs uppercase tracking-wide text-slate-500">Exact match</dt>
              <dd className="mt-1 text-sm font-medium text-slate-900">{percentText(exactMatchRate(report))}</dd>
              <dd className="text-xs text-slate-500">{exactMatchDetail(report)}</dd>
            </div>
            <div>
              <dt className="text-xs uppercase tracking-wide text-slate-500">JSON validity</dt>
              <dd className="mt-1 text-sm font-medium text-slate-900">{percentText(jsonValidityRate(report))}</dd>
              <dd className="text-xs text-slate-500">{jsonValidityDetail(report)}</dd>
            </div>
          </dl>
          {report.unscored_document_ids.length > 0 ? (
            <p className="text-sm text-slate-600">Unscored documents: {report.unscored_document_ids.join(", ")}.</p>
          ) : null}
          <div className="grid gap-6 xl:grid-cols-2">
            <SingleChart
              title="Field accuracy"
              data={fieldSeries(report)}
              unit="%"
              testId={`field-${item.model_role}-${item.split}`}
              color={tone}
              domain={[0, 100]}
            />
            <SingleChart
              title="Latency"
              data={latencySeries(report)}
              unit=" ms"
              testId={`latency-${item.model_role}-${item.split}`}
              color={tone}
            />
            <SingleChart
              title="Error breakdown"
              data={errorSeries(report)}
              unit=""
              testId={`errors-${item.model_role}-${item.split}`}
              color={tone}
            />
          </div>
          <p className="text-sm text-slate-600">Estimated cost: {formatCost(report)}</p>
          <p className="text-xs leading-5 text-slate-500">{report.cost_assumption}</p>
          <InvoiceDetails label={`Invoice-level results for ${item.model_role} ${item.split}`}>
            <InvoiceTable report={report} />
          </InvoiceDetails>
        </>
      )}
    </article>
  );
}

export function BenchmarkView() {
  const { library, error } = useLibrary();
  return (
    <section className="min-w-0 space-y-4">
      <StudyBanner
        library={library}
        title="Benchmark results"
        note="These are the saved Phase 4 reports. Opening this screen does not score invoices again."
      />
      {error ? (
        <p className="rounded-lg bg-rose-50 px-3 py-2 text-sm text-rose-800" role="alert">
          {error}
        </p>
      ) : null}
      {library === null && error === null ? <p className="text-sm text-slate-500">Loading saved reports.</p> : null}
      {library?.reports.map((item) => (
        <ReportPanel key={item.filename} item={item} />
      ))}
    </section>
  );
}

function HistoricalComparison() {
  const { library, error } = useLibrary();
  const teacher = testReport(library, "teacher");
  const student = testReport(library, "student");
  const ready = teacher && student ? { teacher, student } : null;

  return (
    <section className="min-w-0 space-y-4" data-testid="comparison-dashboard">
      <StudyBanner
        library={library}
        title="Model comparison"
        note="Teacher and student numbers below are the completed Phase 4 test reports. This screen does not call the teacher and does not rescore the locked test."
      />
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
        <>
          <div className="grid grid-cols-1 gap-4 sm:grid-cols-2 xl:grid-cols-4" data-testid="performance-kpis">
            <KpiCard
              testId="kpi-teacher-exact"
              label="Teacher exact-match accuracy"
              value={percentText(exactMatchRate(ready.teacher))}
              detail={exactMatchDetail(ready.teacher)}
              tone="blue"
            />
            <KpiCard
              testId="kpi-student-exact"
              label="Student exact-match accuracy"
              value={percentText(exactMatchRate(ready.student))}
              detail={exactMatchDetail(ready.student)}
              tone="teal"
            />
            <KpiCard
              testId="kpi-teacher-json"
              label="Teacher JSON validity"
              value={percentText(jsonValidityRate(ready.teacher))}
              detail={jsonValidityDetail(ready.teacher)}
              tone="blue"
            />
            <KpiCard
              testId="kpi-student-json"
              label="Student JSON validity"
              value={percentText(jsonValidityRate(ready.student))}
              detail={jsonValidityDetail(ready.student)}
              tone="teal"
            />
          </div>
          <ComparisonChart
            title="Field accuracy"
            description="Share of scored fields that match the expected value in the saved test reports."
            data={fieldComparison(ready.teacher, ready.student)}
            unit="%"
            testId="chart-field-accuracy"
            domain={[0, 100]}
          />
          <div className="grid grid-cols-1 gap-4 xl:grid-cols-2">
            <ComparisonChart
              title="Inference latency"
              description="Median, average, and P95 latency from the saved reports. A missing statistic stays blank."
              data={latencyComparison(ready.teacher, ready.student)}
              unit=" ms"
              testId="chart-latency"
            />
            <ComparisonChart
              title="Error breakdown"
              description="Incorrect, missing, and hallucinated field counts from the saved reports."
              data={errorComparison(ready.teacher, ready.student)}
              unit=""
              testId="chart-errors"
            />
          </div>
          <p className="text-sm text-slate-600">
            Estimated cost — Teacher: {formatCost(ready.teacher)} · Student: {formatCost(ready.student)}
          </p>
          <p className="text-xs leading-5 text-slate-500">{ready.student.cost_assumption}</p>
          <InvoiceDetails label="Invoice-level results">
            <div className="grid gap-6 lg:grid-cols-2">
              <div>
                <h3 className="mb-2 text-sm font-semibold text-blue-700">Teacher</h3>
                <InvoiceTable report={ready.teacher} />
              </div>
              <div>
                <h3 className="mb-2 text-sm font-semibold text-teal-700">Student</h3>
                <InvoiceTable report={ready.student} />
              </div>
            </div>
          </InvoiceDetails>
        </>
      ) : null}
    </section>
  );
}

export function ComparisonView() {
  const [tab, setTab] = useState<"historical" | "live">("historical");
  return (
    <div className="space-y-4">
      <div className="flex flex-wrap gap-2" role="tablist" aria-label="Comparison">
        <button
          type="button"
          role="tab"
          aria-selected={tab === "historical"}
          onClick={() => setTab("historical")}
          className={`rounded-lg px-3 py-2 text-sm font-medium ${
            tab === "historical" ? "bg-navy text-white" : "border border-slate-300 bg-white text-slate-700"
          }`}
        >
          Historical benchmark
        </button>
        <button
          type="button"
          role="tab"
          aria-selected={tab === "live"}
          onClick={() => setTab("live")}
          className={`rounded-lg px-3 py-2 text-sm font-medium ${
            tab === "live" ? "bg-navy text-white" : "border border-slate-300 bg-white text-slate-700"
          }`}
        >
          Live comparison
        </button>
      </div>
      {tab === "historical" ? <HistoricalComparison /> : <LiveComparison />}
    </div>
  );
}
