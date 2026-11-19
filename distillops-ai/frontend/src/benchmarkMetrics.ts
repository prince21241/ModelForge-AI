import type { SavedFieldMetric, SavedReportBody } from "./types";

const FIELD_TITLES: Record<string, string> = {
  vendor: "Vendor",
  invoice_number: "Invoice number",
  date: "Date",
  total: "Total",
  currency: "Currency",
};

const FIELD_ORDER = ["vendor", "invoice_number", "date", "total", "currency"];

export type ComparisonPoint = {
  key: string;
  label: string;
  teacher: number | null;
  student: number | null;
  teacherLabel: string;
  studentLabel: string;
};

export type SinglePoint = {
  key: string;
  label: string;
  value: number | null;
  text: string;
};

export function fieldTitle(field: string): string {
  return FIELD_TITLES[field] ?? field;
}

export function orderedFields(fields: string[]): string[] {
  const unique = [...new Set(fields)];
  const known = FIELD_ORDER.filter((field) => unique.includes(field));
  const extra = unique.filter((field) => !FIELD_ORDER.includes(field)).sort();
  return [...known, ...extra];
}

export function toPercent(rate: number | null): number | null {
  if (rate === null || !Number.isFinite(rate)) {
    return null;
  }
  return Math.round(rate * 100);
}

export function exactMatchRate(report: SavedReportBody): number | null {
  const rate = report.end_to_end_exact_match_accuracy;
  if (typeof rate !== "number" || !Number.isFinite(rate)) {
    return null;
  }
  return rate;
}

export function exactMatchDetail(report: SavedReportBody): string {
  if (exactMatchRate(report) === null) {
    return `${report.documents_scored} scored. End-to-end accuracy was not reported.`;
  }
  return `${report.exact_match_count} / ${report.documents_in_split}`;
}

export function jsonValidityRate(report: SavedReportBody): number | null {
  if (report.documents_scored <= 0) {
    return null;
  }
  return report.json_valid_count / report.documents_scored;
}

export function jsonValidityDetail(report: SavedReportBody): string {
  if (report.documents_scored <= 0) {
    return "Not available";
  }
  return `${report.json_valid_count} / ${report.documents_scored}`;
}

export function fieldRate(metric: SavedFieldMetric | undefined): number | null {
  if (!metric) {
    return null;
  }
  if (typeof metric.accuracy === "number" && Number.isFinite(metric.accuracy)) {
    return metric.accuracy;
  }
  if (metric.scored > 0) {
    return metric.correct / metric.scored;
  }
  return null;
}

export function formatMs(value: number | null): string {
  if (value === null || !Number.isFinite(value)) {
    return "Not available";
  }
  return `${Math.round(value)} ms`;
}

export function formatCost(report: SavedReportBody): string {
  if (!report.pricing_configured || report.estimated_cost_usd === null) {
    return "Not configured";
  }
  return `$${report.estimated_cost_usd.toFixed(4)}`;
}

export function percentText(rate: number | null): string {
  const percent = toPercent(rate);
  return percent === null ? "Not available" : `${percent}%`;
}

export function lockedTestLabel(reports: Array<SavedReportBody | null | undefined>): string {
  const sizes = reports.filter((report): report is SavedReportBody => report != null).map((report) => report.documents_in_split);
  if (sizes.length === 0 || new Set(sizes).size !== 1) {
    return "Locked test: Not available";
  }
  return `Locked test: ${sizes[0]} synthetic invoices`;
}

export function displayModelName(report: SavedReportBody): string {
  const id = report.model_id.toLowerCase();
  if (id.includes("gpt-4o-mini")) {
    return "GPT-4o-mini";
  }
  if (id.includes("flan-t5-small")) {
    return "FLAN-T5-small";
  }
  return report.model_id;
}

function metricLabel(metric: SavedFieldMetric | undefined, rate: number | null): string {
  if (!metric || rate === null) {
    return "Not available";
  }
  return `${percentText(rate)} (${metric.correct} / ${metric.scored})`;
}

export function fieldComparison(teacher: SavedReportBody, student: SavedReportBody): ComparisonPoint[] {
  const fields = orderedFields([
    ...teacher.field_metrics.map((metric) => metric.field),
    ...student.field_metrics.map((metric) => metric.field),
  ]);
  return fields.map((field) => {
    const left = teacher.field_metrics.find((metric) => metric.field === field);
    const right = student.field_metrics.find((metric) => metric.field === field);
    const leftRate = fieldRate(left);
    const rightRate = fieldRate(right);
    return {
      key: field,
      label: fieldTitle(field),
      teacher: toPercent(leftRate),
      student: toPercent(rightRate),
      teacherLabel: metricLabel(left, leftRate),
      studentLabel: metricLabel(right, rightRate),
    };
  });
}

export function latencyComparison(teacher: SavedReportBody, student: SavedReportBody): ComparisonPoint[] {
  const rows: Array<[string, string, number | null, number | null]> = [
    ["median", "Median", teacher.latency_ms_median, student.latency_ms_median],
    ["average", "Average", teacher.latency_ms_avg, student.latency_ms_avg],
    ["p95", "P95", teacher.latency_ms_p95, student.latency_ms_p95],
  ];
  return rows.map(([key, label, left, right]) => ({
    key,
    label,
    teacher: left,
    student: right,
    teacherLabel: formatMs(left),
    studentLabel: formatMs(right),
  }));
}

export function errorComparison(teacher: SavedReportBody, student: SavedReportBody): ComparisonPoint[] {
  const rows: Array<[string, string, number, number]> = [
    ["incorrect", "Incorrect fields", teacher.incorrect_field_count, student.incorrect_field_count],
    ["missing", "Missing fields", teacher.missing_field_count, student.missing_field_count],
    ["hallucinated", "Hallucinated fields", teacher.hallucinated_field_count, student.hallucinated_field_count],
  ];
  return rows.map(([key, label, left, right]) => ({
    key,
    label,
    teacher: left,
    student: right,
    teacherLabel: String(left),
    studentLabel: String(right),
  }));
}

export function fieldSeries(report: SavedReportBody): SinglePoint[] {
  return orderedFields(report.field_metrics.map((metric) => metric.field)).map((field) => {
    const metric = report.field_metrics.find((item) => item.field === field);
    const rate = fieldRate(metric);
    return {
      key: field,
      label: fieldTitle(field),
      value: toPercent(rate),
      text: metricLabel(metric, rate),
    };
  });
}

export function latencySeries(report: SavedReportBody): SinglePoint[] {
  const rows: Array<[string, string, number | null]> = [
    ["median", "Median", report.latency_ms_median],
    ["average", "Average", report.latency_ms_avg],
    ["p95", "P95", report.latency_ms_p95],
  ];
  return rows.map(([key, label, value]) => ({
    key,
    label,
    value,
    text: formatMs(value),
  }));
}

export function errorSeries(report: SavedReportBody): SinglePoint[] {
  const rows: Array<[string, string, number]> = [
    ["incorrect", "Incorrect fields", report.incorrect_field_count],
    ["missing", "Missing fields", report.missing_field_count],
    ["hallucinated", "Hallucinated fields", report.hallucinated_field_count],
  ];
  return rows.map(([key, label, value]) => ({
    key,
    label,
    value,
    text: String(value),
  }));
}
