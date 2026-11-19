import { render, screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { afterEach, describe, expect, it, vi } from "vitest";
import App from "./App";
import {
  exactMatchDetail,
  exactMatchRate,
  fieldComparison,
  fieldRate,
  formatMs,
  latencyComparison,
  lockedTestLabel,
  toPercent,
} from "./benchmarkMetrics";
import type { BenchmarkLibrary, SavedFieldMetric, SavedReportBody } from "./types";

const LIBRARY: BenchmarkLibrary = {
  source: "phase4-saved-reports",
  rescored: false,
  reports: [
    {
      available: true,
      filename: "phase4_teacher_test.json",
      model_role: "teacher",
      split: "test",
      detail: null,
      report: {
        status: "completed",
        model_role: "teacher",
        model_id: "gpt-4o-mini",
        split: "test",
        documents_in_split: 5,
        documents_scored: 5,
        unscored_document_ids: [],
        json_valid_count: 5,
        schema_valid_count: 5,
        malformed_json_count: 0,
        exact_match_count: 5,
        exact_match_accuracy: 1,
        end_to_end_exact_match_accuracy: 1,
        field_metrics: [
          { field: "vendor", correct: 5, scored: 5, accuracy: 1, missing: 0, hallucinated: 0, incorrect: 0 },
        ],
        missing_field_count: 0,
        hallucinated_field_count: 0,
        incorrect_field_count: 0,
        latency_ms_avg: 3000,
        latency_ms_median: 1052,
        latency_ms_p95: 8892,
        warmup_excluded: true,
        input_tokens: 3215,
        output_tokens: 201,
        api_calls: 5,
        api_failure_count: 0,
        pricing_configured: false,
        estimated_cost_usd: null,
        cost_assumption: "Teacher price was not configured.",
        rows: [],
      },
    },
    {
      available: true,
      filename: "phase4_student_test.json",
      model_role: "student",
      split: "test",
      detail: null,
      report: {
        status: "completed",
        model_role: "student",
        model_id: "frozen-student",
        split: "test",
        documents_in_split: 5,
        documents_scored: 5,
        unscored_document_ids: [],
        json_valid_count: 5,
        schema_valid_count: 5,
        malformed_json_count: 0,
        exact_match_count: 3,
        exact_match_accuracy: 0.6,
        end_to_end_exact_match_accuracy: 0.6,
        field_metrics: [
          { field: "vendor", correct: 4, scored: 5, accuracy: 0.8, missing: 0, hallucinated: 0, incorrect: 1 },
        ],
        missing_field_count: 0,
        hallucinated_field_count: 0,
        incorrect_field_count: 3,
        latency_ms_avg: 1233,
        latency_ms_median: 1239,
        latency_ms_p95: 1333,
        warmup_excluded: true,
        input_tokens: 0,
        output_tokens: 0,
        api_calls: 0,
        api_failure_count: 0,
        pricing_configured: false,
        estimated_cost_usd: null,
        cost_assumption: "Student price was not configured.",
        rows: [
          {
            document_id: "inv-synth-012",
            template_group: "prose_memo",
            exact_match: false,
            incorrect_fields: ["date"],
            missing_fields: [],
            hallucinated_fields: [],
            latency_ms: 1354,
            prediction: { date: "2025-09-9" },
            expected: { date: "2025-06-09" },
          },
        ],
      },
    },
    {
      available: true,
      filename: "phase4_teacher_val.json",
      model_role: "teacher",
      split: "val",
      detail: null,
      report: {
        status: "cache_incomplete",
        model_role: "teacher",
        model_id: "gpt-4o-mini",
        split: "val",
        documents_in_split: 5,
        documents_scored: 2,
        unscored_document_ids: ["inv-synth-023"],
        json_valid_count: 2,
        schema_valid_count: 2,
        malformed_json_count: 0,
        exact_match_count: 2,
        exact_match_accuracy: 1,
        end_to_end_exact_match_accuracy: null,
        field_metrics: [],
        missing_field_count: 0,
        hallucinated_field_count: 0,
        incorrect_field_count: 0,
        latency_ms_avg: null,
        latency_ms_median: null,
        latency_ms_p95: null,
        warmup_excluded: true,
        input_tokens: 0,
        output_tokens: 0,
        api_calls: 0,
        api_failure_count: 0,
        pricing_configured: false,
        estimated_cost_usd: null,
        cost_assumption: "unset",
        rows: [],
      },
    },
    {
      available: false,
      filename: "phase4_student_val.json",
      model_role: "student",
      split: "val",
      detail: "This saved report is not on this machine.",
      report: null,
    },
  ],
};

function jsonResponse(body: unknown): Response {
  return new Response(JSON.stringify(body), {
    status: 200,
    headers: { "Content-Type": "application/json" },
  });
}

afterEach(() => {
  vi.unstubAllGlobals();
});

function mockBackend() {
  vi.stubGlobal(
    "fetch",
    vi.fn(async (input: RequestInfo | URL) => {
      const url = String(input);
      if (url.endsWith("/api/health")) {
        return jsonResponse({ status: "ok", service: "distillops", student_checkpoint: "epoch-30", student_loaded: false });
      }
      if (url.endsWith("/api/models")) {
        return jsonResponse({
          models: [{ name: "student", identifier: "google/flan-t5-small", checkpoint: "epoch-30", available: true, loaded: false }],
        });
      }
      if (url.endsWith("/api/benchmarks")) {
        return jsonResponse(LIBRARY);
      }
      throw new Error(`unexpected url ${url}`);
    }),
  );
}

function reportAt(index: number): SavedReportBody {
  const report = LIBRARY.reports[index]?.report;
  if (!report) {
    throw new Error(`fixture report ${index} is missing`);
  }
  return report;
}

describe("saved benchmark metrics", () => {
  it("derives percentages from the saved rates and counts", () => {
    const teacher = reportAt(0);
    const student = reportAt(1);
    expect(toPercent(exactMatchRate(teacher))).toBe(100);
    expect(toPercent(exactMatchRate(student))).toBe(60);
    const fields = fieldComparison(teacher, student);
    expect(fields).toEqual([
      {
        key: "vendor",
        label: "Vendor",
        teacher: 100,
        student: 80,
        teacherLabel: "100% (5 / 5)",
        studentLabel: "80% (4 / 5)",
      },
    ]);
    expect(fields.some((point) => point.key === "invoice_number")).toBe(false);
    const latency = latencyComparison(teacher, student);
    expect(latency.map((point) => point.key)).toEqual(["median", "average", "p95"]);
    expect(latency[0]).toMatchObject({ teacher: 1052, student: 1239, teacherLabel: "1052 ms", studentLabel: "1239 ms" });
  });

  it("keeps a missing end-to-end rate and a missing latency as unavailable", () => {
    const validation = reportAt(2);
    expect(validation.exact_match_accuracy).toBe(1);
    expect(exactMatchRate(validation)).toBeNull();
    expect(exactMatchDetail(validation)).toBe("2 scored. End-to-end accuracy was not reported.");
    expect(formatMs(validation.latency_ms_median)).toBe("Not available");
    const missingAccuracy: SavedFieldMetric = {
      field: "date",
      correct: 1,
      scored: 2,
      accuracy: null,
      missing: 0,
      hallucinated: 0,
      incorrect: 1,
    };
    expect(fieldRate(missingAccuracy)).toBe(0.5);
    expect(toPercent(fieldRate(missingAccuracy))).toBe(50);
    expect(fieldRate({ ...missingAccuracy, correct: 0, scored: 0, accuracy: null })).toBeNull();
    expect(lockedTestLabel([reportAt(0), reportAt(1)])).toBe("Locked test: 5 synthetic invoices");
    expect(lockedTestLabel([])).toBe("Locked test: Not available");
  });
});

describe("saved benchmark screens", () => {
  it("shows the frozen test comparison without calling extract", async () => {
    mockBackend();
    const user = userEvent.setup();
    render(<App />);
    await user.click(screen.getByRole("button", { name: "Model Comparison" }));
    expect((await screen.findAllByText("Historical Phase 4 benchmark")).length).toBeGreaterThan(0);
    expect(screen.getByText("Locked test: 5 synthetic invoices")).toBeInTheDocument();
    expect(screen.getByText("Teacher: GPT-4o-mini")).toBeInTheDocument();
    expect(screen.getByText("Student: FLAN-T5-small")).toBeInTheDocument();
    expect(screen.getByText(/too small for reliable production accuracy/)).toBeInTheDocument();
    expect(screen.getByTestId("kpi-teacher-exact")).toHaveTextContent("100%");
    expect(screen.getByTestId("kpi-teacher-exact")).toHaveTextContent("5 / 5");
    expect(screen.getByTestId("kpi-student-exact")).toHaveTextContent("60%");
    expect(screen.getByTestId("kpi-student-exact")).toHaveTextContent("3 / 5");
    expect(screen.getByTestId("kpi-teacher-json")).toHaveTextContent("100%");
    expect(screen.getByTestId("kpi-student-json")).toHaveTextContent("5 / 5");
    expect(screen.getByTestId("chart-field-accuracy-data")).toHaveTextContent("Vendor");
    expect(screen.getByTestId("chart-field-accuracy-data")).toHaveTextContent("80% (4 / 5)");
    expect(screen.getByTestId("chart-field-accuracy-data")).not.toHaveTextContent("Invoice number");
    expect(screen.getByTestId("chart-latency-data")).toHaveTextContent("1052 ms");
    expect(screen.getByTestId("chart-latency-data")).toHaveTextContent("8892 ms");
    expect(screen.getByTestId("chart-errors-data")).toHaveTextContent("Incorrect fields");
    expect(screen.getByTestId("chart-errors-data")).toHaveTextContent("Hallucinated fields");
    expect(screen.getByText(/Teacher: Not configured/)).toBeInTheDocument();
    expect(screen.getByText(/Student: Not configured/)).toBeInTheDocument();
    expect(screen.getByTestId("performance-kpis").className).toContain("grid-cols-1");
    expect(screen.getByTestId("performance-kpis").className).toContain("sm:grid-cols-2");
    expect(screen.getByTestId("chart-field-accuracy").className).toContain("overflow-x-auto");
    expect(screen.getByText(/does not call the teacher/)).toBeInTheDocument();
    expect(vi.mocked(fetch).mock.calls.some((call) => String(call[0]).endsWith("/api/extract"))).toBe(false);
    await user.click(screen.getByText("Invoice-level results"));
    expect(screen.getByText("inv-synth-012")).toBeInTheDocument();
  });

  it("shows a missing report and an incomplete validation score", async () => {
    mockBackend();
    const user = userEvent.setup();
    render(<App />);
    await user.click(screen.getByRole("button", { name: "Benchmark Results" }));
    expect(await screen.findByText("This saved report is not on this machine.")).toBeInTheDocument();
    expect(screen.getAllByText("Historical Phase 4 benchmark").length).toBeGreaterThan(0);
    expect(screen.getByText("Locked test: 5 synthetic invoices")).toBeInTheDocument();
    expect(screen.getByText(/too small for reliable production accuracy/)).toBeInTheDocument();
    const validation = screen.getByTestId("report-teacher-val");
    expect(validation).toHaveTextContent("Not available");
    expect(validation).toHaveTextContent("2 scored. End-to-end accuracy was not reported.");
    expect(screen.getByText(/does not score invoices again/)).toBeInTheDocument();
    await user.click(screen.getByText("Invoice-level results for student test"));
    expect(screen.getByText("inv-synth-012")).toBeInTheDocument();
  });
});