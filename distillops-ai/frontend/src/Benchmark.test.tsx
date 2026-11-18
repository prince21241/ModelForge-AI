import { render, screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { afterEach, describe, expect, it, vi } from "vitest";
import App from "./App";

const LIBRARY = {
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

describe("saved benchmark screens", () => {
  it("shows the frozen test comparison without calling extract", async () => {
    mockBackend();
    const user = userEvent.setup();
    render(<App />);
    await user.click(screen.getByRole("button", { name: "Model Comparison" }));
    expect((await screen.findAllByText("5 / 5")).length).toBeGreaterThan(0);
    expect(screen.getByText("3 / 5")).toBeInTheDocument();
    expect(screen.getByText("4 / 5")).toBeInTheDocument();
    expect(screen.getAllByText("Not configured").length).toBeGreaterThan(0);
    expect(screen.getByText(/does not call the teacher/)).toBeInTheDocument();
    expect(vi.mocked(fetch).mock.calls.some((call) => String(call[0]).endsWith("/api/extract"))).toBe(false);
  });

  it("shows a missing report and an incomplete validation score", async () => {
    mockBackend();
    const user = userEvent.setup();
    render(<App />);
    await user.click(screen.getByRole("button", { name: "Benchmark Results" }));
    expect(await screen.findByText("This saved report is not on this machine.")).toBeInTheDocument();
    expect(screen.getByText("2 scored. End-to-end accuracy was not reported.")).toBeInTheDocument();
    expect(screen.getByText("inv-synth-012")).toBeInTheDocument();
    expect(screen.getByText(/does not score invoices again/)).toBeInTheDocument();
  });
});