import { render, screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { afterEach, describe, expect, it, vi } from "vitest";
import App from "./App";
import type { CompareResponse, CompareStatus } from "./types";

const DISABLED: CompareStatus = {
  feature_enabled: false,
  available: false,
  api_key_configured: false,
  teacher_model: "gpt-4o-mini",
  pricing_configured: false,
  student_pricing_configured: false,
  max_requests: 20,
  budget_usd: null,
  public_deployment: false,
  budget_scope: "process-local",
  detail: "Live teacher comparison is off. Opening this screen does not call the teacher.",
};

const ENABLED: CompareStatus = { ...DISABLED, feature_enabled: true, available: true, api_key_configured: true, detail: "Live teacher comparison is on." };

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
        field_metrics: [{ field: "vendor", correct: 5, scored: 5, accuracy: 1, missing: 0, hallucinated: 0, incorrect: 0 }],
        missing_field_count: 0,
        hallucinated_field_count: 0,
        incorrect_field_count: 0,
        latency_ms_avg: 3000,
        latency_ms_median: 1052,
        latency_ms_p95: 8892,
        warmup_excluded: true,
        input_tokens: 1,
        output_tokens: 1,
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
        model_id: "google/flan-t5-small",
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
        field_metrics: [{ field: "vendor", correct: 4, scored: 5, accuracy: 0.8, missing: 0, hallucinated: 0, incorrect: 1 }],
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
        rows: [],
      },
    },
  ],
};

const MATCH: CompareResponse = {
  teacher: {
    role: "teacher",
    model: "gpt-4o-mini-2024-07-18",
    succeeded: true,
    fields: { vendor: "Cedar Pine", invoice_number: "INV-100", date: "2024-01-15", total: "42.00", currency: "USD" },
    latency_ms: 800,
    input_tokens: 120,
    output_tokens: 30,
    estimated_cost_usd: null,
    pricing_configured: false,
    error: null,
    warnings: [],
  },
  student: {
    role: "student",
    model: "google/flan-t5-small",
    succeeded: true,
    fields: { vendor: "Northwind Desk", invoice_number: "INV-100", date: "2024-01-15", total: "42.00", currency: "USD" },
    latency_ms: 90,
    input_tokens: null,
    output_tokens: null,
    estimated_cost_usd: null,
    pricing_configured: false,
    error: null,
    warnings: [],
  },
  fields: [
    { field: "vendor", label: "Vendor", teacher_value: "Cedar Pine", student_value: "Northwind Desk", status: "disagree" },
    { field: "invoice_number", label: "Invoice number", teacher_value: "INV-100", student_value: "INV-100", status: "agree" },
    { field: "date", label: "Date", teacher_value: "2024-01-15", student_value: "2024-01-15", status: "agree" },
    { field: "total", label: "Total", teacher_value: "42.00", student_value: "42.00", status: "agree" },
    { field: "currency", label: "Currency", teacher_value: "USD", student_value: "USD", status: "agree" },
  ],
  agreement_count: 4,
  fields_compared: 5,
  document_latency_ms: 0,
  extracted_text: "INVOICE",
  filename: null,
  content_type: null,
  page_count: null,
  extraction_method: null,
  partial: false,
  warnings: ["Field differences are model disagreement."],
};

const PARTIAL: CompareResponse = {
  ...MATCH,
  partial: true,
  agreement_count: null,
  fields_compared: 0,
  teacher: { ...MATCH.teacher, succeeded: false, fields: null, error: "The teacher request failed.", input_tokens: null, output_tokens: null },
  fields: MATCH.fields.map((row) => ({ ...row, teacher_value: null, student_value: null, status: "unavailable" })),
};

function jsonResponse(body: unknown, status = 200): Response {
  return new Response(JSON.stringify(body), { status, headers: { "Content-Type": "application/json" } });
}

afterEach(() => {
  vi.unstubAllGlobals();
});

function mockBackend(options?: { status?: CompareStatus; compare?: () => Promise<Response> | Response }) {
  vi.stubGlobal(
    "fetch",
    vi.fn(async (input: RequestInfo | URL, init?: RequestInit) => {
      const url = String(input);
      if (url.endsWith("/api/health")) {
        return jsonResponse({ status: "ok", service: "distillops", student_checkpoint: "epoch-30", student_loaded: false });
      }
      if (url.endsWith("/api/models")) {
        return jsonResponse({ models: [{ name: "student", identifier: "google/flan-t5-small", checkpoint: "epoch-30", available: true, loaded: false }] });
      }
      if (url.endsWith("/api/benchmarks")) {
        return jsonResponse(LIBRARY);
      }
      if (url.endsWith("/api/compare/status")) {
        return jsonResponse(options?.status ?? DISABLED);
      }
      if (url.endsWith("/api/compare")) {
        if (!options?.compare) {
          throw new Error("compare was not expected");
        }
        return options.compare();
      }
      throw new Error(`unexpected url ${url} ${init?.method ?? "GET"}`);
    }),
  );
}

async function openLive() {
  const user = userEvent.setup();
  render(<App />);
  await user.click(screen.getByRole("button", { name: "Model Comparison" }));
  await screen.findByText(/does not call the teacher and does not rescore/);
  await user.click(screen.getByRole("tab", { name: "Live comparison" }));
  return user;
}

describe("live teacher comparison", () => {
  it("keeps the historical benchmark and requires consent before a teacher call", async () => {
    mockBackend({ status: ENABLED, compare: () => jsonResponse(MATCH) });
    const user = await openLive();
    expect(screen.getByText("Live teacher comparison is on.")).toBeInTheDocument();
    await user.click(screen.getByRole("button", { name: "Sample invoice" }));
    await user.click(screen.getByRole("button", { name: "Compare models" }));
    expect(screen.getByRole("dialog")).toHaveTextContent("leaves this computer");
    expect(screen.getByRole("dialog")).toHaveTextContent("may incur charges");
    await user.click(screen.getByRole("button", { name: "Cancel" }));
    expect(screen.queryByRole("dialog")).not.toBeInTheDocument();
    expect(vi.mocked(fetch).mock.calls.some((call) => String(call[0]).endsWith("/api/compare") && call[1]?.method === "POST")).toBe(false);
    await user.click(screen.getByRole("button", { name: "Compare models" }));
    await user.click(screen.getByRole("button", { name: "Send to the teacher" }));
    expect(await screen.findByText("Model disagreement")).toBeInTheDocument();
    expect(screen.getByTestId("live-field-vendor")).toHaveTextContent("Cedar Pine");
    expect(screen.getByTestId("live-field-vendor")).toHaveTextContent("Northwind Desk");
    expect(screen.getAllByText("Fields agree").length).toBe(4);
    expect(screen.getByTestId("live-metrics")).toHaveTextContent("4 of 5");
    expect(screen.getByTestId("live-tokens")).toHaveTextContent("120 in / 30 out");
    expect(screen.getByTestId("live-tokens")).toHaveTextContent("Not available");
    expect(screen.getByTestId("live-teacher-cost")).toHaveTextContent("Not configured");
    expect(screen.getByTestId("live-student-cost")).toHaveTextContent("Not configured");
    expect(screen.getByTestId("live-teacher-latency")).toHaveTextContent("800 ms");
    expect(screen.getByTestId("live-student-latency")).toHaveTextContent("90 ms");
    const post = vi.mocked(fetch).mock.calls.find((call) => String(call[0]).endsWith("/api/compare") && call[1]?.method === "POST");
    expect(JSON.parse(String(post?.[1]?.body))).toMatchObject({ teacher_approved: true });
    await user.click(screen.getByRole("tab", { name: "Historical benchmark" }));
    expect(screen.getByText(/does not call the teacher and does not rescore/)).toBeInTheDocument();
    expect(screen.getByTestId("kpi-teacher-exact")).toHaveTextContent("100%");
    expect(screen.getByTestId("kpi-student-exact")).toHaveTextContent("60%");
  });

  it("shows an unavailable message and does not post when teacher comparison is off", async () => {
    mockBackend();
    await openLive();
    expect(screen.getByRole("status")).toHaveTextContent("Live teacher comparison is off");
    expect(screen.getByRole("button", { name: "Compare models" })).toBeDisabled();
    expect(vi.mocked(fetch).mock.calls.some((call) => String(call[0]).endsWith("/api/compare") && call[1]?.method === "POST")).toBe(false);
  });

  it("shows a loading state and a partial result", async () => {
    let resolveCompare: (response: Response) => void = () => undefined;
    mockBackend({
      status: ENABLED,
      compare: () =>
        new Promise((resolve) => {
          resolveCompare = resolve;
        }),
    });
    const user = await openLive();
    await user.type(screen.getByLabelText("Invoice text"), "INVOICE");
    await user.click(screen.getByRole("button", { name: "Compare models" }));
    await user.click(screen.getByRole("button", { name: "Send to the teacher" }));
    expect(await screen.findByText("Comparing models.")).toBeInTheDocument();
    resolveCompare(jsonResponse(PARTIAL));
    expect(await screen.findByText(/One model did not return fields/)).toBeInTheDocument();
    expect(screen.getByText("The teacher request failed.")).toBeInTheDocument();
    expect(screen.getAllByText("Not available").length).toBeGreaterThan(0);
    expect(screen.getByTestId("live-metrics")).toHaveTextContent("Not available");
  });

  it("shows the backend error and keeps the historical screen available", async () => {
    mockBackend({ status: ENABLED, compare: () => jsonResponse({ detail: "No teacher request was sent." }, 403) });
    const user = await openLive();
    await user.type(screen.getByLabelText("Invoice text"), "INVOICE");
    await user.click(screen.getByRole("button", { name: "Compare models" }));
    await user.click(screen.getByRole("button", { name: "Send to the teacher" }));
    expect(await screen.findByRole("alert")).toHaveTextContent("No teacher request was sent.");
    await user.click(screen.getByRole("tab", { name: "Historical benchmark" }));
    expect(screen.getByText("Locked test: 5 synthetic invoices")).toBeInTheDocument();
  });
});
