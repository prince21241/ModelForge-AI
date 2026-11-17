import { render, screen, waitFor } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { afterEach, describe, expect, it, vi } from "vitest";
import App from "./App";
import { SAMPLE_INVOICE_TEXT } from "./sampleInvoice";

const PREDICTION = {
  model: "student",
  fields: {
    vendor: "Northwind Desk",
    invoice_number: "INV-100",
    date: "2024-01-15",
    total: "42.00",
    currency: "USD",
  },
  latency_ms: 812.4,
  requires_review: true,
  warnings: ["The student is not production-validated. Review every extraction before use."],
};

function jsonResponse(body: unknown, status = 200): Response {
  return new Response(JSON.stringify(body), {
    status,
    headers: { "Content-Type": "application/json" },
  });
}

function mockBackend(extract?: () => Promise<Response> | Response) {
  vi.stubGlobal(
    "fetch",
    vi.fn(async (input: RequestInfo | URL, init?: RequestInit) => {
      const url = String(input);
      if (url.endsWith("/api/health")) {
        return jsonResponse({
          status: "ok",
          service: "distillops",
          student_checkpoint: "models/student/flan-t5-small-dated12/epoch-30",
          student_loaded: false,
        });
      }
      if (url.endsWith("/api/models")) {
        return jsonResponse({
          models: [
            {
              name: "student",
              identifier: "google/flan-t5-small",
              checkpoint: "models/student/flan-t5-small-dated12/epoch-30",
              available: true,
              loaded: false,
            },
          ],
        });
      }
      if (url.endsWith("/api/extract")) {
        if (!extract) {
          throw new Error("extract was not expected");
        }
        return extract();
      }
      throw new Error(`unexpected url ${url} ${init?.method ?? "GET"}`);
    }),
  );
}

afterEach(() => {
  vi.unstubAllGlobals();
});

describe("DistillOps workspace", () => {
  it("renders the dashboard and backend status", async () => {
    mockBackend();
    render(<App />);
    expect(screen.getByRole("heading", { name: "Invoice workspace" })).toBeInTheDocument();
    expect(screen.getByRole("button", { name: /Extract Invoice/ })).toBeInTheDocument();
    expect(await screen.findByText("Backend connected")).toBeInTheDocument();
    expect(screen.getByText(/google\/flan-t5-small/)).toBeInTheDocument();
  });

  it("blocks an empty invoice before calling extract", async () => {
    mockBackend();
    const user = userEvent.setup();
    render(<App />);
    await screen.findByText("Backend connected");
    await user.click(screen.getAllByRole("button", { name: "Extract Fields" })[0]);
    expect(await screen.findByRole("alert")).toHaveTextContent("Enter invoice text before extracting.");
    const fetchMock = vi.mocked(fetch);
    expect(fetchMock.mock.calls.some((call) => String(call[0]).endsWith("/api/extract"))).toBe(false);
  });

  it("fills the input from the sample and shows API fields", async () => {
    mockBackend(() => jsonResponse(PREDICTION));
    const user = userEvent.setup();
    render(<App />);
    await user.click(screen.getAllByRole("button", { name: "Sample invoice" })[0]);
    expect(screen.getAllByLabelText("Invoice text")[0]).toHaveValue(SAMPLE_INVOICE_TEXT);
    expect(screen.getAllByText("Waiting for a model prediction.").length).toBeGreaterThan(0);
    await user.click(screen.getAllByRole("button", { name: "Extract Fields" })[0]);
    expect(await screen.findByText("Needs review")).toBeInTheDocument();
    expect(screen.getByLabelText("Vendor")).toHaveValue("Northwind Desk");
    expect(screen.getByText("812 ms")).toBeInTheDocument();
    expect(screen.getByText(/not production-validated/)).toBeInTheDocument();
    const extractCall = vi.mocked(fetch).mock.calls.find((call) => String(call[0]).endsWith("/api/extract"));
    expect(JSON.parse(String(extractCall?.[1]?.body))).toEqual({
      document_text: SAMPLE_INVOICE_TEXT,
      model: "student",
    });
  });

  it("shows a loading state while the request is open", async () => {
    let resolveExtract: (response: Response) => void = () => undefined;
    mockBackend(
      () =>
        new Promise((resolve) => {
          resolveExtract = resolve;
        }),
    );
    const user = userEvent.setup();
    render(<App />);
    await user.click(screen.getAllByRole("button", { name: "Sample invoice" })[0]);
    await user.click(screen.getAllByRole("button", { name: "Extract Fields" })[0]);
    expect(await screen.findByRole("status")).toHaveTextContent("Running the student");
    expect(screen.getByRole("button", { name: "Extracting fields" })).toBeDisabled();
    resolveExtract(jsonResponse(PREDICTION));
    expect(await screen.findByText("Needs review")).toBeInTheDocument();
  });

  it("shows the backend error detail", async () => {
    mockBackend(() => jsonResponse({ detail: "document_text exceeds the input limit and was not truncated." }, 413));
    const user = userEvent.setup();
    render(<App />);
    await user.type(screen.getAllByLabelText("Invoice text")[0], "INVOICE");
    await user.click(screen.getAllByRole("button", { name: "Extract Fields" })[0]);
    expect(await screen.findByRole("alert")).toHaveTextContent("was not truncated");
  });

  it("shows backend unavailable when the network fails", async () => {
    vi.stubGlobal(
      "fetch",
      vi.fn(async (input: RequestInfo | URL) => {
        const url = String(input);
        if (url.endsWith("/api/health") || url.endsWith("/api/models")) {
          throw new Error("offline");
        }
        throw new Error("offline");
      }),
    );
    const user = userEvent.setup();
    render(<App />);
    expect(await screen.findByText("Backend unavailable")).toBeInTheDocument();
    await user.type(screen.getAllByLabelText("Invoice text")[0], "INVOICE");
    await user.click(screen.getAllByRole("button", { name: "Extract Fields" })[0]);
    await waitFor(() => expect(screen.getByRole("alert")).toHaveTextContent("Backend unavailable."));
  });

  it("keeps the model prediction when a field is edited", async () => {
    mockBackend(() => jsonResponse(PREDICTION));
    const user = userEvent.setup();
    render(<App />);
    await user.click(screen.getAllByRole("button", { name: "Sample invoice" })[0]);
    await user.click(screen.getAllByRole("button", { name: "Extract Fields" })[0]);
    const vendor = await screen.findByLabelText("Vendor");
    await user.clear(vendor);
    await user.type(vendor, "Edited Desk");
    expect(vendor).toHaveValue("Edited Desk");
    expect(screen.getByText("Edited. Model predicted “Northwind Desk”.")).toBeInTheDocument();
    expect(screen.getAllByText("Model prediction").length).toBeGreaterThan(0);
  });
});
