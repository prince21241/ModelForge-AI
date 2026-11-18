import { fireEvent, render, screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { afterEach, describe, expect, it, vi } from "vitest";
import App from "./App";

const DOCUMENT = {
  model: "student",
  fields: {
    vendor: "Northwind Desk",
    invoice_number: "INV-100",
    date: "2024-01-15",
    total: "42.00",
    currency: "USD",
  },
  latency_ms: 812.4,
  model_latency_ms: 812.4,
  document_latency_ms: 20.2,
  total_latency_ms: 900.4,
  requires_review: true,
  warnings: [
    "The student is not production-validated. Review every extraction before use.",
    "This document was read with OCR. OCR can misread characters, and the text was not repaired.",
  ],
  filename: "invoice.pdf",
  content_type: "application/pdf",
  page_count: 2,
  extraction_method: "pdf_text",
  extracted_text: "Vendor: Northwind Desk",
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
    vi.fn(async (input: RequestInfo | URL) => {
      const url = String(input);
      if (url.endsWith("/api/health")) {
        return jsonResponse({
          status: "ok",
          service: "distillops",
          student_checkpoint: "models/student/flan-t5-small-dated12/epoch-30",
          student_loaded: true,
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
              loaded: true,
            },
          ],
        });
      }
      if (url.endsWith("/api/documents/extract")) {
        if (!extract) {
          throw new Error("upload was not expected");
        }
        return extract();
      }
      if (url.endsWith("/api/extract")) {
        throw new Error("paste extract was not expected");
      }
      throw new Error(`unexpected url ${url}`);
    }),
  );
}

afterEach(() => {
  vi.unstubAllGlobals();
});

async function openUpload() {
  const user = userEvent.setup();
  render(<App />);
  await user.click(screen.getByRole("tab", { name: "Upload invoice file" }));
  return user;
}

describe("invoice upload", () => {
  it("selects a file and shows its preview", async () => {
    mockBackend();
    const user = await openUpload();
    const file = new File(["%PDF-1.4"], "invoice.pdf", { type: "application/pdf" });
    await user.upload(screen.getByLabelText("Invoice file"), file);
    expect(screen.getByText("invoice.pdf")).toBeInTheDocument();
    expect(screen.getByText("8 B")).toBeInTheDocument();
    expect(screen.getByTitle("Invoice preview")).toBeInTheDocument();
    await user.click(screen.getByRole("button", { name: "Remove file" }));
    expect(screen.queryByTitle("Invoice preview")).not.toBeInTheDocument();
  });

  it("rejects an unsupported file before upload", async () => {
    mockBackend();
    await openUpload();
    const file = new File(["hello"], "notes.txt", { type: "text/plain" });
    fireEvent.change(screen.getByLabelText("Invoice file"), { target: { files: [file] } });
    expect(screen.getByRole("alert")).toHaveTextContent("Upload a PDF, PNG, JPG, or JPEG.");
    expect(vi.mocked(fetch).mock.calls.some((call) => String(call[0]).includes("/api/documents"))).toBe(false);
  });

  it("rejects a file over 10 MB", async () => {
    mockBackend();
    await openUpload();
    const file = new File(["%PDF"], "big.pdf", { type: "application/pdf" });
    Object.defineProperty(file, "size", { value: 10 * 1024 * 1024 + 1 });
    fireEvent.change(screen.getByLabelText("Invoice file"), { target: { files: [file] } });
    expect(screen.getByRole("alert")).toHaveTextContent("10 MB limit");
  });

  it("accepts a dropped image and previews it", async () => {
    mockBackend();
    await openUpload();
    const file = new File([Uint8Array.from([137, 80, 78, 71])], "invoice.png", { type: "image/png" });
    fireEvent.drop(screen.getByRole("region", { name: "Upload invoice" }), {
      dataTransfer: { files: [file] },
    });
    expect(screen.getByText("invoice.png")).toBeInTheDocument();
    expect(screen.getByAltText("Uploaded invoice preview")).toBeInTheDocument();
  });

  it("shows an upload error from the API", async () => {
    mockBackend(() => jsonResponse({ detail: "The file signature does not match its type." }, 415));
    const user = await openUpload();
    await user.upload(
      screen.getByLabelText("Invoice file"),
      new File(["%PDF"], "invoice.pdf", { type: "application/pdf" }),
    );
    await user.click(screen.getByRole("button", { name: "Extract Fields" }));
    expect(await screen.findByRole("alert")).toHaveTextContent("does not match its type");
  });

  it("shows the loading state while a file is processing", async () => {
    let resolveExtract: (response: Response) => void = () => undefined;
    mockBackend(
      () =>
        new Promise((resolve) => {
          resolveExtract = resolve;
        }),
    );
    const user = await openUpload();
    await user.upload(
      screen.getByLabelText("Invoice file"),
      new File(["%PDF"], "invoice.pdf", { type: "application/pdf" }),
    );
    await user.click(screen.getByRole("button", { name: "Extract Fields" }));
    expect(await screen.findByRole("status")).toHaveTextContent("Reading the document");
    expect(screen.getByRole("button", { name: "Extracting fields" })).toBeDisabled();
    resolveExtract(jsonResponse(DOCUMENT));
    expect(await screen.findByText("Native PDF text")).toBeInTheDocument();
  });

  it("shows the student fields and keeps edits separate from the prediction", async () => {
    mockBackend(() => jsonResponse(DOCUMENT));
    const user = await openUpload();
    const file = new File(["%PDF"], "invoice.pdf", { type: "application/pdf" });
    await user.upload(screen.getByLabelText("Invoice file"), file);
    await user.click(screen.getByRole("button", { name: "Extract Fields" }));
    expect(await screen.findByText("Needs review")).toBeInTheDocument();
    expect(screen.getByText("812 ms")).toBeInTheDocument();
    expect(screen.getByText("20 ms")).toBeInTheDocument();
    expect(screen.getByText("900 ms")).toBeInTheDocument();
    expect(screen.getByText(/read with OCR/)).toBeInTheDocument();
    expect(screen.getByText("Vendor: Northwind Desk")).toBeInTheDocument();
    const vendor = screen.getByLabelText("Vendor");
    await user.clear(vendor);
    await user.type(vendor, "Edited Desk");
    expect(screen.getByText("Edited. Model predicted “Northwind Desk”.")).toBeInTheDocument();
    const extractCall = vi.mocked(fetch).mock.calls.find((call) => String(call[0]).endsWith("/api/documents/extract"));
    expect(extractCall?.[1]?.body).toBeInstanceOf(FormData);
    const body = extractCall?.[1]?.body as FormData;
    expect(body.get("model")).toBe("student");
    expect(body.get("file")).toBeInstanceOf(File);
  });

  it("keeps pasted text when switching to upload and back", async () => {
    mockBackend();
    const user = userEvent.setup();
    render(<App />);
    await user.type(screen.getByLabelText("Invoice text"), "Vendor: Kept Desk");
    await user.click(screen.getByRole("tab", { name: "Upload invoice file" }));
    expect(screen.queryByLabelText("Invoice text")).not.toBeInTheDocument();
    expect(screen.getByRole("region", { name: "Upload invoice" })).toBeInTheDocument();
    await user.click(screen.getByRole("tab", { name: "Paste invoice text" }));
    expect(screen.getByLabelText("Invoice text")).toHaveValue("Vendor: Kept Desk");
  });
});