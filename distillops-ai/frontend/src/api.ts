import type {
  BenchmarkLibrary,
  CompareResponse,
  CompareStatus,
  DocumentExtractResponse,
  ExtractResponse,
  HealthResponse,
  ModelsResponse,
} from "./types";

export class ApiRequestError extends Error {
  readonly status: number;

  constructor(message: string, status: number) {
    super(message);
    this.name = "ApiRequestError";
    this.status = status;
  }
}

export function apiBaseUrl(): string {
  const configured = import.meta.env.VITE_API_BASE_URL;
  if (typeof configured === "string" && configured.trim()) {
    return configured.replace(/\/$/, "");
  }
  return "http://127.0.0.1:8000";
}

export async function fetchHealth(): Promise<HealthResponse> {
  return readJson<HealthResponse>(await request("/api/health"));
}

export async function fetchModels(): Promise<ModelsResponse> {
  return readJson<ModelsResponse>(await request("/api/models"));
}

export async function fetchBenchmarks(): Promise<BenchmarkLibrary> {
  return readJson<BenchmarkLibrary>(await request("/api/benchmarks"));
}

export async function extractInvoice(documentText: string): Promise<ExtractResponse> {
  const response = await request("/api/extract", {
    method: "POST",
    headers: { "Content-Type": "application/json", Accept: "application/json" },
    body: JSON.stringify({ document_text: documentText, model: "student" }),
  });
  return readJson<ExtractResponse>(response);
}

export async function fetchCompareStatus(): Promise<CompareStatus> {
  return readJson<CompareStatus>(await request("/api/compare/status"));
}

export async function compareInvoice(documentText: string): Promise<CompareResponse> {
  const response = await request("/api/compare", {
    method: "POST",
    headers: { "Content-Type": "application/json", Accept: "application/json" },
    body: JSON.stringify({ document_text: documentText, teacher_approved: true }),
  });
  return readJson<CompareResponse>(response);
}

export async function compareDocument(file: File): Promise<CompareResponse> {
  const body = new FormData();
  body.append("file", file);
  body.append("teacher_approved", "true");
  const response = await request("/api/documents/compare", {
    method: "POST",
    headers: { Accept: "application/json" },
    body,
  });
  return readJson<CompareResponse>(response);
}

export async function extractDocument(file: File): Promise<DocumentExtractResponse> {
  const body = new FormData();
  body.append("file", file);
  body.append("model", "student");
  const response = await request("/api/documents/extract", {
    method: "POST",
    headers: { Accept: "application/json" },
    body,
  });
  return readJson<DocumentExtractResponse>(response);
}

async function request(path: string, init?: RequestInit): Promise<Response> {
  let response: Response;
  try {
    response = await fetch(`${apiBaseUrl()}${path}`, init);
  } catch {
    throw new ApiRequestError("Backend unavailable.", 0);
  }
  if (!response.ok) {
    throw new ApiRequestError(await errorDetail(response), response.status);
  }
  return response;
}

async function errorDetail(response: Response): Promise<string> {
  try {
    const body = (await response.json()) as { detail?: unknown };
    if (typeof body.detail === "string" && body.detail.trim()) {
      return body.detail;
    }
  } catch {
    return "The extraction request failed.";
  }
  return "The extraction request failed.";
}

async function readJson<T>(response: Response): Promise<T> {
  return (await response.json()) as T;
}
