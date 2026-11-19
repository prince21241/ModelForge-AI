export type InvoiceFields = {
  vendor: string;
  invoice_number: string;
  date: string;
  total: string;
  currency: string;
};

export type ExtractResponse = {
  model: string;
  fields: InvoiceFields;
  latency_ms: number;
  requires_review: boolean;
  warnings: string[];
};

export type ExtractionMethod = "pdf_text" | "pdf_ocr" | "pdf_mixed" | "image_ocr";

export type DocumentExtractResponse = ExtractResponse & {
  filename: string;
  content_type: string;
  page_count: number;
  extraction_method: ExtractionMethod;
  extracted_text: string;
  document_latency_ms: number;
  model_latency_ms: number;
  total_latency_ms: number;
};

export type InputMode = "text" | "upload";

export type SavedFieldMetric = {
  field: string;
  correct: number;
  scored: number;
  accuracy: number | null;
  missing: number;
  hallucinated: number;
  incorrect: number;
};

export type SavedBenchmarkRow = {
  document_id: string;
  template_group: string;
  exact_match: boolean;
  incorrect_fields: string[];
  missing_fields: string[];
  hallucinated_fields: string[];
  latency_ms: number;
  prediction: Record<string, string>;
  expected: Record<string, string>;
};

export type SavedReportBody = {
  status: string;
  model_role: string;
  model_id: string;
  split: string;
  documents_in_split: number;
  documents_scored: number;
  unscored_document_ids: string[];
  json_valid_count: number;
  schema_valid_count: number;
  malformed_json_count: number;
  exact_match_count: number;
  exact_match_accuracy: number | null;
  end_to_end_exact_match_accuracy: number | null;
  field_metrics: SavedFieldMetric[];
  missing_field_count: number;
  hallucinated_field_count: number;
  incorrect_field_count: number;
  latency_ms_avg: number | null;
  latency_ms_median: number | null;
  latency_ms_p95: number | null;
  warmup_excluded: boolean;
  input_tokens: number;
  output_tokens: number;
  api_calls: number;
  api_failure_count: number;
  pricing_configured: boolean;
  estimated_cost_usd: number | null;
  cost_assumption: string;
  rows: SavedBenchmarkRow[];
};

export type SavedBenchmark = {
  available: boolean;
  filename: string;
  model_role: string;
  split: string;
  detail: string | null;
  report: SavedReportBody | null;
};

export type BenchmarkLibrary = {
  source: string;
  rescored: boolean;
  reports: SavedBenchmark[];
};

export const METHOD_LABELS: Record<ExtractionMethod, string> = {
  pdf_text: "Native PDF text",
  pdf_ocr: "PDF OCR",
  pdf_mixed: "PDF text and OCR",
  image_ocr: "Image OCR",
};

export function isDocumentResult(result: ExtractResponse | null): result is DocumentExtractResponse {
  return result !== null && "extraction_method" in result;
}

export type HealthResponse = {
  status: string;
  service: string;
  student_checkpoint: string;
  student_loaded: boolean;
};

export type ModelInfo = {
  name: string;
  identifier: string;
  checkpoint: string;
  available: boolean;
  loaded: boolean;
};

export type ModelsResponse = {
  models: ModelInfo[];
};

export type ViewId = "dashboard" | "extract" | "comparison" | "benchmarks";

export const FIELD_LABELS: { key: keyof InvoiceFields; label: string }[] = [
  { key: "vendor", label: "Vendor" },
  { key: "invoice_number", label: "Invoice number" },
  { key: "date", label: "Date" },
  { key: "total", label: "Total" },
  { key: "currency", label: "Currency" },
];

export const EMPTY_FIELDS: InvoiceFields = {
  vendor: "",
  invoice_number: "",
  date: "",
  total: "",
  currency: "",
};

export type CompareStatus = {
  feature_enabled: boolean;
  available: boolean;
  api_key_configured: boolean;
  teacher_model: string | null;
  pricing_configured: boolean;
  student_pricing_configured: boolean;
  max_requests: number;
  budget_usd: number | null;
  public_deployment: boolean;
  budget_scope: string;
  detail: string;
};

export type ModelOutcome = {
  role: "teacher" | "student";
  model: string;
  succeeded: boolean;
  fields: InvoiceFields | null;
  latency_ms: number | null;
  input_tokens: number | null;
  output_tokens: number | null;
  estimated_cost_usd: number | null;
  pricing_configured: boolean;
  error: string | null;
  warnings: string[];
};

export type FieldComparison = {
  field: string;
  label: string;
  teacher_value: string | null;
  student_value: string | null;
  status: "agree" | "disagree" | "unavailable";
};

export type CompareResponse = {
  teacher: ModelOutcome;
  student: ModelOutcome;
  fields: FieldComparison[];
  agreement_count: number | null;
  fields_compared: number;
  document_latency_ms: number;
  extracted_text: string;
  filename: string | null;
  content_type: string | null;
  page_count: number | null;
  extraction_method: string | null;
  partial: boolean;
  warnings: string[];
};
