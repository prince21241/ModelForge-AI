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
