import { useRef, useState, type DragEvent } from "react";
import { FileUp, X } from "lucide-react";
import { formatFileSize, MAX_PDF_PAGES, validateUpload } from "../upload";
import { DocumentPreview } from "./DocumentPreview";

type UploadPanelProps = {
  file: File | null;
  onFile: (file: File | null, error?: string) => void;
};

export function UploadPanel({ file, onFile }: UploadPanelProps) {
  const inputRef = useRef<HTMLInputElement>(null);
  const [dragging, setDragging] = useState(false);

  function choose(next: File | undefined) {
    if (!next) {
      return;
    }
    const problem = validateUpload(next);
    if (problem) {
      onFile(null, problem);
      return;
    }
    onFile(next);
  }

  function onDrop(event: DragEvent<HTMLDivElement>) {
    event.preventDefault();
    setDragging(false);
    choose(event.dataTransfer.files?.[0]);
  }

  return (
    <div>
      <div
        role="region"
        aria-label="Upload invoice"
        onDragOver={(event) => {
          event.preventDefault();
          setDragging(true);
        }}
        onDragLeave={() => setDragging(false)}
        onDrop={onDrop}
        className={`rounded-xl border border-dashed px-4 py-6 text-center ${
          dragging ? "border-blue-600 bg-blue-50" : "border-slate-300 bg-slate-50"
        }`}
      >
        <FileUp className="mx-auto h-6 w-6 text-blue-700" aria-hidden="true" />
        <p className="mt-2 text-sm font-medium text-slate-800">Drop a PDF or image here, or browse.</p>
        <p className="mt-1 text-xs text-slate-500">
          PDF, PNG, JPG, or JPEG. Maximum 10 MB. PDFs are limited to {MAX_PDF_PAGES} pages.
        </p>
        <button
          type="button"
          onClick={() => inputRef.current?.click()}
          className="mt-3 rounded-lg border border-slate-300 bg-white px-3 py-2 text-sm font-medium text-slate-700 hover:bg-slate-50"
        >
          Browse files
        </button>
        <input
          ref={inputRef}
          aria-label="Invoice file"
          type="file"
          accept=".pdf,.png,.jpg,.jpeg,application/pdf,image/png,image/jpeg"
          className="sr-only"
          onChange={(event) => {
            choose(event.target.files?.[0]);
            event.target.value = "";
          }}
        />
      </div>
      <p className="mt-3 text-xs text-slate-500">
        OCR and the student can misread invoices. Review every field. There is no confidence score.
      </p>
      {file ? (
        <div className="mt-3 flex items-center justify-between gap-3 rounded-lg border border-slate-200 px-3 py-2 text-sm">
          <span>
            <span className="font-medium text-slate-800">{file.name}</span>
            <span className="ml-2 text-slate-500">{formatFileSize(file.size)}</span>
          </span>
          <button
            type="button"
            onClick={() => onFile(null)}
            className="inline-flex items-center gap-1 rounded-md px-2 py-1 text-slate-600 hover:bg-slate-100"
          >
            <X className="h-4 w-4" aria-hidden="true" />
            Remove file
          </button>
        </div>
      ) : null}
      <DocumentPreview file={file} />
    </div>
  );
}