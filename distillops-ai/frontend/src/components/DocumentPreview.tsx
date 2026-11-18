import { useEffect, useState } from "react";
import { fileExtension } from "../upload";

type DocumentPreviewProps = {
  file: File | null;
};

export function DocumentPreview({ file }: DocumentPreviewProps) {
  const [url, setUrl] = useState<string | null>(null);

  useEffect(() => {
    if (!file) {
      setUrl(null);
      return;
    }
    const next = URL.createObjectURL(file);
    setUrl(next);
    return () => {
      URL.revokeObjectURL(next);
    };
  }, [file]);

  if (!file || !url) {
    return null;
  }
  if (fileExtension(file.name) === ".pdf") {
    return <iframe title="Invoice preview" src={url} className="mt-3 h-72 w-full rounded-xl border border-slate-200 bg-white" />;
  }
  return (
    <img
      alt="Uploaded invoice preview"
      src={url}
      className="mt-3 max-h-72 w-full rounded-xl border border-slate-200 bg-white object-contain"
    />
  );
}