export const MAX_UPLOAD_BYTES = 10 * 1024 * 1024;
export const MAX_PDF_PAGES = 10;

const EXTENSIONS = [".pdf", ".png", ".jpg", ".jpeg"];

export function fileExtension(name: string): string {
  const index = name.lastIndexOf(".");
  return index >= 0 ? name.slice(index).toLowerCase() : "";
}

export function validateUpload(file: File): string | null {
  if (!EXTENSIONS.includes(fileExtension(file.name))) {
    return "Upload a PDF, PNG, JPG, or JPEG.";
  }
  if (file.size === 0) {
    return "The selected file is empty.";
  }
  if (file.size > MAX_UPLOAD_BYTES) {
    return "The file exceeds the 10 MB limit.";
  }
  return null;
}

export function formatFileSize(bytes: number): string {
  if (bytes < 1024) {
    return `${bytes} B`;
  }
  if (bytes < 1024 * 1024) {
    return `${(bytes / 1024).toFixed(1)} KB`;
  }
  return `${(bytes / (1024 * 1024)).toFixed(1)} MB`;
}