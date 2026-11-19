"""Client-facing failures. Messages stay free of stack traces and secrets."""


class ApiError(Exception):
    """An extraction request that should become an HTTP error."""

    def __init__(self, status_code: int, detail: str) -> None:
        self.status_code = status_code
        self.detail = detail
        super().__init__(detail)


class EmptyDocumentError(ApiError):
    def __init__(self) -> None:
        super().__init__(422, "document_text must not be blank.")


class OversizedDocumentError(ApiError):
    def __init__(self, detail: str = "document_text exceeds the input limit and was not truncated.") -> None:
        super().__init__(413, detail)


class UnsupportedModelError(ApiError):
    def __init__(self, model: str) -> None:
        super().__init__(400, f"Unsupported model {model!r}. This API serves the frozen student.")


class ModelLoadError(ApiError):
    def __init__(self) -> None:
        super().__init__(503, "The student model could not be loaded.")


class InferenceFailedError(ApiError):
    def __init__(self) -> None:
        super().__init__(500, "Student inference failed.")


class UnsupportedFileError(ApiError):
    def __init__(self) -> None:
        super().__init__(415, "Unsupported file type. Upload a PDF, PNG, JPG, or JPEG.")


class SignatureMismatchError(ApiError):
    def __init__(self) -> None:
        super().__init__(415, "The file signature does not match its type.")


class EmptyUploadError(ApiError):
    def __init__(self) -> None:
        super().__init__(422, "The uploaded file is empty.")


class OversizedUploadError(ApiError):
    def __init__(self) -> None:
        super().__init__(413, "The upload exceeds the size limit and was not stored.")


class PageLimitError(ApiError):
    def __init__(self) -> None:
        super().__init__(413, "The PDF exceeds the page limit and was not processed.")


class EncryptedPdfError(ApiError):
    def __init__(self) -> None:
        super().__init__(422, "This PDF is encrypted and was not opened.")


class CorruptDocumentError(ApiError):
    def __init__(self) -> None:
        super().__init__(422, "The file is corrupt or is not a readable PDF or image.")


class EmptyExtractedTextError(ApiError):
    def __init__(self) -> None:
        super().__init__(422, "No readable text was found in the document.")


class OcrUnavailableError(ApiError):
    def __init__(self) -> None:
        super().__init__(
            503,
            "Tesseract OCR is not available. Install Tesseract and set TESSERACT_CMD if needed.",
        )


class OcrTimeoutError(ApiError):
    def __init__(self) -> None:
        super().__init__(408, "OCR timed out before the document was read.")


class TooManyPixelsError(ApiError):
    def __init__(self) -> None:
        super().__init__(413, "The image is too large to process safely.")


class TeacherApprovalError(ApiError):
    def __init__(self) -> None:
        super().__init__(
            403,
            "Confirm that this invoice may be sent to the teacher before comparing. No teacher request was sent.",
        )


class TeacherDisabledError(ApiError):
    def __init__(self, detail: str) -> None:
        super().__init__(403, detail)


class TeacherKeyMissingError(ApiError):
    def __init__(self, detail: str = "OPENAI_API_KEY is not set. No teacher request was sent.") -> None:
        super().__init__(503, detail)


class TeacherBudgetError(ApiError):
    def __init__(self, detail: str) -> None:
        super().__init__(429, detail)
