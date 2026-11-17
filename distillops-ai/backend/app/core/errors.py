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
