"""Stable error type shared by every service. `code` values are part of the contract."""

# --- Stable error codes (do not rename; add new ones only) -------------------
INVALID_REQUEST = "invalid_request"            # 400 malformed / missing fields
UNKNOWN_DEMO_TOOL = "unknown_demo_tool"        # 400 selector not in the demo set
NOT_FOUND = "not_found"                        # 404 unknown endpoint
METHOD_NOT_ALLOWED = "method_not_allowed"      # 405
PAYLOAD_TOO_LARGE = "payload_too_large"        # 413
UNSUPPORTED_EXPRESSION = "unsupported_expression"  # 422 calculator fixture miss
QUOTE_NOT_FOUND = "quote_not_found"            # 404 exact quote could not be located
DEPENDENCY_UNAVAILABLE = "dependency_unavailable"  # 503 downstream refused / DNS failed
DEPENDENCY_TIMEOUT = "dependency_timeout"      # 504 downstream too slow
BAD_GATEWAY = "bad_gateway"                    # 502 downstream sent garbage
STORAGE_ERROR = "storage_error"                # 500 SQLite failure
INTERNAL_ERROR = "internal_error"              # 500 unexpected


class ApiError(Exception):
    """Raise anywhere; the HTTP layer turns it into `{"error": {code, message}}`.

    `extra` is merged into the JSON body (e.g. the tool trace of a failed chat).
    """

    def __init__(self, status: int, code: str, message: str, extra: dict | None = None):
        super().__init__(message)
        self.status = status
        self.code = code
        self.message = message
        self.extra = extra or {}
