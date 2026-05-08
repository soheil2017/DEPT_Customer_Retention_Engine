"""Domain-specific exception hierarchy.

Keeping exceptions in one place means the rest of the codebase raises
well-typed errors and the API layer maps them to HTTP status codes cleanly.
"""


class RetentionEngineError(Exception):
    """Base for all domain errors."""


class CustomerNotFoundError(RetentionEngineError):
    def __init__(self, customer_id: str) -> None:
        super().__init__(f"Customer '{customer_id}' not found in database.")
        self.customer_id = customer_id


class ModelNotLoadedError(RetentionEngineError):
    """Raised when the churn model artifact cannot be loaded."""


class LLMError(RetentionEngineError):
    """Raised when the LLM call fails or returns an unparseable response."""


class GuardrailViolationError(RetentionEngineError):
    """Raised when generated content fails brand / safety checks."""

    def __init__(self, violations: list[str]) -> None:
        self.violations = violations
        super().__init__(f"Guardrail violations: {'; '.join(violations)}")
