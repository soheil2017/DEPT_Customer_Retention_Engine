"""Abstract interfaces (Dependency Inversion Principle).

Concrete implementations depend on these abstractions, not on each other.
This makes every service trivially swappable — e.g. replace the CSV
repository with a Postgres one, or swap Anthropic for OpenAI — without
touching the orchestrator or the API layer.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any


@dataclass
class LLMResult:
    """Typed return value from any LLM call.

    Carries the generated text together with token-level usage and latency
    so the tracer can log accurate cost and performance metrics without
    the LLM service needing to know anything about Langfuse.
    """
    text: str
    prompt_tokens: int = 0
    completion_tokens: int = 0
    latency_ms: float = 0.0


class CustomerRepositoryABC(ABC):
    """Defines the contract for fetching customer records."""

    @abstractmethod
    def get_customer(self, customer_id: str) -> dict[str, Any]:
        """Return a dict of feature values keyed by column name.

        Raises:
            CustomerNotFoundError: when the customer_id is absent.
        """

    @abstractmethod
    def list_customer_ids(self) -> list[str]:
        """Return all known customer IDs (useful for health checks / testing)."""


class ChurnPredictorABC(ABC):
    """Defines the contract for churn probability estimation."""

    @abstractmethod
    def predict(self, customer_features: dict[str, Any]) -> float:
        """Return churn probability in [0, 1]."""


class LLMServiceABC(ABC):
    """Defines the contract for LLM-backed text generation."""

    @abstractmethod
    async def generate(self, system_prompt: str, user_prompt: str) -> LLMResult:
        """Return an LLMResult with the generated text, token counts, and latency."""


class GuardrailsABC(ABC):
    """Defines the contract for output safety and brand-compliance checks."""

    @abstractmethod
    def validate(self, raw_llm_output: str, customer_name: str) -> dict[str, str]:
        """Parse and validate LLM output.

        Returns:
            A dict with the validated email fields.

        Raises:
            GuardrailViolationError: when the content fails any check.
        """
