"""Observability / tracing layer — Langfuse integration.

Design goals:
  - Zero coupling to Langfuse outside this file. The rest of the codebase
    only imports `TracerABC`, `LangfuseTracer`, and `NoOpTracer`.
  - `NoOpTracer` is used automatically when Langfuse keys are absent, so
    the system works identically in test / local-dev environments.
  - `LangfuseTracer` wraps every SDK call in try/except so that an
    observability failure can NEVER cause a 500 in the retention workflow.
  - Every step of the retention workflow is instrumented:
      · data retrieval span
      · churn prediction span
      · LLM generation (model, prompt, completion, tokens, latency)
      · guardrail score  (1.0 = pass, 0.0 = fail + violation detail)
  - `flush()` is called at the end of every trace — critical for
    serverless environments (Vercel) where the process may not linger.
"""
from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from typing import Any

from app.services.interfaces import LLMResult

logger = logging.getLogger(__name__)


#  Abstract interface 

class TracerABC(ABC):
    """Observability contract — decouples the orchestrator from any specific
    tracing backend (Langfuse, OpenTelemetry, Datadog, …)."""

    @abstractmethod
    def start_trace(self, customer_id: str, metadata: dict[str, Any]) -> Any:
        """Open a new request-level trace. Returns a trace handle."""

    @abstractmethod
    def log_data_retrieval(self, trace: Any, record: dict[str, Any]) -> None:
        """Log the customer data-retrieval step as a span."""

    @abstractmethod
    def log_prediction(self, trace: Any, probability: float, risk_level: str) -> None:
        """Log the churn model inference step as a span."""

    @abstractmethod
    def log_generation(
        self,
        trace: Any,
        model: str,
        system_prompt: str,
        user_prompt: str,
        result: LLMResult,
    ) -> None:
        """Log the LLM generation as a Langfuse Generation node (captures
        prompt, completion, token usage, and latency)."""

    @abstractmethod
    def log_guardrail(
        self,
        trace: Any,
        passed: bool,
        violations: list[str],
    ) -> None:
        """Log guardrail outcome as a score (1.0 / 0.0) with violation detail."""

    @abstractmethod
    def end_trace(self, trace: Any, output: dict[str, Any]) -> None:
        """Close the trace and flush buffered data to the backend."""


#  No-op implementation (used when Langfuse is not configured) 

class _NoOpTrace:
    """Silent stand-in so the orchestrator never has to check if tracing is on."""
    id: str = "noop"
    def span(self, **_):       return self
    def generation(self, **_): return self
    def event(self, **_):      return self
    def score(self, **_):      return self
    def end(self, **_):        return self
    def update(self, **_):     return self


class NoOpTracer(TracerABC):
    """All methods silently do nothing — used in tests and when Langfuse is
    not configured. Guaranteed zero latency overhead."""

    def start_trace(self, customer_id, metadata):          return _NoOpTrace()
    def log_data_retrieval(self, trace, record):           pass
    def log_prediction(self, trace, probability, risk):    pass
    def log_generation(self, trace, model, sys, usr, res): pass
    def log_guardrail(self, trace, passed, violations):    pass
    def end_trace(self, trace, output):                    pass


#  Langfuse implementation 

class LangfuseTracer(TracerABC):
    """Instruments the retention workflow with Langfuse traces, spans,
    generations, scores, and events.

    Uses the Langfuse SDK v2 API (langfuse>=2.0,<3.0).
    Every SDK call is wrapped in try/except — an observability failure
    must never cause a 500 in the core retention workflow.
    """

    def __init__(self, public_key: str, secret_key: str, host: str) -> None:
        try:
            from langfuse import Langfuse  # lazy import — keeps startup fast if unused
            self._lf = Langfuse(
                public_key=public_key,
                secret_key=secret_key,
                host=host,
            )
            self._enabled = True
            logger.info("Langfuse tracer initialised (host=%s).", host)
        except Exception as exc:
            logger.warning("Langfuse init failed — tracing disabled: %s", exc)
            self._lf = None
            self._enabled = False

    #  TracerABC implementation 

    def start_trace(self, customer_id: str, metadata: dict[str, Any]) -> Any:
        if not self._enabled:
            return _NoOpTrace()
        try:
            return self._lf.trace(
                name="retention-workflow",
                user_id=customer_id,
                input={"customer_id": customer_id},
                metadata=metadata,
                tags=["retention-engine"],
            )
        except Exception as exc:
            logger.warning("Langfuse start_trace failed: %s", exc)
            return _NoOpTrace()

    def log_data_retrieval(self, trace: Any, record: dict[str, Any]) -> None:
        if not self._enabled:
            return
        try:
            span = trace.span(name="data-retrieval", input={"customer_id": record.get("customerID")})
            span.end(output={
                "tenure":          record.get("tenure"),
                "contract":        record.get("Contract"),
                "monthly_charges": record.get("MonthlyCharges"),
            })
        except Exception as exc:
            logger.warning("Langfuse log_data_retrieval failed: %s", exc)

    def log_prediction(self, trace: Any, probability: float, risk_level: str) -> None:
        if not self._enabled:
            return
        try:
            span = trace.span(name="churn-prediction")
            span.end(output={"probability": probability, "risk_level": risk_level})
        except Exception as exc:
            logger.warning("Langfuse log_prediction failed: %s", exc)

    def log_generation(
        self,
        trace: Any,
        model: str,
        system_prompt: str,
        user_prompt: str,
        result: LLMResult,
    ) -> None:
        if not self._enabled:
            return
        try:
            gen = trace.generation(
                name="email-generation",
                model=model,
                input=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user",   "content": user_prompt},
                ],
                model_parameters={"max_tokens": 1024, "response_format": "json_object"},
            )
            gen.end(
                output=result.text,
                usage={
                    "input":  result.prompt_tokens,
                    "output": result.completion_tokens,
                    "unit":   "TOKENS",
                },
                metadata={"latency_ms": round(result.latency_ms, 2)},
            )
        except Exception as exc:
            logger.warning("Langfuse log_generation failed: %s", exc)

    def log_guardrail(self, trace: Any, passed: bool, violations: list[str]) -> None:
        if not self._enabled:
            return
        try:
            trace.score(
                name="guardrail-compliance",
                value=1.0 if passed else 0.0,
                comment="All checks passed" if passed else "; ".join(violations),
            )
            if not passed:
                trace.event(
                    name="guardrail-violation",
                    input={"violations": violations},
                    metadata={"violation_count": len(violations)},
                )
                logger.warning("Guardrail violations logged to Langfuse: %s", violations)
        except Exception as exc:
            logger.warning("Langfuse log_guardrail failed: %s", exc)

    def end_trace(self, trace: Any, output: dict[str, Any]) -> None:
        if not self._enabled:
            return
        try:
            trace.update(output=output)
            # flush() blocks until all queued events are delivered —
            # essential in serverless (Vercel) where the process may freeze
            # immediately after the HTTP response is sent.
            self._lf.flush()
            logger.info("Langfuse trace flushed (output=%s).", output.get("status"))
        except Exception as exc:
            logger.warning("Langfuse end_trace failed: %s", exc)
