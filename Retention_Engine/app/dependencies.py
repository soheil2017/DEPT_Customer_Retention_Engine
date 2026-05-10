"""FastAPI dependency providers.

All heavy objects (model, CSV index, LLM client, Langfuse tracer) are
initialised once in the lifespan hook and stored on app.state.  The
Depends() functions here pull them out for injection into route handlers.
"""
from __future__ import annotations

from fastapi import Request

from app.services.churn_predictor import SklearnChurnPredictor
from app.services.customer_repository import CSVCustomerRepository
from app.services.guardrails import VodafoneGuardrails
from app.services.llm_service import OpenAILLMService
from app.services.retention_orchestrator import RetentionOrchestrator
from app.services.tracer import TracerABC


def get_orchestrator(request: Request) -> RetentionOrchestrator:
    """Inject the pre-built RetentionOrchestrator from application state."""
    return request.app.state.orchestrator


def get_repository(request: Request) -> CSVCustomerRepository:
    """Inject the customer repository (useful for utility endpoints)."""
    return request.app.state.repository


def get_tracer(request: Request) -> TracerABC:
    """Inject the active tracer (LangfuseTracer or NoOpTracer)."""
    return request.app.state.tracer
