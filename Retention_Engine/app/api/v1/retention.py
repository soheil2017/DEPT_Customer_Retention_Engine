"""Retention Engine API endpoints (v1).

Routes are kept thin — they validate input/output shapes and translate
domain exceptions to HTTP status codes.  All business logic lives in the
RetentionOrchestrator.

Optional header: X-OpenAI-Key
  If provided, that key is used for this request (real LLM generation).
  If absent, the server-configured service is used (OpenAI or demo mode).
"""
from __future__ import annotations

import logging
from typing import Optional

from fastapi import APIRouter, Depends, Header, HTTPException, status

from app.core.exceptions import (
    CustomerNotFoundError,
    GuardrailViolationError,
    LLMError,
    ModelNotLoadedError,
)
from app.dependencies import get_orchestrator, get_repository
from app.schemas.retention import RetentionResponse
from app.services.retention_orchestrator import RetentionOrchestrator

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/retention", tags=["Retention Engine"])


@router.get(
    "/{customer_id}",
    response_model=RetentionResponse,
    summary="Assess churn risk and generate a retention offer",
    responses={
        200: {"description": "Risk assessment (+ retention email if high-risk)"},
        404: {"description": "Customer not found"},
        503: {"description": "Model or LLM unavailable"},
        422: {"description": "Guardrail violation — LLM output rejected"},
    },
)
async def get_retention_assessment(
    customer_id: str,
    orchestrator: RetentionOrchestrator = Depends(get_orchestrator),
    x_openai_key: Optional[str] = Header(
        default=None,
        description="Optional OpenAI API key. When supplied, real LLM generation is used "
                    "for this request regardless of server configuration.",
    ),
) -> RetentionResponse:
    """
    Execute the full retention workflow for a customer:

    1. **Data retrieval** — fetch customer attributes from the database.
    2. **Churn inference** — score with the trained ML model.
    3. **Strategic routing**:
       - `risk_level: low` → returns `status: healthy` immediately.
       - `risk_level: high` → triggers LLM email generation with brand guardrails.

    **Demo mode:** if no `OPENAI_API_KEY` is configured on the server and no
    `X-OpenAI-Key` header is provided, a template-based email is returned so
    the full pipeline can be evaluated without any API credentials.
    """
    try:
        return await orchestrator.process(customer_id, api_key_override=x_openai_key or None)

    except CustomerNotFoundError as exc:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(exc))

    except ModelNotLoadedError as exc:
        logger.error("Model unavailable: %s", exc)
        raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail=str(exc))

    except LLMError as exc:
        logger.error("LLM call failed: %s", exc)
        raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail=str(exc))

    except GuardrailViolationError as exc:
        logger.warning("Guardrail violation for customer %s: %s", customer_id, exc.violations)
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail={"message": "Generated content failed brand guardrails.", "violations": exc.violations},
        )


@router.get(
    "/",
    summary="List available customer IDs",
    tags=["Retention Engine"],
)
async def list_customers(
    repository=Depends(get_repository),
) -> dict:
    """Return the full list of customer IDs available in the database."""
    ids = repository.list_customer_ids()
    return {"total": len(ids), "customer_ids": ids}
