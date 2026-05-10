"""Health and readiness endpoints."""
from __future__ import annotations

from fastapi import APIRouter, Depends, Request

from app.dependencies import get_repository

router = APIRouter(tags=["Infrastructure"])


@router.get("/health", summary="Liveness probe")
async def health() -> dict:
    """Returns 200 as long as the process is alive."""
    return {"status": "ok"}


@router.get("/ready", summary="Readiness probe")
async def ready(
    request: Request,
    repository=Depends(get_repository),
) -> dict:
    """Returns 200 when the model and customer database are fully loaded."""
    model_loaded = hasattr(request.app.state, "orchestrator")
    db_size = len(repository.list_customer_ids())
    return {
        "status": "ready" if model_loaded and db_size > 0 else "not_ready",
        "model_loaded": model_loaded,
        "customer_records": db_size,
    }
