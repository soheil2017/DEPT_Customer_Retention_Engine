"""FastAPI application factory.

The lifespan context manager initialises all heavy dependencies once on
startup and tears them down cleanly on shutdown — no module-level singletons,
no cold-start latency per request.
"""
from __future__ import annotations

import logging
import logging.config
from contextlib import asynccontextmanager
from typing import AsyncGenerator

from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse

from app.api.v1 import health, retention

_FRONTEND_PATH = Path(__file__).resolve().parent.parent / "frontend" / "index.html"
from app.core.config import settings
from app.services.churn_predictor import SklearnChurnPredictor
from app.services.customer_repository import CSVCustomerRepository
from app.services.guardrails import VodafoneGuardrails
from app.services.llm_service import DemoLLMService, OpenAILLMService
from app.services.retention_orchestrator import RetentionOrchestrator
from app.services.tracer import LangfuseTracer, NoOpTracer

#  Logging setup 
logging.basicConfig(
    level=getattr(logging, settings.log_level.upper(), logging.INFO),
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
)
logger = logging.getLogger(__name__)


#  Lifespan: startup / shutdown 
@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    logger.info("=== Retention Engine starting (env=%s) ===", settings.app_env)

    # Initialise all services — fail fast if artifacts are missing
    repository = CSVCustomerRepository(settings.customer_db_path)
    predictor  = SklearnChurnPredictor(settings.model_path)
    guardrails = VodafoneGuardrails()

    # Auto-select LLM service — demo mode when no API key is configured
    if settings.openai_api_key:
        llm_service = OpenAILLMService(settings)
        demo_mode = False
        logger.info("OpenAI LLM service enabled (model=%s).", settings.llm_model)
    else:
        llm_service = DemoLLMService()
        demo_mode = True
        logger.warning(
            "OPENAI_API_KEY not set — running in DEMO MODE. "
            "Emails are template-generated; all other pipeline stages are fully live."
        )

    # Langfuse tracer — enabled only when both keys are present
    if settings.langfuse_enabled:
        tracer = LangfuseTracer(
            public_key=settings.langfuse_public_key,
            secret_key=settings.langfuse_secret_key,
            host=settings.langfuse_host,
        )
        logger.info(
            "Langfuse observability enabled (host=%s, public_key=%s...).",
            settings.langfuse_host,
            settings.langfuse_public_key[:12],
        )
    else:
        tracer = NoOpTracer()
        logger.warning(
            "Langfuse keys not set — using NoOpTracer. "
            "Set LANGFUSE_PUBLIC_KEY and LANGFUSE_SECRET_KEY to enable tracing."
        )

    app.state.tracer     = tracer
    app.state.repository = repository
    app.state.orchestrator = RetentionOrchestrator(
        repository=repository,
        predictor=predictor,
        llm_service=llm_service,
        guardrails=guardrails,
        settings=settings,
        tracer=tracer,
        demo_mode=demo_mode,
    )

    logger.info("All services initialised. API ready.")
    yield

    logger.info("=== Retention Engine shutting down ===")


#  App factory 
def create_app() -> FastAPI:
    app = FastAPI(
        title="Vodafone Proactive Retention Engine",
        description="""
A modular AI service that operationalises the Vodafone churn prediction model
into a fully autonomous retention workflow.

## How it works

1. **Fetch** — retrieve the customer's profile from the database.
2. **Score** — run the trained sklearn model to get a churn probability.
3. **Route**:
   - `churn_probability < 0.5` → return `status: healthy` immediately (no LLM cost).
   - `churn_probability ≥ 0.5` → generate a personalised retention email via OpenAI,
     validated through a 7-check brand guardrail before returning.

## Demo mode

No OpenAI API key? The service runs in **demo mode** automatically —
template-based emails are generated and every other pipeline stage
(ML inference, guardrails, Langfuse tracing) runs fully.

Alternatively, pass your own key per-request via the `X-OpenAI-Key` header.

## Example customer IDs to try

| Customer ID   | Expected result |
|---------------|-----------------|
| `1053-YWGNE`  | High-risk → retention email generated |
| `3170-YWWJE`  | Low-risk → healthy, no email |

## Source code

[github.com/soheil2017/Customer_churn_DEPT](https://github.com/soheil2017/Customer_churn_DEPT)
        """,
        version="1.0.0",
        contact={
            "name": "Vodafone Retention Engine — DEPT® PoC",
        },
        license_info={
            "name": "Private — Case Assignment",
        },
        lifespan=lifespan,
        docs_url="/docs",
        redoc_url="/redoc",
    )

    # CORS — restrict in production
    origins = ["*"] if not settings.is_production else []
    app.add_middleware(
        CORSMiddleware,
        allow_origins=origins,
        allow_methods=["GET"],
        allow_headers=["*"],
    )

    # Frontend — served directly by FastAPI so a single Vercel function handles everything
    @app.get("/", response_class=HTMLResponse, include_in_schema=False)
    async def frontend():
        return _FRONTEND_PATH.read_text()

    # Routers
    app.include_router(health.router)
    app.include_router(retention.router, prefix="/api/v1")

    return app


app = create_app()
