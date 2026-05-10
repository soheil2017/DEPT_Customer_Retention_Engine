"""Retention workflow orchestrator.

Coordinates all services and owns the Langfuse trace lifecycle for every
request.  Each step is instrumented as a span or generation — if Langfuse
is not configured a NoOpTracer is injected and no overhead is incurred.

"""
from __future__ import annotations

import asyncio
import logging
from typing import Any

from app.core.config import Settings
from app.core.exceptions import GuardrailViolationError
from app.schemas.retention import (
    CustomerProfile,
    RetentionEmail,
    RetentionResponse,
    RiskLevel,
)
from app.services.customer_repository import CSVCustomerRepository
from app.services.interfaces import (
    ChurnPredictorABC,
    CustomerRepositoryABC,
    GuardrailsABC,
    LLMServiceABC,
)
from app.services.llm_service import OpenAILLMService
from app.services.tracer import NoOpTracer, TracerABC

logger = logging.getLogger(__name__)


class RetentionOrchestrator:
    """Coordinates all services to produce a retention decision."""

    def __init__(
        self,
        repository: CustomerRepositoryABC,
        predictor: ChurnPredictorABC,
        llm_service: LLMServiceABC,
        guardrails: GuardrailsABC,
        settings: Settings,
        tracer: TracerABC | None = None,
        demo_mode: bool = False,
    ) -> None:
        self._repository = repository
        self._predictor  = predictor
        self._llm        = llm_service
        self._guardrails = guardrails
        self._threshold  = settings.churn_threshold
        self._tracer     = tracer or NoOpTracer()
        self._demo_mode  = demo_mode

    async def process(self, customer_id: str, api_key_override: str | None = None) -> RetentionResponse:
        """Execute the full retention workflow for a given customer.

        Args:
            customer_id: The customer to assess.
            api_key_override: If provided, a one-off OpenAILLMService is created
                              with this key for the duration of this request only.
                              When absent, the server-configured LLM (real or demo)
                              is used unchanged.
        """
        # Resolve which LLM service to use for this request
        if api_key_override:
            llm = OpenAILLMService.from_api_key(api_key_override)
            demo_mode_effective = False
        else:
            llm = self._llm
            demo_mode_effective = self._demo_mode

        #  Open Langfuse trace 
        trace = self._tracer.start_trace(
            customer_id=customer_id,
            metadata={"threshold": self._threshold},
        )

        try:
            #  Step 1: Data retrieval 
            record = await asyncio.to_thread(self._repository.get_customer, customer_id)
            self._tracer.log_data_retrieval(trace, record)

            #  Step 2: Churn inference 
            churn_prob = await asyncio.to_thread(self._predictor.predict, record)
            risk_level = RiskLevel.HIGH if churn_prob >= self._threshold else RiskLevel.LOW
            self._tracer.log_prediction(trace, churn_prob, risk_level.value)

            logger.info(
                "Customer %s | prob=%.4f | risk=%s", customer_id, churn_prob, risk_level.value
            )

            profile = self._build_profile(customer_id, record)

            #  Step 3: Strategic routing 
            if risk_level == RiskLevel.LOW:
                response = RetentionResponse(
                    customer_id=customer_id,
                    churn_probability=churn_prob,
                    risk_level=risk_level,
                    status="healthy",
                    customer_profile=profile,
                    retention_email=None,
                    demo_mode=demo_mode_effective,
                )
                self._tracer.end_trace(trace, {"status": "healthy", "churn_probability": churn_prob})
                return response

            #  Step 4: Personalised email generation 
            email = await self._generate_email(customer_id, record, churn_prob, profile, trace, llm)

            response = RetentionResponse(
                customer_id=customer_id,
                churn_probability=churn_prob,
                risk_level=risk_level,
                status="at_risk",
                customer_profile=profile,
                retention_email=email,
                demo_mode=demo_mode_effective,
            )
            self._tracer.end_trace(trace, {"status": "at_risk", "churn_probability": churn_prob})
            return response

        except Exception:
            # Ensure the trace is always closed, even on unexpected errors
            self._tracer.end_trace(trace, {"status": "error"})
            raise

    # ── Private helpers ───────────────────────────────────────────────────────

    async def _generate_email(
        self,
        customer_id: str,
        record: dict[str, Any],
        churn_prob: float,
        profile: CustomerProfile,
        trace: Any,
        llm: LLMServiceABC | None = None,
    ) -> RetentionEmail:
        llm = llm or self._llm
        system_prompt = OpenAILLMService.system_prompt()
        user_prompt   = self._build_user_prompt(customer_id, record, churn_prob, profile)

        # LLM call — result carries text + token counts + latency for the tracer
        result = await llm.generate(system_prompt, user_prompt)
        self._tracer.log_generation(trace, llm._model if hasattr(llm, "_model") else "unknown",
                                    system_prompt, user_prompt, result)

        # Guardrail validation — outcome logged as a Langfuse score
        try:
            validated_fields = self._guardrails.validate(result.text, customer_id)
            self._tracer.log_guardrail(trace, passed=True, violations=[])
        except GuardrailViolationError as exc:
            self._tracer.log_guardrail(trace, passed=False, violations=exc.violations)
            raise

        return RetentionEmail(**validated_fields)

    @staticmethod
    def _build_user_prompt(
        customer_id: str,
        record: dict[str, Any],
        churn_prob: float,
        profile: CustomerProfile,
    ) -> str:
        services_text = (
            "\n".join(f"  - {s}" for s in profile.active_services)
            if profile.active_services
            else "  - Standard plan (no add-ons)"
        )
        return f"""Write a personalised retention email for the following Vodafone customer.

CUSTOMER DETAILS:
- Customer ID: {customer_id}
- Tenure: {profile.tenure_months} months with Vodafone
- Contract type: {profile.contract_type}
- Monthly spend: £{profile.monthly_charges:.2f}
- Churn risk score: {churn_prob:.0%} (HIGH)

ACTIVE SERVICES:
{services_text}

INSTRUCTIONS:
- Address the customer using their ID ({customer_id}) in the greeting since we do not
  have their first name.
- Highlight 2-3 benefits specifically relevant to their active services above.
- Propose a relevant loyalty gesture (e.g. free upgrade, priority support, bonus data)
  appropriate to their tenure and contract — do NOT invent specific percentage discounts.
- Follow the Vodafone tone of voice rules in your system prompt exactly.
- Return only the JSON object — no markdown, no preamble."""

    @staticmethod
    def _build_profile(customer_id: str, record: dict[str, Any]) -> CustomerProfile:
        active_services = CSVCustomerRepository.get_active_services(record)
        return CustomerProfile(
            customer_id=customer_id,
            tenure_months=int(record.get("tenure", 0)),
            contract_type=str(record.get("Contract", "Unknown")),
            monthly_charges=float(record.get("MonthlyCharges", 0.0)),
            active_services=active_services,
        )
