"""Pydantic schemas for API request / response contracts.

These are the *only* models the HTTP layer touches — internal service types
are kept separate so we can evolve them independently (Open/Closed Principle).
"""
from __future__ import annotations

from enum import Enum
from typing import Optional

from pydantic import BaseModel, Field


class RiskLevel(str, Enum):
    LOW = "low"
    HIGH = "high"


class RetentionEmail(BaseModel):
    """Structured email produced by the LLM, validated by the guardrails layer."""

    subject_line: str = Field(..., description="Friendly, personalised subject line")
    greeting: str = Field(..., description="Warm opening address to the customer")
    introduction: str = Field(..., description="Brief explanation of the email's purpose")
    body: str = Field(..., description="Main content: tailored offers / benefits")
    call_to_action: str = Field(..., description="Clear, compelling next step for the customer")
    closing: str = Field(..., description="Warm, appreciative sign-off")
    signature: str = Field(..., description="Team signature line")


class CustomerProfile(BaseModel):
    """Snapshot of the customer returned alongside the risk assessment."""

    customer_id:     str   = Field(..., description="Unique customer identifier")
    tenure_months:   int   = Field(..., description="Months the customer has been with Vodafone")
    contract_type:   str   = Field(..., description="Contract type: Month-to-month, One year, or Two year")
    monthly_charges: float = Field(..., description="Current monthly spend in GBP")
    active_services: list[str] = Field(..., description="List of active add-on services")


class RetentionResponse(BaseModel):
    """Top-level API response for the /retention/{customer_id} endpoint."""

    customer_id: str
    churn_probability: float = Field(..., ge=0.0, le=1.0)
    risk_level: RiskLevel
    status: str
    customer_profile: CustomerProfile
    retention_email: Optional[RetentionEmail] = Field(
        default=None,
        description="Only present when risk_level is 'high'",
    )
    demo_mode: bool = Field(
        default=False,
        description="True when no OpenAI key is configured — email is template-generated, not LLM-generated",
    )
