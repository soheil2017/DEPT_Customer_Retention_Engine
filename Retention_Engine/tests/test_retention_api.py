"""Integration tests for the retention API endpoints.

The LLM is mocked in all tests so the suite runs without an Anthropic key
or network access.  Real model + real CSV are used.
"""
from __future__ import annotations

import json
from unittest.mock import AsyncMock

import pytest

from app.services.interfaces import LLMResult


def test_health_endpoint(client):
    resp = client.get("/health")
    assert resp.status_code == 200
    assert resp.json()["status"] == "ok"


def test_ready_endpoint(client):
    resp = client.get("/ready")
    assert resp.status_code == 200
    data = resp.json()
    assert data["model_loaded"] is True
    assert data["customer_records"] > 0


def test_list_customers(client):
    resp = client.get("/api/v1/retention/")
    assert resp.status_code == 200
    data = resp.json()
    assert "customer_ids" in data
    assert data["total"] > 0


def test_low_risk_customer_returns_healthy(client, real_repository, real_predictor):
    # Find a customer with a low predicted probability
    low_risk_id = None
    for cid in real_repository.list_customer_ids():
        prob = real_predictor.predict(real_repository.get_customer(cid))
        if prob < 0.5:
            low_risk_id = cid
            break

    if low_risk_id is None:
        pytest.skip("No low-risk customer found in the test dataset.")

    resp = client.get(f"/api/v1/retention/{low_risk_id}")
    assert resp.status_code == 200
    data = resp.json()
    assert data["risk_level"] == "low"
    assert data["status"] == "healthy"
    assert data["retention_email"] is None


def test_high_risk_customer_returns_email(client, mock_llm_service, real_repository, real_predictor):
    # Find a customer with a high predicted probability
    high_risk_id = None
    for cid in real_repository.list_customer_ids():
        prob = real_predictor.predict(real_repository.get_customer(cid))
        if prob >= 0.5:
            high_risk_id = cid
            break

    if high_risk_id is None:
        pytest.skip("No high-risk customer found in the test dataset.")

    # Set mock to return a valid email for this customer
    mock_llm_service.generate = AsyncMock(return_value=LLMResult(
        text=json.dumps({
            "subject_line": f"A special offer just for you, {high_risk_id}!",
            "greeting": f"Hi {high_risk_id}, we have something special for you.",
            "introduction": "We value your loyalty and want to say thank you.",
            "body": "Here is what we have lined up:\n- Priority support\n- Bonus data on your plan",
            "call_to_action": "Click here to explore your exclusive Vodafone benefits today.",
            "closing": "Thank you for being a valued Vodafone customer. We look forward to serving you.",
            "signature": "Warm regards, Vodafone Customer Care Team",
        }),
        prompt_tokens=150,
        completion_tokens=300,
    ))

    resp = client.get(f"/api/v1/retention/{high_risk_id}")
    assert resp.status_code == 200
    data = resp.json()
    assert data["risk_level"] == "high"
    assert data["status"] == "at_risk"
    assert data["retention_email"] is not None
    email = data["retention_email"]
    assert all(k in email for k in ["subject_line", "greeting", "body", "call_to_action"])


def test_unknown_customer_returns_404(client):
    resp = client.get("/api/v1/retention/UNKNOWN-CUSTOMER-XYZ")
    assert resp.status_code == 404


def test_response_schema_has_profile(client, real_repository):
    cid = real_repository.list_customer_ids()[0]
    resp = client.get(f"/api/v1/retention/{cid}")
    assert resp.status_code == 200
    data = resp.json()
    profile = data["customer_profile"]
    assert "tenure_months" in profile
    assert "contract_type" in profile
    assert "monthly_charges" in profile
    assert "active_services" in profile


def test_churn_probability_in_valid_range(client, real_repository):
    cid = real_repository.list_customer_ids()[0]
    resp = client.get(f"/api/v1/retention/{cid}")
    assert resp.status_code == 200
    prob = resp.json()["churn_probability"]
    assert 0.0 <= prob <= 1.0
