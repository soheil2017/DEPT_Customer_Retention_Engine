"""Tests for VodafoneGuardrails — all checks run without LLM or network calls."""
import json

import pytest

from app.core.exceptions import GuardrailViolationError
from app.services.guardrails import VodafoneGuardrails

CUSTOMER_ID = "CUST-9999"


def _make_json(**overrides) -> str:
    base = {
        "subject_line": f"A special offer just for you, {CUSTOMER_ID}!",
        "greeting": f"Hi {CUSTOMER_ID}, welcome back!",
        "introduction": "We value your loyalty and want to say thank you.",
        "body": "Here is what we have lined up:\n- Priority support\n- Bonus data",
        "call_to_action": "Click here to explore your exclusive Vodafone benefits.",
        "closing": "Thank you for being a valued customer. We look forward to serving you.",
        "signature": "Warm regards, Vodafone Customer Care Team",
    }
    base.update(overrides)
    return json.dumps(base)


@pytest.fixture
def g():
    return VodafoneGuardrails()


def test_valid_email_passes(g):
    result = g.validate(_make_json(), CUSTOMER_ID)
    assert "subject_line" in result
    assert "greeting" in result


def test_invalid_json_raises(g):
    with pytest.raises(GuardrailViolationError) as exc_info:
        g.validate("not json at all", CUSTOMER_ID)
    assert any("JSON" in v for v in exc_info.value.violations)


def test_missing_field_raises(g):
    data = json.loads(_make_json())
    del data["call_to_action"]
    with pytest.raises(GuardrailViolationError) as exc_info:
        g.validate(json.dumps(data), CUSTOMER_ID)
    assert any("call_to_action" in v for v in exc_info.value.violations)


def test_placeholder_raises(g):
    with pytest.raises(GuardrailViolationError) as exc_info:
        g.validate(_make_json(greeting="Hi [Name], welcome back!"), CUSTOMER_ID)
    assert any("placeholder" in v for v in exc_info.value.violations)


def test_churn_mention_raises(g):
    with pytest.raises(GuardrailViolationError) as exc_info:
        g.validate(_make_json(body="We noticed you might churn soon."), CUSTOMER_ID)
    assert any("churn" in v for v in exc_info.value.violations)


def test_competitor_mention_raises(g):
    with pytest.raises(GuardrailViolationError) as exc_info:
        g.validate(_make_json(body="Unlike O2, we offer great value."), CUSTOMER_ID)
    assert any("competitor" in v for v in exc_info.value.violations)


def test_cancellation_mention_raises(g):
    with pytest.raises(GuardrailViolationError) as exc_info:
        g.validate(_make_json(introduction="We noticed your cancellation request."), CUSTOMER_ID)
    assert any("cancellation" in v for v in exc_info.value.violations)


def test_markdown_fences_stripped(g):
    raw = "```json\n" + _make_json() + "\n```"
    result = g.validate(raw, CUSTOMER_ID)
    assert "subject_line" in result


def test_too_short_field_raises(g):
    with pytest.raises(GuardrailViolationError) as exc_info:
        g.validate(_make_json(body="Short"), CUSTOMER_ID)
    assert any("body" in v and "short" in v.lower() for v in exc_info.value.violations)
