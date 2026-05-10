"""Brand-compliance and safety guardrails for LLM-generated content.

The guardrails layer is intentionally separated from the LLM service so it
can be unit-tested in isolation and extended without modifying generation logic
(Open/Closed Principle).

Checks performed:
  1. JSON parseability and required field presence (structural integrity).
  2. No unreplaced template placeholders (e.g. "[Name]").
  3. Customer name appears in the greeting (personalisation requirement).
  4. No prohibited / negative-churn language that would alarm the customer.
  5. No competitor brand mentions.
  6. All required email sections are non-trivially populated (min-length).
"""
from __future__ import annotations

import json
import logging
import re
from typing import Any

from app.core.exceptions import GuardrailViolationError
from app.services.interfaces import GuardrailsABC

logger = logging.getLogger(__name__)

#  Configuration constants 

REQUIRED_FIELDS = [
    "subject_line",
    "greeting",
    "introduction",
    "body",
    "call_to_action",
    "closing",
    "signature",
]

MIN_FIELD_LENGTH = 10  # characters — guards against empty / trivially short fields

# Patterns that should never appear in customer-facing copy
_PROHIBITED_PATTERNS: list[tuple[str, str]] = [
    (r"\bchurn\b", "must not mention 'churn'"),
    (r"\bcancel(?:l?ation)?\b", "must not mention cancellation"),
    (r"\bterminate\b|\btermination\b", "must not mention termination"),
    (r"\bworst\b|\bterrible\b|\bawful\b", "must not use strongly negative language"),
    (r"\bEE\b|\bO2\b|\bThree\b|\bBT\b|\bSky\b|\bVirgin Media\b",
     "must not name competitors"),
]

# Template placeholders the LLM sometimes forgets to substitute
_PLACEHOLDER_RE = re.compile(r"\[(?:Name|Customer Name|First Name|Last Name)\]", re.IGNORECASE)


class VodafoneGuardrails(GuardrailsABC):
    """Validates LLM output against Vodafone brand and safety rules."""

    def validate(self, raw_llm_output: str, customer_name: str) -> dict[str, str]:
        """Parse JSON output and run all checks.

        Args:
            raw_llm_output: Raw string returned by the LLM (expected to be JSON).
            customer_name:  The customer's identifier used for personalisation checks.

        Returns:
            A clean dict with validated email field values.

        Raises:
            GuardrailViolationError: if any check fails.
        """
        violations: list[str] = []

        #  1. Parse JSON 
        parsed = self._parse_json(raw_llm_output, violations)
        if parsed is None:
            raise GuardrailViolationError(violations)

        #  2. Required fields 
        self._check_required_fields(parsed, violations)

        #  3. Minimum content length 
        self._check_field_lengths(parsed, violations)

        #  4. No unreplaced placeholders 
        self._check_placeholders(parsed, violations)

        #  5. Personalisation (customer name in greeting) 
        self._check_personalisation(parsed, customer_name, violations)

        #  6. Prohibited language and competitor mentions 
        self._check_prohibited_patterns(parsed, violations)

        if violations:
            logger.warning("Guardrail violations detected: %s", violations)
            raise GuardrailViolationError(violations)

        logger.info("Guardrails passed for customer '%s'.", customer_name)
        return {field: str(parsed.get(field, "")) for field in REQUIRED_FIELDS}

    #  Private helpers 

    @staticmethod
    def _parse_json(raw: str, violations: list[str]) -> dict[str, Any] | None:
        # Strip markdown code fences that some models add
        cleaned = re.sub(r"```(?:json)?", "", raw).strip()
        try:
            parsed = json.loads(cleaned)
            if not isinstance(parsed, dict):
                violations.append("LLM output is not a JSON object.")
                return None
            return parsed
        except json.JSONDecodeError as exc:
            violations.append(f"LLM output is not valid JSON: {exc}")
            return None

    @staticmethod
    def _check_required_fields(parsed: dict, violations: list[str]) -> None:
        for field in REQUIRED_FIELDS:
            if field not in parsed:
                violations.append(f"Missing required email field: '{field}'.")

    @staticmethod
    def _check_field_lengths(parsed: dict, violations: list[str]) -> None:
        for field in REQUIRED_FIELDS:
            value = str(parsed.get(field, "")).strip()
            if len(value) < MIN_FIELD_LENGTH:
                violations.append(
                    f"Field '{field}' is too short ({len(value)} chars, min {MIN_FIELD_LENGTH})."
                )

    @staticmethod
    def _check_placeholders(parsed: dict, violations: list[str]) -> None:
        for field in REQUIRED_FIELDS:
            value = str(parsed.get(field, ""))
            if _PLACEHOLDER_RE.search(value):
                violations.append(
                    f"Field '{field}' contains an unreplaced template placeholder."
                )

    @staticmethod
    def _check_personalisation(parsed: dict, customer_name: str, violations: list[str]) -> None:
        greeting = str(parsed.get("greeting", "")).lower()
        # Use the last portion of a customer_id as a fallback name hint
        name_hint = customer_name.split("-")[-1].lower() if "-" in customer_name else customer_name.lower()
        if name_hint not in greeting and customer_name.lower() not in greeting:
            violations.append(
                "Greeting does not appear to address the customer by their identifier."
            )

    @staticmethod
    def _check_prohibited_patterns(parsed: dict, violations: list[str]) -> None:
        full_text = " ".join(str(v) for v in parsed.values()).lower()
        for pattern, reason in _PROHIBITED_PATTERNS:
            if re.search(pattern, full_text, re.IGNORECASE):
                violations.append(f"Content {reason}.")
