"""LLM service implementations.

OpenAILLMService  — production path, requires OPENAI_API_KEY.
DemoLLMService    — zero-dependency fallback used when no API key is set.
                    Generates a realistic, personalised email from a template
                    using the actual customer data in the user prompt, so
                    reviewers can exercise the full workflow without a key.
"""
from __future__ import annotations

import json
import logging
import re
import time

import openai

from app.core.config import Settings
from app.core.exceptions import LLMError
from app.services.interfaces import LLMResult, LLMServiceABC

logger = logging.getLogger(__name__)

_SYSTEM_PROMPT = """You are a specialist retention copywriter for Vodafone.
Your job is to write personalised retention emails that follow the Vodafone
Tone of Voice Guidelines exactly.

TONE OF VOICE RULES:
- Friendly, warm, and conversational — never cold or corporate.
- Clear and concise — use short sentences and bullet points in the body.
- Positive and reassuring — focus on benefits, not problems.
- Professional — respectful, accurate, never pushy or alarmist.

PROHIBITED:
- Do NOT mention churn, cancellation, or the customer potentially leaving.
- Do NOT reference competitor brands (EE, O2, Three, BT, Sky, Virgin Media).
- Do NOT use strongly negative language (worst, terrible, awful).
- Do NOT invent specific monetary discounts (e.g. "50% off") unless derived
  from the customer's actual data provided to you.
- Do NOT leave template placeholders like [Name] in your output.
- Do NOT use markdown syntax of any kind — no **bold**, no *italic*, no #headers,
  no markdown bullet syntax. Use plain text only. For bullet points in the body
  field use a dash followed by a space (- ) at the start of each line.

REQUIRED EMAIL STRUCTURE — respond with ONLY valid JSON, no markdown fences:
{
  "subject_line": "<Friendly, enticing, personalised subject>",
  "greeting": "<Warm opening that uses the customer identifier>",
  "introduction": "<1-2 sentences: why you are writing>",
  "body": "<Main content with bullet-pointed benefits tailored to their services>",
  "call_to_action": "<One clear, compelling next step>",
  "closing": "<Warm, appreciative sign-off>",
  "signature": "Warm regards, Vodafone Customer Care Team"
}

Return ONLY the JSON object. No commentary before or after it."""


class OpenAILLMService(LLMServiceABC):
    """Calls the OpenAI Chat Completions API with async I/O."""

    def __init__(self, settings: Settings) -> None:
        self._model = settings.llm_model
        self._client = openai.AsyncOpenAI(api_key=settings.openai_api_key)

    @classmethod
    def from_api_key(cls, api_key: str, model: str = "gpt-4o-mini") -> "OpenAILLMService":
        """Create a one-off instance from a bare API key (e.g. per-request override)."""
        instance = object.__new__(cls)
        instance._model = model
        instance._client = openai.AsyncOpenAI(api_key=api_key)
        return instance

    async def generate(self, system_prompt: str, user_prompt: str) -> LLMResult:
        """Send a request to the LLM and return an LLMResult with full metadata."""
        logger.info("Calling LLM model '%s'.", self._model)
        t0 = time.perf_counter()
        try:
            response = await self._client.chat.completions.create(
                model=self._model,
                max_tokens=1024,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user",   "content": user_prompt},
                ],
                response_format={"type": "json_object"},
            )
            latency_ms = (time.perf_counter() - t0) * 1000
            raw = response.choices[0].message.content or ""
            usage = response.usage

            logger.info(
                "LLM response received | latency=%.0fms | tokens in=%d out=%d",
                latency_ms,
                usage.prompt_tokens if usage else 0,
                usage.completion_tokens if usage else 0,
            )
            return LLMResult(
                text=raw,
                prompt_tokens=usage.prompt_tokens if usage else 0,
                completion_tokens=usage.completion_tokens if usage else 0,
                latency_ms=latency_ms,
            )
        except openai.APIError as exc:
            raise LLMError(f"OpenAI API error: {exc}") from exc
        except Exception as exc:
            raise LLMError(f"Unexpected LLM error: {exc}") from exc

    @staticmethod
    def system_prompt() -> str:
        return _SYSTEM_PROMPT


class DemoLLMService(LLMServiceABC):
    """Template-based email generator — no API key required.

    Parses the customer details from the user prompt and fills a
    Vodafone-compliant email template.  Every guardrail check still runs
    on the output, so the full validation pipeline is exercised.
    """

    _MODEL_NAME = "demo-template-v1"

    async def generate(self, system_prompt: str, user_prompt: str) -> LLMResult:
        logger.info("DemoLLMService: generating template email (no API key required).")
        t0 = time.perf_counter()

        customer_id = self._extract(user_prompt, r"Customer ID:\s*(\S+)")
        tenure      = self._extract(user_prompt, r"Tenure:\s*(\d+)")
        contract    = self._extract(user_prompt, r"Contract type:\s*(.+)")
        spend       = self._extract(user_prompt, r"Monthly spend:\s*£?([\d.]+)")
        services    = self._extract_services(user_prompt)

        body_bullets = self._build_body(services, tenure, contract)

        email = {
            "subject_line": f"A special thank-you just for you, {customer_id}!",
            "greeting": f"Hi {customer_id},",
            "introduction": (
                f"You've been with us for {tenure} months and we truly value your loyalty. "
                "We'd love to make sure you're getting everything Vodafone has to offer."
            ),
            "body": body_bullets,
            "call_to_action": (
                "Get in touch with our dedicated loyalty team today to claim your benefits — "
                "we're here to help."
            ),
            "closing": (
                "Thank you for being such a valued part of the Vodafone family. "
                "We look forward to continuing to serve you."
            ),
            "signature": "Warm regards, Vodafone Customer Care Team",
        }

        latency_ms = (time.perf_counter() - t0) * 1000
        return LLMResult(
            text=json.dumps(email),
            prompt_tokens=0,
            completion_tokens=0,
            latency_ms=latency_ms,
        )

    #  Helpers 

    @staticmethod
    def _extract(text: str, pattern: str, default: str = "valued customer") -> str:
        m = re.search(pattern, text, re.IGNORECASE)
        return m.group(1).strip() if m else default

    @staticmethod
    def _extract_services(user_prompt: str) -> list[str]:
        """Pull the active services bullet list from the user prompt."""
        match = re.search(r"ACTIVE SERVICES:(.*?)INSTRUCTIONS:", user_prompt, re.DOTALL)
        if not match:
            return []
        lines = [l.strip().lstrip("- ") for l in match.group(1).splitlines() if l.strip().startswith("-")]
        return lines

    @staticmethod
    def _build_body(services: list[str], tenure: str, contract: str) -> str:
        bullets = ["Here is what we have lined up exclusively for you:"]

        # Service-specific offers
        service_str = " ".join(services).lower()
        if "streaming" in service_str:
            bullets.append("• Free upgrade to our premium entertainment bundle — more content, same price.")
        if "fiber" in service_str or "dsl" in service_str:
            bullets.append("• Complimentary broadband speed boost for the next 3 months.")
        if "techsupport" in service_str or "tech support" in service_str:
            bullets.append("• Priority access to our expert Tech Support team — no waiting.")
        if not services or len(bullets) == 1:
            bullets.append("• Exclusive loyalty data bonus added to your current plan.")
            bullets.append("• Priority customer support — skip the queue, every time.")

        # Tenure-based offer
        try:
            months = int(tenure)
            if months >= 24:
                bullets.append("• As a long-standing customer, you qualify for our Platinum Loyalty Reward.")
            elif months >= 12:
                bullets.append("• A special loyalty thank-you: an extra data boost on us.")
        except ValueError:
            pass

        return "\n".join(bullets)

    @staticmethod
    def system_prompt() -> str:
        return _SYSTEM_PROMPT
