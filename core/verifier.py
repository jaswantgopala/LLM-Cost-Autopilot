"""
Phase 3: Async Quality Verification Loop.
Uses LLM-as-judge to check if a cheaper model's response was good enough,
and triggers escalation to a higher-tier model if not.
"""

import json
import re
from dataclasses import dataclass
from typing import Optional

from core.registry import get_model
from core.interface import send_request
from core.models import LLMResponse

JUDGE_MODEL_KEY = "gpt-oss-120b-groq"  # highest-tier model acts as judge
QUALITY_THRESHOLD = 4  # out of 5; below this triggers escalation


@dataclass
class VerificationResult:
    prompt: str
    original_model_key: str
    original_response: str
    quality_score: int          # 1-5, from the judge
    reasoning: str
    passed: bool                 # score >= QUALITY_THRESHOLD
    escalated: bool = False
    escalated_model_key: Optional[str] = None
    escalated_response: Optional[str] = None
    cost_delta: float = 0.0


JUDGE_PROMPT_TEMPLATE = """You are a strict quality judge for AI responses.

Original request:
{prompt}

Response to evaluate:
{response}

Rate this response's quality on a scale of 1-5, where:
5 = Excellent, fully correct and complete
4 = Good, minor issues but acceptable
3 = Mediocre, noticeable gaps or errors
2 = Poor, significant problems
1 = Failing, wrong or unusable

Respond ONLY with valid JSON in this exact format, no other text:
{{"score": <int 1-5>, "reasoning": "<one sentence explanation>"}}
"""


def _parse_judge_output(text: str) -> tuple[int, str]:
    """Extract score and reasoning from the judge's JSON response, with fallback."""
    try:
        # strip markdown code fences if present
        cleaned = re.sub(r"^```(json)?|```$", "", text.strip(), flags=re.MULTILINE).strip()
        data = json.loads(cleaned)
        return int(data["score"]), str(data["reasoning"])
    except (json.JSONDecodeError, KeyError, ValueError):
        # fallback: try to find a digit 1-5 in the text
        match = re.search(r"\b([1-5])\b", text)
        score = int(match.group(1)) if match else 3  # default to mediocre if unparseable
        return score, f"[unparsed judge output] {text[:200]}"


def verify_response(prompt: str, original_response: LLMResponse, original_model_key: str) -> VerificationResult:
    """
    Send the same prompt + the cheap model's response to the judge model
    for quality scoring. This is the core of the async verification loop.
    """
    judge_config = get_model(JUDGE_MODEL_KEY)

    judge_prompt = JUDGE_PROMPT_TEMPLATE.format(
        prompt=prompt,
        response=original_response.text,
    )

    judge_response = send_request(judge_prompt, judge_config)
    score, reasoning = _parse_judge_output(judge_response.text)

    passed = score >= QUALITY_THRESHOLD

    return VerificationResult(
        prompt=prompt,
        original_model_key=original_model_key,
        original_response=original_response.text,
        quality_score=score,
        reasoning=reasoning,
        passed=passed,
    )


def escalate(verification: VerificationResult, escalation_model_key: str = JUDGE_MODEL_KEY) -> VerificationResult:
    """
    Re-run the prompt on a higher-tier model and attach the escalated result.
    Called only when verify_response() found the original response failed.
    """
    escalation_config = get_model(escalation_model_key)
    escalated_response = send_request(verification.prompt, escalation_config)

    original_cost_config = get_model(verification.original_model_key)
    # we don't have the original response's exact token counts here, so cost_delta
    # is computed by the caller (who has both LLMResponse objects) — see scripts/test_verifier.py

    verification.escalated = True
    verification.escalated_model_key = escalation_model_key
    verification.escalated_response = escalated_response.text

    return verification, escalated_response