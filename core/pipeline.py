"""
End-to-end pipeline: route -> respond -> verify -> escalate if needed -> log.
This is the single function the API layer (Phase 5) will call.
"""

from core.router import Router
from core.interface import send_request
from core.verifier import verify_response, escalate
from core.logger import log_request, init_db

_router = None


def get_router() -> Router:
    global _router
    if _router is None:
        _router = Router()
    return _router


def process_request(prompt: str, verify: bool = True, route_text: str | None = None) -> dict:
    """
    Full pipeline for handling one prompt.
    Returns a dict with the final response and all metadata.
    """
    router = get_router()
    decision = router.route(route_text or prompt)
    model_config = decision["model_config"]

    response = send_request(prompt, model_config)

    result = {
        "prompt": prompt,
        "predicted_tier": decision["predicted_tier"],
        "routed_model_key": decision["routed_model_key"],
        "response_text": response.text,
        "cost_usd": response.cost_usd,
        "latency_ms": response.latency_ms,
        "escalated": False,
        "escalated_model_key": None,
        "final_response_text": response.text,
        "final_cost_usd": response.cost_usd,
        "input_tokens": response.input_tokens,
        "output_tokens": response.output_tokens,
    }

    quality_score = None
    quality_reasoning = None
    passed = None

    if verify:
        verification = verify_response(prompt, response, decision["routed_model_key"])
        quality_score = verification.quality_score
        quality_reasoning = verification.reasoning
        passed = verification.passed

        if not passed:
            verification, escalated_response = escalate(verification)
            result["escalated"] = True
            result["escalated_model_key"] = verification.escalated_model_key
            result["final_response_text"] = escalated_response.text
            result["final_cost_usd"] = response.cost_usd + escalated_response.cost_usd

            log_request(
                prompt=prompt,
                predicted_tier=decision["predicted_tier"],
                routed_model_key=decision["routed_model_key"],
                response_text=response.text,
                input_tokens=response.input_tokens,
                output_tokens=response.output_tokens,
                cost_usd=response.cost_usd,
                latency_ms=response.latency_ms,
                quality_score=quality_score,
                quality_reasoning=quality_reasoning,
                passed_verification=passed,
                escalated=True,
                escalated_model_key=verification.escalated_model_key,
                escalated_cost_usd=escalated_response.cost_usd,
            )
            return result

    log_request(
        prompt=prompt,
        predicted_tier=decision["predicted_tier"],
        routed_model_key=decision["routed_model_key"],
        response_text=response.text,
        input_tokens=response.input_tokens,
        output_tokens=response.output_tokens,
        cost_usd=response.cost_usd,
        latency_ms=response.latency_ms,
        quality_score=quality_score,
        quality_reasoning=quality_reasoning,
        passed_verification=passed,
    )

    return result