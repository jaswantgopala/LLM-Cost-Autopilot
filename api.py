"""
Phase 5: FastAPI service exposing the LLM Cost Autopilot as an API.
Run with: uvicorn api:app --reload
"""

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from typing import Optional
import yaml

from core.pipeline import process_request, get_router
from core.registry import MODEL_REGISTRY, get_model
from core.logger import init_db, get_summary_stats

app = FastAPI(
    title="LLM Cost Autopilot",
    description="Intelligent request routing to minimize LLM API costs while maintaining quality",
    version="1.0.0",
)

ROUTING_CONFIG_PATH = "config/routing.yaml"

init_db()


# --- Request/Response schemas ---

class CompletionRequest(BaseModel):
    prompt: str = Field(..., description="The user's prompt/request")
    verify: bool = Field(True, description="Whether to run async quality verification")


class CompletionResponse(BaseModel):
    prompt: str
    response: str
    predicted_tier: int
    routed_model: str
    escalated: bool
    escalated_model: Optional[str] = None
    cost_usd: float


class RoutingConfigUpdate(BaseModel):
    tier: int = Field(..., ge=1, le=3, description="Tier number (1, 2, or 3)")
    model_key: str = Field(..., description="Model registry key to route this tier to")
    fallback_model_key: Optional[str] = Field(None, description="Fallback model key")


# --- Endpoints ---

@app.post("/v1/completions", response_model=CompletionResponse)
def create_completion(request: CompletionRequest):
    """
    Main routing endpoint. The user doesn't choose the model — the router does.
    Returns the response along with metadata on which model was selected and why.
    """
    try:
        result = process_request(request.prompt, verify=request.verify)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

    return CompletionResponse(
        prompt=result["prompt"],
        response=result["final_response_text"],
        predicted_tier=result["predicted_tier"],
        routed_model=result["routed_model_key"],
        escalated=result["escalated"],
        escalated_model=result.get("escalated_model_key"),
        cost_usd=result["final_cost_usd"],
    )


@app.get("/v1/models")
def list_models():
    """List all available models in the registry with their pricing."""
    return {
        key: {
            "provider": cfg.provider,
            "model_id": cfg.model_id,
            "cost_per_input_token": cfg.cost_per_input_token,
            "cost_per_output_token": cfg.cost_per_output_token,
            "avg_latency_ms": cfg.avg_latency_ms,
            "quality_tier": cfg.quality_tier.value,
        }
        for key, cfg in MODEL_REGISTRY.items()
    }


@app.get("/v1/stats")
def get_stats():
    """Cost savings summary — total spend, escalation rate, per-model breakdown."""
    stats = get_summary_stats()
    return stats


@app.put("/v1/routing-config")
def update_routing_config(update: RoutingConfigUpdate):
    """
    Update the tier-to-model mapping without redeploying.
    Validates the model key exists before writing to the YAML file.
    """
    # validate model key exists
    try:
        get_model(update.model_key)
    except KeyError:
        raise HTTPException(status_code=400, detail=f"Unknown model_key: {update.model_key}")

    if update.fallback_model_key:
        try:
            get_model(update.fallback_model_key)
        except KeyError:
            raise HTTPException(status_code=400, detail=f"Unknown fallback_model_key: {update.fallback_model_key}")

    with open(ROUTING_CONFIG_PATH, "r") as f:
        config = yaml.safe_load(f)

    tier_key = f"tier_{update.tier}"
    if tier_key not in config["routing"]:
        raise HTTPException(status_code=400, detail=f"Unknown tier: {update.tier}")

    config["routing"][tier_key]["model_key"] = update.model_key
    if update.fallback_model_key:
        config["routing"][tier_key]["fallback_model_key"] = update.fallback_model_key

    with open(ROUTING_CONFIG_PATH, "w") as f:
        yaml.dump(config, f, default_flow_style=False, sort_keys=False)

    # force the router to reload the updated config on next request
    import core.pipeline
    core.pipeline._router = None

    return {"status": "updated", "tier": update.tier, "new_config": config["routing"][tier_key]}


@app.get("/")
def root():
    return {
        "service": "LLM Cost Autopilot",
        "endpoints": ["/v1/completions", "/v1/models", "/v1/stats", "/v1/routing-config", "/docs"],
    }