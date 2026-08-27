"""
Model Registry for LLM Cost Autopilot
Free-tier providers: Groq (cloud, fast), Gemini (cloud), Ollama (local).
"""

from dataclasses import dataclass
from enum import Enum
from typing import Dict, Optional


class QualityTier(str, Enum):
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"


@dataclass(frozen=True)
class ModelConfig:
    provider: str                # "groq", "gemini", "ollama"
    model_id: str
    cost_per_input_token: float  # $0 for all free-tier options
    cost_per_output_token: float
    avg_latency_ms: int
    quality_tier: QualityTier

    def estimate_cost(self, input_tokens: int, output_tokens: int) -> float:
        return (
            input_tokens * self.cost_per_input_token
            + output_tokens * self.cost_per_output_token
        )


# ---------------------------------------------------------------------------
# All models below are free-tier / local — cost_per_token is $0.
# Since routing logic needs SOME cost differential to be meaningful, we use
# a "shadow cost" model: what these models WOULD cost on paid tiers, based
# on their paid equivalents. This keeps your dashboard's "cost saved" story
# intact even though you're not actually being billed during development.
# You can flip `use_shadow_pricing` off later once you add real paid keys.
# ---------------------------------------------------------------------------

MODEL_REGISTRY: Dict[str, ModelConfig] = {
        "gpt-oss-120b-groq": ModelConfig(
        provider="groq",
        model_id="openai/gpt-oss-120b",
        cost_per_input_token=0.59 / 1_000_000,   # shadow price, similar tier to old 70b
        cost_per_output_token=0.79 / 1_000_000,
        avg_latency_ms=500,
        quality_tier=QualityTier.HIGH,
    ),
    "gpt-oss-20b-groq": ModelConfig(
        provider="groq",
        model_id="openai/gpt-oss-20b",
        cost_per_input_token=0.05 / 1_000_000,
        cost_per_output_token=0.08 / 1_000_000,
        avg_latency_ms=250,
        quality_tier=QualityTier.MEDIUM,
    ),
       "gemini-flash": ModelConfig(
        provider="gemini",
        model_id="gemini-3.6-flash",
        cost_per_input_token=0.075 / 1_000_000,
        cost_per_output_token=0.30 / 1_000_000,
        avg_latency_ms=900,
        quality_tier=QualityTier.MEDIUM,
    ),
    "llama3-local": ModelConfig(
        provider="ollama",
        model_id="llama3",
        cost_per_input_token=0.0,
        cost_per_output_token=0.0,
        avg_latency_ms=3500,
        quality_tier=QualityTier.LOW,
    ),
}


def get_model(name: str) -> ModelConfig:
    if name not in MODEL_REGISTRY:
        raise KeyError(
            f"Model '{name}' not found in registry. "
            f"Available: {list(MODEL_REGISTRY.keys())}"
        )
    return MODEL_REGISTRY[name]


def list_models(tier: Optional[QualityTier] = None) -> Dict[str, ModelConfig]:
    if tier is None:
        return MODEL_REGISTRY
    return {k: v for k, v in MODEL_REGISTRY.items() if v.quality_tier == tier}