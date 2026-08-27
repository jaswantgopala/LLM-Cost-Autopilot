"""
Standardized response schema for LLM Cost Autopilot.
"""

from dataclasses import dataclass
from typing import Optional


@dataclass
class LLMResponse:
    text: str                # the model's output
    model_id: str             # e.g. "gpt-4o", "claude-sonnet-4-6"
    provider: str              # "openai", "anthropic", "ollama"
    input_tokens: int
    output_tokens: int
    latency_ms: float
    cost_usd: float
    raw_response: Optional[dict] = None  # original provider response, for debugging

    @property
    def total_tokens(self) -> int:
        return self.input_tokens + self.output_tokens

    def __repr__(self) -> str:
        return (
            f"LLMResponse(model={self.model_id}, "
            f"tokens={self.total_tokens}, "
            f"cost=${self.cost_usd:.6f}, "
            f"latency={self.latency_ms:.0f}ms)"
        )