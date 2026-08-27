"""
Complexity tier definitions and feature extraction for routing.
"""

import re
from dataclasses import dataclass
from enum import IntEnum


class ComplexityTier(IntEnum):
    SIMPLE = 1      # reformatting, extraction, basic Q&A from provided context
    MODERATE = 2    # summarization, classification, structured analysis
    COMPLEX = 3      # multi-step reasoning, creative generation, nuanced judgment


# Keywords that hint at each tier — used as engineered features, not sole classifiers
COMPLEX_SIGNAL_WORDS = [
    "analyze", "compare", "evaluate", "argue", "critique", "design",
    "strategize", "synthesize", "reason", "justify", "recommend",
    "write a story", "brainstorm", "debate", "why", "trade-off",
]

MODERATE_SIGNAL_WORDS = [
    "summarize", "classify", "categorize", "extract themes",
    "structure", "organize", "outline", "identify",
]

SIMPLE_SIGNAL_WORDS = [
    "extract", "format", "list", "translate", "what is", "define",
    "convert", "rewrite",
]


@dataclass
class PromptFeatures:
    token_count: int
    has_complex_signal: bool
    has_moderate_signal: bool
    has_simple_signal: bool
    num_constraints: int          # rough count of "must", "should", "don't", etc.
    has_context_provided: bool    # e.g. prompt includes a block of text to work from
    output_format_complexity: int # 0 = free text, 1 = list/simple structure, 2 = JSON/table/multi-section


CONSTRAINT_WORDS = ["must", "should", "don't", "avoid", "only", "make sure", "ensure", "never", "always"]
STRUCTURED_OUTPUT_HINTS = ["json", "table", "bullet", "numbered list", "format:", "schema"]


def extract_features(prompt: str) -> PromptFeatures:
    """Extract routing-relevant features from a raw prompt string."""
    lower = prompt.lower()

    token_count = len(prompt.split())  # rough proxy; swap for a real tokenizer later if needed

    has_complex_signal = any(word in lower for word in COMPLEX_SIGNAL_WORDS)
    has_moderate_signal = any(word in lower for word in MODERATE_SIGNAL_WORDS)
    has_simple_signal = any(word in lower for word in SIMPLE_SIGNAL_WORDS)

    num_constraints = sum(lower.count(word) for word in CONSTRAINT_WORDS)

    # crude heuristic: a long quoted/pasted block or explicit "context:" marker
    has_context_provided = bool(
        re.search(r'context\s*:', lower) or len(prompt) > 400 and ('"' in prompt or "'" in prompt)
    )

    if any(hint in lower for hint in ["json", "schema"]):
        output_format_complexity = 2
    elif any(hint in lower for hint in ["table", "bullet", "numbered list", "format:"]):
        output_format_complexity = 1
    else:
        output_format_complexity = 0

    return PromptFeatures(
        token_count=token_count,
        has_complex_signal=has_complex_signal,
        has_moderate_signal=has_moderate_signal,
        has_simple_signal=has_simple_signal,
        num_constraints=num_constraints,
        has_context_provided=has_context_provided,
        output_format_complexity=output_format_complexity,
    )


def features_to_vector(features: PromptFeatures) -> list[float]:
    """Convert PromptFeatures into a numeric vector for the classifier."""
    return [
        features.token_count,
        int(features.has_complex_signal),
        int(features.has_moderate_signal),
        int(features.has_simple_signal),
        features.num_constraints,
        int(features.has_context_provided),
        features.output_format_complexity,
    ]


FEATURE_NAMES = [
    "token_count",
    "has_complex_signal",
    "has_moderate_signal",
    "has_simple_signal",
    "num_constraints",
    "has_context_provided",
    "output_format_complexity",
]