"""
Unified Model Interface — single entry point for calling any provider.
"""

import os
import time
from dotenv import load_dotenv

from groq import Groq
from google import genai as google_genai
import httpx

from core.registry import ModelConfig
from core.models import LLMResponse

load_dotenv()

_groq_client = Groq(api_key=os.getenv("GROQ_API_KEY"))
_gemini_client = google_genai.Client(api_key=os.getenv("GEMINI_API_KEY"))
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")


def send_request(prompt: str, model_config: ModelConfig, system: str = "") -> LLMResponse:
    start = time.perf_counter()

    if model_config.provider == "groq":
        raw = _call_groq(prompt, model_config, system)
        input_tokens = raw.usage.prompt_tokens
        output_tokens = raw.usage.completion_tokens
        text = raw.choices[0].message.content
        raw_dict = raw.model_dump()

    elif model_config.provider == "gemini":
        raw, input_tokens, output_tokens, text = _call_gemini(prompt, model_config, system)
        raw_dict = {"text": text}

    elif model_config.provider == "ollama":
        raw, input_tokens, output_tokens = _call_ollama(prompt, model_config, system)
        text = raw.get("response", "")
        raw_dict = raw

    else:
        raise ValueError(f"Unknown provider: {model_config.provider}")

    latency_ms = (time.perf_counter() - start) * 1000
    cost = model_config.estimate_cost(input_tokens, output_tokens)

    return LLMResponse(
        text=text,
        model_id=model_config.model_id,
        provider=model_config.provider,
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        latency_ms=latency_ms,
        cost_usd=cost,
        raw_response=raw_dict,
    )


def _call_groq(prompt: str, model_config: ModelConfig, system: str):
    messages = []
    if system:
        messages.append({"role": "system", "content": system})
    messages.append({"role": "user", "content": prompt})

    return _groq_client.chat.completions.create(
        model=model_config.model_id,
        messages=messages,
    )


def _call_gemini(prompt: str, model_config: ModelConfig, system: str):
    config = {}
    if system:
        config["system_instruction"] = system

    response = _gemini_client.models.generate_content(
        model=model_config.model_id,
        contents=prompt,
        config=config if config else None,
    )

    input_tokens = response.usage_metadata.prompt_token_count
    output_tokens = response.usage_metadata.candidates_token_count
    text = response.text

    return response, input_tokens, output_tokens, text


def _call_ollama(prompt: str, model_config: ModelConfig, system: str):
    full_prompt = f"{system}\n\n{prompt}" if system else prompt

    response = httpx.post(
        f"{OLLAMA_BASE_URL}/api/generate",
        json={
            "model": model_config.model_id,
            "prompt": full_prompt,
            "stream": False,
        },
        timeout=120.0,
    )
    response.raise_for_status()
    data = response.json()

    input_tokens = data.get("prompt_eval_count", 0)
    output_tokens = data.get("eval_count", 0)

    return data, input_tokens, output_tokens