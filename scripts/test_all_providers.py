"""
Phase 1, Step 3: Test every provider in the registry with the same prompts.
Logs outputs, cost, and latency to validate the abstraction layer.
"""

import sys
import os
import json
from datetime import datetime

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.registry import MODEL_REGISTRY
from core.interface import send_request

TEST_PROMPTS = [
    "What is the capital of France?",
    "Summarize this in one sentence: The stock market fell 3% today amid inflation concerns.",
    "Extract the name and email from: 'Contact John Doe at john@example.com for details.'",
    "Write a haiku about autumn.",
    "Is this sentence positive or negative? 'The service was absolutely terrible.'",
    "Explain quantum entanglement in simple terms.",
    "List 3 benefits of exercise.",
    "Translate 'good morning' to Spanish.",
    "Compare Python and JavaScript in two sentences.",
    "What is 15% of 240?",
]


def run_tests():
    results = []

    for model_key, model_config in MODEL_REGISTRY.items():
        print(f"\n{'='*60}")
        print(f"Testing model: {model_key} ({model_config.provider})")
        print(f"{'='*60}")

        for i, prompt in enumerate(TEST_PROMPTS, 1):
            try:
                response = send_request(prompt, model_config)
                print(f"[{i}/10] cost=${response.cost_usd:.6f} "
                      f"latency={response.latency_ms:.0f}ms "
                      f"tokens={response.total_tokens}")

                results.append({
                    "model_key": model_key,
                    "model_id": response.model_id,
                    "provider": response.provider,
                    "prompt": prompt,
                    "output": response.text,
                    "input_tokens": response.input_tokens,
                    "output_tokens": response.output_tokens,
                    "cost_usd": response.cost_usd,
                    "latency_ms": response.latency_ms,
                    "timestamp": datetime.utcnow().isoformat(),
                    "error": None,
                })

            except Exception as e:
                print(f"[{i}/10] ERROR: {e}")
                results.append({
                    "model_key": model_key,
                    "model_id": model_config.model_id,
                    "provider": model_config.provider,
                    "prompt": prompt,
                    "output": None,
                    "input_tokens": None,
                    "output_tokens": None,
                    "cost_usd": None,
                    "latency_ms": None,
                    "timestamp": datetime.utcnow().isoformat(),
                    "error": str(e),
                })

    return results


def save_results(results, path="test_results/baseline_results.json"):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved {len(results)} results to {path}")


def print_summary(results):
    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")

    by_model = {}
    for r in results:
        key = r["model_key"]
        by_model.setdefault(key, {"total_cost": 0, "count": 0, "errors": 0, "latencies": []})
        by_model[key]["count"] += 1
        if r["error"]:
            by_model[key]["errors"] += 1
        else:
            by_model[key]["total_cost"] += r["cost_usd"]
            by_model[key]["latencies"].append(r["latency_ms"])

    for model_key, stats in by_model.items():
        avg_latency = sum(stats["latencies"]) / len(stats["latencies"]) if stats["latencies"] else 0
        print(f"{model_key:20s} total_cost=${stats['total_cost']:.6f} "
              f"avg_latency={avg_latency:.0f}ms "
              f"errors={stats['errors']}/{stats['count']}")


if __name__ == "__main__":
    results = run_tests()
    save_results(results)
    print_summary(results)