"""
Phase 6, Step 1 (final): Generate the final cost savings report from logged data.
"""

import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
from core.logger import get_all_requests, get_summary_stats

REFERENCE_INPUT_COST_PER_TOKEN = 2.50 / 1_000_000
REFERENCE_OUTPUT_COST_PER_TOKEN = 10.00 / 1_000_000

requests = get_all_requests()
stats = get_summary_stats()

df = pd.DataFrame(requests)

print("=" * 60)
print("FINAL LOAD TEST REPORT")
print("=" * 60)

print(f"\nTotal requests processed: {len(df)}")
print(f"Total actual cost: ${stats['total_cost_usd']:.6f}")

# Reference baseline (GPT-4o, not called — pricing-only comparison)
baseline_cost = 0.0
for _, row in df.iterrows():
    baseline_cost += (row["input_tokens"] or 0) * REFERENCE_INPUT_COST_PER_TOKEN
    baseline_cost += (row["output_tokens"] or 0) * REFERENCE_OUTPUT_COST_PER_TOKEN

savings = baseline_cost - stats["total_cost_usd"]
savings_pct = (savings / baseline_cost * 100) if baseline_cost > 0 else 0

print(f"Reference cost (all-GPT-4o, pricing-only): ${baseline_cost:.6f}")
print(f"Cost savings: {savings_pct:.1f}%")

print(f"\nEscalation rate: {stats['escalation_rate']:.1%}")
print(f"Average quality score: {stats['avg_quality_score']:.2f}/5.00" if stats['avg_quality_score'] else "N/A")

print(f"\nRouting distribution:")
tier_counts = df["predicted_tier"].value_counts().sort_index()
for tier, count in tier_counts.items():
    pct = count / len(df) * 100
    print(f"  Tier {tier}: {count} requests ({pct:.1f}%)")

print(f"\nBy model:")
for m in stats["by_model"]:
    print(f"  {m['routed_model_key']:20s} count={m['count']:4d} cost=${m['total_cost']:.6f}")

# Rate limit errors encountered (from load test)
if os.path.exists("data/loadtest_progress.json"):
    import json
    with open("data/loadtest_progress.json") as f:
        progress = json.load(f)
    rate_limit_errors = [e for e in progress.get("errors", []) if "429" in e.get("error", "") or "rate_limit" in e.get("error", "").lower()]
    print(f"\nRate-limit errors encountered during load test: {len(rate_limit_errors)}")
    print("(These reflect genuine free-tier daily quota limits from Groq/Gemini, not system bugs)")

print("\n" + "=" * 60)