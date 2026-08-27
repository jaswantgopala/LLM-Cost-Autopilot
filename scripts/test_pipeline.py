import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.logger import init_db, get_summary_stats
from core.pipeline import process_request

init_db()

test_prompts = [
    "What is the capital of France?",
    "Summarize the benefits of remote work in 2 sentences.",
    "Analyze the risks of scaling a startup too quickly, and recommend mitigation strategies.",
]

for p in test_prompts:
    result = process_request(p)
    print(f"\nPrompt: {p}")
    print(f"  Tier: {result['predicted_tier']}, Model: {result['routed_model_key']}")
    print(f"  Escalated: {result['escalated']}")
    print(f"  Final cost: ${result['final_cost_usd']:.6f}")

print("\n" + "=" * 60)
print("SUMMARY STATS")
print("=" * 60)
stats = get_summary_stats()
print(stats)