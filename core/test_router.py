import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.router import Router

router = Router()

test_prompts = [
    "What is the capital of France?",
    "Summarize this article in 3 bullet points: [some text here]",
    "Analyze the trade-offs between microservices and monoliths, and recommend an approach for a 5-person startup.",
]

for p in test_prompts:
    decision = router.route(p)
    print(f"\nPrompt: {p}")
    print(f"  Tier: {decision['predicted_tier']} ({decision['tier_name']})")
    print(f"  Routed to: {decision['routed_model_key']} ({decision['model_config'].model_id})")