import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.router import Router
from core.verifier import verify_response, escalate, QUALITY_THRESHOLD
from core.interface import send_request

router = Router()

test_prompt = "What is the capital of France?"

# Step 1: route + get the actual response from the routed (cheap) model
decision = router.route(test_prompt)
model_config = decision["model_config"]
original_response = send_request(test_prompt, model_config)

print(f"Prompt: {test_prompt}")
print(f"Routed to: {decision['routed_model_key']}")
print(f"Response: {original_response.text}")
print(f"Cost: ${original_response.cost_usd:.6f}")

# Step 2: verify
verification = verify_response(test_prompt, original_response, decision["routed_model_key"])
print(f"\nQuality score: {verification.quality_score}/5")
print(f"Reasoning: {verification.reasoning}")
print(f"Passed (>= {QUALITY_THRESHOLD}): {verification.passed}")

# Step 3: escalate if needed
if not verification.passed:
    print("\n--- ESCALATING ---")
    verification, escalated_response = escalate(verification)
    cost_delta = escalated_response.cost_usd - original_response.cost_usd
    print(f"Escalated to: {verification.escalated_model_key}")
    print(f"Escalated response: {verification.escalated_response}")
    print(f"Cost delta: ${cost_delta:.6f}")


# Test 2: force a harder prompt that a tiny model might get wrong,
# to confirm the escalation path actually works
print("\n" + "=" * 60)
print("Test 2: Escalation path")
print("=" * 60)

hard_prompt = "A farmer has 17 sheep. All but 9 die. Then he buys triple the number of sheep that survived, then sells 8. How many sheep does he have now?"

decision2 = router.route(hard_prompt)
model_config2 = decision2["model_config"]
original_response2 = send_request(hard_prompt, model_config2)

print(f"Prompt: {hard_prompt}")
print(f"Routed to: {decision2['routed_model_key']}")
print(f"Response: {original_response2.text}")

verification2 = verify_response(hard_prompt, original_response2, decision2["routed_model_key"])
print(f"\nQuality score: {verification2.quality_score}/5")
print(f"Reasoning: {verification2.reasoning}")
print(f"Passed: {verification2.passed}")

if not verification2.passed:
    print("\n--- ESCALATING ---")
    verification2, escalated_response2 = escalate(verification2)
    cost_delta2 = escalated_response2.cost_usd - original_response2.cost_usd
    print(f"Escalated to: {verification2.escalated_model_key}")
    print(f"Escalated response: {verification2.escalated_response}")
    print(f"Cost delta: ${cost_delta2:.6f}")
else:
    print("(No escalation triggered — model answered well enough)")    