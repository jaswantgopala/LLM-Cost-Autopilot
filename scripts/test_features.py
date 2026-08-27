import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.complexity import extract_features, features_to_vector

test_prompts = [
    "What is the capital of France?",
    "Summarize this article in 3 bullet points.",
    "Analyze the trade-offs between microservices and monoliths, and recommend an approach for a 5-person startup.",
]

for p in test_prompts:
    f = extract_features(p)
    print(f"\nPrompt: {p}")
    print(f"Features: {f}")
    print(f"Vector: {features_to_vector(f)}")