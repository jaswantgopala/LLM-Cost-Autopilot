"""
Batch runner: sends a realistic mix of prompts through the full pipeline
to populate the dashboard with meaningful data.
"""

import sys, os, time
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.logger import init_db, get_summary_stats
from core.pipeline import process_request

init_db()

# Realistic mix: ~50% simple, ~30% moderate, ~20% complex
simple_prompts = [
    "What is the capital of Japan?",
    "Extract the email from: 'Reach me at anna@example.com'",
    "Translate 'thank you' to French.",
    "What is 25% of 400?",
    "List 3 primary colors.",
    "Define the word 'ubiquitous'.",
    "Convert 10 kg to pounds.",
    "What year did the Berlin Wall fall?",
    "What is the plural of 'cactus'?",
    "Give a synonym for 'happy'.",
    "What is the chemical symbol for silver?",
    "Extract the phone number from: 'Call 555-987-6543'",
    "What is the opposite of 'ancient'?",
    "How many continents are there?",
    "What is the square root of 144?",
]

moderate_prompts = [
    "Summarize this in one sentence: Remote work has increased productivity for many companies but created challenges around team cohesion and onboarding.",
    "Classify this review as positive or negative: 'The food was cold and the service was slow.'",
    "Categorize these items into fruits or vegetables: apple, carrot, banana, spinach",
    "Identify the main theme of this sentence: 'The company's stock fell after missing earnings expectations.'",
    "Outline the steps to set up a home Wi-Fi network.",
    "Summarize the difference between TCP and UDP in 2 sentences.",
    "Classify this email as spam or not spam: 'You've won a free vacation, click now!'",
    "Identify the sentiment of this tweet: 'This new phone update completely ruined my battery life.'",
    "Outline a simple weekly meal plan for a vegetarian diet.",
]

complex_prompts = [
    "Analyze the trade-offs between building vs buying software for a mid-size company, and give a recommendation.",
    "Write a short story about a robot learning to paint.",
    "Debate the ethical implications of AI-generated art replacing human artists.",
    "Design a basic system architecture for a real-time chat application, considering scalability.",
    "Evaluate the risks of a startup relying on a single cloud provider, and suggest mitigation strategies.",
]

all_prompts = simple_prompts + moderate_prompts + complex_prompts

print(f"Running {len(all_prompts)} prompts through the pipeline...\n")

for i, prompt in enumerate(all_prompts, 1):
    try:
        result = process_request(prompt)
        status = "ESCALATED" if result["escalated"] else "ok"
        print(f"[{i}/{len(all_prompts)}] tier={result['predicted_tier']} "
              f"model={result['routed_model_key']:20s} "
              f"cost=${result['final_cost_usd']:.6f} [{status}]")
    except Exception as e:
        print(f"[{i}/{len(all_prompts)}] ERROR: {e}")
    time.sleep(0.5)  # small pause to be gentle on free-tier rate limits

print("\n" + "=" * 60)
print("FINAL SUMMARY")
print("=" * 60)
stats = get_summary_stats()
print(f"Total requests: {stats['total_requests']}")
print(f"Total cost: ${stats['total_cost_usd']:.6f}")
print(f"Escalation rate: {stats['escalation_rate']:.1%}")
print(f"Avg quality score: {stats['avg_quality_score']:.2f}" if stats['avg_quality_score'] else "N/A")
print("\nBy model:")
for m in stats["by_model"]:
    print(f"  {m['routed_model_key']:20s} count={m['count']:3d} cost=${m['total_cost']:.6f}")