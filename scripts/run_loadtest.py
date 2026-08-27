"""
Phase 6, Step 1: Load test — run 500 diverse prompts through the pipeline.
Designed to be resumable and gentle on free-tier rate limits.
"""

import sys, os, json, time
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.logger import init_db, get_summary_stats
from core.pipeline import process_request

PROMPTS_PATH = "data/loadtest_prompts.json"
PROGRESS_PATH = "data/loadtest_progress.json"
DELAY_SECONDS = 1.0  # gentle pacing for free-tier rate limits

init_db()


def load_progress():
    if os.path.exists(PROGRESS_PATH):
        with open(PROGRESS_PATH, "r") as f:
            return json.load(f)
    return {"completed_index": -1, "errors": []}


def save_progress(progress):
    with open(PROGRESS_PATH, "w") as f:
        json.dump(progress, f, indent=2)


def main():
    with open(PROMPTS_PATH, "r", encoding="utf-8") as f:
        prompts = json.load(f)

    progress = load_progress()
    start_index = progress["completed_index"] + 1

    if start_index > 0:
        print(f"Resuming from index {start_index} (out of {len(prompts)})")
    else:
        print(f"Starting fresh load test: {len(prompts)} prompts")

    success_count = 0
    error_count = 0

    for i in range(start_index, len(prompts)):
        prompt = prompts[i]
        try:
            result = process_request(prompt)
            status = "ESCALATED" if result["escalated"] else "ok"
            print(f"[{i+1}/{len(prompts)}] tier={result['predicted_tier']} "
                  f"model={result['routed_model_key']:20s} "
                  f"cost=${result['final_cost_usd']:.6f} [{status}]")
            success_count += 1
        except Exception as e:
            print(f"[{i+1}/{len(prompts)}] ERROR: {e}")
            progress["errors"].append({"index": i, "prompt": prompt, "error": str(e)})
            error_count += 1

        progress["completed_index"] = i
        save_progress(progress)
        time.sleep(DELAY_SECONDS)

    print("\n" + "=" * 60)
    print("LOAD TEST COMPLETE")
    print("=" * 60)
    print(f"Total processed: {len(prompts) - start_index}")
    print(f"Successes: {success_count}, Errors: {error_count}")

    stats = get_summary_stats()
    print(f"\nOverall totals (cumulative across all runs):")
    print(f"Total requests logged: {stats['total_requests']}")
    print(f"Total cost: ${stats['total_cost_usd']:.6f}")
    print(f"Escalation rate: {stats['escalation_rate']:.1%}")
    print(f"Avg quality score: {stats['avg_quality_score']:.2f}" if stats['avg_quality_score'] else "N/A")


if __name__ == "__main__":
    main()