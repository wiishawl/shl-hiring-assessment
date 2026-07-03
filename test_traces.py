"""
Test harness: replays each trace's real user messages against your OWN
running /chat endpoint (http://127.0.0.1:8000/chat), letting YOUR agent
generate its own replies at each turn (not the reference agent's replies).
After the full conversation, compares your agent's FINAL recommendations
against the trace's labeled ground-truth shortlist and computes Recall@10.

IMPORTANT: make sure your FastAPI server is running (uvicorn main:app)
in another terminal before running this script.
"""
import json
import requests

API_URL = "http://127.0.0.1:8000/chat"


def normalize(name):
    return name.strip().lower()


import time

def run_trace(trace):
    messages = []
    final_recommendations = []

    for user_msg in trace['user_turns']:
        messages.append({"role": "user", "content": user_msg})
        try:
            resp = requests.post(API_URL, json={"messages": messages}, timeout=32)
            resp.raise_for_status()
            data = resp.json()
        except Exception as e:
            print(f"    ERROR calling /chat: {e}")
            break

        messages.append({"role": "assistant", "content": data.get("reply", "")})
        if data.get("recommendations"):
            final_recommendations = data["recommendations"]
        time.sleep(2)  # small pause between turns to ease per-minute rate limit pressure during testing

    return final_recommendations, messages


def compute_recall(predicted, ground_truth, k=10):
    predicted_top_k = predicted[:k]
    predicted_names = {normalize(p['name']) for p in predicted_top_k}
    gt_names = {normalize(g['name']) for g in ground_truth}
    if not gt_names:
        return None
    hit = len(predicted_names & gt_names)
    return hit / len(gt_names), hit, len(gt_names)


def main():
    with open('traces_parsed.json', encoding='utf-8') as f:
        traces = json.load(f)

    recalls = []
    print("Checking server is up...")
    try:
        health = requests.get("http://127.0.0.1:8000/health", timeout=5)
        print("  /health:", health.json())
    except Exception as e:
        print(f"  Could not reach server: {e}")
        print("  Make sure `uvicorn main:app` is running in another terminal.")
        return

    for trace in traces:
        print(f"\n=== {trace['trace_id']} ({len(trace['user_turns'])} turns) ===")
        predicted, full_convo = run_trace(trace)
        recall, hit, total = compute_recall(predicted, trace['ground_truth'])

        print(f"  Ground truth ({total}): {[g['name'] for g in trace['ground_truth']]}")
        print(f"  Predicted ({len(predicted)}): {[p['name'] for p in predicted]}")
        if recall is not None:
            print(f"  Recall@10: {recall:.3f}  ({hit}/{total} found)")
            recalls.append(recall)
        else:
            print("  Skipped (no ground truth)")
        time.sleep(3)  # pause between traces too

    if recalls:
        mean_recall = sum(recalls) / len(recalls)
        print(f"\n{'='*50}")
        print(f"MEAN RECALL@10 across {len(recalls)} traces: {mean_recall:.3f}")
        print(f"{'='*50}")


if __name__ == '__main__':
    main()