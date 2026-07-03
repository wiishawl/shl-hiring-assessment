"""
Builds a semantic (real embedding) retrieval index over the SHL catalog,
using Google's Gemini embedding API (gemini-embedding-001).

Improvements over v1:
- Automatically retries on rate-limit (429) errors with backoff
- Saves progress every 20 items to embedding_progress.pkl, so if the
  script crashes or you stop it, re-running picks up where it left off
  instead of re-embedding everything from scratch
"""
import json
import os
import time
import pickle
import numpy as np
from dotenv import load_dotenv
from google import genai
from google.genai import types
from google.genai.errors import ClientError

load_dotenv()
client = genai.Client(api_key=os.environ["GEMINI_API_KEY"])

MODEL = "gemini-embedding-001"
PROGRESS_FILE = "embedding_progress.pkl"
FINAL_FILE = "embedding_index.pkl"

with open('catalog_clean.json', encoding='utf-8') as f:
    catalog = json.load(f)


def build_doc_text(item):
    name = item.get('name', '')
    desc = item.get('description', '')
    job_levels = ' '.join(item.get('job_levels', []))
    keys = ' '.join(item.get('keys', []))
    return f"{name}. {desc} Suitable for: {job_levels}. Category: {keys}."


def embed_text_with_retry(text, task_type="RETRIEVAL_DOCUMENT", max_retries=8):
    """Call Gemini's embedding API, retrying with backoff on rate-limit errors."""
    delay = 5  # start with a 5 second wait, doubling each retry
    for attempt in range(max_retries):
        try:
            result = client.models.embed_content(
                model=MODEL,
                contents=text,
                config=types.EmbedContentConfig(
                    task_type=task_type,
                    output_dimensionality=768,
                ),
            )
            return result.embeddings[0].values
        except ClientError as e:
            if "RESOURCE_EXHAUSTED" in str(e) or "429" in str(e):
                print(f"    Rate limited, waiting {delay}s before retry ({attempt+1}/{max_retries})...")
                time.sleep(delay)
                delay = min(delay * 2, 60)  # cap backoff at 60s
            else:
                raise
    raise RuntimeError("Failed after max retries due to repeated rate limiting")


def load_progress():
    if os.path.exists(PROGRESS_FILE):
        with open(PROGRESS_FILE, 'rb') as f:
            data = pickle.load(f)
        print(f"Resuming from saved progress: {len(data['vectors'])}/{len(catalog)} already done")
        return data['vectors']
    return []


def save_progress(vectors):
    with open(PROGRESS_FILE, 'wb') as f:
        pickle.dump({'vectors': vectors}, f)


def main():
    vectors = load_progress()
    start_idx = len(vectors)

    if start_idx >= len(catalog):
        print("All items already embedded from a previous run.")
    else:
        print(f"Embedding items {start_idx} to {len(catalog)-1} with {MODEL}...")
        for i in range(start_idx, len(catalog)):
            item = catalog[i]
            text = build_doc_text(item)
            vec = embed_text_with_retry(text, task_type="RETRIEVAL_DOCUMENT")
            vectors.append(vec)

            if (i + 1) % 20 == 0 or i == len(catalog) - 1:
                save_progress(vectors)
                print(f"  {i + 1}/{len(catalog)} embedded (progress saved)")

            time.sleep(0.5)  # slower pace to respect free-tier per-minute limits

    # Finalize
    vectors_arr = np.array(vectors, dtype=np.float32)
    print("Vector matrix shape:", vectors_arr.shape)

    with open(FINAL_FILE, 'wb') as f:
        pickle.dump({
            'vectors': vectors_arr,
            'catalog': catalog,
        }, f)

    print(f"Saved {FINAL_FILE}")

    # Clean up progress file now that we're done
    if os.path.exists(PROGRESS_FILE):
        os.remove(PROGRESS_FILE)
        print("Removed temporary progress file")


if __name__ == '__main__':
    main()