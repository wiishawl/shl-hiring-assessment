"""
Multi-query retrieval over the SHL embedding index.

Instead of embedding one blended query (which drowns out weaker signals in
compound requests like "Java developer with stakeholder communication
skills"), this searches each facet of the requirement SEPARATELY, then
merges the results — boosting items that show up as relevant to more than
one facet.

This file is also structured to be imported directly by the FastAPI app
later (see `retrieve()` at the bottom).
"""
import os
import pickle
import numpy as np
from dotenv import load_dotenv
from google import genai
from google.genai import types

load_dotenv()
_api_key = os.environ.get("GEMINI_API_KEY")
if not _api_key:
    raise RuntimeError(
        "GEMINI_API_KEY environment variable is not set. Set it in your "
        ".env file locally, or in your hosting platform's environment "
        "variable settings when deployed."
    )
client = genai.Client(api_key=_api_key)
MODEL = "gemini-embedding-001"

try:
    with open('embedding_index.pkl', 'rb') as f:
        _data = pickle.load(f)
except FileNotFoundError:
    raise RuntimeError(
        "embedding_index.pkl not found. This file must be present in the "
        "working directory at startup (built by build_embeddings.py) and "
        "must be included in the deployment (not gitignored)."
    )
except Exception as e:
    raise RuntimeError(f"Failed to load embedding_index.pkl: {e}")

_vectors = _data['vectors']
_catalog = _data['catalog']
_norms = np.linalg.norm(_vectors, axis=1, keepdims=True)
_normalized_vectors = _vectors / _norms


import time
from google.genai.errors import ClientError


def _embed_query(text):
    delay = 2
    max_retries = 2
    for attempt in range(max_retries):
        try:
            result = client.models.embed_content(
                model=MODEL,
                contents=text,
                config=types.EmbedContentConfig(
                    task_type="RETRIEVAL_QUERY",
                    output_dimensionality=768,
                ),
            )
            vec = np.array(result.embeddings[0].values, dtype=np.float32)
            return vec / np.linalg.norm(vec)
        except ClientError as e:
            if ("RESOURCE_EXHAUSTED" in str(e) or "429" in str(e)) and attempt < max_retries - 1:
                time.sleep(delay)
                delay = min(delay * 2, 4)
            else:
                raise


def _single_search(query, top_k=10):
    """One embedding search. Returns list of (index, score)."""
    query_vec = _embed_query(query)
    scores = _normalized_vectors @ query_vec
    top_indices = np.argsort(scores)[::-1][:top_k]
    return [(int(idx), float(scores[idx])) for idx in top_indices]


def retrieve(facets, top_k_per_facet=10, final_top_n=10):
    """
    facets: list of strings, each representing one distinct requirement
            e.g. ["Java programming ability", "stakeholder communication skills"]
    Returns: list of catalog items (dicts) ranked by merged relevance,
             length <= final_top_n

    IMPORTANT: raw cosine similarity scores are NOT comparable across
    different facet queries (different queries have different score
    distributions -- e.g. "Java" searches score higher on average than
    "stakeholder communication" searches, simply because of how the
    catalog and embedding space are shaped, not because Java matches are
    "more relevant"). Sorting everything together by raw score would let
    a high-scoring facet crowd out a low-scoring-but-still-relevant facet
    entirely -- confirmed by testing.

    Fix: convert each facet's results to a RANK (1st place, 2nd place...)
    instead of a raw score, and reserve roughly equal slots per facet.
    Items relevant to multiple facets still get boosted to the top.
    """
    # Run one search per facet, keep results as ordered rank lists
    per_facet_results = []  # list of lists of (idx, rank, raw_score)
    for facet in facets:
        results = _single_search(facet, top_k=top_k_per_facet)
        per_facet_results.append([
            (idx, rank, score) for rank, (idx, score) in enumerate(results)
        ])

    # Track combined info per catalog item
    combined = {}  # idx -> {"best_rank": int, "hit_count": int, "facets": [...], "raw_score": float}
    for facet, results in zip(facets, per_facet_results):
        for idx, rank, score in results:
            if idx not in combined:
                combined[idx] = {"best_rank": rank, "hit_count": 1, "facets": [facet], "raw_score": score}
            else:
                combined[idx]["best_rank"] = min(combined[idx]["best_rank"], rank)
                combined[idx]["hit_count"] += 1
                combined[idx]["facets"].append(facet)
                combined[idx]["raw_score"] = max(combined[idx]["raw_score"], score)

    # Round-robin selection: guarantees every facet contributes results,
    # instead of one facet's naturally-higher scores dominating everything.
    selected_idx_order = []
    seen = set()
    max_rank = max(len(r) for r in per_facet_results)
    for rank_slot in range(max_rank):
        for results in per_facet_results:
            if rank_slot < len(results):
                idx = results[rank_slot][0]
                if idx not in seen:
                    seen.add(idx)
                    selected_idx_order.append(idx)
        if len(selected_idx_order) >= final_top_n * 2:  # gather a bit extra before final trim
            break

    # Within the round-robin pool, push multi-facet matches (hit_count > 1) to the front
    selected_idx_order.sort(key=lambda idx: (-combined[idx]["hit_count"], combined[idx]["best_rank"]))
    final_indices = selected_idx_order[:final_top_n]

    output = []
    for idx in final_indices:
        data = combined[idx]
        item = _catalog[idx]
        output.append({
            "name": item["name"],
            "url": item["link"],
            "test_type": item["test_type"],
            "score": round(data["raw_score"], 4),
            "matched_facets": data["facets"],  # useful for debugging, not part of final API output
        })
    return output


if __name__ == '__main__':
    # Test: the exact compound query that single-vector search failed on,
    # now decomposed manually into two facets.
    print("=== Multi-query: Java developer + stakeholder communication ===")
    results = retrieve([
        "Java programming ability",
        "stakeholder communication and interpersonal skills",
    ], top_k_per_facet=8, final_top_n=8)
    for r in results:
        print(f"  {r['score']:.3f}  hits={len(r['matched_facets'])}  {r['name']}  [{','.join(r['test_type'])}]  <- {r['matched_facets']}")