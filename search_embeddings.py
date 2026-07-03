"""
Loads the Gemini embedding index and lets us search it, using the same
test queries we ran against the TF-IDF baseline for a direct comparison.
"""
import os
import pickle
import numpy as np
from dotenv import load_dotenv
from google import genai
from google.genai import types

load_dotenv()
client = genai.Client(api_key=os.environ["GEMINI_API_KEY"])
MODEL = "gemini-embedding-001"

with open('embedding_index.pkl', 'rb') as f:
    data = pickle.load(f)

vectors = data['vectors']  # shape (377, 768)
catalog = data['catalog']

# Normalize catalog vectors once, so we can use dot product as cosine similarity
norms = np.linalg.norm(vectors, axis=1, keepdims=True)
normalized_vectors = vectors / norms


def embed_query(text):
    result = client.models.embed_content(
        model=MODEL,
        contents=text,
        config=types.EmbedContentConfig(
            task_type="RETRIEVAL_QUERY",  # different task type for queries vs documents
            output_dimensionality=768,
        ),
    )
    vec = np.array(result.embeddings[0].values, dtype=np.float32)
    return vec / np.linalg.norm(vec)


def search(query, top_k=10):
    query_vec = embed_query(query)
    scores = normalized_vectors @ query_vec  # cosine similarity via dot product
    top_indices = np.argsort(scores)[::-1][:top_k]
    results = []
    for idx in top_indices:
        item = catalog[idx]
        results.append({
            'name': item['name'],
            'url': item['link'],
            'test_type': item['test_type'],
            'score': round(float(scores[idx]), 4),
        })
    return results


if __name__ == '__main__':
    test_queries = [
        "Java developer with stakeholder communication skills",
        "customer service representative for contact center",
        "personality assessment for sales role",
        "entry level data entry accuracy",
    ]
    for q in test_queries:
        print(f"\n=== Query: {q!r} ===")
        for r in search(q, top_k=5):
            print(f"  {r['score']:.3f}  {r['name']}  [{','.join(r['test_type'])}]").