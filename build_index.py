"""
Builds a TF-IDF retrieval index over the SHL Individual Test Solutions catalog.
"""
import json
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# --- Load cleaned catalog ---
with open('catalog_clean.json', encoding='utf-8') as f:
    catalog = json.load(f)

# --- Build the searchable text per item ---
def build_doc_text(item):
    name = item.get('name', '')
    desc = item.get('description', '')
    job_levels = ' '.join(item.get('job_levels', []))
    keys = ' '.join(item.get('keys', []))
    return f"{name} {name} {name} {desc} {job_levels} {keys}"

doc_texts = [build_doc_text(item) for item in catalog]

# --- Fit TF-IDF vectorizer ---
vectorizer = TfidfVectorizer(
    stop_words='english',
    ngram_range=(1, 2),
    max_features=5000,
    sublinear_tf=True,
)
doc_matrix = vectorizer.fit_transform(doc_texts)

print(f"Built TF-IDF matrix: {doc_matrix.shape[0]} items x {doc_matrix.shape[1]} features")

# --- Save everything needed for retrieval at runtime ---
with open('retrieval_index.pkl', 'wb') as f:
    pickle.dump({
        'vectorizer': vectorizer,
        'doc_matrix': doc_matrix,
        'catalog': catalog,
    }, f)

print("Saved retrieval_index.pkl")


def search(query, top_k=10):
    """Given a text query, return the top_k most relevant catalog items."""
    query_vec = vectorizer.transform([query])
    scores = cosine_similarity(query_vec, doc_matrix).flatten()
    top_indices = scores.argsort()[::-1][:top_k]
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
            print(f"  {r['score']:.3f}  {r['name']}  [{','.join(r['test_type'])}]")