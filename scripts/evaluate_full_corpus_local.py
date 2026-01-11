#!/usr/bin/env python3
"""
Evaluate search on full Wikipedia corpus by reading directly from GCS.
This script runs locally but uses the full corpus indices from GCP bucket.
"""

import sys
import os
import json
import math
from collections import Counter

# Add backend to path
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(SCRIPT_DIR)
BACKEND_DIR = os.path.join(PROJECT_DIR, 'backend')
sys.path.insert(0, BACKEND_DIR)
sys.path.insert(0, PROJECT_DIR)

from inverted_index_gcp import InvertedIndex
from pre_processing import tokenize_and_process

# Configuration
GCS_BUCKET = "206969750_bucket"
INDEX_DIR = "postings_gcp"
N = 6000000  # Approximate Wikipedia document count

print("="*80)
print("FULL CORPUS EVALUATION (Running Locally with GCS Indices)")
print("="*80)

# Load body index from GCS
print(f"\nLoading body index from gs://{GCS_BUCKET}/{INDEX_DIR}/...")
BODY_INDEX = InvertedIndex.read_index(INDEX_DIR, 'index', GCS_BUCKET)
print(f"✓ Body index loaded: {len(BODY_INDEX.df):,} terms")

# Try to load metadata
try:
    from google.cloud import storage
    import pickle

    client = storage.Client()
    bucket = client.bucket(GCS_BUCKET)
    blob = bucket.blob('indices/metadata.pkl')
    metadata_bytes = blob.download_as_bytes()
    METADATA = pickle.loads(metadata_bytes)
    print(f"✓ Metadata loaded: {METADATA['num_docs']} documents")
    N = METADATA['num_docs']
except Exception as e:
    print(f"⚠ Could not load metadata, using default N={N}")
    METADATA = {'num_docs': N, 'doc_titles': {}, 'doc_norms': {}}

def compute_tfidf_cosine_gcs(query_tokens):
    """Compute TF-IDF cosine similarity using GCS posting files."""
    if not query_tokens:
        return {}

    # 1. Compute query TF-IDF vector
    query_counts = Counter(query_tokens)
    query_tfidf = {}

    for term, count in query_counts.items():
        if term in BODY_INDEX.df:
            df = BODY_INDEX.df[term]
            idf = math.log(N / df)
            query_tfidf[term] = count * idf

    if not query_tfidf:
        return {}

    query_norm = math.sqrt(sum(v**2 for v in query_tfidf.values()))
    if query_norm == 0:
        return {}

    # 2. Get candidates from GCS postings
    candidates = {}
    for term in query_tfidf.keys():
        try:
            posting_list = BODY_INDEX.read_a_posting_list(INDEX_DIR, term, GCS_BUCKET)
            for doc_id, tf in posting_list:
                if doc_id not in candidates:
                    candidates[doc_id] = {}
                candidates[doc_id][term] = tf
        except:
            continue

    # 3. Compute cosine similarity
    scores = {}
    for doc_id, doc_terms in candidates.items():
        dot_product = 0.0
        doc_tfidf_squared = 0.0

        for term, tf in doc_terms.items():
            if term in query_tfidf:
                df = BODY_INDEX.df[term]
                idf = math.log(N / df)
                doc_tfidf = tf * idf
                dot_product += query_tfidf[term] * doc_tfidf
                doc_tfidf_squared += doc_tfidf ** 2

        doc_norm = METADATA['doc_norms'].get(doc_id, math.sqrt(doc_tfidf_squared))

        if doc_norm > 0:
            scores[doc_id] = dot_product / (query_norm * doc_norm)

    return scores

def average_precision_at_k(relevant_docs, retrieved_docs, k=10):
    """Compute Average Precision at K."""
    if not relevant_docs:
        return 0.0

    retrieved_docs = retrieved_docs[:k]
    num_relevant = 0
    precision_sum = 0.0

    for i, doc_id in enumerate(retrieved_docs, 1):
        if str(doc_id) in relevant_docs:
            num_relevant += 1
            precision_at_i = num_relevant / i
            precision_sum += precision_at_i

    if num_relevant == 0:
        return 0.0

    ap = precision_sum / min(len(relevant_docs), k)
    return ap

# Load queries
print("\nLoading queries...")
queries_path = os.path.join(PROJECT_DIR, 'data', 'queries_train.json')
with open(queries_path, 'r') as f:
    queries = json.load(f)
print(f"✓ Loaded {len(queries)} queries")

print("\n" + "="*80)
print("EVALUATION IN PROGRESS")
print("="*80)

results = {
    'map_at_k': 0.0,
    'k': 10,
    'num_queries': len(queries),
    'query_scores': []
}

total_ap = 0.0

for i, (query_text, relevant_docs) in enumerate(queries.items(), 1):
    print(f"\n[{i}/{len(queries)}] {query_text}")

    # Tokenize
    query_tokens = tokenize_and_process(query_text, remove_stops=True, stem=True)
    print(f"  Tokens: {query_tokens}")

    # Search
    scores = compute_tfidf_cosine_gcs(query_tokens)

    # Sort results
    sorted_docs = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    retrieved_ids = [str(doc_id) for doc_id, _ in sorted_docs[:10]]

    # Compute AP@10
    relevant_set = set(str(doc_id) for doc_id in relevant_docs)
    ap_at_10 = average_precision_at_k(relevant_set, retrieved_ids, k=10)

    total_ap += ap_at_10

    print(f"  Retrieved: {len(retrieved_ids)} docs")
    print(f"  AP@10: {ap_at_10:.4f}")

    if ap_at_10 == 0.0:
        print(f"  ⚠️  No relevant docs in top-10")
    elif ap_at_10 >= 0.3:
        print(f"  ✓ High precision!")

    results['query_scores'].append({
        'query': query_text,
        'ap_at_k': ap_at_10,
        'num_retrieved': len(retrieved_ids)
    })

# Compute MAP@10
map_at_10 = total_ap / len(queries)
results['map_at_k'] = map_at_10

# Summary
print("\n" + "="*80)
print("EVALUATION COMPLETE")
print("="*80)
print(f"\nFull Wikipedia Corpus:")
print(f"  Terms in vocabulary: {len(BODY_INDEX.df):,}")
print(f"  Documents: ~{N:,}")
print(f"\nResults:")
print(f"  Queries evaluated: {len(queries)}")
print(f"  MAP@10: {map_at_10:.4f}")

zero_ap = sum(1 for q in results['query_scores'] if q['ap_at_k'] == 0.0)
high_ap = sum(1 for q in results['query_scores'] if q['ap_at_k'] >= 0.3)

print(f"\nQuery Distribution:")
print(f"  Zero AP@10: {zero_ap}/{len(queries)} ({zero_ap/len(queries)*100:.1f}%)")
print(f"  High AP@10 (≥0.3): {high_ap}/{len(queries)} ({high_ap/len(queries)*100:.1f}%)")

# Save results
output_file = 'evaluation_results_full_corpus.json'
with open(output_file, 'w') as f:
    json.dump(results, f, indent=2)

print(f"\n✓ Results saved to: {output_file}")

# Check threshold
threshold = 0.1
if map_at_10 >= threshold:
    print(f"\n✅ SUCCESS: MAP@10 = {map_at_10:.4f} ≥ {threshold}")
else:
    print(f"\n❌ FAILED: MAP@10 = {map_at_10:.4f} < {threshold}")

print("\n" + "="*80)
