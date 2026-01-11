#!/usr/bin/env python3
"""
Analyze failing queries using the PRODUCTION search endpoint.
This script directly calls the production search_frontend.py functions.
"""

import sys
import os

# Add backend to path BEFORE any imports
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(SCRIPT_DIR)
BACKEND_DIR = os.path.join(PROJECT_DIR, 'backend')
sys.path.insert(0, BACKEND_DIR)
sys.path.insert(0, PROJECT_DIR)

import search_frontend
from pre_processing import tokenize_and_process
import json

# Configuration
INDEX_DIR = os.path.join(PROJECT_DIR, 'indices_mini')
N = 42  # Total documents in mini corpus

# Load all indices using the production function
# This loads into search_frontend's global variables
print("Loading indices using production load_indices()...")
search_frontend.load_indices(INDEX_DIR)

# Get references to the loaded indices
body_index = search_frontend.BODY_INDEX
title_index = search_frontend.TITLE_INDEX
anchor_index = search_frontend.ANCHOR_INDEX
METADATA = search_frontend.METADATA

# Failing queries from evaluation
FAILING_QUERIES = [
    "Mount Everest climbing expeditions",
    "Great Fire of London 1666",
    "Robotics automation industry",
    "Wright brothers first flight",
    "Renaissance architecture Florence Italy",
    "Silk Road trade cultural exchange",
    "Green Revolution agriculture yield",
    "Roman aqueducts engineering innovation",
    "Coffee history Ethiopia trade",
    "Ballet origins France Russia"
]

def analyze_query(query_text):
    """Analyze a single query using production search."""
    print(f"\n{'='*80}")
    print(f"Query: {query_text}")
    print(f"{'='*80}")

    # Tokenize using production tokenization (same as search endpoint)
    query_tokens = tokenize_and_process(query_text, remove_stops=True, stem=True)
    print(f"\nTokenized query: {query_tokens}")
    print(f"Number of terms: {len(query_tokens)}")

    # Check which terms are in vocabulary
    print("\nTerm vocabulary check:")
    oov_terms = []
    for term in query_tokens:
        in_body = term in body_index.df
        in_title = term in title_index.df
        in_anchor = term in anchor_index.df

        if not (in_body or in_title or in_anchor):
            oov_terms.append(term)
            print(f"  ❌ '{term}' - OUT OF VOCABULARY")
        else:
            print(f"  ✓ '{term}' - Body:{in_body} Title:{in_title} Anchor:{in_anchor}")
            if in_body:
                df = body_index.df[term]
                idf = body_index.idf[term] if hasattr(body_index, 'idf') and term in body_index.idf else None
                idf_str = f"{idf:.3f}" if idf is not None else "N/A"
                print(f"     Body: DF={df}, IDF={idf_str}")

    # Get search results using production function
    final_scores = search_frontend.multi_field_fusion(
        query_tokens, INDEX_DIR, N, METADATA,
        w_body=0.40, w_title=0.25, w_anchor=0.15,
        w_pagerank=0.15, w_pageviews=0.05
    )

    # Sort and show top results
    sorted_results = sorted(final_scores.items(), key=lambda x: x[1], reverse=True)[:10]

    print(f"\nTop 10 results:")
    if not sorted_results:
        print("  ⚠️  NO RESULTS RETURNED")
    else:
        for rank, (doc_id, score) in enumerate(sorted_results, 1):
            title = METADATA.get(doc_id, {}).get('title', 'Unknown')
            print(f"  {rank}. Doc {doc_id}: {score:.4f} - {title[:60]}...")

    # Summary
    print("\nSummary:")
    print(f"  OOV terms: {len(oov_terms)}/{len(query_tokens)}")
    if oov_terms:
        print(f"  Missing: {oov_terms}")
    print(f"  Results returned: {len(sorted_results)}")
    print(f"  Top score: {sorted_results[0][1]:.4f}" if sorted_results else "  Top score: 0.0000")

    return {
        'query': query_text,
        'tokens': query_tokens,
        'oov_terms': oov_terms,
        'oov_ratio': len(oov_terms) / len(query_tokens) if query_tokens else 0,
        'num_results': len(sorted_results),
        'top_score': sorted_results[0][1] if sorted_results else 0.0,
        'top_results': sorted_results[:5]
    }

# Main analysis
print("="*80)
print("ANALYZING FAILING QUERIES (Production Search Endpoint)")
print("="*80)

all_results = []
for query in FAILING_QUERIES:
    result = analyze_query(query)
    all_results.append(result)

# Overall summary
print("\n" + "="*80)
print("OVERALL SUMMARY")
print("="*80)

total_oov = sum(len(r['oov_terms']) for r in all_results)
total_tokens = sum(len(r['tokens']) for r in all_results)
queries_with_oov = sum(1 for r in all_results if r['oov_terms'])

print(f"\nTotal OOV terms: {total_oov}/{total_tokens} ({total_oov/total_tokens*100:.1f}%)")
print(f"Queries with OOV: {queries_with_oov}/10")

print(f"\nQueries by number of results returned:")
for r in all_results:
    print(f"  {r['query'][:50]:50s} - {r['num_results']} results (top score: {r['top_score']:.4f})")

# Save detailed results
output_file = 'failing_queries_analysis.json'
with open(output_file, 'w') as f:
    json.dump(all_results, f, indent=2)

print(f"\n✓ Detailed analysis saved to: {output_file}")
