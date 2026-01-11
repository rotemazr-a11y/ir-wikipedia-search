#!/usr/bin/env python3
"""
Deep dive analysis of a single query to understand retrieval behavior.
Shows detailed scoring breakdown for each retrieved document.
"""

import sys
import os

# Add backend to path
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(SCRIPT_DIR)
BACKEND_DIR = os.path.join(PROJECT_DIR, 'backend')
sys.path.insert(0, BACKEND_DIR)
sys.path.insert(0, PROJECT_DIR)

import search_frontend
from pre_processing import tokenize_and_process
import json
import math

# Configuration
INDEX_DIR = os.path.join(PROJECT_DIR, 'indices_mini')
N = 42

# Load indices
print("Loading indices...")
search_frontend.load_indices(INDEX_DIR)
print()

# Get references
body_index = search_frontend.BODY_INDEX
title_index = search_frontend.TITLE_INDEX
anchor_index = search_frontend.ANCHOR_INDEX
METADATA = search_frontend.METADATA

def compute_body_tfidf_verbose(query_tokens, doc_id):
    """Compute body TF-IDF for a document with detailed breakdown."""
    doc_vector = {}
    query_vector = {}

    print(f"\n  Body TF-IDF Breakdown for Doc {doc_id}:")

    for term in query_tokens:
        # Compute IDF on the fly (same as production)
        if term in body_index.df:
            df = body_index.df[term]
            idf = math.log(N / df)  # Natural log, same as production
            query_idf = idf
            query_vector[term] = query_idf
            print(f"    '{term}': DF={df}, IDF={idf:.3f}", end="")

            # Document TF-IDF
            posting_list = body_index.read_a_posting_list(INDEX_DIR, term)
            doc_tf_idf = None
            for doc, tf in posting_list:
                if doc == doc_id:
                    doc_tf_idf = tf * idf
                    doc_vector[term] = doc_tf_idf
                    break

            if doc_tf_idf:
                print(f", Doc TF-IDF={doc_tf_idf:.3f} ✓")
            else:
                print(f", NOT in doc ✗")
        else:
            print(f"    '{term}': NOT in vocabulary ✗")

    # Compute cosine similarity
    if not doc_vector:
        return 0.0

    dot_product = sum(doc_vector.values())
    query_norm = math.sqrt(sum(v**2 for v in query_vector.values()))

    # Get document norm (from metadata)
    doc_norm = 1.0
    if 'doc_norms' in METADATA and doc_id in METADATA['doc_norms']:
        doc_norm = METADATA['doc_norms'][doc_id]

    if query_norm == 0 or doc_norm == 0:
        return 0.0

    cosine_sim = dot_product / (query_norm * doc_norm)

    print(f"  → Cosine similarity: {cosine_sim:.4f}")
    print(f"     (dot={dot_product:.3f}, q_norm={query_norm:.3f}, d_norm={doc_norm:.3f})")

    return cosine_sim

def check_title_match(query_tokens, doc_id):
    """Check if query terms appear in document title."""
    print(f"\n  Title Matching for Doc {doc_id}:")

    matches = []
    for term in query_tokens:
        if term in title_index.df:
            posting_list = title_index.read_a_posting_list(INDEX_DIR, term)
            doc_ids = [doc for doc, _ in posting_list]
            if doc_id in doc_ids:
                matches.append(term)
                print(f"    '{term}': IN TITLE ✓")
            else:
                print(f"    '{term}': not in title")
        else:
            print(f"    '{term}': not in title index")

    match_ratio = len(matches) / len(query_tokens) if query_tokens else 0
    print(f"  → Title match score: {match_ratio:.3f} ({len(matches)}/{len(query_tokens)} terms)")

    return match_ratio

def check_anchor_match(query_tokens, doc_id):
    """Check if query terms appear in anchor text."""
    print(f"\n  Anchor Text Matching for Doc {doc_id}:")

    matches = []
    for term in query_tokens:
        if term in anchor_index.df:
            posting_list = anchor_index.read_a_posting_list(INDEX_DIR, term)
            doc_ids = [doc for doc, _ in posting_list]
            if doc_id in doc_ids:
                matches.append(term)
                print(f"    '{term}': IN ANCHORS ✓")
            else:
                print(f"    '{term}': not in anchors")
        else:
            print(f"    '{term}': not in anchor index")

    match_ratio = len(matches) / len(query_tokens) if query_tokens else 0
    print(f"  → Anchor match score: {match_ratio:.3f} ({len(matches)}/{len(query_tokens)} terms)")

    return match_ratio

def analyze_query_detailed(query_text, top_k=5):
    """Detailed analysis of a single query."""
    print("="*80)
    print(f"DEEP DIVE: {query_text}")
    print("="*80)

    # Tokenize
    query_tokens = tokenize_and_process(query_text, remove_stops=True, stem=True)
    print(f"\nTokenized: {query_tokens}")
    print(f"Number of terms: {len(query_tokens)}")

    # Check vocabulary
    print("\n" + "-"*80)
    print("VOCABULARY CHECK")
    print("-"*80)

    for term in query_tokens:
        in_body = term in body_index.df
        in_title = term in title_index.df
        in_anchor = term in anchor_index.df

        if in_body:
            df = body_index.df[term]
            idf = math.log(N / df)  # Compute IDF same as production
            print(f"  ✓ '{term}': DF={df}, IDF={idf:.3f}")
        else:
            print(f"  ✗ '{term}': OUT OF VOCABULARY")

    # Get production search results
    print("\n" + "-"*80)
    print("PRODUCTION SEARCH RESULTS")
    print("-"*80)

    final_scores = search_frontend.multi_field_fusion(
        query_tokens, INDEX_DIR, N, METADATA,
        w_body=0.40, w_title=0.25, w_anchor=0.15,
        w_pagerank=0.15, w_pageviews=0.05
    )

    sorted_results = sorted(final_scores.items(), key=lambda x: x[1], reverse=True)[:top_k]

    print(f"\nTop {top_k} results:")
    for rank, (doc_id, score) in enumerate(sorted_results, 1):
        title = METADATA['doc_titles'].get(doc_id, "Unknown")
        print(f"\n{rank}. Doc {doc_id} (Score: {score:.4f})")
        print(f"   Title: {title}")

    # Detailed breakdown for each top result
    print("\n" + "="*80)
    print("DETAILED SCORING BREAKDOWN")
    print("="*80)

    for rank, (doc_id, final_score) in enumerate(sorted_results, 1):
        title = METADATA['doc_titles'].get(doc_id, "Unknown")

        print(f"\n{'='*80}")
        print(f"Rank #{rank}: Doc {doc_id} - {title}")
        print(f"Final Score: {final_score:.4f}")
        print('='*80)

        # Body score
        body_score = compute_body_tfidf_verbose(query_tokens, doc_id)

        # Title score
        title_score = check_title_match(query_tokens, doc_id)

        # Anchor score
        anchor_score = check_anchor_match(query_tokens, doc_id)

        # PageRank and PageViews
        pagerank = search_frontend.PAGERANK.get(doc_id, 0) if search_frontend.PAGERANK else 0
        pageviews = search_frontend.PAGEVIEWS.get(doc_id, 0) if search_frontend.PAGEVIEWS else 0

        print(f"\n  PageRank: {pagerank}")
        print(f"  PageViews: {pageviews}")

        # Compute expected final score
        expected_score = (
            0.40 * body_score +
            0.25 * title_score +
            0.15 * anchor_score +
            0.15 * pagerank +
            0.05 * pageviews
        )

        print(f"\n  WEIGHTED SCORE BREAKDOWN:")
        print(f"    Body (0.40):     {0.40 * body_score:.4f}  ({body_score:.4f})")
        print(f"    Title (0.25):    {0.25 * title_score:.4f}  ({title_score:.4f})")
        print(f"    Anchor (0.15):   {0.15 * anchor_score:.4f}  ({anchor_score:.4f})")
        print(f"    PageRank (0.15): {0.15 * pagerank:.4f}  ({pagerank:.4f})")
        print(f"    PageViews (0.05):{0.05 * pageviews:.4f}  ({pageviews:.4f})")
        print(f"    ──────────────────────────")
        print(f"    Expected Total:  {expected_score:.4f}")
        print(f"    Actual Total:    {final_score:.4f}")

        if abs(expected_score - final_score) > 0.001:
            print(f"    ⚠ Discrepancy: {abs(expected_score - final_score):.4f}")

    # Load ground truth
    print("\n" + "="*80)
    print("GROUND TRUTH COMPARISON")
    print("="*80)

    with open(os.path.join(PROJECT_DIR, 'data', 'queries_train.json'), 'r') as f:
        queries_train = json.load(f)

    if query_text in queries_train:
        relevant_docs = set(queries_train[query_text])
        retrieved_docs = set(str(doc_id) for doc_id, _ in sorted_results)

        print(f"\nRelevant documents (total {len(relevant_docs)}):")
        print(f"  Sample: {list(relevant_docs)[:10]}")

        print(f"\nRetrieved documents:")
        print(f"  {list(retrieved_docs)}")

        overlap = relevant_docs & retrieved_docs
        print(f"\nOverlap: {len(overlap)}/{len(relevant_docs)}")
        if overlap:
            print(f"  Relevant docs found: {overlap}")
        else:
            print(f"  ✗ NO RELEVANT DOCUMENTS IN TOP {top_k}")

            # Check if ANY relevant docs are in corpus
            corpus_docs = set(str(doc_id) for doc_id in METADATA['doc_titles'].keys())
            relevant_in_corpus = relevant_docs & corpus_docs

            print(f"\nRelevant docs in entire corpus: {len(relevant_in_corpus)}/{len(relevant_docs)}")
            if relevant_in_corpus:
                print(f"  Docs: {relevant_in_corpus}")
            else:
                print(f"  ✗ NONE OF THE RELEVANT DOCUMENTS ARE IN THE CORPUS")
                print(f"     This query CANNOT achieve AP@10 > 0 with current corpus!")

if __name__ == "__main__":
    # Query to analyze
    query = "Mount Everest climbing expeditions"

    if len(sys.argv) > 1:
        query = " ".join(sys.argv[1:])

    analyze_query_detailed(query, top_k=5)
