#!/usr/bin/env python3
"""
Deep analysis of query results to understand MAP@10 performance
"""
import json
from search_runtime import initialize_engine, get_engine
from pre_processing import tokenize_and_process

print("Initializing engine...")
initialize_engine()
engine = get_engine()

# Load queries
with open('queries_train.json', 'r') as f:
    queries = json.load(f)

print("\n" + "="*80)
print("DETAILED QUERY ANALYSIS")
print("="*80)

# Analyze first 5 queries in detail
for idx, (query_text, true_docs) in enumerate(list(queries.items())[:5]):
    print(f"\n{'='*80}")
    print(f"Query #{idx+1}: {query_text}")
    print(f"Expected docs: {true_docs[:3]}")  # First 3 expected
    print(f"{'='*80}")
    
    # Tokenize
    tokens_body = tokenize_and_process(query_text, remove_stops=True, stem=True)
    tokens_title = tokenize_and_process(query_text, remove_stops=True, stem=False)
    
    print(f"Body tokens:  {tokens_body}")
    print(f"Title tokens: {tokens_title}")
    
    # Get component scores
    body_scores = engine.search_body_bm25(tokens_body)
    title_scores = engine.search_title(tokens_title)
    
    print(f"\nBody matches:  {len(body_scores)}")
    print(f"Title matches: {len(title_scores)}")
    
    # Check if expected docs are found
    true_docs_set = set(str(d) for d in true_docs)
    
    found_in_title = []
    found_in_body = []
    
    for doc_str in true_docs_set:
        doc_id = int(doc_str)
        if doc_id in title_scores:
            found_in_title.append((doc_id, title_scores[doc_id]))
        if doc_id in body_scores:
            found_in_body.append((doc_id, body_scores[doc_id]))
    
    print(f"\nExpected docs found in TITLE: {len(found_in_title)}/{len(true_docs)}")
    for doc_id, score in found_in_title[:3]:
        title = engine.doc_titles.get(doc_id, "NO TITLE")
        print(f"  Doc {doc_id} (score={score:.2f}): {title[:80]}")
    
    print(f"\nExpected docs found in BODY: {len(found_in_body)}/{len(true_docs)}")
    for doc_id, score in found_in_body[:3]:
        title = engine.doc_titles.get(doc_id, "NO TITLE")
        print(f"  Doc {doc_id} (score={score:.2f}): {title[:80]}")
    
    # Get final results
    results = engine.search(query_text, top_n=10)
    
    print(f"\nTop 10 Results:")
    for rank, (doc_id, title, score) in enumerate(results, 1):
        is_relevant = "✓ RELEVANT" if str(doc_id) in true_docs_set else "✗ Not relevant"
        print(f"  {rank}. [{score:.2f}] {is_relevant}")
        print(f"      {title[:70]}")
    
    # Calculate precision@10 for this query
    relevant_in_top10 = sum(1 for doc_id, _, _ in results if str(doc_id) in true_docs_set)
    print(f"\n📊 Relevant in top 10: {relevant_in_top10}/10")
    print(f"📊 Precision@10: {relevant_in_top10/10:.2%}")

print("\n" + "="*80)
print("SUMMARY")
print("="*80)

# Quick statistics
title_coverage = 0
body_coverage = 0
total_queries = 0

for query_text, true_docs in queries.items():
    tokens_title = tokenize_and_process(query_text, remove_stops=True, stem=False)
    title_scores = engine.search_title(tokens_title)
    
    true_docs_set = set(int(d) for d in true_docs)
    
    # Count how many expected docs appear in title index
    found_in_title = sum(1 for doc_id in true_docs_set if doc_id in title_scores)
    
    if found_in_title > 0:
        title_coverage += 1
    
    total_queries += 1

print(f"Queries with at least 1 relevant doc in title index: {title_coverage}/{total_queries} ({title_coverage/total_queries*100:.1f}%)")
print(f"\nThis indicates the main bottleneck for MAP@10 improvement.")