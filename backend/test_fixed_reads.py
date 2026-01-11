#!/usr/bin/env python3
"""
Quick test to verify title index reading works now
"""
from search_runtime import initialize_engine, get_engine
from pre_processing import tokenize_and_process

print("Initializing engine...")
initialize_engine()
engine = get_engine()

print("\n" + "="*60)
print("TESTING TITLE INDEX READS")
print("="*60)

# Test queries that should have title matches
test_queries = [
    "Stonehenge prehistoric monument",
    "Coffee history Ethiopia",
    "Photography invention Daguerre",
    "Ballet origins France Russia"
]

for query in test_queries:
    print(f"\nQuery: {query}")
    tokens = tokenize_and_process(query, remove_stops=True, stem=False)
    print(f"  Tokens: {tokens}")
    
    title_scores = engine.search_title(tokens)
    print(f"  Title matches: {len(title_scores)}")
    
    if title_scores:
        # Show top 3
        sorted_results = sorted(title_scores.items(), key=lambda x: x[1], reverse=True)[:3]
        for doc_id, score in sorted_results:
            title = engine.doc_titles.get(doc_id, f"Doc {doc_id}")
            print(f"    [{score}] {title}")
    else:
        print(f"    ⚠️  NO RESULTS")

print("\n" + "="*60)
print("FULL SEARCH TEST")
print("="*60)

query = "Stonehenge prehistoric monument"
print(f"\nQuery: {query}")
results = engine.search(query, top_n=5)

if results:
    print(f"Found {len(results)} results:")
    for doc_id, title, score in results:
        print(f"  [{score:.2f}] {title}")
else:
    print("  ⚠️  NO RESULTS")

print("\n✓ Test complete!")