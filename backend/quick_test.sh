#!/bin/bash
# Quick test commands for the search engine

echo "Testing Search Engine Setup"
echo "=============================="
echo ""

echo "Test 1: Check imports"
python3 -c "from search_runtime import SearchEngine; print('✓ Imports OK')" || exit 1

echo ""
echo "Test 2: Initialize engine and check posting list"
python3 << 'EOF'
from search_runtime import initialize_engine

# Initialize
engine = initialize_engine('indices_mini', '206969750_bucket')
print(f"✓ Engine initialized")
print(f"  Body terms: {len(engine.body_index.df):,}")
print(f"  Title terms: {len(engine.title_index.df):,}")
print(f"  Anchor terms: {len(engine.anchor_index.df):,}")

# Test reading a posting list
if engine.body_index.posting_locs:
    test_term = list(engine.body_index.posting_locs.keys())[0]
    print(f"\n  Testing term: '{test_term}'")

    posting_list = engine.body_index.read_a_posting_list(
        '.',  # GCS root
        test_term,
        bucket_name='206969750_bucket'
    )

    if posting_list:
        print(f"  ✓ Read {len(posting_list)} postings from GCS")
    else:
        print(f"  ✗ Empty posting list")
else:
    print("  ✗ No terms in index")
EOF

echo ""
echo "Test 3: Search query"
python3 << 'EOF'
from search_runtime import get_engine

engine = get_engine()
results = engine.search("python", top_n=5)

if results:
    print(f"✓ Search returned {len(results)} results")
    for i, (doc_id, title, score) in enumerate(results[:3], 1):
        print(f"  {i}. [{doc_id}] {title[:50]}...")
else:
    print("⚠ No results (check if indices have data)")
EOF

echo ""
echo "=============================="
echo "All tests completed!"
echo "Start server with: python3 search_frontend.py"
