"""
Test script for diagnostic search module.

This script tests the diagnostic search functions on a single query
to verify the instrumentation is working correctly.
"""

import sys
import json
import pickle
from pathlib import Path

# Add both backend and parent directory to path
# This is needed for pickle to find the backend module
parent_dir = Path(__file__).parent.parent
backend_path = parent_dir / 'backend'
sys.path.insert(0, str(parent_dir))
sys.path.insert(0, str(backend_path))

# Import backend modules
from backend.inverted_index_gcp import InvertedIndex
from diagnostic_search import search_with_full_diagnostics


def load_indices(index_dir):
    """Load all indices and metadata."""
    print(f"Loading indices from {index_dir}...")

    # Load inverted indices
    body_index = InvertedIndex.read_index(index_dir, 'body_index')
    print(f"✓ Body index loaded: {len(body_index.df)} terms")

    title_index = InvertedIndex.read_index(index_dir, 'title_index')
    print(f"✓ Title index loaded: {len(title_index.df)} terms")

    anchor_index = InvertedIndex.read_index(index_dir, 'anchor_index')
    print(f"✓ Anchor index loaded: {len(anchor_index.df)} terms")

    # Load metadata
    metadata_path = Path(index_dir) / 'metadata.pkl'
    with open(metadata_path, 'rb') as f:
        metadata = pickle.load(f)
    print(f"✓ Metadata loaded: {metadata['num_docs']} documents")

    # Load PageRank
    try:
        pagerank_path = Path(index_dir) / 'pagerank.pkl'
        with open(pagerank_path, 'rb') as f:
            pagerank = pickle.load(f)
        print(f"✓ PageRank loaded: {len(pagerank)} scores")
    except FileNotFoundError:
        print("⚠ PageRank not found")
        pagerank = {}

    # Load page views
    try:
        pageviews_path = Path(index_dir) / 'pageviews.pkl'
        with open(pageviews_path, 'rb') as f:
            pageviews = pickle.load(f)
        print(f"✓ Page views loaded: {len(pageviews)} entries")
    except FileNotFoundError:
        print("⚠ Page views not found")
        pageviews = {}

    return body_index, title_index, anchor_index, metadata, pagerank, pageviews


def main():
    """Test diagnostic search on a single query."""
    # Configuration
    index_dir = Path(__file__).parent.parent / 'indices_mini'

    # Test query (one that performs well)
    test_query = "DNA double helix discovery Watson Crick"

    print("\n" + "="*70)
    print("DIAGNOSTIC SEARCH TEST")
    print("="*70)
    print(f"Query: {test_query}")
    print()

    # Load indices
    body_index, title_index, anchor_index, metadata, pagerank, pageviews = load_indices(index_dir)

    print("\n" + "-"*70)
    print("Running diagnostic search...")
    print("-"*70)

    # Run diagnostic search
    report = search_with_full_diagnostics(
        query=test_query,
        body_index=body_index,
        title_index=title_index,
        anchor_index=anchor_index,
        index_dir=str(index_dir),
        metadata=metadata,
        pagerank=pagerank,
        pageviews=pageviews,
        w_body=0.40,
        w_title=0.25,
        w_anchor=0.15,
        w_pagerank=0.15,
        w_pageviews=0.05,
        remove_stops=True,
        stem=True
    )

    # Display summary results
    print("\n" + "="*70)
    print("DIAGNOSTIC REPORT SUMMARY")
    print("="*70)

    print("\n[Query Info]")
    print(f"  Original: {report['query_info']['original_query']}")
    print(f"  Tokens: {report['query_info']['tokens']}")
    print(f"  Unique terms: {report['query_info']['unique_terms']}")

    print("\n[Search Summary]")
    print(f"  Total results: {report['summary']['num_results']}")
    print(f"  Max score: {report['summary']['max_score']:.4f}")
    print(f"  Avg score: {report['summary']['avg_score']:.4f}")

    print("\n[Field Matches]")
    field_summary = report['field_diagnostics']['summary']
    print(f"  Body matches: {field_summary.get('num_body_matches', 0)}")
    print(f"  Title matches: {field_summary.get('num_title_matches', 0)}")
    print(f"  Anchor matches: {field_summary.get('num_anchor_matches', 0)}")

    print("\n[Top 5 Results]")
    for result in report['ranking']['top_100'][:5]:
        print(f"  {result['rank']}. [{result['doc_id']}] {result['title'][:60]}... (score: {result['score']:.4f})")

    # Show term-level diagnostics
    if 'body_diagnostics' in report['field_diagnostics']:
        body_diag = report['field_diagnostics']['body_diagnostics']
        if 'term_stats' in body_diag:
            print("\n[Term Statistics]")
            for term, stats in body_diag['term_stats'].items():
                if stats['in_vocabulary']:
                    print(f"  {term:15s} | DF: {stats['df']:6d} | IDF: {stats['idf']:.3f} | Matches: {stats['num_matched_docs']:4d}")
                else:
                    print(f"  {term:15s} | OOV (out of vocabulary)")

    # Save full report
    output_path = Path(__file__).parent / 'test_diagnostic_output.json'
    with open(output_path, 'w') as f:
        # Convert to JSON-serializable format
        json.dump(report, f, indent=2, default=str)
    print(f"\n✓ Full diagnostic report saved to: {output_path}")

    print("\n" + "="*70)
    print("TEST COMPLETED SUCCESSFULLY!")
    print("="*70)


if __name__ == '__main__':
    main()
