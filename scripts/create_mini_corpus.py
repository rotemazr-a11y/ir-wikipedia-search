"""
Mini Corpus Selection Script

This script extracts a small subset of Wikipedia document IDs from the training queries
to create a TINY test corpus (100-500 docs) for rapid local testing.

Strategy:
1. Extract top-10 relevant docs per training query
2. Add optional high-PageRank "landmark" pages
3. Save doc IDs to data/mini_corpus_doc_ids.json

Usage:
    python scripts/create_mini_corpus.py
"""

import json
from pathlib import Path
from typing import Set, List

# Landmark Wikipedia pages (high PageRank, broad topics)
# These are common articles that might appear in many queries
LANDMARK_PAGES = [
    "5043734",    # Barack Obama
    "18630637",   # World War II
    "25445",      # Paris
    "3434750",    # New York City
    "645042",     # United States
    "4913064",    # London
    "22989",      # Physics
    "9239",       # Chemistry
    "18963",      # Mathematics
    "5862",       # Biology
    "18237",      # Computer science
    "13692155",   # Earth
    "2513178",    # Sun
    "25105",      # Moon
    "4768",       # Ancient Rome
    "27785",      # Science
    "21131",      # Music
    "586",        # Art
    "19648",      # Literature
    "18611516",   # Geography
]


def load_training_queries(queries_path: str = "data/queries_train.json") -> dict:
    """Load training queries from JSON file."""
    with open(queries_path, 'r', encoding='utf-8') as f:
        return json.load(f)


def select_mini_corpus_ids(queries: dict,
                           top_k_per_query: int = 10,
                           include_landmarks: bool = True) -> Set[str]:
    """
    Select document IDs for the mini corpus.

    Parameters:
    -----------
    queries : dict
        Dictionary mapping query text to list of relevant doc IDs
    top_k_per_query : int, default=10
        Number of top relevant docs to select per query
    include_landmarks : bool, default=True
        Whether to add landmark pages

    Returns:
    --------
    set of str
        Set of document IDs to include in mini corpus
    """
    mini_corpus_ids = set()

    # Extract top-k relevant docs per query
    print(f"\nExtracting top-{top_k_per_query} docs per query...")
    for query_text, relevant_ids in queries.items():
        # Take first top_k docs
        selected = relevant_ids[:top_k_per_query]
        mini_corpus_ids.update(selected)
        print(f"  '{query_text[:50]}...': {len(selected)} docs")

    print(f"\nTotal from queries: {len(mini_corpus_ids)} unique documents")

    # Add landmark pages
    if include_landmarks:
        print(f"\nAdding {len(LANDMARK_PAGES)} landmark pages...")
        mini_corpus_ids.update(LANDMARK_PAGES)
        print(f"Total after landmarks: {len(mini_corpus_ids)} unique documents")

    return mini_corpus_ids


def save_corpus_ids(corpus_ids: Set[str], output_path: str = "data/mini_corpus_doc_ids.json"):
    """Save corpus document IDs to JSON file."""
    # Convert set to sorted list for readability
    corpus_list = sorted(list(corpus_ids))

    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(corpus_list, f, indent=2)

    print(f"\n✓ Saved {len(corpus_list)} document IDs to {output_path}")


def print_corpus_stats(corpus_ids: Set[str], queries: dict):
    """Print statistics about the mini corpus."""
    print("\n" + "="*70)
    print("MINI CORPUS STATISTICS")
    print("="*70)
    print(f"Total documents in mini corpus: {len(corpus_ids)}")
    print(f"Number of training queries: {len(queries)}")

    # Calculate coverage
    total_relevant = sum(len(rel_ids) for rel_ids in queries.values())
    covered = sum(1 for rel_ids in queries.values()
                  for doc_id in rel_ids if doc_id in corpus_ids)
    coverage_pct = (covered / total_relevant * 100) if total_relevant > 0 else 0

    print(f"Total relevant docs in queries: {total_relevant}")
    print(f"Covered by mini corpus: {covered} ({coverage_pct:.1f}%)")

    # Per-query coverage
    print(f"\nPer-query coverage:")
    for query_text, relevant_ids in list(queries.items())[:5]:  # Show first 5
        in_corpus = sum(1 for doc_id in relevant_ids if doc_id in corpus_ids)
        print(f"  '{query_text[:40]}...': {in_corpus}/{len(relevant_ids)} docs")
    print(f"  ... ({len(queries)-5} more queries)")

    print("="*70)


def main():
    """Main execution function."""
    print("\n" + "="*70)
    print("MINI CORPUS SELECTION SCRIPT")
    print("="*70)

    # Load training queries
    queries_path = "data/queries_train.json"
    print(f"\nLoading training queries from: {queries_path}")
    queries = load_training_queries(queries_path)
    print(f"✓ Loaded {len(queries)} training queries")

    # Select mini corpus
    mini_corpus_ids = select_mini_corpus_ids(
        queries,
        top_k_per_query=10,  # TINY corpus: 10 docs per query
        include_landmarks=True
    )

    # Print statistics
    print_corpus_stats(mini_corpus_ids, queries)

    # Save to file
    output_path = "data/mini_corpus_doc_ids.json"
    save_corpus_ids(mini_corpus_ids, output_path)

    print("\n" + "="*70)
    print("MINI CORPUS SELECTION COMPLETE!")
    print("="*70)
    print(f"\nNext steps:")
    print(f"1. Use this doc ID list to filter Wikipedia dump during indexing")
    print(f"2. Build indices with only these {len(mini_corpus_ids)} documents")
    print(f"3. Test search endpoints locally with <1 second response times")
    print()


if __name__ == "__main__":
    main()
