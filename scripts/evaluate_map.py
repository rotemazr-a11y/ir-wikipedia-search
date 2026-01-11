"""
MAP@10 Evaluation Script

Evaluates Mean Average Precision at 10 on the training queries.

Usage:
    python scripts/evaluate_map.py --server http://localhost:8080 --queries data/queries_train.json
"""

import json
import requests
import argparse
from typing import List, Dict, Tuple
import numpy as np


def average_precision_at_k(ranked_results: List[str],
                            ground_truth: List[str],
                            k: int = 10) -> float:
    """
    Compute Average Precision at K.

    Parameters:
    -----------
    ranked_results : list of str
        Document IDs in ranked order (best to worst)
    ground_truth : list of str
        Relevant document IDs
    k : int, default=10
        Cutoff rank

    Returns:
    --------
    float
        Average Precision at K

    Formula:
        AP@K = (1/min(K, |relevant|)) * Σ_{i=1}^{K} P(i) × rel(i)

    Where:
        - P(i) = precision at rank i = (# relevant docs in top i) / i
        - rel(i) = 1 if doc at rank i is relevant, 0 otherwise
    """
    if not ground_truth:
        return 0.0

    relevant_set = set(ground_truth)
    relevant_count = 0
    precision_sum = 0.0

    for i, doc_id in enumerate(ranked_results[:k], start=1):
        if doc_id in relevant_set:
            relevant_count += 1
            precision_at_i = relevant_count / i
            precision_sum += precision_at_i

    # Normalize by min(k, |relevant|)
    norm_factor = min(k, len(ground_truth))
    ap_at_k = precision_sum / norm_factor if norm_factor > 0 else 0.0

    return ap_at_k


def query_endpoint(server_url: str, endpoint: str, query: str) -> List[Tuple[str, str]]:
    """
    Query a search endpoint.

    Parameters:
    -----------
    server_url : str
        Base server URL (e.g., http://localhost:8080)
    endpoint : str
        Endpoint name (e.g., 'search_body', 'search', 'search_title')
    query : str
        Query text

    Returns:
    --------
    list of tuples
        List of (doc_id, title) tuples
    """
    url = f"{server_url}/{endpoint}"
    params = {'query': query}

    try:
        response = requests.get(url, params=params, timeout=60)
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        print(f"✗ Error querying {endpoint}: {e}")
        return []


def evaluate_queries(server_url: str,
                      queries: Dict[str, List[str]],
                      endpoint: str = 'search',
                      k: int = 10,
                      verbose: bool = True) -> Tuple[float, List[Tuple[str, float]]]:
    """
    Evaluate all queries and compute MAP@K.

    Parameters:
    -----------
    server_url : str
        Base server URL
    queries : dict
        Dictionary mapping query text to list of relevant doc IDs
    endpoint : str, default='search'
        Endpoint to test
    k : int, default=10
        Cutoff rank
    verbose : bool, default=True
        Print detailed results per query

    Returns:
    --------
    tuple
        (MAP@K, list of (query, AP@K) pairs)
    """
    ap_scores = []
    query_details = []

    print(f"\nEvaluating {len(queries)} queries on /{endpoint} endpoint...")
    print(f"Metric: Average Precision @ {k}")
    print("="*70 + "\n")

    for i, (query_text, relevant_ids) in enumerate(queries.items(), start=1):
        # Query the endpoint
        results = query_endpoint(server_url, endpoint, query_text)

        # Extract doc IDs from results (first element of tuple)
        ranked_doc_ids = [str(doc_id) for doc_id, title in results]

        # Compute AP@K
        ap_score = average_precision_at_k(ranked_doc_ids, relevant_ids, k=k)
        ap_scores.append(ap_score)
        query_details.append((query_text, ap_score))

        if verbose:
            # Show top 5 results
            top_5 = ranked_doc_ids[:5]
            relevant_in_top_5 = [doc_id for doc_id in top_5 if doc_id in relevant_ids]

            print(f"[{i}/{len(queries)}] Query: '{query_text[:50]}...'")
            print(f"  AP@{k}: {ap_score:.4f}")
            print(f"  Relevant docs: {len(relevant_ids)}")
            print(f"  Returned docs: {len(ranked_doc_ids)}")
            print(f"  Top-5: {top_5}")
            print(f"  Relevant in top-5: {relevant_in_top_5}")
            print()

    # Compute MAP@K
    map_score = np.mean(ap_scores) if ap_scores else 0.0

    return map_score, query_details


def print_summary(map_score: float,
                   query_details: List[Tuple[str, float]],
                   k: int = 10):
    """Print evaluation summary."""
    print("\n" + "="*70)
    print("EVALUATION SUMMARY")
    print("="*70)
    print(f"MAP@{k}: {map_score:.4f}")
    print(f"Total queries: {len(query_details)}")

    # Pass/fail threshold
    threshold = 0.1
    if map_score >= threshold:
        print(f"✓ PASS: MAP@{k} ≥ {threshold} (requirement met)")
    else:
        print(f"✗ FAIL: MAP@{k} < {threshold} (requirement NOT met)")

    # Show best and worst queries
    sorted_queries = sorted(query_details, key=lambda x: x[1], reverse=True)

    print(f"\nBest 5 queries:")
    for query, ap in sorted_queries[:5]:
        print(f"  {ap:.4f} - '{query[:50]}...'")

    print(f"\nWorst 5 queries:")
    for query, ap in sorted_queries[-5:]:
        print(f"  {ap:.4f} - '{query[:50]}...'")

    print("="*70 + "\n")


def main():
    """Main execution."""
    parser = argparse.ArgumentParser(description='Evaluate MAP@10 on training queries')
    parser.add_argument('--server', default='http://localhost:8080',
                        help='Server URL (default: http://localhost:8080)')
    parser.add_argument('--queries', default='data/queries_train.json',
                        help='Path to queries JSON file')
    parser.add_argument('--endpoint', default='search',
                        choices=['search', 'search_body', 'search_title', 'search_anchor'],
                        help='Endpoint to evaluate')
    parser.add_argument('--k', type=int, default=10,
                        help='Cutoff rank (default: 10)')
    parser.add_argument('--quiet', action='store_true',
                        help='Quiet mode (no per-query output)')

    args = parser.parse_args()

    print("\n" + "="*70)
    print("MAP@K EVALUATION SCRIPT")
    print("="*70)
    print(f"Server: {args.server}")
    print(f"Endpoint: /{args.endpoint}")
    print(f"Queries file: {args.queries}")
    print(f"Cutoff K: {args.k}")

    # Load queries
    print(f"\nLoading queries from {args.queries}...")
    with open(args.queries, 'r', encoding='utf-8') as f:
        queries = json.load(f)
    print(f"✓ Loaded {len(queries)} queries")

    # Evaluate
    map_score, query_details = evaluate_queries(
        server_url=args.server,
        queries=queries,
        endpoint=args.endpoint,
        k=args.k,
        verbose=not args.quiet
    )

    # Print summary
    print_summary(map_score, query_details, k=args.k)

    # Save results
    results = {
        'map_at_k': map_score,
        'k': args.k,
        'endpoint': args.endpoint,
        'num_queries': len(queries),
        'query_scores': [
            {'query': q, 'ap_at_k': ap} for q, ap in query_details
        ]
    }

    output_file = f'evaluation_results_{args.endpoint}_map{args.k}.json'
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2)

    print(f"Results saved to {output_file}")


if __name__ == "__main__":
    main()
