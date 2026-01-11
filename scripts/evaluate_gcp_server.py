#!/usr/bin/env python3
"""
Evaluate search performance on GCP server with full corpus.
"""

import requests
import json
import argparse
from collections import defaultdict
import time

def load_queries(queries_file):
    """Load queries and ground truth."""
    with open(queries_file, 'r') as f:
        queries = json.load(f)
    return queries

def search_query(server_url, query_text, endpoint='search'):
    """Query the GCP search server."""
    try:
        response = requests.get(
            f"{server_url}/{endpoint}",
            params={'query': query_text},
            timeout=30
        )
        response.raise_for_status()
        results = response.json()

        # Results are list of [doc_id, title] tuples
        # Extract just the doc_ids
        doc_ids = [str(doc_id) for doc_id, title in results]
        return doc_ids

    except requests.exceptions.RequestException as e:
        print(f"Error querying server: {e}")
        return []

def average_precision_at_k(relevant_docs, retrieved_docs, k=10):
    """
    Compute Average Precision at K.

    Parameters:
    -----------
    relevant_docs : set
        Set of relevant document IDs (strings)
    retrieved_docs : list
        Ordered list of retrieved document IDs (strings)
    k : int
        Cutoff position

    Returns:
    --------
    float : AP@K score
    """
    if not relevant_docs:
        return 0.0

    # Consider only top-k results
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

    # AP is average of precisions at ranks where relevant docs appear
    ap = precision_sum / min(len(relevant_docs), k)
    return ap

def evaluate_server(server_url, queries, k=10, endpoint='search'):
    """
    Evaluate search server performance.

    Parameters:
    -----------
    server_url : str
        Base URL of search server (e.g., http://34.123.45.67:8080)
    queries : dict
        Query text -> list of relevant doc IDs
    k : int
        Cutoff for evaluation (default 10)
    endpoint : str
        Search endpoint to use (default 'search')

    Returns:
    --------
    dict : Evaluation results
    """
    print(f"Evaluating {len(queries)} queries against {server_url}/{endpoint}")
    print(f"Cutoff: top-{k} results")
    print("="*80)

    results = {
        'map_at_k': 0.0,
        'k': k,
        'endpoint': endpoint,
        'server_url': server_url,
        'num_queries': len(queries),
        'query_scores': []
    }

    total_ap = 0.0
    query_times = []

    for i, (query_text, relevant_docs) in enumerate(queries.items(), 1):
        print(f"\n[{i}/{len(queries)}] {query_text}")

        # Query server
        start_time = time.time()
        retrieved_docs = search_query(server_url, query_text, endpoint)
        query_time = time.time() - start_time
        query_times.append(query_time)

        # Compute AP@K
        relevant_set = set(str(doc_id) for doc_id in relevant_docs)
        ap_at_k = average_precision_at_k(relevant_set, retrieved_docs, k)

        total_ap += ap_at_k

        # Log results
        print(f"  Retrieved: {len(retrieved_docs)} docs")
        print(f"  AP@{k}: {ap_at_k:.4f}")
        print(f"  Time: {query_time:.2f}s")

        if ap_at_k == 0.0:
            print(f"  ⚠️  No relevant docs in top-{k}")

        # Store per-query results
        results['query_scores'].append({
            'query': query_text,
            'ap_at_k': ap_at_k,
            'num_retrieved': len(retrieved_docs),
            'num_relevant': len(relevant_docs),
            'query_time': query_time
        })

    # Compute MAP@K
    map_at_k = total_ap / len(queries)
    results['map_at_k'] = map_at_k
    results['avg_query_time'] = sum(query_times) / len(query_times)
    results['total_time'] = sum(query_times)

    # Summary
    print("\n" + "="*80)
    print("EVALUATION SUMMARY")
    print("="*80)
    print(f"Server: {server_url}")
    print(f"Endpoint: /{endpoint}")
    print(f"Queries: {len(queries)}")
    print(f"Cutoff: top-{k}")
    print(f"\nMAP@{k}: {map_at_k:.4f}")
    print(f"Average query time: {results['avg_query_time']:.2f}s")
    print(f"Total evaluation time: {results['total_time']:.1f}s")

    # Query performance distribution
    zero_ap = sum(1 for q in results['query_scores'] if q['ap_at_k'] == 0.0)
    high_ap = sum(1 for q in results['query_scores'] if q['ap_at_k'] >= 0.3)

    print(f"\nQuery Performance:")
    print(f"  Zero AP@{k}: {zero_ap}/{len(queries)} ({zero_ap/len(queries)*100:.1f}%)")
    print(f"  High AP@{k} (≥0.3): {high_ap}/{len(queries)} ({high_ap/len(queries)*100:.1f}%)")

    # Top and bottom queries
    sorted_queries = sorted(results['query_scores'], key=lambda x: x['ap_at_k'], reverse=True)

    print(f"\nTop 5 queries:")
    for q in sorted_queries[:5]:
        print(f"  {q['query'][:60]:60s} AP@{k}={q['ap_at_k']:.4f}")

    print(f"\nBottom 5 queries:")
    for q in sorted_queries[-5:]:
        print(f"  {q['query'][:60]:60s} AP@{k}={q['ap_at_k']:.4f}")

    return results

def main():
    parser = argparse.ArgumentParser(description='Evaluate GCP search server')
    parser.add_argument('--server-url', required=True,
                       help='Base URL of search server (e.g., http://34.123.45.67:8080)')
    parser.add_argument('--queries', required=True,
                       help='Path to queries JSON file')
    parser.add_argument('--output', default='evaluation_results_gcp.json',
                       help='Output file for results')
    parser.add_argument('--k', type=int, default=10,
                       help='Cutoff for MAP@K evaluation (default: 10)')
    parser.add_argument('--endpoint', default='search',
                       help='Search endpoint to evaluate (default: search)')

    args = parser.parse_args()

    # Load queries
    print(f"Loading queries from {args.queries}...")
    queries = load_queries(args.queries)
    print(f"Loaded {len(queries)} queries\n")

    # Test server connectivity
    print(f"Testing connection to {args.server_url}...")
    try:
        response = requests.get(f"{args.server_url}/{args.endpoint}",
                              params={'query': 'test'},
                              timeout=10)
        response.raise_for_status()
        print(f"✓ Server is reachable\n")
    except requests.exceptions.RequestException as e:
        print(f"✗ Cannot connect to server: {e}")
        print("\nTroubleshooting:")
        print("  1. Check if server is running: gcloud compute instances list")
        print("  2. Check firewall rules: gcloud compute firewall-rules list")
        print("  3. Test manually: curl 'http://IP:8080/search?query=test'")
        return 1

    # Run evaluation
    results = evaluate_server(args.server_url, queries, args.k, args.endpoint)

    # Save results
    with open(args.output, 'w') as f:
        json.dump(results, f, indent=2)

    print(f"\n✓ Results saved to {args.output}")

    # Check if threshold met
    threshold = 0.1
    if results['map_at_k'] >= threshold:
        print(f"\n✅ SUCCESS: MAP@{args.k} = {results['map_at_k']:.4f} ≥ {threshold}")
    else:
        print(f"\n❌ FAILED: MAP@{args.k} = {results['map_at_k']:.4f} < {threshold}")

    return 0

if __name__ == '__main__':
    exit(main())
