#!/usr/bin/env python3
"""
Evaluation script for the search engine.
Tests all endpoints and computes P@5, P@10, F1@30.

Usage:
    # Test against local Flask server:
    python evaluate_search.py --url http://localhost:8080
    
    # Test against GCP:
    python evaluate_search.py --url http://YOUR_EXTERNAL_IP:8080
    
    # Test directly (import engine, no HTTP):
    python evaluate_search.py --direct
"""

import json
import time
import argparse
import requests
from pathlib import Path
from typing import List, Dict, Tuple


def load_queries(queries_path: str = "queries_train.json") -> Dict[str, List[str]]:
    """Load training queries and their relevant documents."""
    path = Path(queries_path)
    if not path.exists():
        # Try parent directory
        path = Path(__file__).parent.parent / "queries_train.json"
    
    with open(path, 'r') as f:
        return json.load(f)


def precision_at_k(relevant: List[str], retrieved: List, k: int) -> float:
    """
    Precision@K: fraction of top-K results that are relevant.
    """
    relevant_set = {str(d).strip() for d in relevant}
    retrieved_k = [str(r[0] if isinstance(r, (list, tuple)) else r).strip() 
                   for r in retrieved[:k]]
    
    hits = sum(1 for doc in retrieved_k if doc in relevant_set)
    return hits / k if k > 0 else 0.0


def recall_at_k(relevant: List[str], retrieved: List, k: int) -> float:
    """
    Recall@K: fraction of relevant docs found in top-K.
    """
    relevant_set = {str(d).strip() for d in relevant}
    retrieved_k = [str(r[0] if isinstance(r, (list, tuple)) else r).strip() 
                   for r in retrieved[:k]]
    
    hits = sum(1 for doc in retrieved_k if doc in relevant_set)
    return hits / len(relevant_set) if relevant_set else 0.0


def f1_at_k(relevant: List[str], retrieved: List, k: int) -> float:
    """
    F1@K: harmonic mean of Precision@K and Recall@K.
    """
    p = precision_at_k(relevant, retrieved, k)
    r = recall_at_k(relevant, retrieved, k)
    return 2 * p * r / (p + r) if (p + r) > 0 else 0.0


def average_precision_at_k(relevant: List[str], retrieved: List, k: int) -> float:
    """
    Average Precision@K (AP@K).
    """
    relevant_set = {str(d).strip() for d in relevant}
    retrieved_k = [str(r[0] if isinstance(r, (list, tuple)) else r).strip() 
                   for r in retrieved[:k]]
    
    hits = 0
    sum_precs = 0.0
    
    for i, doc_id in enumerate(retrieved_k):
        if doc_id in relevant_set:
            hits += 1
            sum_precs += hits / (i + 1)
    
    # Normalize by min(|relevant|, k)
    denominator = min(len(relevant_set), k)
    return sum_precs / denominator if denominator > 0 else 0.0


class SearchEvaluator:
    """Evaluator that can test via HTTP or direct import."""
    
    def __init__(self, base_url: str = None, direct: bool = False):
        self.base_url = base_url
        self.direct = direct
        self.engine = None
        
        if direct:
            self._init_direct_engine()
    
    def _init_direct_engine(self):
        """Initialize engine directly for faster testing."""
        import sys
        sys.path.insert(0, str(Path(__file__).parent.parent / "frontend"))
        from search_frontend import engine
        engine.load_all()
        self.engine = engine
    
    def query_search(self, query: str) -> Tuple[List, float]:
        """Query the main /search endpoint."""
        start = time.time()
        
        if self.direct:
            from search_frontend import combined_search
            results = combined_search(query, top_k=100)
        else:
            resp = requests.get(f"{self.base_url}/search", params={"query": query})
            results = resp.json()
        
        elapsed = time.time() - start
        return results, elapsed
    
    def query_search_body(self, query: str) -> Tuple[List, float]:
        """Query /search_body endpoint."""
        start = time.time()
        
        if self.direct:
            from search_frontend import tokenize_no_stem, compute_tfidf_cosine, engine, BODY_INDEX_PATH
            tokens = tokenize_no_stem(query)
            results = compute_tfidf_cosine(tokens, engine.body_index, BODY_INDEX_PATH, 100)
            results = [(doc_id, engine.doc_titles.get(doc_id, "")) for doc_id, _ in results]
        else:
            resp = requests.get(f"{self.base_url}/search_body", params={"query": query})
            results = resp.json()
        
        elapsed = time.time() - start
        return results, elapsed
    
    def query_search_title(self, query: str) -> Tuple[List, float]:
        """Query /search_title endpoint."""
        start = time.time()
        
        if self.direct:
            from search_frontend import tokenize_no_stem, binary_ranking, engine, TITLE_INDEX_PATH
            tokens = tokenize_no_stem(query)
            results = binary_ranking(tokens, engine.title_index, TITLE_INDEX_PATH)
            results = [(doc_id, engine.doc_titles.get(doc_id, "")) for doc_id, _ in results]
        else:
            resp = requests.get(f"{self.base_url}/search_title", params={"query": query})
            results = resp.json()
        
        elapsed = time.time() - start
        return results, elapsed


def evaluate_endpoint(evaluator: SearchEvaluator, queries: Dict, endpoint: str = "search"):
    """Evaluate a specific endpoint."""
    
    query_method = {
        "search": evaluator.query_search,
        "search_body": evaluator.query_search_body,
        "search_title": evaluator.query_search_title,
    }.get(endpoint, evaluator.query_search)
    
    total_p5 = 0.0
    total_p10 = 0.0
    total_f1_30 = 0.0
    total_ap10 = 0.0
    total_time = 0.0
    count = len(queries)
    
    print(f"\n{'='*70}")
    print(f"Evaluating: /{endpoint}")
    print(f"{'='*70}")
    print(f"{'#':<3} | {'Query':<35} | {'P@5':<5} | {'P@10':<5} | {'F1@30':<5} | {'Time':<6}")
    print("-" * 70)
    
    for i, (query_text, relevant_docs) in enumerate(queries.items(), 1):
        try:
            results, elapsed = query_method(query_text)
            
            p5 = precision_at_k(relevant_docs, results, 5)
            p10 = precision_at_k(relevant_docs, results, 10)
            f1_30 = f1_at_k(relevant_docs, results, 30)
            ap10 = average_precision_at_k(relevant_docs, results, 10)
            
            total_p5 += p5
            total_p10 += p10
            total_f1_30 += f1_30
            total_ap10 += ap10
            total_time += elapsed
            
            print(f"{i:<3} | {query_text[:35]:<35} | {p5:.3f} | {p10:.3f} | {f1_30:.3f} | {elapsed:.2f}s")
        
        except Exception as e:
            print(f"{i:<3} | {query_text[:35]:<35} | ERROR: {e}")
    
    print("-" * 70)
    print(f"\n📊 RESULTS for /{endpoint}:")
    print(f"   Average P@5:     {total_p5/count:.4f}")
    print(f"   Average P@10:    {total_p10/count:.4f}  (minimum required: > 0.1)")
    print(f"   Average F1@30:   {total_f1_30/count:.4f}")
    print(f"   Average AP@10:   {total_ap10/count:.4f}")
    print(f"   Avg Query Time:  {total_time/count:.3f}s")
    print(f"   Total Time:      {total_time:.2f}s")
    
    # Grading metric: harmonic mean of P@5 and F1@30
    harmonic = 2 * (total_p5/count) * (total_f1_30/count) / ((total_p5/count) + (total_f1_30/count)) if (total_p5 + total_f1_30) > 0 else 0
    print(f"\n   🎯 Harmonic(P@5, F1@30): {harmonic:.4f}  (grading metric)")
    
    return {
        "p5": total_p5/count,
        "p10": total_p10/count,
        "f1_30": total_f1_30/count,
        "ap10": total_ap10/count,
        "avg_time": total_time/count,
        "harmonic": harmonic
    }


def main():
    parser = argparse.ArgumentParser(description="Evaluate search engine")
    parser.add_argument("--url", default="http://localhost:8080", help="Base URL of search engine")
    parser.add_argument("--direct", action="store_true", help="Test directly without HTTP")
    parser.add_argument("--queries", default="queries_train.json", help="Path to queries file")
    parser.add_argument("--endpoint", default="all", choices=["search", "search_body", "search_title", "all"])
    args = parser.parse_args()
    
    print("=" * 70)
    print("SEARCH ENGINE EVALUATION")
    print("=" * 70)
    
    queries = load_queries(args.queries)
    print(f"Loaded {len(queries)} queries from {args.queries}")
    
    if args.direct:
        print("Mode: Direct (importing engine)")
    else:
        print(f"Mode: HTTP ({args.url})")
    
    evaluator = SearchEvaluator(base_url=args.url, direct=args.direct)
    
    results = {}
    endpoints = ["search", "search_body", "search_title"] if args.endpoint == "all" else [args.endpoint]
    
    for endpoint in endpoints:
        results[endpoint] = evaluate_endpoint(evaluator, queries, endpoint)
    
    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    for endpoint, metrics in results.items():
        print(f"/{endpoint}: P@10={metrics['p10']:.3f}, Harmonic={metrics['harmonic']:.3f}, Time={metrics['avg_time']:.2f}s")


if __name__ == "__main__":
    main()
