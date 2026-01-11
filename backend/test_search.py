import sys
try:
    import importlib.metadata as im
    if not hasattr(im, 'packages_distributions'):
        im.packages_distributions = lambda: {}
except:
    pass

import json
import time
from pathlib import Path
from search_runtime import get_engine, initialize_engine

def calculate_ap(relevant_docs, retrieved_docs, k=10):
    if not relevant_docs: return 0.0
    relevant_set = {str(d).strip() for d in relevant_docs}
    retrieved_k = retrieved_docs[:k]
    hits = 0
    sum_precs = 0
    for i, doc_id in enumerate(retrieved_k):
        if str(doc_id).strip() in relevant_set:
            hits += 1
            sum_precs += hits / (i + 1)
    denominator = min(len(relevant_set), k)
    return sum_precs / denominator if denominator > 0 else 0.0

def calculate_p_at_k(relevant_docs, retrieved_docs, k=10):
    relevant_set = {str(d).strip() for d in relevant_docs}
    retrieved_k = retrieved_docs[:k]
    hits = len([d for d in retrieved_k if str(d).strip() in relevant_set])
    return hits / k

def main():
    print("--- Initializing Engine ---")
    initialize_engine()
    engine = get_engine()

    queries_path = Path('queries_train.json')
    with open(queries_path, 'r') as f:
        queries = json.load(f)

    print(f"Evaluating {len(queries)} queries...")
    total_ap, total_p10, count = 0, 0, 0
    
    for query_text, true_docs in queries.items():
        results = engine.search(query_text, top_n=100)
        retrieved_ids = [res[0] for res in results]
        
        total_ap += calculate_ap(true_docs, retrieved_ids, k=10)
        total_p10 += calculate_p_at_k(true_docs, retrieved_ids, k=10)
        count += 1
        if count % 10 == 0: print(f"Done {count}/{len(queries)}...")

    print("\n" + "═"*40)
    print("        FINAL EVALUATION RESULTS")
    print("═"*40)
    print(f"MAP@10:         {total_ap/count:.4f}")
    print(f"Mean P@10:      {total_p10/count:.4f}")
    print(f"Total Queries:  {count}")
    print("═"*40)

if __name__ == "__main__":
    main()