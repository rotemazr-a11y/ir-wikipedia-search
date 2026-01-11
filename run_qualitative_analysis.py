import json
import time
import statistics
from search_runtime import get_engine, initialize_engine

def run_performance_summary():
    # 1. טעינת השאילתות
    try:
        with open('queries_train.json', 'r') as f:
            queries_dict = json.load(f)
    except FileNotFoundError:
        print("Error: queries_train.json not found!")
        return

    # 2. אתחול המנוע
    print("--- Initializing Engine ---")
    initialize_engine()
    engine = get_engine()

    all_stats = []
    
    print(f"\nEvaluating {len(queries_dict)} queries... (Please wait, this may take time due to GCS latency)")

    for query_text, actual_ids in queries_dict.items():
        actual_set = set(map(int, actual_ids))
        
        # מדידת זמן שאילתה
        start_q = time.time()
        results = engine.search(query_text, top_n=10)
        latency = time.time() - start_q
        
        # חישוב P@10
        retrieved_ids = [int(r[0]) for r in results]
        hits = sum(1 for rid in retrieved_ids if rid in actual_set)
        p_at_10 = hits / 10
        
        all_stats.append({
            "query": query_text,
            "p_at_10": p_at_10,
            "latency": latency
        })

    # 3. עיבוד נתונים לסיכום
    all_stats.sort(key=lambda x: x['p_at_10'], reverse=True)
    avg_p10 = statistics.mean([s['p_at_10'] for s in all_stats])
    avg_lat = statistics.mean([s['latency'] for s in all_stats])

    # 4. הדפסת הדוח המבוקש
    print("\n" + "="*85)
    print(f"{'TOP 10 BEST PERFORMING QUERIES':^85}")
    print("="*85)
    print(f"{'#':<4} | {'Query':<45} | {'P@10':<10} | {'Latency':<12}")
    print("-" * 85)
    
    for i, s in enumerate(all_stats[:10], 1):
        print(f"{i:<4} | {s['query'][:45]:<45} | {s['p_at_10']:<10.2f} | {s['latency']:>8.3f}s")

    print("\n" + "="*85)
    print(f"{'QUERIES WITH ZERO PRECISION (P@10 = 0.0)':^85}")
    print("="*85)
    zero_queries = [s for s in all_stats if s['p_at_10'] == 0]
    if not zero_queries:
        print("No queries with zero precision found!")
    else:
        for i, s in enumerate(zero_queries[:10], 1): # מציג עד 10 כאלו
            print(f"{i:<4} | {s['query'][:45]:<45} | {s['p_at_10']:<10.2f} | {s['latency']:>8.3f}s")
        if len(zero_queries) > 10:
            print(f"... and {len(zero_queries) - 10} more queries with zero precision.")

    print("\n" + "="*85)
    print(f"{'OVERALL SYSTEM METRICS':^85}")
    print("="*85)
    print(f"Mean Precision@10: {avg_p10:.4f}")
    print(f"Mean Latency:      {avg_lat:.4f} seconds")
    print("="*85)

if __name__ == "__main__":
    run_performance_summary()