import json
import time
import os
from pathlib import Path
from search_runtime import get_engine, initialize_engine

def calculate_ap(relevant_docs, retrieved_docs, k=10):
    """
    מחשב Average Precision עבור שאילתה בודדת.
    הפונקציה ממירה הכל למחרוזות כדי למנוע טעויות של השוואת מספר למחרוזת.
    """
    if not relevant_docs or not retrieved_docs:
        return 0.0
    
    # המרה ל-Set של מחרוזות לחיפוש מהיר (O(1))
    relevant_set = {str(doc_id) for doc_id in relevant_docs}
    hits = 0
    sum_precs = 0
    
    # בדיקת k התוצאות הראשונות שהוחזרו מהמנוע
    for i, doc_id in enumerate(retrieved_docs[:k]):
        if str(doc_id) in relevant_set:
            hits += 1
            sum_precs += hits / (i + 1)
            
    if hits == 0:
        return 0.0
    
    # חילוק במספר המסמכים הרלוונטיים (לפי הגדרת MAP@10 המקובלת)
    return sum_precs / min(len(relevant_docs), k)

def main():
    # --- שלב 1: אתחול המנוע ---
    print("Loading engine... (This might take a minute)")
    try:
        initialize_engine()
        engine = get_engine()
    except Exception as e:
        print(f"Failed to initialize engine: {e}")
        return

    # --- שלב 2: טעינת קובץ השאילתות ---
    script_dir = Path(__file__).parent.absolute()
    queries_path = script_dir / 'queries_train.json'

    if not queries_path.exists():
        # ניסיון למצוא בתיקיות משנה נפוצות
        for p in [Path('queries_train.json'), Path('data/queries_train.json')]:
            if p.exists():
                queries_path = p
                break

    if not queries_path.exists():
        print(f"Error: Could not find queries_train.json")
        return

    print(f"Loading queries from: {queries_path}")
    with open(queries_path, 'r') as f:
        queries = json.load(f)

    # --- שלב 3: הרצת ההערכה ---
    print(f"Starting evaluation on {len(queries)} queries...")
    total_ap = 0
    start_time = time.time()

    for idx, (query_text, true_docs) in enumerate(queries.items()):
        # המנוע מחזיר רשימה של טאפלים: (doc_id, title, score)
        results = engine.search(query_text, top_n=10)
        
        # חילוץ מזהה המסמך בלבד והמרה למחרוזת
        retrieved_ids = [str(res[0]) for res in results]
        
        # חישוב AP והוספה לסכום
        ap = calculate_ap(true_docs, retrieved_ids, k=10)
        total_ap += ap

        # הדפסת התקדמות כל 10 שאילתות
        if (idx + 1) % 10 == 0:
            print(f"Processed {idx + 1}/{len(queries)} queries...")

    # --- שלב 4: סיכום תוצאות ---
    mean_ap = total_ap / len(queries) if queries else 0
    duration = time.time() - start_time

    print("\n" + "="*30)
    print("       EVALUATION REPORT")
    print("="*30)
    print(f"MAP@10:         {mean_ap:.4f}")
    print(f"Total Queries:  {len(queries)}")
    print(f"Total Time:     {duration:.2f} seconds")
    print(f"Avg Time/Query: {duration/len(queries) if queries else 0:.4f}s")
    print("="*30)

if __name__ == "__main__":
    main()