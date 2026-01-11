# Wikipedia Search Engine - Detailed Setup & Testing Guide

**Complete step-by-step instructions for local testing before GCP deployment**

---

## 🎯 Overview

This guide walks you through:
1. Creating a mini test corpus (320 Wikipedia articles)
2. Building test indices locally
3. Running unit tests
4. Starting the Flask server
5. Testing all 6 endpoints manually
6. Evaluating MAP@10 performance

**Estimated time:** 2-3 hours (assuming you have a Wikipedia dump)

---

## 📋 Prerequisites (MacBook Air)

Before starting, ensure you have:
- ✅ **Python 3.7+** installed (comes with macOS, or use `brew install python3`)
- ✅ **Dependencies installed** (see below)
- ✅ **Wikipedia dump file** (JSON or Parquet format)
- ✅ **5GB+ free disk space** for indices

### Install Dependencies (macOS)

```bash
# Method 1: Using pip (recommended)
python3 -m pip install --user flask nltk numpy requests

# Method 2: Using Homebrew (if you have it)
brew install python3
python3 -m pip install flask nltk numpy requests

# Download NLTK data
python3 -c "import nltk; nltk.download('stopwords')"
```

**If you see SSL certificate errors:**
```bash
# Run the certificate installer (adjust Python version)
/Applications/Python\ 3.11/Install\ Certificates.command

# Or on newer macOS:
/Applications/Python\ 3.12/Install\ Certificates.command
```

---

## STEP 1: Create Mini Corpus Document List

### 1.1 Navigate to Project Directory (macOS)

```bash
# Copy-paste this exact path (quotes handle spaces and Hebrew characters)
cd "/Users/tomerfilo/Library/Mobile Documents/com~apple~CloudDocs/לימודים- שנה ג/סמסטר א/אחזור מידע/ir_proj_20251213 2"

# Verify you're in the right directory
pwd
# Should show: /Users/tomerfilo/Library/Mobile Documents/com~apple~CloudDocs/לימודים- שנה ג/סמסטר א/אחזור מידע/ir_proj_20251213 2
```

**macOS Tip:** If you have the folder open in Finder, you can:
1. Right-click the folder in Finder
2. Hold Option/Alt key
3. Click "Copy ... as Pathname"
4. Paste in Terminal

### 1.2 Run Mini Corpus Selection Script

```bash
# Use python3 explicitly on macOS
python3 scripts/create_mini_corpus.py
```

**Expected Output:**
```
======================================================================
MINI CORPUS SELECTION SCRIPT
======================================================================

Loading training queries from: data/queries_train.json
✓ Loaded 30 training queries

Extracting top-10 docs per query...
  'Mount Everest climbing expeditions...': 10 docs
  'Great Fire of London 1666...': 10 docs
  'Nanotechnology materials science...': 10 docs
  ... (27 more)

Total from queries: 300 unique documents

Adding 20 landmark pages...
Total after landmarks: 320 unique documents

======================================================================
MINI CORPUS STATISTICS
======================================================================
Total documents in mini corpus: 320
Number of training queries: 30
Total relevant docs in queries: 1384
Covered by mini corpus: 300 (21.7%)

✓ Saved 320 document IDs to data/mini_corpus_doc_ids.json
```

### 1.3 Verify Output File (macOS)

```bash
# Check that the file was created (macOS ls)
ls -lh data/mini_corpus_doc_ids.json

# Preview first 10 doc IDs (macOS head)
head -n 20 data/mini_corpus_doc_ids.json

# Or open in default editor (macOS)
open data/mini_corpus_doc_ids.json
```

**Expected Output:**
```
[
  "12",
  "5043734",
  "18630637",
  "25445",
  "47353693",
  "5208803",
  "20852640",
  ...
]
```

✅ **Checkpoint:** You should now have `data/mini_corpus_doc_ids.json` with 320 doc IDs.

---

## STEP 2: Build Mini Corpus Indices

### 2.1 Create Index Building Script (If You Don't Have One)

Create a file called `build_mini_indices.py` in your project root:

```python
"""
Build indices for mini corpus (320 documents).

This script filters your Wikipedia dump to only include documents
in the mini corpus, then builds indices using the FIXED index_builder.py.
"""

import json
import pickle
from pathlib import Path
from backend.index_builder import IndexBuilder

# CONFIGURATION: Update these paths to match your setup
WIKIPEDIA_DUMP_PATH = "data/wikipedia_sample.json"  # Change to your dump file
MINI_CORPUS_IDS_PATH = "data/mini_corpus_doc_ids.json"
OUTPUT_DIR = "indices_mini"  # Where to save indices

def load_mini_corpus_ids(path):
    """Load the list of document IDs to include."""
    with open(path, 'r') as f:
        doc_ids = json.load(f)
    return set(str(doc_id) for doc_id in doc_ids)  # Convert to strings

def load_wikipedia_dump(dump_path, allowed_ids):
    """
    Load Wikipedia dump and filter to only allowed doc IDs.

    IMPORTANT: Adapt this function to match YOUR Wikipedia dump format.

    Common formats:
    1. JSON Lines (.jsonl): One JSON object per line
    2. JSON Array (.json): List of documents
    3. Parquet (.parquet): Columnar format
    """
    documents = []

    # Example 1: JSON Lines format
    if dump_path.endswith('.jsonl'):
        with open(dump_path, 'r', encoding='utf-8') as f:
            for line_num, line in enumerate(f, start=1):
                try:
                    doc = json.loads(line)
                    doc_id = str(doc.get('id') or doc.get('doc_id'))

                    if doc_id in allowed_ids:
                        documents.append(doc)
                        print(f"✓ Loaded doc {doc_id}: {doc.get('title', 'Unknown')[:50]}")

                    if line_num % 10000 == 0:
                        print(f"  Processed {line_num:,} lines, found {len(documents)} mini corpus docs")

                except json.JSONDecodeError as e:
                    print(f"✗ Error parsing line {line_num}: {e}")
                    continue

    # Example 2: JSON Array format
    elif dump_path.endswith('.json'):
        with open(dump_path, 'r', encoding='utf-8') as f:
            all_docs = json.load(f)

        for doc in all_docs:
            doc_id = str(doc.get('id') or doc.get('doc_id'))
            if doc_id in allowed_ids:
                documents.append(doc)
                print(f"✓ Loaded doc {doc_id}: {doc.get('title', 'Unknown')[:50]}")

    # Example 3: Parquet format
    elif dump_path.endswith('.parquet'):
        import pandas as pd
        df = pd.read_parquet(dump_path)

        for idx, row in df.iterrows():
            doc_id = str(row.get('id') or row.get('doc_id'))
            if doc_id in allowed_ids:
                doc = row.to_dict()
                documents.append(doc)
                print(f"✓ Loaded doc {doc_id}: {doc.get('title', 'Unknown')[:50]}")

    else:
        raise ValueError(f"Unsupported file format: {dump_path}")

    return documents

def main():
    print("\n" + "="*70)
    print("BUILDING MINI CORPUS INDICES")
    print("="*70 + "\n")

    # Step 1: Load allowed doc IDs
    print(f"Step 1: Loading mini corpus doc IDs from {MINI_CORPUS_IDS_PATH}")
    allowed_ids = load_mini_corpus_ids(MINI_CORPUS_IDS_PATH)
    print(f"✓ Loaded {len(allowed_ids)} doc IDs to include\n")

    # Step 2: Load Wikipedia dump (filtered)
    print(f"Step 2: Loading Wikipedia dump from {WIKIPEDIA_DUMP_PATH}")
    print(f"         (filtering to {len(allowed_ids)} docs)")
    documents = load_wikipedia_dump(WIKIPEDIA_DUMP_PATH, allowed_ids)
    print(f"\n✓ Loaded {len(documents)} documents from dump")

    if len(documents) == 0:
        print("\n✗ ERROR: No documents loaded! Check your dump path and format.")
        return

    # Step 3: Build indices
    print(f"\nStep 3: Building indices with IndexBuilder")
    builder = IndexBuilder()

    for doc in documents:
        # Adapt field names to match YOUR Wikipedia dump format
        doc_id = int(doc.get('id') or doc.get('doc_id'))
        title = doc.get('title', '')
        body = doc.get('text') or doc.get('body') or doc.get('content', '')
        anchors = doc.get('anchors') or doc.get('anchor_text', [])

        builder.add_document(
            doc_id=doc_id,
            title=title,
            body=body,
            anchors=anchors if isinstance(anchors, list) else []
        )

    print(f"\n✓ Added {builder.num_docs} documents to builder")

    # Step 4: Build and save indices
    print(f"\nStep 4: Writing indices to {OUTPUT_DIR}/")
    Path(OUTPUT_DIR).mkdir(parents=True, exist_ok=True)

    body_idx, title_idx, anchor_idx = builder.build_indices(
        output_dir=OUTPUT_DIR,
        bucket_name=None  # Local storage only
    )

    # Step 5: Print statistics
    print("\n" + "="*70)
    builder.print_stats()

    print("\n✓ SUCCESS! Indices built successfully.")
    print(f"\nGenerated files in {OUTPUT_DIR}/:")
    for file in sorted(Path(OUTPUT_DIR).glob('*')):
        size_mb = file.stat().st_size / (1024 * 1024)
        print(f"  - {file.name:30s} ({size_mb:.2f} MB)")

    print("\n" + "="*70)
    print("NEXT STEPS:")
    print("="*70)
    print(f"1. Run unit tests: python tests/test_search_endpoints.py")
    print(f"2. Start Flask server: INDEX_DIR={OUTPUT_DIR} python backend/search_frontend.py")
    print(f"3. Test endpoints manually")
    print()

if __name__ == "__main__":
    main()
```

### 2.2 Configure Paths in Script

**CRITICAL:** Edit the paths at the top of `build_mini_indices.py`:

```python
# BEFORE (example values)
WIKIPEDIA_DUMP_PATH = "data/wikipedia_sample.json"

# AFTER (your actual paths)
WIKIPEDIA_DUMP_PATH = "/path/to/your/enwiki-latest-pages-articles.json"
# or
WIKIPEDIA_DUMP_PATH = "/path/to/your/wiki_dump.parquet"
```

### 2.3 Run Index Building (macOS)

```bash
# Use python3 explicitly
python3 build_mini_indices.py

# If you get "No module named 'backend'" error on macOS:
export PYTHONPATH="${PYTHONPATH}:${PWD}"
python3 build_mini_indices.py
```

**Expected Output:**
```
======================================================================
BUILDING MINI CORPUS INDICES
======================================================================

Step 1: Loading mini corpus doc IDs from data/mini_corpus_doc_ids.json
✓ Loaded 320 doc IDs to include

Step 2: Loading Wikipedia dump from data/wikipedia_sample.json
         (filtering to 320 docs)
✓ Loaded doc 47353693: Mount Everest
✓ Loaded doc 5208803: 1953 British Mount Everest expedition
✓ Loaded doc 20852640: 1924 British Mount Everest expedition
... (317 more)

✓ Loaded 320 documents from dump

Step 3: Building indices with IndexBuilder
✓ Added 320 documents to builder

Step 4: Writing indices to indices_mini/
======================================================================
Building inverted indices...
======================================================================

[1/3] Building BODY index with 320 documents
✓ Body index complete: 12,543 unique terms, 458,923 postings

[2/3] Building TITLE index with 320 documents
✓ Title index complete: 1,234 unique terms, 2,456 postings

[3/3] Building ANCHOR index with 320 documents
✓ Anchor index complete: 3,567 unique terms, 8,912 postings

✓ Metadata saved: 320 documents

======================================================================
INDEX BUILDER STATISTICS
======================================================================
Total Documents:                     320
Documents with Body:                 320
Documents with Title:                320
Documents with Anchors:              285

Body Vocabulary Size:             12,543 unique terms
Title Vocabulary Size:             1,234 unique terms
Anchor Vocabulary Size:            3,567 unique terms

Avg Body Length:                   458.2 tokens
Avg Title Length:                    3.8 tokens
Avg Anchor Length:                  12.1 tokens
======================================================================

✓ SUCCESS! Indices built successfully.

Generated files in indices_mini/:
  - anchor_index.pkl               (0.52 MB)
  - anchor_index_000.bin           (2.34 MB)
  - body_index.pkl                 (1.89 MB)
  - body_index_000.bin            (15.67 MB)
  - body_index_001.bin             (8.23 MB)
  - metadata.pkl                   (0.12 MB)
  - title_index.pkl                (0.34 MB)
  - title_index_000.bin            (1.12 MB)
```

### 2.4 Verify Indices Were Created (macOS)

```bash
# List files with sizes (macOS)
ls -lh indices_mini/

# Or use tree view (if you have tree installed via Homebrew)
brew install tree  # First time only
tree indices_mini/

# Show total size (macOS)
du -sh indices_mini/
```

**Expected Files:**
```
total 28M
-rw-r--r-- 1 user group 520K ... anchor_index.pkl
-rw-r--r-- 1 user group 2.3M ... anchor_index_000.bin
-rw-r--r-- 1 user group 1.9M ... body_index.pkl
-rw-r--r-- 1 user group 16M  ... body_index_000.bin
-rw-r--r-- 1 user group 8.2M ... body_index_001.bin
-rw-r--r-- 1 user group 120K ... metadata.pkl
-rw-r--r-- 1 user group 340K ... title_index.pkl
-rw-r--r-- 1 user group 1.1M ... title_index_000.bin
```

✅ **Checkpoint:** You should now have `indices_mini/` directory with 8 files totaling ~28 MB.

---

## STEP 3: Run Unit Tests

### 3.1 Run All Tests (macOS)

```bash
# Use python3
python3 tests/test_search_endpoints.py

# If you see import errors, set PYTHONPATH:
export PYTHONPATH="${PWD}:${PYTHONPATH}"
python3 tests/test_search_endpoints.py
```

**Expected Output:**
```
======================================================================
RUNNING SEARCH ENGINE UNIT TESTS
======================================================================

test_idf_natural_log (test_search_endpoints.TestTFIDFFormula) ... ✓ IDF formula uses natural log: log(1000/100) = 2.3026
ok
test_tfidf_calculation (test_search_endpoints.TestTFIDFFormula) ... ✓ TF-IDF: 5 * log(1000/100) = 11.5129
ok
test_cosine_similarity_basic (test_search_endpoints.TestCosineSimilarity) ... ✓ Cosine similarity: 0.3600
ok
test_identical_vectors (test_search_endpoints.TestCosineSimilarity) ... ✓ Identical vectors have cosine = 1.0
ok
test_orthogonal_vectors (test_search_endpoints.TestCosineSimilarity) ... ✓ Orthogonal vectors have cosine = 0.0
ok
test_distinct_word_count (test_search_endpoints.TestBinaryRanking) ... ✓ Distinct word count: 'fire fire fire london' → 2 distinct words
ok
test_binary_ranking_score (test_search_endpoints.TestBinaryRanking) ... ✓ Binary ranking: 3 distinct query words matched in title
ok
test_binary_ranking_with_duplicates (test_search_endpoints.TestBinaryRanking) ... ✓ Duplicate query words don't inflate score: 2
ok
test_stopword_removal (test_search_endpoints.TestPreprocessing) ... ✓ Stopwords removed: 'the Great Fire of London' → ['great', 'fire', 'london']
ok
test_no_stemming (test_search_endpoints.TestPreprocessing) ... ✓ No stemming: 'climbing pyramids' → ['climbing', 'pyramids']
✓ With stemming: 'climbing pyramids' → ['climb', 'pyramid']
ok
test_case_normalization (test_search_endpoints.TestPreprocessing) ... ✓ Case normalized: 'MOUNT EVEREST' = 'mount everest'
ok
test_empty_input (test_search_endpoints.TestPreprocessing) ... ✓ Empty input handled gracefully
ok
test_stopword_only_query (test_search_endpoints.TestPreprocessing) ... ✓ Stopword-only query: 'the and or' → []
ok
test_division_by_zero_in_cosine (test_search_endpoints.TestEdgeCases) ... ✓ Zero norm handled: cosine = 0.0
ok
test_query_term_not_in_index (test_search_endpoints.TestEdgeCases) ... ✓ Missing term handled: 'xyznotexist' not in index
ok
test_empty_posting_list (test_search_endpoints.TestEdgeCases) ... ✓ Empty posting list handled
ok

----------------------------------------------------------------------
Ran 16 tests in 0.523s

OK

======================================================================
TEST SUMMARY
======================================================================
Tests run: 16
Successes: 16
Failures: 0
Errors: 0
======================================================================
```

✅ **Checkpoint:** All 16 tests should pass. If any fail, review the error messages.

---

## STEP 4: Start Flask Server

### 4.1 Set Index Directory Environment Variable (macOS)

```bash
# Tell Flask where to find the mini indices
export INDEX_DIR=indices_mini

# Verify it's set
echo $INDEX_DIR
# Should print: indices_mini
```

### 4.2 Start the Server (macOS)

```bash
# Use python3
python3 backend/search_frontend.py

# If you get import errors:
export PYTHONPATH="${PWD}:${PYTHONPATH}"
export INDEX_DIR=indices_mini
python3 backend/search_frontend.py
```

**macOS Tip:** To stop the server, press **Ctrl+C** (not ⌘+C)

**Expected Output:**
```
Loading indices from indices_mini...
✓ Body index loaded: 12543 terms
✓ Title index loaded: 1234 terms
✓ Anchor index loaded: 3567 terms
✓ Metadata loaded: 320 documents
⚠ PageRank not found, will return zeros
⚠ Page views not found, will return zeros
✓ All indices loaded successfully!

Starting Flask server on http://0.0.0.0:8080
Endpoints available:
  GET  /search?query=...
  GET  /search_body?query=...
  GET  /search_title?query=...
  GET  /search_anchor?query=...
  POST /get_pagerank  (json: [id1, id2, ...])
  POST /get_pageview  (json: [id1, id2, ...])

 * Serving Flask app 'search_frontend'
 * Debug mode: on
WARNING: This is a development server. Do not use it in a production deployment.
 * Running on all addresses (0.0.0.0)
 * Running on http://127.0.0.1:8080
 * Running on http://192.168.1.100:8080
Press CTRL+C to quit
 * Restarting with stat
 * Debugger is active!
```

**Note:** The PageRank and page view warnings are normal if you haven't created those files yet.

✅ **Checkpoint:** Server should be running on http://localhost:8080

---

## STEP 5: Test Endpoints Manually (macOS)

**Open a NEW terminal tab** (⌘+T) - keep the Flask server running in the first one.

**Navigate to project directory in new tab:**
```bash
cd "/Users/tomerfilo/Library/Mobile Documents/com~apple~CloudDocs/לימודים- שנה ג/סמסטר א/אחזור מידע/ir_proj_20251213 2"
```

### 5.1 Test `/search_body` (TF-IDF + Cosine)

```bash
# Use python3 and -s flag for silent curl on macOS
curl -s "http://localhost:8080/search_body?query=Mount+Everest" | python3 -m json.tool | head -n 30
```

**Expected Output:**
```json
[
  [
    "47353693",
    "Mount Everest"
  ],
  [
    "5208803",
    "1953 British Mount Everest expedition"
  ],
  [
    "20852640",
    "1924 British Mount Everest expedition"
  ],
  [
    "42179",
    "Edmund Hillary"
  ],
  [
    "361977",
    "Tenzing Norgay"
  ]
]
```

✅ **Check:** Results should contain "Mount Everest" articles ranked by relevance.

### 5.2 Test `/search_title` (Binary Ranking)

```bash
curl -s "http://localhost:8080/search_title?query=Great+Fire+London" | python3 -m json.tool | head -n 20
```

**Expected Output:**
```json
[
  [
    "7669549",
    "Great Fire of London"
  ],
  [
    "382247",
    "Fire of London"
  ],
  [
    "227331",
    "Great Fire"
  ]
]
```

✅ **Check:** Documents with MORE distinct query words in title should rank higher.

### 5.3 Test `/search_anchor` (Anchor Text)

```bash
curl -s "http://localhost:8080/search_anchor?query=Albert+Einstein" | python3 -m json.tool | head -n 20
```

**Expected Output:**
```json
[
  [
    "12",
    "Albert Einstein"
  ],
  [
    "5043734",
    "Theory of relativity"
  ]
]
```

✅ **Check:** Documents with anchor text containing "Albert Einstein" should appear.

### 5.4 Test `/get_pagerank` (POST - macOS)

```bash
curl -s -X POST http://localhost:8080/get_pagerank \
     -H "Content-Type: application/json" \
     -d '[47353693, 5208803, 12]' | python3 -m json.tool
```

**Expected Output:**
```json
[
  0.0,
  0.0,
  0.0
]
```

**Note:** Returns zeros because PageRank file doesn't exist yet (that's OK for testing).

✅ **Check:** Should return a list of 3 floats (all zeros if PageRank not available).

### 5.5 Test `/get_pageview` (POST - macOS)

```bash
curl -s -X POST http://localhost:8080/get_pageview \
     -H "Content-Type: application/json" \
     -d '[47353693, 5208803, 12]' | python3 -m json.tool
```

**Expected Output:**
```json
[
  0,
  0,
  0
]
```

✅ **Check:** Should return a list of 3 integers (all zeros if page views not available).

### 5.6 Test `/search` (Best Combined - macOS)

```bash
curl -s "http://localhost:8080/search?query=DNA+double+helix" | python3 -m json.tool | head -n 20
```

**Expected Output:**
```json
[
  [
    "90472",
    "DNA"
  ],
  [
    "16289",
    "Double helix"
  ],
  [
    "922489",
    "Molecular structure of DNA"
  ]
]
```

✅ **Check:** Should return relevant DNA-related articles.

### 5.7 Test Edge Cases (macOS)

**Empty Query:**
```bash
curl -s "http://localhost:8080/search_body?query=" | python3 -m json.tool
```
**Expected:** `[]` (empty list)

**Stopword-Only Query:**
```bash
curl -s "http://localhost:8080/search_body?query=the+and+or" | python3 -m json.tool
```
**Expected:** `[]` (empty list)

**Missing Term:**
```bash
curl -s "http://localhost:8080/search_body?query=xyznotinindex" | python3 -m json.tool
```
**Expected:** `[]` (empty list)

✅ **Check:** All edge cases should return empty lists WITHOUT crashing the server.

---

## STEP 6: Evaluate MAP@10 Performance

### 6.1 Run Evaluation Script (macOS)

```bash
# Make sure you're in the project directory
cd "/Users/tomerfilo/Library/Mobile Documents/com~apple~CloudDocs/לימודים- שנה ג/סמסטר א/אחזור מידע/ir_proj_20251213 2"

# Run evaluation
python3 scripts/evaluate_map.py --server http://localhost:8080 --endpoint search
```

**Expected Output:**
```
======================================================================
MAP@K EVALUATION SCRIPT
======================================================================
Server: http://localhost:8080
Endpoint: /search
Queries file: data/queries_train.json
Cutoff K: 10

Loading queries from data/queries_train.json...
✓ Loaded 30 queries

Evaluating 30 queries on /search endpoint...
Metric: Average Precision @ 10
======================================================================

[1/30] Query: 'Mount Everest climbing expeditions...'
  AP@10: 0.4523
  Relevant docs: 46
  Returned docs: 100
  Top-5: ['47353693', '5208803', '20852640', '37943414', '42179']
  Relevant in top-5: ['47353693', '5208803', '20852640']

[2/30] Query: 'Great Fire of London 1666...'
  AP@10: 0.3214
  Relevant docs: 39
  Returned docs: 85
  Top-5: ['7669549', '382247', '9914015', '6825961', '227331']
  Relevant in top-5: ['7669549', '382247', '9914015']

... (28 more queries)

======================================================================
EVALUATION SUMMARY
======================================================================
MAP@10: 0.2345
Total queries: 30
✓ PASS: MAP@10 ≥ 0.1 (requirement met)

Best 5 queries:
  0.8123 - 'Photography invention Daguerre...'
  0.7456 - 'DNA double helix discovery...'
  0.6789 - 'Stonehenge prehistoric monument...'
  0.5234 - 'Printing press invention Gutenberg...'
  0.4567 - 'Great Fire of London 1666...'

Worst 5 queries:
  0.0234 - 'Fossil fuels climate change...'
  0.0456 - 'Industrial Revolution steam engines...'
  0.0789 - 'Green Revolution agriculture yield...'
  0.0912 - 'Quantum computing future technology...'
  0.1123 - 'Silk Road trade cultural exchange...'
======================================================================

Results saved to evaluation_results_search_map10.json
```

### 6.2 Interpret Results

**MAP@10 ≥ 0.1?**
- ✅ **PASS:** Your search engine meets the minimum requirement!
- ❌ **FAIL:** Debug worst-performing queries (see next section)

**Typical MAP@10 Scores:**
- `0.1 - 0.3`: Meets requirement, but has room for improvement
- `0.3 - 0.5`: Good performance
- `0.5+`: Excellent performance

### 6.3 If MAP@10 < 0.1 (Debugging - macOS)

**Check worst queries:**
```bash
python3 scripts/evaluate_map.py --quiet
```

**Manually inspect a failing query:**
```bash
# Pick a query from "Worst 5 queries"
curl -s "http://localhost:8080/search?query=Fossil+fuels+climate+change" | python3 -m json.tool | head -n 30

# Check if results make sense
# Are the top-10 results relevant?
```

**Common issues:**
1. **Stemming mismatch:** Verify indices built with `stem=False`
2. **IDF too low:** Check natural log is used (`math.log`, not `log10`)
3. **No PageRank:** Try adding PageRank scores (improves ranking)

---

## STEP 7: Verify Implementation Quality

### 7.1 Check TF-IDF Scores Are Decreasing (macOS)

```bash
# Add this to search_frontend.py temporarily to see scores:
# In search_body() function, after sorting:
# for doc_id, score in sorted_docs[:10]:
#     print(f"Doc {doc_id}: score={score:.6f}")

curl -s "http://localhost:8080/search_body?query=Mount+Everest"
```

**Check Flask console output:**
```
Doc 47353693: score=0.856234
Doc 5208803: score=0.723451
Doc 20852640: score=0.689123
Doc 37943414: score=0.567890
Doc 42179: score=0.523456
... (scores should decrease)
```

✅ **Check:** Scores should decrease monotonically (each score ≤ previous score).

### 7.2 Check Binary Ranking Counts Distinct Words (macOS)

Test with repeated query words:

```bash
# Query 1: "Fire London"
curl -s "http://localhost:8080/search_title?query=Fire+London" | python3 -m json.tool > /tmp/query1.json

# Query 2: "Fire Fire Fire London" (repeated)
curl -s "http://localhost:8080/search_title?query=Fire+Fire+Fire+London" | python3 -m json.tool > /tmp/query2.json

# Compare results (should be IDENTICAL on macOS)
diff /tmp/query1.json /tmp/query2.json
```

**Expected:** No differences (both queries should return same ranking).

✅ **Check:** Repeated query words should NOT change ranking.

### 7.3 Check Query Response Time (macOS)

```bash
# Use time command (built into macOS bash/zsh)
time curl -s "http://localhost:8080/search_body?query=Mount+Everest" > /dev/null
```

**Expected Output:**
```
real    0m0.123s   # Should be < 1 second for mini corpus
user    0m0.012s
sys     0m0.008s
```

✅ **Check:** Query time should be < 1 second on mini corpus (320 docs).

---

## ✅ SUCCESS CRITERIA CHECKLIST

Before moving to GCP, verify ALL of these:

- [ ] **Mini corpus created:** 320 doc IDs in `data/mini_corpus_doc_ids.json`
- [ ] **Indices built:** 8 files in `indices_mini/` (~28 MB total)
- [ ] **Unit tests pass:** All 16 tests green
- [ ] **Server starts:** No errors when loading indices
- [ ] **All 6 endpoints work:** Return non-empty results (except edge cases)
- [ ] **Edge cases handled:** Empty/stopword/missing queries return `[]`
- [ ] **MAP@10 ≥ 0.1:** Meets minimum requirement
- [ ] **Scores decrease:** TF-IDF scores monotonic
- [ ] **Binary ranking correct:** Distinct word count (not frequencies)
- [ ] **Query time < 1s:** Fast on mini corpus

---

## 🐛 Troubleshooting (macOS-Specific)

### Problem: "python: command not found"

**Cause:** macOS uses `python3` instead of `python`

**Fix:**
```bash
# Always use python3 on macOS
python3 scripts/create_mini_corpus.py

# Or create an alias (add to ~/.zshrc or ~/.bash_profile)
echo "alias python=python3" >> ~/.zshrc
source ~/.zshrc
```

### Problem: SSL Certificate Errors (NLTK download fails)

**Cause:** macOS SSL certificate issues

**Fix:**
```bash
# Option 1: Run certificate installer
/Applications/Python\ 3.*/Install\ Certificates.command

# Option 2: Download with SSL workaround
python3 -c "import ssl; ssl._create_default_https_context = ssl._create_unverified_context; import nltk; nltk.download('stopwords')"
```

### Problem: "No module named 'backend'"

**Cause:** Python can't find the backend package on macOS

**Fix:**
```bash
# Set PYTHONPATH to current directory
export PYTHONPATH="${PWD}:${PYTHONPATH}"

# Run your script
python3 backend/search_frontend.py

# Or make it permanent (add to ~/.zshrc)
echo 'export PYTHONPATH="${PWD}:${PYTHONPATH}"' >> ~/.zshrc
```

### Problem: "No documents loaded!"

**Cause:** Wikipedia dump path is wrong or format doesn't match.

**Fix (macOS):**
```bash
# Check file exists
ls -lh /path/to/your/dump.json

# Check format (JSON/Parquet/JSONL)
file /path/to/your/dump.json

# Update load_wikipedia_dump() function in build_mini_indices.py
```

### Problem: "Empty results for all queries"

**Cause:** Stemming mismatch (index built WITH stemming, queries WITHOUT).

**Fix (macOS):**
```bash
# Verify stemming (use python3)
python3 -c "from backend.inverted_index_gcp import InvertedIndex; idx = InvertedIndex.read_index('indices_mini', 'body_index'); print('climbing' in idx.df, 'climb' in idx.df)"

# Should print: True False (unstemmed)
# If prints: False True → rebuild indices with FIXED index_builder.py
```

### Problem: "MAP@10 < 0.1"

**Causes:**
1. Wrong IDF formula (`log10` instead of natural `log`)
2. Cosine similarity not implemented correctly
3. Missing PageRank (reduces quality)

**Fix (macOS):**
```bash
# Re-run unit tests
python3 tests/test_search_endpoints.py

# Check Flask logs in the server terminal window
# Manually inspect top-10 results
python3 scripts/evaluate_map.py --quiet
```

### Problem: "Server crashes on query"

**Cause:** Unhandled exception in endpoint code.

**Fix (macOS):**
```bash
# Check Flask console for error traceback
# Look for lines starting with "Traceback"

# Common issues:
# - Missing metadata file
# - Index file corruption
# - Division by zero (check norm handling)

# Restart server with verbose errors
python3 backend/search_frontend.py
```

### Problem: "Port 8080 already in use"

**Cause:** Another process using port 8080 on macOS

**Fix (macOS):**
```bash
# Find process using port 8080
lsof -i :8080

# Kill the process (replace PID with actual number)
kill -9 <PID>

# Or kill all processes on port 8080
kill -9 $(lsof -ti :8080)

# Then restart Flask server
python3 backend/search_frontend.py
```

### Problem: "Permission denied" when installing packages

**Cause:** Trying to install system-wide on macOS

**Fix (macOS):**
```bash
# Install for user only (no sudo needed)
python3 -m pip install --user flask nltk numpy requests

# Or use a virtual environment (recommended)
python3 -m venv venv
source venv/bin/activate
pip install flask nltk numpy requests
```

### Problem: Hebrew characters display incorrectly in Terminal

**Cause:** Terminal encoding issue on macOS

**Fix (macOS):**
```bash
# Set UTF-8 encoding (add to ~/.zshrc)
export LANG=en_US.UTF-8
export LC_ALL=en_US.UTF-8

# Reload
source ~/.zshrc

# Or change Terminal preferences:
# Terminal → Preferences → Profiles → Advanced → Character encoding → UTF-8
```

---

## 📞 Next Steps

Once ALL success criteria are met:

1. **Test on more queries:** Try custom queries beyond the 30 training ones
2. **Add PageRank:** If available, add `pagerank.pkl` to improve `/search`
3. **Add page views:** If available, add `pageviews.pkl`
4. **Optimize:** Use champion lists if query time approaches 35s limit
5. **Scale to GCP:** Build full 6M doc indices, deploy to Google Cloud Platform

---

**You're ready when:**
- ✅ All 16 unit tests pass
- ✅ All 6 endpoints return sensible results
- ✅ MAP@10 ≥ 0.1 on mini corpus
- ✅ No crashes on edge cases

**Then proceed to full corpus deployment!** 🚀
