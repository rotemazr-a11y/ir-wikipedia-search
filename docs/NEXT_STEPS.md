# IR Wikipedia Search Engine - Project Status

> **Last Updated**: January 18, 2026  
> **Session Context**: For continuing work in new sessions

---

## 🎯 Project Overview

Building a Wikipedia search engine with inverted indices, PageRank, and PageView scoring.

### Key Reference Documents
| Document | Location | Purpose |
|----------|----------|---------|
| **Project Requirements** | `docs/ir_final_rpoject.md` | Full assignment specification |
| **GCP Instructions** | `docs/working_with_gcp.md` | Dataproc setup, bucket config |
| **Claude Prompt** | `docs/CLAUDE_PROMPT.md` | Original planning context |
| **Training Queries** | `queries_train.json` | 30 queries for evaluation |

### GCP Configuration
- **Project**: `durable-tracer-479509-q2`
- **Bucket**: `bucket_207916263`
- **Cluster**: `ir-cluster` (us-central1)
- **Corpus**: `gs://bucket_207916263/corpus` (6.3M Wikipedia docs as Parquet)

---

## ✅ Work Completed

### 1. Indexing Pipeline (`indexing/`)

| File | Description | Status |
|------|-------------|--------|
| `pre_processing.py` | Tokenization (RE_WORD regex), hardcoded stopwords (NLTK-compatible), optional Porter stemming | ✅ Fixed (removed NLTK dependency for GCP) |
| `spimi_block_builder.py` | SPIMI algorithm with memory-bounded blocks (250MB body, 100MB title, 150MB anchor), k-way merge | ✅ Complete |
| `pyspark_index_builder.py` | Main Spark driver using `mapPartitions` to avoid OOM, writes 3 indices | ✅ Complete |
| `inverted_index_gcp.py` | InvertedIndex class with MultiFileWriter for binary posting lists (.bin + .pkl) | ✅ Complete |

### 2. PageRank, PageView & Metadata (`indexing/`)

| File | Description | Status |
|------|-------------|--------|
| `compute_pagerank.py` | Iterative DataFrame PageRank (10 iterations, damping=0.85, checkpoint every 3 iters) | ✅ Written, needs testing after index |
| `compute_pageviews.py` | Parse Wikipedia pageview dumps, aggregate by doc_id | ✅ Written, needs testing after index |
| `compute_doc_metadata.py` | Extract doc_titles.pkl and doc_lengths.pkl from corpus | ✅ NEW - needed for frontend |

### 3. Search Frontend (`frontend/`)

| File | Description | Status |
|------|-------------|--------|
| `search_frontend.py` | Flask API with 6 endpoints, TF-IDF cosine similarity, binary ranking, combined search | ✅ Complete |

### 4. Tests (`tests/`)

| File | Description | Status |
|------|-------------|--------|
| `test_index_builder.py` | 15 unit tests for SPIMI and index building | ✅ 15/15 passed |
| `test_index_structure.py` | Structure validation tests, GCS output validator | ✅ 10 passed, 16 skipped (no GCS locally) |
| `test_local_pipeline.py` | End-to-end local integration test | ✅ Complete |

---

## 🔄 Current Status: Indexing Job Running ✅

### Job Details
- **Job ID**: `94fb8087280740b2bf8310ead1155444`
- **Cluster**: `ir-cluster`
- **Status**: 🔄 Running - Processing body index (submitted Jan 18, 2026 ~14:20 UTC)
- **Expected Duration**: 30-60 minutes for full indexing

### Issues Fixed This Session

1. **NLTK Stopwords Error** (Fixed)
   - Problem: `PermissionError: [Errno 13] Permission denied: '/home/nltk_data'`
   - Solution: Hardcoded stopwords in `pre_processing.py` (removed NLTK dependency)

2. **Job Hanging on Parquet Read** (Fixed)
   - Problem: Job stuck at "Reading corpus from Parquet files..." for 20+ minutes
   - Cause: `docs_rdd.count()` triggered expensive full scan before processing
   - Solution: Removed `count()` call, use lazy evaluation

### Commands Used (Working)
```bash
# From local machine
cd /Users/rotemazriel/Documents/University/2026/Semester\ A/IR/Assignments/PROJECT/indexing

# Create deps zip with fixed code
zip deps.zip pre_processing.py spimi_block_builder.py inverted_index_gcp.py

# Upload to GCS
gsutil cp deps.zip gs://bucket_207916263/code/
gsutil cp pyspark_index_builder.py gs://bucket_207916263/code/

# Submit job
gcloud dataproc jobs submit pyspark \
    --cluster=ir-cluster \
    --region=us-central1 \
    --py-files=gs://bucket_207916263/code/deps.zip \
    gs://bucket_207916263/code/pyspark_index_builder.py \
    -- \
    --input gs://bucket_207916263/corpus \
    --output gs://bucket_207916263/indices \
    --bucket bucket_207916263 \
    --partitions 200
```

### Expected Timeline
| Phase | Duration | Log Message |
|-------|----------|-------------|
| Reading corpus metadata | 1-3 min | `Reading corpus from Parquet files...` |
| Corpus loaded | instant | `Corpus loaded: 6348910 documents, 200 partitions` |
| SPIMI processing | 30-60 min | `Processing body index...` + stage progress |
| Writing indices | 5-10 min | GCS write operations |

---

## 📋 Next Steps After Indexing Completes

### Step 1: Verify Index Output
```bash
gsutil ls gs://bucket_207916263/indices/
gsutil ls gs://bucket_207916263/indices/body_index/
```

### Step 2: Extract Document Metadata (REQUIRED)
```bash
# Upload script
gsutil cp indexing/compute_doc_metadata.py gs://bucket_207916263/code/

# Submit job - generates doc_titles.pkl and doc_lengths.pkl
gcloud dataproc jobs submit pyspark \
    gs://bucket_207916263/code/compute_doc_metadata.py \
    --cluster=ir-cluster \
    --region=us-central1 \
    -- \
    --input gs://bucket_207916263/corpus \
    --bucket bucket_207916263
```

### Step 3: Run PageRank Job
```bash
# Upload script
gsutil cp indexing/compute_pagerank.py gs://bucket_207916263/code/

# Submit job
gcloud dataproc jobs submit pyspark \
    gs://bucket_207916263/code/compute_pagerank.py \
    --cluster=ir-cluster \
    --region=us-central1 \
    -- \
    --input gs://bucket_207916263/corpus \
    --output pagerank/pagerank.pkl \
    --bucket bucket_207916263
```

### Step 4: Run PageView Job
```bash
# Download pageview dump (~4GB) from Wikipedia
# https://dumps.wikimedia.org/other/pageview_complete/monthly/2021/2021-08/pageviews-202108-user.bz2
# Then upload to GCS:
# gsutil cp pageviews-202108-user.bz2 gs://bucket_207916263/

gsutil cp indexing/compute_pageviews.py gs://bucket_207916263/code/

gcloud dataproc jobs submit pyspark \
    gs://bucket_207916263/code/compute_pageviews.py \
    --cluster=ir-cluster \
    --region=us-central1 \
    -- \
    --input gs://bucket_207916263/pageviews-202108-user.bz2 \
    --output pageviews/pageviews.pkl \
    --bucket bucket_207916263
```

### Step 5: Verify All Files Exist
```bash
gsutil ls gs://bucket_207916263/
# Expected files:
# - indices/body_index/
# - indices/title_index/
# - indices/anchor_index/
# - doc_titles.pkl
# - doc_lengths.pkl
# - pagerank/pagerank.pkl
# - pageviews/pageviews.pkl
```

### Step 6: Test with GCS Validator
```bash
python3 tests/test_index_structure.py --validate-gcs --bucket bucket_207916263
```

### Step 5: Deploy Search Frontend
```bash
gsutil cp frontend/search_frontend.py gs://bucket_207916263/code/

# On VM:
pip install flask google-cloud-storage
python search_frontend.py
```

---

## 📁 Project Structure

```
PROJECT/
├── docs/
│   ├── CLAUDE_PROMPT.md          # Original planning context
│   ├── ir_final_rpoject.md       # Assignment requirements
│   ├── working_with_gcp.md       # GCP setup instructions
│   └── NEXT_STEPS.md             # This file
├── indexing/
│   ├── pre_processing.py         # Tokenizer (hardcoded stopwords)
│   ├── spimi_block_builder.py    # SPIMI algorithm
│   ├── pyspark_index_builder.py  # Main Spark driver
│   ├── inverted_index_gcp.py     # Index class + GCS I/O
│   ├── compute_pagerank.py       # PageRank calculator ⏳
│   ├── compute_pageviews.py      # PageView processor ⏳
│   └── deps.zip                  # Bundled dependencies for GCP
├── frontend/
│   └── search_frontend.py        # Flask API (6 endpoints)
├── tests/
│   ├── test_index_builder.py     # Unit tests
│   ├── test_index_structure.py   # Structure + GCS validator
│   └── test_local_pipeline.py    # Integration test
├── scripts/                      # Utility scripts
└── queries_train.json            # 30 training queries
```

---

## 🔧 Expected GCS Output Structure

```
gs://bucket_207916263/
├── corpus/                       # Input: Wikipedia parquet files
├── code/                         # Uploaded Python scripts
│   ├── deps.zip
│   ├── pyspark_index_builder.py
│   ├── compute_pagerank.py
│   └── compute_pageviews.py
├── indices/                      # Output: Inverted indices
│   ├── body_index/
│   │   ├── body_index.pkl        # Metadata (df, posting_locs)
│   │   └── body_index_*.bin      # Binary posting lists
│   ├── title_index/
│   │   ├── title_index.pkl
│   │   └── title_index_*.bin
│   └── anchor_index/
│       ├── anchor_index.pkl
│       └── anchor_index_*.bin
├── pagerank/
│   └── pagerank.pkl              # Dict[doc_id, float]
├── pageviews/
│   └── pageviews.pkl             # Dict[doc_id, int]
├── doc_titles.pkl                # Dict[doc_id, str]
└── doc_lengths.pkl               # Dict[doc_id, int]
```

---

## ⚠️ Important Reminders

### Don't Forget!
1. **Delete cluster when done**: 
   ```bash
   gcloud dataproc clusters delete ir-cluster --region=us-central1 --quiet
   ```
2. **Check job status**:
   ```bash
   gcloud dataproc jobs describe 9b9ba55709f2468e8d7242be307c7a4f --region=us-central1
   ```

### Common Issues & Fixes

| Issue | Solution |
|-------|----------|
| NLTK stopwords error | ✅ Fixed: Using hardcoded stopwords |
| OOM errors | Reduce partitions, use `mapPartitions` not `flatMap` |
| GCS path errors | Always use `gs://bucket_207916263/...` format |
| Import errors on workers | Ensure all deps in `deps.zip` |

---

## 📊 Search Frontend Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/search` | GET | Combined search (TF-IDF + PageRank + PageView) |
| `/search_body` | GET | TF-IDF cosine similarity on body |
| `/search_title` | GET | Binary ranking on title matches |
| `/search_anchor` | GET | Binary ranking on anchor text |
| `/get_pagerank` | POST | PageRank scores for doc IDs |
| `/get_pageview` | POST | Page view counts for doc IDs |
| `/health` | GET | Health check |

---

## 🧪 PageRank & PageView Testing (Pending)

### What's Ready
- `compute_pagerank.py` - Iterative PageRank with DataFrames
- `compute_pageviews.py` - Pageview dump parser

### Tests Needed After Index Completes
1. Verify PageRank produces values in [0, 1]
2. Check top PageRank docs are high-quality articles
3. Verify PageView counts match expected magnitudes
4. Integration test with search_frontend.py

### Test Commands (After Index)
```bash
# Download sample outputs
gsutil cp gs://bucket_207916263/pagerank/pagerank.pkl ./
gsutil cp gs://bucket_207916263/pageviews/pageviews.pkl ./

# Inspect locally
python3 -c "
import pickle
with open('pagerank.pkl', 'rb') as f:
    pr = pickle.load(f)
print(f'PageRank entries: {len(pr)}')
print(f'Top 10: {sorted(pr.items(), key=lambda x: -x[1])[:10]}')
"
```
