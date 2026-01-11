# Production Deployment Guide - Direct GCS Access

## Overview

The search engine has been refactored for production deployment with **100% direct GCS access**. No local file downloads are required.

## Key Changes

### 1. **search_runtime.py** - Complete Refactor

#### Hardcoded GCS Paths (Lines 20-38)
```python
BUCKET_NAME = '206969750_bucket'

# Index paths in GCS
BODY_INDEX_PATH = 'postings_out/body/index.pkl'
TITLE_INDEX_PATH = 'postings_gcp/title/index.pkl'
ANCHOR_INDEX_PATH = 'postings_gcp/anchor/index.pkl'

# Binary posting lists directories
BODY_POSTINGS_DIR = 'postings_out/body/'
TITLE_POSTINGS_DIR = 'postings_gcp/title/'
ANCHOR_POSTINGS_DIR = 'postings_gcp/anchor/'
```

#### GCS Client Singleton (Lines 61-72)
- Single global GCS client initialized once
- Avoids overhead of multiple client instances
- Thread-safe access pattern

#### Graceful Missing File Handling (Lines 75-108)
```python
def load_pickle_from_gcs(blob_path: str, required: bool = True):
    """
    Load pickle from GCS with graceful error handling.

    - required=True: Raises error if missing
    - required=False: Returns {} if missing (no crashes)
    """
```

**Missing Files Handled Gracefully:**
- `doc_len.pkl` → Uses defaults: `num_docs=6344269`, `avg_len=450.0`
- `titles.pkl` → Returns `f"Document {doc_id}"` as fallback
- `pagerank.pkl` → Returns `0.0` for missing scores
- `pageviews.pkl` → Returns `0` for missing counts

#### Direct GCS Binary Reading (Lines 207-267)
```python
def read_posting_list(self, term: str, posting_locs: list, df: int,
                      postings_dir: str) -> List[Tuple[int, int]]:
    """
    Read posting lists directly from GCS .bin files.
    Uses blob.download_as_bytes(start=offset, end=offset+n_bytes)
    """
```

**No local storage used** - All reads are byte-range requests directly from GCS.

#### Comprehensive Error Handling (Lines 278-321)
Every search method wrapped in try-except:
- `search_body_bm25()` - Returns `{}` on BM25 calculation errors
- `search_title()` - Returns `{}` on title search errors
- `search_anchor()` - Returns `{}` on anchor search errors
- `search()` - Returns `[]` on fusion errors

**Result:** Flask workers never crash due to missing metadata or failed calculations.

### 2. **search_frontend.py** - Production Hardening

#### Query Preprocessing Applied Everywhere (Lines 111-122, 169-178, 226-234)
All endpoints now apply `tokenize_and_process()` to queries:

```python
# /search_body
query_tokens = tokenize_and_process(
    raw_query.lower(),
    remove_stops=True,
    stem=True  # Body index uses stemming
)

# /search_title and /search_anchor
query_tokens = tokenize_and_process(
    raw_query.lower(),
    remove_stops=True,
    stem=False  # Title/anchor don't use stemming
)
```

#### Comprehensive Logging
Every endpoint logs:
- Raw query received
- Preprocessed tokens
- Number of results returned
- Any errors encountered (with full stack traces)

#### Error Recovery
All endpoints return empty lists `[]` on errors instead of crashing:
```python
except Exception as e:
    logger.error(f"[ENDPOINT] Error: {e}", exc_info=True)
    return jsonify([])
```

## Deployment Steps

### 1. Verify GCS File Paths

Ensure these files exist in your bucket:

```bash
gsutil ls gs://206969750_bucket/postings_out/body/
# Should show: index.pkl, *.bin files

gsutil ls gs://206969750_bucket/postings_gcp/title/
# Should show: index.pkl, *.bin files

gsutil ls gs://206969750_bucket/postings_gcp/anchor/
# Should show: index.pkl, *.bin files
```

### 2. Deploy to GCP VM

```bash
# Copy files to VM
gcloud compute scp backend/* YOUR_VM_NAME:~/backend/ --zone YOUR_ZONE

# SSH into VM
gcloud compute ssh YOUR_VM_NAME --zone YOUR_ZONE

# Install dependencies
pip3 install google-cloud-storage flask nltk

# Start server
cd ~/backend
python3 search_frontend.py
```

### 3. Test Endpoints

```bash
# Test main search
curl "http://YOUR_VM_IP:8080/search?query=machine+learning"

# Test body search
curl "http://YOUR_VM_IP:8080/search_body?query=python+programming"

# Test title search
curl "http://YOUR_VM_IP:8080/search_title?query=artificial+intelligence"

# Test PageRank
curl -X POST "http://YOUR_VM_IP:8080/get_pagerank" \
  -H "Content-Type: application/json" \
  -d '[12, 345, 6789]'
```

## Configuration Constants

All paths are hardcoded at the top of `search_runtime.py` (lines 20-52). To change paths, edit these constants:

```python
# Change bucket name
BUCKET_NAME = 'your_bucket_name'

# Change index paths
BODY_INDEX_PATH = 'your/path/to/body/index.pkl'
TITLE_INDEX_PATH = 'your/path/to/title/index.pkl'
ANCHOR_INDEX_PATH = 'your/path/to/anchor/index.pkl'

# Change posting lists directories
BODY_POSTINGS_DIR = 'your/path/to/body/'
TITLE_POSTINGS_DIR = 'your/path/to/title/'
ANCHOR_POSTINGS_DIR = 'your/path/to/anchor/'
```

## Performance Optimizations

1. **Singleton GCS Client** - Initialized once, reused for all requests
2. **Parallel Index Search** - ThreadPoolExecutor searches body/title/anchor simultaneously
3. **Byte-Range Requests** - Only downloads needed bytes from .bin files
4. **Lazy Loading** - Indices loaded once at startup, not per request

## Error Handling Strategy

| Component | Missing File Behavior |
|-----------|---------------------|
| Body Index | **Fails** - Required for search |
| Title Index | **Fails** - Required for search |
| Anchor Index | **Fails** - Required for search |
| doc_len.pkl | Uses defaults (num_docs=6.3M, avg_len=450) |
| titles.pkl | Returns "Document {doc_id}" |
| pagerank.pkl | Returns 0.0 for all docs |
| pageviews.pkl | Returns 0 for all docs |

## Troubleshooting

### Issue: "No module named 'google.cloud'"
```bash
pip3 install google-cloud-storage
```

### Issue: "Permission denied" when accessing GCS
```bash
# Set credentials
export GOOGLE_APPLICATION_CREDENTIALS="/path/to/service-account-key.json"

# Or use VM default service account (recommended)
gcloud compute instances set-service-account YOUR_VM_NAME \
  --service-account YOUR_SERVICE_ACCOUNT \
  --scopes cloud-platform
```

### Issue: Empty results `[]`
Check logs for:
1. Query preprocessing: Are tokens being generated?
2. Index loading: Did all indices load successfully?
3. Posting list reading: Are .bin files accessible?

```bash
# View logs
journalctl -u search-engine -f  # If using systemd
# OR
python3 search_frontend.py 2>&1 | tee search.log
```

## Production Checklist

- [x] No local file operations (all GCS direct access)
- [x] Graceful handling of missing metadata files
- [x] Query preprocessing applied to all endpoints
- [x] Comprehensive error handling (no worker crashes)
- [x] Singleton GCS client (no overhead)
- [x] All paths hardcoded as constants
- [x] Logging at INFO level for all operations
- [x] Returns empty lists instead of errors

## Architecture Summary

```
┌─────────────────┐
│  Flask Request  │
└────────┬────────┘
         │
         ▼
┌─────────────────────────────┐
│ tokenize_and_process(query) │ ← Lowercase, stopword removal, stemming
└────────┬────────────────────┘
         │
         ▼
┌──────────────────────────────┐
│  SearchEngine.search(tokens) │
└────────┬─────────────────────┘
         │
         ├──► search_body_bm25() ──┬──► read_posting_list() ──► GCS: postings_out/body/*.bin
         │                          └──► BM25 scoring with doc_lengths fallback
         │
         ├──► search_title() ───────┬──► read_posting_list() ──► GCS: postings_gcp/title/*.bin
         │                          └──► Term-count ranking
         │
         └──► search_anchor() ──────┬──► read_posting_list() ──► GCS: postings_gcp/anchor/*.bin
                                    └──► Term-count ranking
         │
         ▼
┌─────────────────────┐
│  Score Fusion       │ ← 0.5*Body + 0.3*Title + 0.2*Anchor
└────────┬────────────┘
         │
         ▼
┌─────────────────────┐
│  Top 100 Results    │
└─────────────────────┘
```

## Next Steps

1. **Monitor Performance**: Track query latency and GCS read times
2. **Optimize Weights**: Tune `WEIGHT_BODY`, `WEIGHT_TITLE`, `WEIGHT_ANCHOR` for better results
3. **Add Caching**: Consider caching frequently accessed posting lists
4. **Scale Horizontally**: Deploy multiple Flask workers behind load balancer
