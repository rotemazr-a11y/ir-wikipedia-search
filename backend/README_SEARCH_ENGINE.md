# Wikipedia Search Engine - Complete Guide

## Overview

This is a high-performance Wikipedia search engine running on Google Cloud Platform with the following architecture:

- **Frontend**: Flask REST API (`search_frontend.py`)
- **Runtime**: Pre-loaded search engine (`search_runtime.py`)
- **Ranking**: BM25 for body, term-count for title/anchor
- **Fusion**: Weighted score combination (0.5 Body + 0.3 Title + 0.2 Anchor)
- **Storage**: GCS bucket with `.bin` files in root, `.pkl` files local
- **Authentication**: GCS SDK with service account (no public access)

## Architecture

```
search_frontend.py (Flask API)
    ↓
search_runtime.py (SearchEngine class)
    ↓
inverted_index_gcp.py (InvertedIndex, MultiFileReader)
    ↓
GCS Bucket: gs://206969750_bucket/ (ROOT - no subdirectories)
    - body_index_0_000.bin
    - body_index_0_001.bin
    - title_index_0_000.bin
    - anchors_index_0_000.bin
    - ... (all .bin files together in root)

Local VM: indices_mini/ (FLAT - no subdirectories)
    - body_index.pkl
    - title_index.pkl
    - anchors_index.pkl
    - metadata.pkl
    - pagerank.pkl
    - pageviews.pkl
```

## Critical Fixes Implemented

### 1. Path Alignment Fix
**Problem**: Code was looking for `.bin` files in subdirectories, but they're in bucket root.

**Solution**: Use `INDEX_GCS_DIR = '.'` to search in bucket root.

```python
# In search_runtime.py
INDEX_GCS_DIR = '.'  # Root of bucket where .bin files are stored

# When reading posting lists
posting_list = self.body_index.read_a_posting_list(
    self.INDEX_GCS_DIR,  # Uses '.' to read from bucket root
    term,
    bucket_name=self.bucket_name
)
```

### 2. GCS Authentication Fix
**Problem**: Bucket has Public Access Prevention enabled.

**Solution**: Use `google-cloud-storage` SDK with service account authentication.

```python
# In search_runtime.py
self.storage_client = storage.Client()  # Authenticates with VM service account
```

### 3. Query Tokenization Fix
**Problem**: Queries weren't lowercased to match index terms.

**Solution**: Always use `.lower()` before tokenization.

```python
query_tokens = tokenize_and_process(
    query.lower(),  # CRITICAL: Match index term casing
    remove_stops=True,
    stem=True
)
```

### 4. Empty Results Fix
**Problem**: Incorrect path handling caused empty results.

**Solution**: Proper separation of local `.pkl` paths and GCS `.bin` paths.

```python
# .pkl files: Load from flat local directory (no subdirectories)
self.body_index = InvertedIndex.read_index(
    self.local_index_dir,  # Local: indices_mini/ (FLAT)
    "body_index",
    bucket_name=None  # No bucket needed for local files
)

# .bin files: Read from GCS bucket root (all .bin files together)
posting_list = self.body_index.read_a_posting_list(
    '.',  # GCS root directory (FLAT - no subdirectories)
    term,
    bucket_name='206969750_bucket'  # Use GCS with auth
)
```

## Configuration

### Constants (in `search_runtime.py`)

```python
BUCKET_NAME = '206969750_bucket'        # Your GCS bucket
INDEX_LOCAL_DIR = 'indices_mini'        # Local .pkl files
INDEX_GCS_DIR = '.'                     # Root of bucket for .bin files

# BM25 Parameters
BM25_K1 = 1.5
BM25_B = 0.75

# Fusion Weights
WEIGHT_BODY = 0.5
WEIGHT_TITLE = 0.3
WEIGHT_ANCHOR = 0.2
```

### Constants (in `search_frontend.py`)

```python
HOST = '0.0.0.0'
PORT = 8080
LOCAL_INDEX_DIR = 'indices_mini'
BUCKET_NAME = '206969750_bucket'
```

## File Structure

```
backend/
├── search_frontend.py          # Flask REST API
├── search_runtime.py           # Search engine core (NEW)
├── inverted_index_gcp.py       # Index I/O with GCS
├── pre_processing.py           # Tokenization
├── pyspark_index_builder.py   # Index building (Spark)
└── indices_mini/               # Local index files (FLAT - no subdirectories)
    ├── body_index.pkl
    ├── title_index.pkl
    ├── anchors_index.pkl
    ├── metadata.pkl
    ├── pagerank.pkl
    └── pageviews.pkl
```

## API Endpoints

### 1. Main Search (Multi-Index Fusion)
```bash
GET /search?query=machine+learning
```

**Response**: `[["doc_id", "title"], ...]`

**Logic**:
1. Parallel search across body (BM25), title (count), anchor (count)
2. Weighted fusion: `0.5*body + 0.3*title + 0.2*anchor`
3. Return top 100 results

### 2. Body Search (BM25)
```bash
GET /search_body?query=neural+networks
```

**Response**: `[["doc_id", "title"], ...]`

**Logic**: BM25 ranking on body index with stemming

### 3. Title Search (Term Count)
```bash
GET /search_title?query=machine+learning
```

**Response**: `[["doc_id", "title"], ...]`

**Logic**: Simple term-count ranking on title index (no stemming)

### 4. Anchor Search (Term Count)
```bash
GET /search_anchor?query=python+programming
```

**Response**: `[["doc_id", "title"], ...]`

**Logic**: Simple term-count ranking on anchor index (no stemming)

### 5. PageRank Scores
```bash
POST /get_pagerank
Content-Type: application/json

[12, 34, 56]
```

**Response**: `[0.5, 0.3, 0.7]`

### 6. PageView Counts
```bash
POST /get_pageview
Content-Type: application/json

[12, 34, 56]
```

**Response**: `[1000, 500, 2000]`

## Running the Server

### On GCP VM

```bash
# 1. SSH to your VM
gcloud compute ssh your-vm-name --zone=us-central1-a

# 2. Navigate to backend directory
cd /path/to/backend/

# 3. Ensure indices are downloaded
# .pkl files should be in indices_mini/
# .bin files should be in gs://206969750_bucket/ (root)

# 4. Run the server
python3 search_frontend.py
```

### Expected Startup Log

```
================================================================================
Starting Wikipedia Search Engine
================================================================================
Host: 0.0.0.0
Port: 8080
Local index directory: indices_mini
GCS bucket: 206969750_bucket
GCS .bin files location: root of bucket (.)
================================================================================
Initializing search engine...
Initializing SearchEngine...
Local index dir: indices_mini
GCS bucket: 206969750_bucket
GCS .bin files location: root of bucket (.)
✓ GCS client initialized with service account
✓ Body index loaded: 45231 terms
✓ Title index loaded: 12453 terms
✓ Anchor index loaded: 8932 terms
✓ Metadata loaded: 6831681 titles
✓ SearchEngine initialization complete
  Total documents: 6,831,681
  Titles loaded: 6,831,681
✓ Search engine ready
================================================================================
Starting Flask server...
================================================================================
 * Running on http://0.0.0.0:8080
```

## Testing

```bash
# Test main search
curl "http://localhost:8080/search?query=machine+learning"

# Test body search
curl "http://localhost:8080/search_body?query=neural+networks"

# Test title search
curl "http://localhost:8080/search_title?query=python"

# Test anchor search
curl "http://localhost:8080/search_anchor?query=programming"

# Test PageRank
curl -X POST http://localhost:8080/get_pagerank \
  -H "Content-Type: application/json" \
  -d '[12, 34, 56]'
```

## Key Differences from Old Code

| Feature | Old Code | New Code |
|---------|----------|----------|
| Architecture | Monolithic | Separated runtime + frontend |
| Index Loading | At every request | Once at startup |
| Parallelism | ThreadPoolExecutor in route | Built into SearchEngine |
| GCS Path | Subdirectories | Root (`.`) |
| Query Tokenization | Missing `.lower()` | Always `.lower()` |
| Ranking | TF-IDF cosine | BM25 for body |
| Error Handling | 500 errors | Returns `[]` |
| GCS Access | HTTP (fails with PAP) | SDK with auth |

## Troubleshooting

### Empty Results `[]`

**Cause**: Index files not found or path mismatch

**Fix**:
1. Check `.pkl` files exist in `indices_mini/` (FLAT directory):
   ```bash
   ls indices_mini/
   # Should show: body_index.pkl, title_index.pkl, anchors_index.pkl, metadata.pkl
   ```
2. Check `.bin` files exist in `gs://206969750_bucket/` (root, all together):
   ```bash
   gsutil ls gs://206969750_bucket/*.bin | head
   # Should show: body_index_0_000.bin, title_index_0_000.bin, etc.
   ```
3. Verify bucket name is correct
4. Check VM service account has Storage Object Viewer role

### 500 Internal Server Error

**Cause**: Missing dependencies or index files

**Fix**:
```bash
pip install flask google-cloud-storage nltk
python3 -c "import nltk; nltk.download('stopwords')"
```

### GCS Permission Denied

**Cause**: VM service account lacks permissions

**Fix**:
```bash
# Grant Storage Object Viewer role to VM service account
gcloud projects add-iam-policy-binding assignment3-479509 \
  --member="serviceAccount:YOUR-VM-SA@assignment3-479509.iam.gserviceaccount.com" \
  --role="roles/storage.objectViewer"
```

### Slow Search (> 2 seconds)

**Cause**: Index not pre-loaded or too many terms

**Fix**:
- Ensure `initialize_engine()` runs at startup (not per-request)
- Check logs for "✓ Search engine ready"
- Consider adding query term limit

## Performance Characteristics

- **Startup Time**: 30-60 seconds (one-time index loading)
- **Query Latency**: 50-200ms (parallel search + fusion)
- **Memory Usage**: ~2-4 GB (pre-loaded indices)
- **Throughput**: ~50-100 queries/second (threaded Flask)

## Gaia's Principles Applied

✓ **Pre-loaded Engine**: All indices loaded once at startup
✓ **Parallel Search**: ThreadPoolExecutor for body/title/anchor
✓ **BM25 Ranking**: Proper BM25 implementation for body search
✓ **Score Fusion**: Weighted combination of multiple signals
✓ **Safety**: Try-except blocks prevent 500 errors
✓ **GCS SDK**: Proper authentication with storage.Client()
✓ **Path Correctness**: Root directory (`.`) for .bin files
✓ **Tokenization**: Always lowercase queries to match index

## Next Steps

1. **Test with sample queries** to verify results are non-empty
2. **Monitor logs** for any errors during index loading
3. **Optimize BM25 parameters** (k1, b) based on evaluation metrics
4. **Adjust fusion weights** (body/title/anchor) for best results
5. **Add caching** for frequent queries (optional)
6. **Implement query expansion** (optional)

## Support

If you encounter issues:
1. Check the startup logs for ✓ vs ✗ indicators
2. Verify all paths and bucket names are correct
3. Ensure VM service account has proper GCS permissions
4. Test with simple queries first (e.g., "python")
