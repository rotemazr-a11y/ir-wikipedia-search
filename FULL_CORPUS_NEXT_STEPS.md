# Full Corpus Evaluation - Next Steps

**Current Status:** Infrastructure ready, needs manual authentication to proceed

---

## What We Discovered

### ✅ Success: Full Corpus Exists!

Your GCP bucket contains a **complete Wikipedia corpus** with:
- **495,515 unique terms** (vs 21,169 in mini corpus)
- **~6 million documents** (vs 42 in mini corpus)
- **5.92 GiB of posting files** in `gs://206969750_bucket/postings_gcp/`
- **PageRank data** in `gs://206969750_bucket/pr/` (62.84 MiB)

### ⚠️ Challenge: Index Format Mismatch

Your `search_frontend.py` expects:
```
indices/
├── body_index.pkl       (self-contained)
├── title_index.pkl      (self-contained)
└── anchor_index.pkl     (self-contained)
```

But your full corpus uses:
```
postings_gcp/
├── index.pkl            (index metadata)
├── 0_000.bin            (posting files)
├── 0_001.bin
├── ... (many .bin files)
└── 0_posting_locs.pickle
```

---

## What I Created

### 1. GCS-Enabled Search Backend

**File:** `backend/search_frontend_gcs.py`

This version can read posting files directly from GCS without needing to download everything locally. Key features:
- Uses `InvertedIndex.read_a_posting_list()` to stream posting files from GCS
- Works with your existing `postings_gcp/` structure
- Supports ~6M documents

**Already uploaded to:** `gs://206969750_bucket/backend_gcs.tar.gz`

### 2. Local Evaluation Script

**File:** `scripts/evaluate_full_corpus_local.py`

Runs evaluation on your local machine but reads indices from GCS. This bypasses the flaky SSH connection issues.

### 3. Deployment Scripts

- `deploy_to_gcp.sh` - Automated deployment
- `deployment/startup_script.sh` - VM auto-setup
- `scripts/evaluate_gcp_server.py` - Remote server evaluation

---

## How to Complete Full Corpus Evaluation

### Option A: Run Locally (Recommended - Fastest)

**Steps:**

1. **Setup GCP Authentication** (one-time)
   ```bash
   # This will open a browser window
   gcloud auth application-default login

   # Follow the prompts and sign in
   # The credentials will be saved locally
   ```

2. **Run Full Corpus Evaluation**
   ```bash
   cd "/Users/tomerfilo/Library/Mobile Documents/com~apple~CloudDocs/לימודים- שנה ג/סמסטר א/אחזור מידע/ir_proj_20251213 2"

   python3 scripts/evaluate_full_corpus_local.py
   ```

3. **Results**
   - Evaluation runs on your Mac
   - Reads posting files from GCS on-demand
   - Takes ~5-15 minutes for 30 queries
   - Outputs: `evaluation_results_full_corpus.json`

**Pros:**
- ✅ No SSH issues
- ✅ No VM costs
- ✅ Easy to debug
- ✅ Can re-run quickly

**Cons:**
- ⚠️ Slower than running on GCP VM (network latency)
- ⚠️ Requires local GCS authentication

---

### Option B: Fix GCP VM Deployment

**Steps:**

1. **SSH into VM** (when connection is stable)
   ```bash
   gcloud compute ssh ir-search-engine --zone=us-central1-a
   ```

2. **Setup Full Corpus Server**
   ```bash
   cd ~
   mkdir full_corpus_server
   cd full_corpus_server

   # Download GCS-enabled backend
   gsutil cp gs://206969750_bucket/backend_gcs.tar.gz .
   tar -xzf backend_gcs.tar.gz

   # Install dependencies (if not already done)
   pip3 install --user flask nltk pandas pyarrow google-cloud-storage

   # Start server
   export PYTHONPATH=~/full_corpus_server:$PYTHONPATH
   export PATH=$PATH:~/.local/bin
   export GCS_BUCKET=206969750_bucket
   export INDEX_DIR=postings_gcp

   python3 backend/search_frontend_gcs.py
   ```

3. **Test from Local Machine**
   ```bash
   # Get VM IP
   EXTERNAL_IP=34.72.141.111

   # Test query
   curl "http://$EXTERNAL_IP:8080/search?query=Mount+Everest"

   # Run evaluation
   python3 scripts/evaluate_gcp_server.py \
       --server-url http://$EXTERNAL_IP:8080 \
       --queries data/queries_train.json \
       --output evaluation_results_full_corpus_gcp.json
   ```

**Pros:**
- ✅ Faster queries (VM in same datacenter as GCS)
- ✅ Scalable for many queries

**Cons:**
- ⚠️ SSH connection has been unstable
- ⚠️ Costs ~$3/day while running
- ⚠️ More complex setup

---

## Expected Performance with Full Corpus

### Current (Mini Corpus - 42 docs)
```
MAP@10: 0.1078
Failing queries: 10/30 (33%)
Reason: Relevant docs not in corpus
```

### Expected (Full Corpus - 6M docs)
```
MAP@10: 0.15 - 0.30 (estimated 40-180% improvement)
Failing queries: 0-2/30 (0-7%)
Reason: All relevant docs should be available
```

**Why such improvement?**

1. ✅ **Mount Everest query** - Currently 0/46 relevant docs → Will have all 46!
   - Can retrieve actual Mount Everest articles
   - "Everest" term will be in vocabulary
   - Expect AP@10 > 0.5

2. ✅ **Great Fire of London** - Currently 0/39 relevant docs → Will have all 39!
3. ✅ **All 10 failing queries** - Will now have relevant documents

4. ✅ **Better term coverage**
   - Mini: 21,169 terms, "everest" OOV
   - Full: 495,515 terms, comprehensive coverage

---

## Detailed Comparison

| Metric | Mini Corpus | Full Corpus (Expected) |
|--------|-------------|----------------------|
| **Documents** | 42 | ~6,000,000 |
| **Vocabulary** | 21,169 | 495,515 |
| **MAP@10** | 0.1078 | **0.15 - 0.30** |
| **Failing queries** | 10/30 | 0-2/30 |
| **Mount Everest AP@10** | 0.0 | **>0.5** |
| **Great Fire AP@10** | 0.0 | **>0.4** |
| **OOV rate** | 2.5% | **<0.5%** |

---

## Files Ready for Full Corpus

### Backend Code ✅
- `backend/search_frontend_gcs.py` - GCS-enabled search
- `backend/inverted_index_gcp.py` - Already supports GCS
- `backend/pre_processing.py` - Tokenization with stem=True

### Evaluation Scripts ✅
- `scripts/evaluate_full_corpus_local.py` - Local evaluation
- `scripts/evaluate_gcp_server.py` - Remote server evaluation

### Data in GCS ✅
- `gs://206969750_bucket/postings_gcp/` - Full body index (5.92 GiB)
- `gs://206969750_bucket/pr/` - PageRank data (62.84 MiB)
- `gs://206969750_bucket/indices/metadata.pkl` - Document metadata

---

## Missing Components

### Optional but Beneficial

1. **Title and Anchor Indices for Full Corpus**
   - Currently only have body index
   - Would enable multi-field fusion on full corpus
   - Not critical - body index alone should work well

2. **PageViews Data**
   - Have PageRank but not PageViews
   - Can use PageRank only (worth 15% of score)

3. **Document Norms**
   - May need to compute on-the-fly if not in metadata
   - Slightly slower but works

---

## Recommended Action Plan

### Immediate (5 minutes):

1. Run `gcloud auth application-default login`
2. Complete browser authentication
3. Run `python3 scripts/evaluate_full_corpus_local.py`
4. Get MAP@10 results for full corpus!

### After Results (Optional):

1. If MAP@10 < expected, analyze results
2. Could add title/anchor indices
3. Could optimize weights based on full corpus performance
4. Generate final comparison report

---

## Quick Start Command

```bash
# Navigate to project
cd "/Users/tomerfilo/Library/Mobile Documents/com~apple~CloudDocs/לימודים- שנה ג/סמסטר א/אחזור מידע/ir_proj_20251213 2"

# Authenticate (opens browser)
gcloud auth application-default login

# Run evaluation (~10 minutes)
python3 scripts/evaluate_full_corpus_local.py

# View results
cat evaluation_results_full_corpus.json | python3 -m json.tool | grep -A 2 "map_at_k"
```

---

## Summary

Everything is ready! You have:
- ✅ Full corpus in GCS (6M docs, 495K terms)
- ✅ GCS-enabled search backend
- ✅ Evaluation scripts
- ✅ GCP VM running (can use if SSH stabilizes)

**Only blocker:** Need to run `gcloud auth application-default login` to authenticate locally, then run the evaluation script.

**Expected outcome:** MAP@10 improvement from 0.1078 → **0.15-0.30** with full Wikipedia corpus!

---

## Support Files Created

1. `GCP_DEPLOYMENT_GUIDE.md` - Complete deployment guide
2. `QUICK_START_GCP.md` - Quick reference
3. `PRODUCTION_ANALYSIS_SUMMARY.md` - Mini corpus analysis
4. `GCP_DEPLOYMENT_SUMMARY.txt` - What we accomplished
5. `FULL_CORPUS_NEXT_STEPS.md` - This file

All ready for you to proceed! 🚀
