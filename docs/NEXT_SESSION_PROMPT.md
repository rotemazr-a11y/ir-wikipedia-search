# IR Wikipedia Search Engine - Next Session Prompt

> **Last Updated**: January 18, 2026, ~18:30 UTC
> **Copy this entire file into a new Claude session to continue**

---

## 🎯 Quick Summary

I'm building a **Wikipedia search engine** for my IR course final project. The testing period is **January 20-22, 2026** (2 days away).

### Current Status
- ✅ All code complete and tested locally
- 🔄 **Indexing job running on GCP** (Job ID: `94fb8087280740b2bf8310ead1155444`)
- ⏳ After indexing: run 3 more jobs (doc_metadata, pagerank, pageviews)
- ⏳ Then deploy frontend to GCP VM

---

## 📁 Project Structure

```
PROJECT/
├── docs/
│   ├── NEXT_SESSION_PROMPT.md      # THIS FILE - copy to new sessions
│   ├── POST_INDEXING_AND_DEPLOYMENT.md  # Full deployment guide
│   ├── ir_final_rpoject.md         # Assignment requirements
│   └── working_with_gcp.md         # GCP setup instructions
├── indexing/
│   ├── pyspark_index_builder.py    # Main Spark indexing driver
│   ├── spimi_block_builder.py      # SPIMI algorithm implementation
│   ├── inverted_index_gcp.py       # InvertedIndex class for GCS
│   ├── pre_processing.py           # Tokenization (no NLTK)
│   ├── compute_doc_metadata.py     # Generates doc_titles.pkl, doc_lengths.pkl
│   ├── compute_pagerank.py         # PageRank calculation
│   └── compute_pageviews.py        # PageView aggregation
├── frontend/
│   ├── search_frontend.py          # Flask API (6 endpoints)
│   └── run_frontend_in_colab.ipynb # Colab notebook for frontend
├── scripts/
│   ├── startup_script_gcp.sh       # VM startup configuration
│   ├── run_frontend_in_gcp.sh      # VM creation command
│   └── deploy_to_gcp.sh            # Alternative deployment
├── tests/
│   ├── evaluate_search.py          # P@5, P@10, F1@30 evaluation
│   └── test_*.py                   # Unit tests (15/15 passed)
└── queries_train.json              # 30 training queries
```

---

## 🔧 GCP Configuration

```bash
PROJECT_NAME="durable-tracer-479509-q2"
BUCKET_NAME="bucket_207916263"
CLUSTER_NAME="ir-cluster"
REGION="us-central1"
ZONE="us-central1-a"
GOOGLE_ACCOUNT_NAME="rotemazr"
```

### GitHub Repository
https://github.com/RotemAzriel/wikipedia-search-engine

---

## 📊 Current Job Status (Check This First!)

### Indexing Job
```bash
# Check if indexing is complete
gcloud dataproc jobs describe 94fb8087280740b2bf8310ead1155444 --region=us-central1 --format="yaml(status)"

# Expected output when done:
# status:
#   state: DONE
```

### Expected Duration
- Started: Jan 18, 2026 ~14:20 UTC
- Estimated completion: Jan 18, 2026 ~17:00-18:00 UTC (3-4 hours total)
- Map phase: ~155/200 partitions complete as of last check

### Check Output in GCS
```bash
# When job completes, indices should be here:
gsutil ls gs://bucket_207916263/indices/
# Should show: body_index/, title_index/, anchor_index/

# Each index folder should have:
gsutil ls gs://bucket_207916263/indices/body_index/
# body_index.pkl, posting_locs_body_index.pickle, multiple .bin files
```

---

## 📋 Post-Indexing Steps (In Order)

### Step 1: Compute Document Metadata
**Purpose**: Generate `doc_titles.pkl` and `doc_lengths.pkl` (needed by frontend)

```bash
cd "/Users/rotemazriel/Documents/University/2026/Semester A/IR/Assignments/PROJECT/indexing"

# Upload script
gsutil cp compute_doc_metadata.py gs://bucket_207916263/

# Submit job
gcloud dataproc jobs submit pyspark \
    --cluster=ir-cluster \
    --region=us-central1 \
    gs://bucket_207916263/compute_doc_metadata.py \
    -- \
    --input gs://bucket_207916263/corpus \
    --output gs://bucket_207916263 \
    --bucket bucket_207916263

# Verify output (after ~15-30 min)
gsutil ls -l gs://bucket_207916263/doc_titles.pkl
gsutil ls -l gs://bucket_207916263/doc_lengths.pkl
```

### Step 2: Compute PageRank
**Purpose**: Generate `pagerank/pagerank.pkl` from Wikipedia link graph

```bash
gsutil cp compute_pagerank.py gs://bucket_207916263/

gcloud dataproc jobs submit pyspark \
    --cluster=ir-cluster \
    --region=us-central1 \
    --driver-memory=8g \
    --executor-memory=8g \
    --properties=spark.executor.memoryOverhead=2g \
    gs://bucket_207916263/compute_pagerank.py \
    -- \
    --input gs://bucket_207916263/corpus \
    --output gs://bucket_207916263/pagerank \
    --bucket bucket_207916263 \
    --iterations 10

# Verify (after ~30-60 min)
gsutil ls -l gs://bucket_207916263/pagerank/pagerank.pkl
```

### Step 3: Compute PageViews
**Purpose**: Generate `pageviews/pageviews.pkl` from Wikipedia pageview dump

```bash
gsutil cp compute_pageviews.py gs://bucket_207916263/

gcloud dataproc jobs submit pyspark \
    --cluster=ir-cluster \
    --region=us-central1 \
    --driver-memory=8g \
    --executor-memory=8g \
    gs://bucket_207916263/compute_pageviews.py \
    -- \
    --pageviews gs://bucket_207916263/pageviews-202108-user.bz2 \
    --corpus gs://bucket_207916263/corpus \
    --output gs://bucket_207916263/pageviews \
    --bucket bucket_207916263

# Verify (after ~20-40 min)
gsutil ls -l gs://bucket_207916263/pageviews/pageviews.pkl
```

---

## 🖥️ GCP VM Deployment

### Prerequisites Check
```bash
# All these files must exist in GCS:
gsutil ls gs://bucket_207916263/indices/body_index/body_index.pkl
gsutil ls gs://bucket_207916263/indices/title_index/title_index.pkl
gsutil ls gs://bucket_207916263/indices/anchor_index/anchor_index.pkl
gsutil ls gs://bucket_207916263/doc_titles.pkl
gsutil ls gs://bucket_207916263/doc_lengths.pkl
gsutil ls gs://bucket_207916263/pagerank/pagerank.pkl
gsutil ls gs://bucket_207916263/pageviews/pageviews.pkl
```

### Deploy VM
```bash
# Create static IP
gcloud compute addresses create ir-search-ip \
    --project=durable-tracer-479509-q2 \
    --region=us-central1

# Get the IP
INSTANCE_IP=$(gcloud compute addresses describe ir-search-ip --region=us-central1 --format="get(address)")
echo "Your IP: $INSTANCE_IP"

# Create firewall rule
gcloud compute firewall-rules create default-allow-http-8080 \
    --allow tcp:8080 \
    --source-ranges 0.0.0.0/0 \
    --target-tags http-server

# Upload frontend files
cd "/Users/rotemazriel/Documents/University/2026/Semester A/IR/Assignments/PROJECT"
gsutil cp frontend/search_frontend.py gs://bucket_207916263/frontend/
gsutil cp indexing/inverted_index_gcp.py gs://bucket_207916263/frontend/

# Create VM
cd scripts
gcloud compute instances create ir-search-engine \
    --zone=us-central1-a \
    --machine-type=e2-standard-2 \
    --network-interface=address=$INSTANCE_IP,network-tier=PREMIUM,subnet=default \
    --metadata-from-file startup-script=startup_script_gcp.sh \
    --scopes=https://www.googleapis.com/auth/cloud-platform \
    --tags=http-server
```

### Start Server (SSH into VM)
```bash
gcloud compute ssh rotemazr@ir-search-engine --zone us-central1-a

# Inside VM:
nohup ~/venv/bin/python ~/search_frontend.py > ~/frontend.log 2>&1 &
curl http://127.0.0.1:8080/health
exit
```

### Test From Local
```bash
curl "http://$INSTANCE_IP:8080/search?query=machine+learning"
```

---

## ⚠️ Common Errors & Solutions

### Indexing Job Errors

| Error | Cause | Solution |
|-------|-------|----------|
| `NLTK Permission denied` | NLTK tries to write to read-only path | Fixed - hardcoded stopwords in pre_processing.py |
| `Job stuck reading Parquet` | count() caused full scan | Fixed - removed count() call |
| `OOM during shuffle` | Too much data in memory | Fixed - SPIMI with memory-bounded blocks |
| Memory warnings (BlockManager) | Normal - data spilling to disk | Not critical, job continues |

### Post-Indexing Job Errors

| Error | Cause | Solution |
|-------|-------|----------|
| `ModuleNotFoundError` | Missing dependencies | Upload deps.zip with required modules |
| `FileNotFoundError: corpus` | Wrong input path | Use `gs://bucket_207916263/corpus` |
| `KeyError in PageRank` | doc_id mismatch | Ensure corpus has `id` column |

### Frontend Errors

| Error | Cause | Solution |
|-------|-------|----------|
| `Index file not found` | Files not in expected GCS path | Check path structure matches frontend expectations |
| `Connection refused` | Server not running or wrong port | SSH and check `ps aux \| grep python` |
| `500 Internal Error` | Python exception | Check `~/frontend.log` on VM |

---

## 🔍 Verification Commands

### Check All Required Files Exist
```bash
echo "=== Checking indices ===" && \
gsutil ls gs://bucket_207916263/indices/body_index/body_index.pkl && \
gsutil ls gs://bucket_207916263/indices/title_index/title_index.pkl && \
gsutil ls gs://bucket_207916263/indices/anchor_index/anchor_index.pkl && \
echo "=== Checking metadata ===" && \
gsutil ls gs://bucket_207916263/doc_titles.pkl && \
gsutil ls gs://bucket_207916263/doc_lengths.pkl && \
echo "=== Checking PageRank/PageView ===" && \
gsutil ls gs://bucket_207916263/pagerank/pagerank.pkl && \
gsutil ls gs://bucket_207916263/pageviews/pageviews.pkl
```

### Monitor Running Job
```bash
JOB_ID="94fb8087280740b2bf8310ead1155444"  # Replace with current job

# Check status
gcloud dataproc jobs describe $JOB_ID --region=us-central1 --format="yaml(status)"

# Get YARN progress
gcloud dataproc jobs describe $JOB_ID --region=us-central1 --format="value(yarnApplications)"

# Check worker logs
gcloud compute ssh ir-cluster-m --zone=us-central1-a --command="yarn logs -applicationId application_1768742238803_0004 -log_files stderr 2>/dev/null | tail -50"
```

---

## 📝 Remaining Tasks

1. **Wait for indexing job to complete** (~1-2 hours remaining)
2. **Run post-indexing jobs** (in order: metadata, pagerank, pageviews)
3. **Deploy frontend to GCP VM**
4. **Test all 6 endpoints**
5. **Run evaluation script** (`tests/evaluate_search.py`)
6. **Write report** (4 pages)
7. **Create slides** (3-5 slides)
8. **Make GCS bucket public** (before testing period)

### Deadline
**Testing Period**: January 20-22, 2026 (12:00 - 12:00)

---

## 📎 Key Files Reference

| Purpose | Local Path | GCS Path |
|---------|------------|----------|
| Deployment guide | `docs/POST_INDEXING_AND_DEPLOYMENT.md` | - |
| Main indexer | `indexing/pyspark_index_builder.py` | `gs://bucket_207916263/code/` |
| Frontend | `frontend/search_frontend.py` | `gs://bucket_207916263/frontend/` |
| Index class | `indexing/inverted_index_gcp.py` | `gs://bucket_207916263/frontend/` |
| Training queries | `queries_train.json` | - |
| Requirements | `docs/ir_final_rpoject.md` | - |
