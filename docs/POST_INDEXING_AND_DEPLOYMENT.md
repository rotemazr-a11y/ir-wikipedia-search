# Post-Indexing Jobs & GCP Deployment Guide

This document provides step-by-step instructions for running all post-indexing jobs and deploying the search engine to GCP.

---

## Table of Contents

1. [Prerequisites](#prerequisites)
2. [Post-Indexing Jobs](#post-indexing-jobs)
   - [Job 1: Compute Document Metadata](#job-1-compute-document-metadata)
   - [Job 2: Compute PageRank](#job-2-compute-pagerank)
   - [Job 3: Compute PageViews](#job-3-compute-pageviews)
3. [Verify All Files in GCS](#verify-all-files-in-gcs)
4. [GCP Deployment](#gcp-deployment)
   - [Step 1: Upload Frontend Files](#step-1-upload-frontend-files)
   - [Step 2: Create Static IP](#step-2-create-static-ip)
   - [Step 3: Create Firewall Rule](#step-3-create-firewall-rule)
   - [Step 4: Create VM Instance](#step-4-create-vm-instance)
   - [Step 5: SSH and Start Server](#step-5-ssh-and-start-server)
   - [Step 6: Test the Search Engine](#step-6-test-the-search-engine)
5. [Troubleshooting](#troubleshooting)
6. [Cleanup Commands](#cleanup-commands)

---

## Prerequisites

### GCP Configuration
```bash
PROJECT_NAME="durable-tracer-479509-q2"
BUCKET_NAME="bucket_207916263"
CLUSTER_NAME="ir-cluster"
REGION="us-central1"
ZONE="us-central1-a"
```

### Verify Indexing Job Completed
```bash
# Check job status
gcloud dataproc jobs describe 94fb8087280740b2bf8310ead1155444 --region=us-central1

# Verify indices exist in GCS
gsutil ls gs://bucket_207916263/indices/
# Should show: body_index/, title_index/, anchor_index/
```

### Verify Cluster is Running
```bash
gcloud dataproc clusters list --region=us-central1
# If not running, create it:
# gcloud dataproc clusters create ir-cluster --region=us-central1 --num-workers=2
```

---

## Post-Indexing Jobs

### Job 1: Compute Document Metadata

**Purpose:** Generate `doc_titles.pkl` and `doc_lengths.pkl` from the corpus.

**Output Files:**
- `gs://bucket_207916263/doc_titles.pkl` - Mapping of doc_id → title
- `gs://bucket_207916263/doc_lengths.pkl` - Mapping of doc_id → word count

#### Step 1.1: Upload the script
```bash
cd "/Users/rotemazriel/Documents/University/2026/Semester A/IR/Assignments/PROJECT/indexing"

gsutil cp compute_doc_metadata.py gs://bucket_207916263/
```

#### Step 1.2: Submit the job
```bash
gcloud dataproc jobs submit pyspark \
    --cluster=ir-cluster \
    --region=us-central1 \
    gs://bucket_207916263/compute_doc_metadata.py \
    -- \
    --input gs://bucket_207916263/corpus \
    --output gs://bucket_207916263 \
    --bucket bucket_207916263
```

#### Step 1.3: Monitor progress
```bash
# Get job ID from the output, then:
gcloud dataproc jobs describe JOB_ID --region=us-central1

# Or check worker logs:
gcloud compute ssh ir-cluster-w-0 --zone=us-central1-a -- \
    "sudo tail -20 /var/log/hadoop-yarn/userlogs/*/container_*/stderr 2>/dev/null | tail -20"
```

#### Step 1.4: Verify output
```bash
gsutil ls -l gs://bucket_207916263/doc_titles.pkl
gsutil ls -l gs://bucket_207916263/doc_lengths.pkl
```

**Expected Duration:** 15-30 minutes

---

### Job 2: Compute PageRank

**Purpose:** Calculate PageRank scores from Wikipedia link graph.

**Output File:**
- `gs://bucket_207916263/pagerank/pagerank.pkl` - Mapping of doc_id → PageRank score

#### Step 2.1: Upload the script
```bash
cd "/Users/rotemazriel/Documents/University/2026/Semester A/IR/Assignments/PROJECT/indexing"

gsutil cp compute_pagerank.py gs://bucket_207916263/
```

#### Step 2.2: Submit the job
```bash
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
```

#### Step 2.3: Monitor progress
```bash
gcloud dataproc jobs describe JOB_ID --region=us-central1
```

#### Step 2.4: Verify output
```bash
gsutil ls -l gs://bucket_207916263/pagerank/pagerank.pkl
```

**Expected Duration:** 30-60 minutes (depends on link graph size)

---

### Job 3: Compute PageViews

**Purpose:** Process Wikipedia pageview data from August 2021.

**Input File:**
- `gs://bucket_207916263/pageviews-202108-user.bz2` (already uploaded)

**Output File:**
- `gs://bucket_207916263/pageviews/pageviews.pkl` - Mapping of doc_id → view count

#### Step 3.1: Upload the script
```bash
cd "/Users/rotemazriel/Documents/University/2026/Semester A/IR/Assignments/PROJECT/indexing"

gsutil cp compute_pageviews.py gs://bucket_207916263/
```

#### Step 3.2: Submit the job
```bash
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
```

#### Step 3.3: Verify output
```bash
gsutil ls -l gs://bucket_207916263/pageviews/pageviews.pkl
```

**Expected Duration:** 20-40 minutes

---

## Verify All Files in GCS

After all jobs complete, verify everything is in place:

```bash
echo "=== Checking all required files ==="

# Indices
echo "Indices:"
gsutil ls gs://bucket_207916263/indices/body_index/ | head -5
gsutil ls gs://bucket_207916263/indices/title_index/ | head -5
gsutil ls gs://bucket_207916263/indices/anchor_index/ | head -5

# Metadata
echo -e "\nMetadata:"
gsutil ls -l gs://bucket_207916263/doc_titles.pkl
gsutil ls -l gs://bucket_207916263/doc_lengths.pkl

# PageRank
echo -e "\nPageRank:"
gsutil ls -l gs://bucket_207916263/pagerank/pagerank.pkl

# PageViews
echo -e "\nPageViews:"
gsutil ls -l gs://bucket_207916263/pageviews/pageviews.pkl

# Frontend
echo -e "\nFrontend files:"
gsutil ls gs://bucket_207916263/frontend/
```

### Expected File Structure
```
gs://bucket_207916263/
├── corpus/                          # Wikipedia corpus (Parquet)
├── indices/
│   ├── body_index/                  # Body inverted index
│   │   ├── body_index.pkl
│   │   └── posting_locs_body_index.pickle
│   ├── title_index/                 # Title inverted index
│   │   ├── title_index.pkl
│   │   └── posting_locs_title_index.pickle
│   └── anchor_index/                # Anchor inverted index
│       ├── anchor_index.pkl
│       └── posting_locs_anchor_index.pickle
├── pagerank/
│   └── pagerank.pkl                 # PageRank scores
├── pageviews/
│   └── pageviews.pkl                # Page view counts
├── doc_titles.pkl                   # Document titles
├── doc_lengths.pkl                  # Document lengths
├── frontend/
│   ├── search_frontend.py           # Flask app
│   └── inverted_index_gcp.py        # Index reader
└── pageviews-202108-user.bz2        # Raw pageview data
```

---

## GCP Deployment

### Configuration Variables
```bash
# Set these in your terminal
INSTANCE_NAME="ir-search-engine"
REGION="us-central1"
ZONE="us-central1-a"
PROJECT_NAME="durable-tracer-479509-q2"
IP_NAME="ir-search-ip"
GOOGLE_ACCOUNT_NAME="rotemazr"
BUCKET_NAME="bucket_207916263"
```

---

### Step 1: Upload Frontend Files

Ensure the latest frontend files are in GCS:

```bash
cd "/Users/rotemazriel/Documents/University/2026/Semester A/IR/Assignments/PROJECT"

# Upload search frontend
gsutil cp frontend/search_frontend.py gs://bucket_207916263/frontend/

# Upload inverted index module
gsutil cp indexing/inverted_index_gcp.py gs://bucket_207916263/frontend/

# Verify
gsutil ls gs://bucket_207916263/frontend/
```

---

### Step 2: Create Static IP

```bash
# Create a static external IP address
gcloud compute addresses create $IP_NAME \
    --project=$PROJECT_NAME \
    --region=$REGION

# Get the IP address
gcloud compute addresses list

# Store the IP for later use
INSTANCE_IP=$(gcloud compute addresses describe $IP_NAME --region=$REGION --format="get(address)")
echo "Your external IP: $INSTANCE_IP"
```

**⚠️ Note:** Save this IP address - you'll need it for the report and for testing.

---

### Step 3: Create Firewall Rule

```bash
# Allow HTTP traffic on port 8080
gcloud compute firewall-rules create default-allow-http-8080 \
    --allow tcp:8080 \
    --source-ranges 0.0.0.0/0 \
    --target-tags http-server

# Verify
gcloud compute firewall-rules list --filter="name=default-allow-http-8080"
```

---

### Step 4: Create VM Instance

```bash
# Navigate to scripts directory
cd "/Users/rotemazriel/Documents/University/2026/Semester A/IR/Assignments/PROJECT/scripts"

# Create the VM with startup script
gcloud compute instances create $INSTANCE_NAME \
    --zone=$ZONE \
    --machine-type=e2-standard-2 \
    --network-interface=address=$INSTANCE_IP,network-tier=PREMIUM,subnet=default \
    --metadata-from-file startup-script=startup_script_gcp.sh \
    --scopes=https://www.googleapis.com/auth/cloud-platform \
    --tags=http-server

# Monitor startup progress (Ctrl+C when done, ~5 minutes)
gcloud compute instances tail-serial-port-output $INSTANCE_NAME --zone $ZONE
```

**Machine Type Options:**
| Type | vCPU | RAM | Cost/hour | Recommended For |
|------|------|-----|-----------|-----------------|
| e2-micro | 0.25 | 1GB | ~$0.01 | Testing only |
| e2-small | 0.5 | 2GB | ~$0.02 | Light testing |
| e2-medium | 1 | 4GB | ~$0.04 | Development |
| **e2-standard-2** | 2 | 8GB | ~$0.08 | **Production** |
| e2-standard-4 | 4 | 16GB | ~$0.16 | High traffic |

---

### Step 5: SSH and Start Server

```bash
# Verify instance is running
gcloud compute instances list --filter="name=$INSTANCE_NAME"

# SSH into the VM
gcloud compute ssh $GOOGLE_ACCOUNT_NAME@$INSTANCE_NAME --zone $ZONE
```

**Inside the VM, run:**

```bash
# Check files were downloaded
ls -la ~

# Verify Python environment
~/venv/bin/python --version

# Check required packages
~/venv/bin/python -c "import flask, pandas, numpy; print('All packages OK')"

# Start the search engine (background process)
nohup ~/venv/bin/python ~/search_frontend.py > ~/frontend.log 2>&1 &

# Verify it's running
sleep 3
curl "http://127.0.0.1:8080/health"

# Check logs if needed
tail -f ~/frontend.log
```

**Exit the VM:**
```bash
exit
```

---

### Step 6: Test the Search Engine

From your local machine:

```bash
# Get your external IP
INSTANCE_IP=$(gcloud compute addresses describe $IP_NAME --region=$REGION --format="get(address)")

# Test health endpoint
curl "http://$INSTANCE_IP:8080/health"

# Test search endpoint
curl "http://$INSTANCE_IP:8080/search?query=hello+world"

# Test body search
curl "http://$INSTANCE_IP:8080/search_body?query=machine+learning"

# Test title search
curl "http://$INSTANCE_IP:8080/search_title?query=python+programming"

# Test anchor search
curl "http://$INSTANCE_IP:8080/search_anchor?query=wikipedia"

# Test PageRank (POST request)
curl -X POST "http://$INSTANCE_IP:8080/get_pagerank" \
    -H "Content-Type: application/json" \
    -d "[1, 2, 3, 4, 5]"

# Test PageView (POST request)
curl -X POST "http://$INSTANCE_IP:8080/get_pageview" \
    -H "Content-Type: application/json" \
    -d "[1, 2, 3, 4, 5]"
```

### Run Evaluation Script

```bash
cd "/Users/rotemazriel/Documents/University/2026/Semester A/IR/Assignments/PROJECT"

python tests/evaluate_search.py --url "http://$INSTANCE_IP:8080"
```

---

## Troubleshooting

### Check VM Logs
```bash
# SSH into VM
gcloud compute ssh $GOOGLE_ACCOUNT_NAME@$INSTANCE_NAME --zone $ZONE

# Check Flask logs
tail -100 ~/frontend.log

# Check startup script logs
sudo journalctl -u google-startup-scripts.service

# Check if process is running
ps aux | grep python
```

### Restart the Server
```bash
# SSH into VM, then:
pkill -f search_frontend.py
nohup ~/venv/bin/python ~/search_frontend.py > ~/frontend.log 2>&1 &
```

### Common Issues

| Issue | Solution |
|-------|----------|
| Connection refused | Check firewall rule exists, server is running |
| 500 Internal Error | Check ~/frontend.log for Python errors |
| Index not loading | Verify files exist in GCS with correct paths |
| Slow queries | Use larger machine type (e2-standard-4) |
| Out of memory | Increase machine type or optimize index loading |

---

## Cleanup Commands

**⚠️ Run these after the grading period (January 22+) to avoid charges:**

```bash
# Delete VM instance
gcloud compute instances delete -q $INSTANCE_NAME --zone=$ZONE

# Verify no instances running
gcloud compute instances list

# Delete firewall rule
gcloud compute firewall-rules delete -q default-allow-http-8080

# Delete static IP (releases the address)
gcloud compute addresses delete -q $IP_NAME --region=$REGION

# Delete Dataproc cluster (if no longer needed)
gcloud dataproc clusters delete ir-cluster --region=$REGION

# Optional: Delete GCS data (only if you have a backup!)
# gsutil -m rm -r gs://bucket_207916263/indices/
# gsutil -m rm gs://bucket_207916263/*.pkl
```

---

## Quick Reference

### Important URLs
- **Search Engine:** `http://YOUR_IP:8080/search?query=...`
- **GCP Console:** https://console.cloud.google.com/compute/instances

### Key Commands
```bash
# Check VM status
gcloud compute instances list

# SSH to VM
gcloud compute ssh rotemazr@ir-search-engine --zone us-central1-a

# Check job status
gcloud dataproc jobs list --region=us-central1

# Check GCS files
gsutil ls gs://bucket_207916263/
```

### Testing Period
- **Start:** Tuesday, January 20, 2026 at 12:00 (noon)
- **End:** Thursday, January 22, 2026 at 12:00 (noon)

**Make sure your VM is running before the testing period starts!**
