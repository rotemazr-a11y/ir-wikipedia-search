# Quick Start: Deploy to GCP

**Current Status:** Local MAP@10 = 0.1078 (mini corpus, 42 docs)
**Goal:** Test on full Wikipedia corpus to improve performance

---

## Prerequisites

1. **GCP Account** with billing enabled
2. **gcloud CLI** installed and configured
3. **Full corpus indices** in GCP Storage bucket

---

## One-Command Deployment

```bash
# Set your bucket name
export GCS_BUCKET="your-bucket-name-here"

# Run deployment script
./deploy_to_gcp.sh
```

That's it! The script will:
1. ✅ Package your backend code
2. ✅ Upload to GCS
3. ✅ Create VM instance
4. ✅ Setup firewall rules
5. ✅ Auto-start search server
6. ✅ Wait for server to be ready

---

## Manual Deployment (Step by Step)

### Step 1: Upload Backend Code

```bash
cd "/Users/tomerfilo/Library/Mobile Documents/com~apple~CloudDocs/לימודים- שנה ג/סמסטר א/אחזור מידע/ir_proj_20251213 2"

# Package backend
tar -czf backend.tar.gz backend/

# Upload to GCS
gsutil cp backend.tar.gz gs://YOUR_BUCKET_NAME/
```

### Step 2: Create VM Instance

```bash
# Create instance with automated setup
gcloud compute instances create ir-search-engine \
    --zone=us-central1-a \
    --machine-type=e2-standard-4 \
    --image-family=debian-11 \
    --image-project=debian-cloud \
    --boot-disk-size=50GB \
    --scopes=https://www.googleapis.com/auth/cloud-platform \
    --metadata=GCS_BUCKET=YOUR_BUCKET_NAME \
    --metadata-from-file=startup-script=deployment/startup_script.sh

# Create firewall rule
gcloud compute firewall-rules create allow-flask \
    --allow tcp:8080 \
    --source-ranges 0.0.0.0/0
```

### Step 3: Get Server IP

```bash
# Get external IP
gcloud compute instances describe ir-search-engine \
    --zone=us-central1-a \
    --format='get(networkInterfaces[0].accessConfigs[0].natIP)'

# Store in variable
EXTERNAL_IP=$(gcloud compute instances describe ir-search-engine \
    --zone=us-central1-a \
    --format='get(networkInterfaces[0].accessConfigs[0].natIP)')

echo "Server at: http://$EXTERNAL_IP:8080"
```

### Step 4: Test Server

```bash
# Wait 2-3 minutes for startup, then test
curl "http://$EXTERNAL_IP:8080/search?query=Mount+Everest" | python3 -m json.tool
```

Expected output:
```json
[
  ["47353693", "Mount Everest"],
  ["5208803", "1953 British Mount Everest expedition"],
  ...
]
```

### Step 5: Run Full Evaluation

```bash
# Evaluate all 30 queries
python3 scripts/evaluate_gcp_server.py \
    --server-url http://$EXTERNAL_IP:8080 \
    --queries data/queries_train.json \
    --output evaluation_results_gcp_full.json

# View results
cat evaluation_results_gcp_full.json | python3 -m json.tool | grep -A 2 "map_at_k"
```

---

## Troubleshooting

### Server not responding?

```bash
# Check if instance is running
gcloud compute instances list

# View server logs
gcloud compute ssh ir-search-engine --zone=us-central1-a \
    --command 'sudo journalctl -u ir-search -n 100'

# Check server status
gcloud compute ssh ir-search-engine --zone=us-central1-a \
    --command 'sudo systemctl status ir-search'

# Restart server
gcloud compute ssh ir-search-engine --zone=us-central1-a \
    --command 'sudo systemctl restart ir-search'
```

### Indices not loading?

```bash
# SSH into instance
gcloud compute ssh ir-search-engine --zone=us-central1-a

# Check indices directory
ls -lh /home/ir_search_engine/indices/

# Check which files are missing
for f in body_index.pkl title_index.pkl anchor_index.pkl metadata.pkl pagerank.pkl pageviews.pkl; do
    if [ -f "/home/ir_search_engine/indices/$f" ]; then
        echo "✓ $f"
    else
        echo "✗ $f MISSING"
    fi
done

# Download missing indices
gsutil cp gs://YOUR_BUCKET_NAME/indices/*.pkl /home/ir_search_engine/indices/

# Restart server
sudo systemctl restart ir-search
```

### Memory errors?

```bash
# Use larger machine
gcloud compute instances stop ir-search-engine --zone=us-central1-a
gcloud compute instances set-machine-type ir-search-engine \
    --machine-type=e2-highmem-4 \
    --zone=us-central1-a
gcloud compute instances start ir-search-engine --zone=us-central1-a
```

---

## Cost Management

### Stop instance when not in use

```bash
# Stop (preserves disk, ~$2/day savings)
gcloud compute instances stop ir-search-engine --zone=us-central1-a

# Start again
gcloud compute instances start ir-search-engine --zone=us-central1-a
```

### Delete instance completely

```bash
# Delete (cannot recover data!)
gcloud compute instances delete ir-search-engine --zone=us-central1-a
```

### Estimated costs

- **e2-standard-4:** $0.13/hour = $3.12/day
- **Storage (50GB):** $2/month
- **Network:** ~$0.10/day

**Total for 1 day of testing: ~$3-4**

---

## Expected Performance

### Current (Mini Corpus)
- Corpus: 42 documents
- MAP@10: 0.1078
- Failing queries: 10/30 (33%)
- Reason: Relevant docs not in corpus

### Expected (Full Corpus)
- Corpus: 6.4M+ documents
- MAP@10: **0.15 - 0.25** (estimated)
- Failing queries: **0-2/30** (0-7%)
- All relevant docs should be in corpus

### Why improvement?
1. ✅ All 46 Mount Everest articles now available
2. ✅ All ground truth docs in corpus
3. ✅ PageRank/PageViews contributing (if available)
4. ✅ Better term coverage

---

## Files Created

- ✅ [GCP_DEPLOYMENT_GUIDE.md](GCP_DEPLOYMENT_GUIDE.md) - Detailed guide
- ✅ [QUICK_START_GCP.md](QUICK_START_GCP.md) - This file
- ✅ [deploy_to_gcp.sh](deploy_to_gcp.sh) - Automated deployment
- ✅ [deployment/startup_script.sh](deployment/startup_script.sh) - VM setup
- ✅ [scripts/evaluate_gcp_server.py](scripts/evaluate_gcp_server.py) - Evaluation

---

## Next Steps

1. Set `GCS_BUCKET` environment variable
2. Run `./deploy_to_gcp.sh`
3. Wait 2-3 minutes for startup
4. Run evaluation script
5. Compare with mini corpus results
6. Celebrate improved MAP@10! 🎉

---

## Common Commands Reference

```bash
# Deployment
export GCS_BUCKET="your-bucket"
./deploy_to_gcp.sh

# Get IP
EXTERNAL_IP=$(gcloud compute instances describe ir-search-engine \
    --zone=us-central1-a \
    --format='get(networkInterfaces[0].accessConfigs[0].natIP)')

# Test
curl "http://$EXTERNAL_IP:8080/search?query=test" | python3 -m json.tool

# Evaluate
python3 scripts/evaluate_gcp_server.py \
    --server-url http://$EXTERNAL_IP:8080 \
    --queries data/queries_train.json

# Logs
gcloud compute ssh ir-search-engine --zone=us-central1-a \
    --command 'sudo journalctl -u ir-search -f'

# SSH
gcloud compute ssh ir-search-engine --zone=us-central1-a

# Stop
gcloud compute instances stop ir-search-engine --zone=us-central1-a

# Delete
gcloud compute instances delete ir-search-engine --zone=us-central1-a
```
