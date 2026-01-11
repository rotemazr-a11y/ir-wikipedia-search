#!/bin/bash
#
# Quick Deployment Script for GCP
# Packages and uploads backend code, then creates/starts GCP VM
#

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo -e "${GREEN}=========================================${NC}"
echo -e "${GREEN}IR Search Engine - GCP Deployment${NC}"
echo -e "${GREEN}=========================================${NC}"
echo ""

# Configuration (CHANGE THESE!)
GCS_BUCKET="${GCS_BUCKET:-YOUR_BUCKET_NAME}"
INSTANCE_NAME="${INSTANCE_NAME:-ir-search-engine}"
ZONE="${ZONE:-us-central1-a}"
MACHINE_TYPE="${MACHINE_TYPE:-e2-standard-4}"

# Check if bucket is configured
if [ "$GCS_BUCKET" = "YOUR_BUCKET_NAME" ]; then
    echo -e "${RED}ERROR: Please set your GCS bucket name${NC}"
    echo ""
    echo "Either:"
    echo "  1. Edit this script and change GCS_BUCKET variable"
    echo "  2. Or run: export GCS_BUCKET=your-bucket-name"
    echo ""
    exit 1
fi

echo "Configuration:"
echo "  Bucket: $GCS_BUCKET"
echo "  Instance: $INSTANCE_NAME"
echo "  Zone: $ZONE"
echo "  Machine: $MACHINE_TYPE"
echo ""

# Step 1: Package backend code
echo -e "${YELLOW}Step 1: Packaging backend code...${NC}"
if [ -d "backend" ]; then
    tar -czf backend.tar.gz backend/
    echo -e "${GREEN}✓ Created backend.tar.gz${NC}"
else
    echo -e "${RED}ERROR: backend/ directory not found${NC}"
    exit 1
fi

# Step 2: Upload to GCS
echo ""
echo -e "${YELLOW}Step 2: Uploading to GCS...${NC}"
gsutil cp backend.tar.gz gs://$GCS_BUCKET/ && \
    echo -e "${GREEN}✓ Uploaded backend.tar.gz${NC}" || \
    { echo -e "${RED}ERROR: Failed to upload${NC}"; exit 1; }

# Check if indices exist in GCS
echo ""
echo -e "${YELLOW}Step 3: Checking for indices in GCS...${NC}"
if gsutil ls gs://$GCS_BUCKET/indices/*.pkl >/dev/null 2>&1; then
    echo -e "${GREEN}✓ Indices found in gs://$GCS_BUCKET/indices/${NC}"

    # List what's there
    echo ""
    echo "Available indices:"
    gsutil ls -lh gs://$GCS_BUCKET/indices/*.pkl | awk '{print "  " $3 " - " $1}'

    # Check for important files
    echo ""
    if gsutil ls gs://$GCS_BUCKET/indices/pagerank.pkl >/dev/null 2>&1; then
        echo -e "${GREEN}  ✓ pagerank.pkl found${NC}"
    else
        echo -e "${YELLOW}  ⚠ pagerank.pkl missing (will use zeros)${NC}"
    fi

    if gsutil ls gs://$GCS_BUCKET/indices/pageviews.pkl >/dev/null 2>&1; then
        echo -e "${GREEN}  ✓ pageviews.pkl found${NC}"
    else
        echo -e "${YELLOW}  ⚠ pageviews.pkl missing (will use zeros)${NC}"
    fi
else
    echo -e "${RED}WARNING: No indices found in gs://$GCS_BUCKET/indices/${NC}"
    echo ""
    echo "You need to upload your indices first:"
    echo "  gsutil -m cp indices_full/*.pkl gs://$GCS_BUCKET/indices/"
    echo ""
    read -p "Continue anyway? (y/N) " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        exit 1
    fi
fi

# Step 4: Check if instance exists
echo ""
echo -e "${YELLOW}Step 4: Checking for existing instance...${NC}"
if gcloud compute instances describe $INSTANCE_NAME --zone=$ZONE >/dev/null 2>&1; then
    echo -e "${YELLOW}Instance $INSTANCE_NAME already exists${NC}"
    echo ""
    read -p "Delete and recreate? (y/N) " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        echo "Deleting instance..."
        gcloud compute instances delete $INSTANCE_NAME --zone=$ZONE --quiet
        echo -e "${GREEN}✓ Deleted${NC}"
    else
        echo "Skipping instance creation"
        echo ""
        echo -e "${YELLOW}To manually update the server:${NC}"
        echo "  gcloud compute ssh $INSTANCE_NAME --zone=$ZONE"
        echo "  cd /home/ir_search_engine"
        echo "  gsutil cp gs://$GCS_BUCKET/backend.tar.gz ."
        echo "  tar -xzf backend.tar.gz"
        echo "  sudo systemctl restart ir-search"
        exit 0
    fi
fi

# Step 5: Create firewall rule
echo ""
echo -e "${YELLOW}Step 5: Creating firewall rule...${NC}"
if gcloud compute firewall-rules describe allow-flask >/dev/null 2>&1; then
    echo -e "${GREEN}✓ Firewall rule 'allow-flask' already exists${NC}"
else
    gcloud compute firewall-rules create allow-flask \
        --allow tcp:8080 \
        --source-ranges 0.0.0.0/0 \
        --description "Allow Flask server on port 8080" && \
        echo -e "${GREEN}✓ Created firewall rule${NC}"
fi

# Step 6: Create instance with startup script
echo ""
echo -e "${YELLOW}Step 6: Creating GCP instance...${NC}"
gcloud compute instances create $INSTANCE_NAME \
    --zone=$ZONE \
    --machine-type=$MACHINE_TYPE \
    --image-family=debian-11 \
    --image-project=debian-cloud \
    --boot-disk-size=50GB \
    --scopes=https://www.googleapis.com/auth/cloud-platform \
    --metadata=GCS_BUCKET=$GCS_BUCKET \
    --metadata-from-file=startup-script=deployment/startup_script.sh && \
    echo -e "${GREEN}✓ Instance created${NC}"

# Step 7: Wait for server to start
echo ""
echo -e "${YELLOW}Step 7: Waiting for server to start...${NC}"
echo "(This may take 2-3 minutes for first-time setup)"

# Get external IP
EXTERNAL_IP=$(gcloud compute instances describe $INSTANCE_NAME \
    --zone=$ZONE \
    --format='get(networkInterfaces[0].accessConfigs[0].natIP)')

echo "Instance IP: $EXTERNAL_IP"
echo ""

# Wait for server to be ready
MAX_ATTEMPTS=30
ATTEMPT=1
while [ $ATTEMPT -le $MAX_ATTEMPTS ]; do
    echo -n "Attempt $ATTEMPT/$MAX_ATTEMPTS: "

    if curl -s --max-time 5 "http://$EXTERNAL_IP:8080/search?query=test" >/dev/null 2>&1; then
        echo -e "${GREEN}Server is UP!${NC}"
        break
    else
        echo "Waiting..."
        sleep 10
        ATTEMPT=$((ATTEMPT + 1))
    fi
done

if [ $ATTEMPT -gt $MAX_ATTEMPTS ]; then
    echo -e "${RED}WARNING: Server didn't respond after $MAX_ATTEMPTS attempts${NC}"
    echo ""
    echo "Check logs with:"
    echo "  gcloud compute ssh $INSTANCE_NAME --zone=$ZONE --command 'sudo journalctl -u ir-search -n 50'"
else
    # Test the server
    echo ""
    echo -e "${GREEN}=========================================${NC}"
    echo -e "${GREEN}Deployment Successful!${NC}"
    echo -e "${GREEN}=========================================${NC}"
    echo ""
    echo "Server URL: http://$EXTERNAL_IP:8080"
    echo ""
    echo "Test query:"
    echo "  curl 'http://$EXTERNAL_IP:8080/search?query=Mount+Everest' | python3 -m json.tool"
    echo ""
    echo "Run evaluation:"
    echo "  python3 scripts/evaluate_gcp_server.py \\"
    echo "      --server-url http://$EXTERNAL_IP:8080 \\"
    echo "      --queries data/queries_train.json \\"
    echo "      --output evaluation_results_gcp_full.json"
    echo ""
    echo "View server logs:"
    echo "  gcloud compute ssh $INSTANCE_NAME --zone=$ZONE --command 'sudo journalctl -u ir-search -f'"
    echo ""
    echo "SSH into instance:"
    echo "  gcloud compute ssh $INSTANCE_NAME --zone=$ZONE"
    echo ""
    echo "Stop instance (to save costs):"
    echo "  gcloud compute instances stop $INSTANCE_NAME --zone=$ZONE"
    echo ""
    echo -e "${GREEN}=========================================${NC}"
fi

# Cleanup
rm -f backend.tar.gz
echo ""
echo "Cleaned up local backend.tar.gz"
