#!/bin/bash
# GCP Deployment Script for Wikipedia Index Builder
# Project: durable-tracer-479509-q2
# Bucket: bucket_207916263

set -e  # Exit on error

# ============================================================================
# CONFIGURATION
# ============================================================================
export PROJECT_ID="durable-tracer-479509-q2"
export BUCKET_NAME="bucket_207916263"
export REGION="us-central1"
export ZONE="us-central1-a"
export CLUSTER_NAME="ir-cluster"

echo "=============================================="
echo "GCP Wikipedia Index Builder Deployment"
echo "=============================================="
echo "Project: ${PROJECT_ID}"
echo "Bucket:  ${BUCKET_NAME}"
echo "Region:  ${REGION}"
echo "=============================================="

# ============================================================================
# STEP 1: Set Project
# ============================================================================
echo ""
echo "[1/6] Setting GCP project..."
gcloud config set project ${PROJECT_ID}

# ============================================================================
# STEP 2: Create Bucket (if not exists)
# ============================================================================
echo ""
echo "[2/6] Creating/verifying bucket..."
if gcloud storage buckets describe gs://${BUCKET_NAME} &>/dev/null; then
    echo "  Bucket gs://${BUCKET_NAME} already exists"
else
    echo "  Creating bucket gs://${BUCKET_NAME}..."
    gcloud storage buckets create gs://${BUCKET_NAME} \
        --project=${PROJECT_ID} \
        --location=${REGION}
fi

# ============================================================================
# STEP 3: Upload Code Files
# ============================================================================
echo ""
echo "[3/6] Uploading code files..."

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}"

# Create dependencies zip
echo "  Creating deps.zip..."
zip -j deps.zip \
    pre_processing.py \
    spimi_block_builder.py \
    inverted_index_gcp.py

# Upload files
echo "  Uploading to gs://${BUCKET_NAME}/code/..."
gcloud storage cp deps.zip gs://${BUCKET_NAME}/code/
gcloud storage cp pyspark_index_builder.py gs://${BUCKET_NAME}/code/

# Upload initialization script (graphframes.sh from ASS3 if available)
if [ -f "../ASS3/colab/graphframes.sh" ]; then
    gcloud storage cp ../ASS3/colab/graphframes.sh gs://${BUCKET_NAME}/
elif [ -f "../ASS3/gcp/graphframes.sh" ]; then
    gcloud storage cp ../ASS3/gcp/graphframes.sh gs://${BUCKET_NAME}/
else
    echo "  Warning: graphframes.sh not found, creating minimal init script..."
    echo '#!/bin/bash' > /tmp/init.sh
    echo 'pip install nltk' >> /tmp/init.sh
    gcloud storage cp /tmp/init.sh gs://${BUCKET_NAME}/init.sh
fi

echo "  Code uploaded successfully!"

# ============================================================================
# STEP 4: Copy Wikipedia Data (if not already present)
# ============================================================================
echo ""
echo "[4/6] Checking Wikipedia corpus data..."

# Check if data already exists
if gcloud storage ls gs://${BUCKET_NAME}/corpus/ &>/dev/null; then
    echo "  Corpus data already exists in gs://${BUCKET_NAME}/corpus/"
    echo "  Skipping data transfer..."
else
    echo "  Corpus data not found. Starting transfer from course bucket..."
    echo "  This may take 10-20 minutes..."
    
    # Use transfer job for large data
    gcloud storage cp -r \
        "gs://wikidata20210801_preprocessed/*" \
        "gs://${BUCKET_NAME}/corpus/"
    
    echo "  Data transfer complete!"
fi

# ============================================================================
# STEP 5: Create Dataproc Cluster
# ============================================================================
echo ""
echo "[5/6] Creating Dataproc cluster..."

# Check if cluster exists
if gcloud dataproc clusters describe ${CLUSTER_NAME} --region=${REGION} &>/dev/null; then
    echo "  Cluster ${CLUSTER_NAME} already exists"
else
    echo "  Creating cluster ${CLUSTER_NAME}..."
    
    # Determine init script path
    if gcloud storage ls gs://${BUCKET_NAME}/graphframes.sh &>/dev/null; then
        INIT_SCRIPT="gs://${BUCKET_NAME}/graphframes.sh"
    else
        INIT_SCRIPT="gs://${BUCKET_NAME}/init.sh"
    fi
    
    gcloud dataproc clusters create ${CLUSTER_NAME} \
        --enable-component-gateway \
        --region ${REGION} \
        --zone ${ZONE} \
        --project ${PROJECT_ID} \
        --master-machine-type n1-standard-4 \
        --master-boot-disk-size 100 \
        --num-workers 2 \
        --worker-machine-type n1-standard-4 \
        --worker-boot-disk-size 100 \
        --image-version 2.0-debian10 \
        --scopes 'https://www.googleapis.com/auth/cloud-platform' \
        --initialization-actions "${INIT_SCRIPT}"
    
    echo "  Cluster created successfully!"
fi

# ============================================================================
# STEP 6: Submit Index Build Job
# ============================================================================
echo ""
echo "[6/6] Ready to submit Spark job!"
echo ""
echo "Run this command to start the index build:"
echo ""
echo "gcloud dataproc jobs submit pyspark \\"
echo "    --cluster=${CLUSTER_NAME} \\"
echo "    --region=${REGION} \\"
echo "    --py-files=gs://${BUCKET_NAME}/code/deps.zip \\"
echo "    gs://${BUCKET_NAME}/code/pyspark_index_builder.py \\"
echo "    -- \\"
echo "    --input gs://${BUCKET_NAME}/corpus \\"
echo "    --output gs://${BUCKET_NAME}/indices \\"
echo "    --bucket ${BUCKET_NAME} \\"
echo "    --partitions 200"
echo ""
echo "=============================================="
echo "Deployment setup complete!"
echo "=============================================="
echo ""
echo "⚠️  IMPORTANT: Delete cluster when done to save credits:"
echo "    gcloud dataproc clusters delete ${CLUSTER_NAME} --region=${REGION} --quiet"
