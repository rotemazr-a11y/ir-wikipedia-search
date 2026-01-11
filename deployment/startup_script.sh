#!/bin/bash
#
# GCP VM Startup Script
# Automatically sets up the search server environment
#
# Usage:
#   gcloud compute instances create ir-search-engine \
#       --metadata-from-file startup-script=deployment/startup_script.sh \
#       --machine-type=e2-standard-4 \
#       --zone=us-central1-a
#

set -e  # Exit on error

echo "========================================="
echo "IR Search Engine - Automated Setup"
echo "========================================="

# Configuration
BUCKET_NAME="${GCS_BUCKET:-YOUR_BUCKET_NAME}"  # Override with --metadata GCS_BUCKET=...
PROJECT_DIR="/home/ir_search_engine"
INDEX_DIR="$PROJECT_DIR/indices"

# Update system
echo "Updating system packages..."
sudo apt-get update
sudo apt-get install -y python3 python3-pip wget

# Install Python dependencies
echo "Installing Python dependencies..."
pip3 install --break-system-packages flask nltk pyarrow google-cloud-storage requests

# Download NLTK data
echo "Downloading NLTK data..."
python3 -c "
import nltk
import ssl
try:
    _create_unverified_https_context = ssl._create_unverified_context
except AttributeError:
    pass
else:
    ssl._create_default_https_context = _create_unverified_https_context

nltk.download('stopwords')
nltk.download('punkt')
"

# Create project directory
echo "Creating project directory..."
mkdir -p "$PROJECT_DIR"
cd "$PROJECT_DIR"

# Download backend code from GCS
echo "Downloading backend code from gs://$BUCKET_NAME/..."
gsutil cp gs://$BUCKET_NAME/backend.tar.gz . || {
    echo "ERROR: Failed to download backend.tar.gz"
    echo "Make sure you've uploaded it with:"
    echo "  gsutil cp backend.tar.gz gs://$BUCKET_NAME/"
    exit 1
}

tar -xzf backend.tar.gz

# Download indices
echo "Downloading indices from GCS..."
mkdir -p "$INDEX_DIR"
gsutil -m cp gs://$BUCKET_NAME/indices/*.pkl "$INDEX_DIR/" || {
    echo "WARNING: Failed to download some indices"
    echo "Server may not function correctly"
}

# Verify indices
echo "Verifying downloaded indices..."
for index in body_index.pkl title_index.pkl anchor_index.pkl metadata.pkl; do
    if [ -f "$INDEX_DIR/$index" ]; then
        echo "  ✓ $index"
    else
        echo "  ✗ $index MISSING"
    fi
done

# Check for optional files
for optional in pagerank.pkl pageviews.pkl; do
    if [ -f "$INDEX_DIR/$optional" ]; then
        echo "  ✓ $optional (optional)"
    else
        echo "  ⚠ $optional missing (will use zeros)"
    fi
done

# Set environment variables
export INDEX_DIR="$INDEX_DIR"
export FLASK_ENV=production

# Create systemd service for auto-start
echo "Creating systemd service..."
sudo tee /etc/systemd/system/ir-search.service > /dev/null <<EOF
[Unit]
Description=IR Search Engine Flask Server
After=network.target

[Service]
Type=simple
User=$USER
WorkingDirectory=$PROJECT_DIR
Environment="INDEX_DIR=$INDEX_DIR"
Environment="FLASK_ENV=production"
ExecStart=/usr/bin/python3 $PROJECT_DIR/backend/search_frontend.py
Restart=always
RestartSec=10

[Install]
WantedBy=multi-user.target
EOF

# Enable and start service
sudo systemctl daemon-reload
sudo systemctl enable ir-search.service
sudo systemctl start ir-search.service

echo ""
echo "========================================="
echo "Setup Complete!"
echo "========================================="
echo "Server should be running on port 8080"
echo ""
echo "To check status:"
echo "  sudo systemctl status ir-search"
echo ""
echo "To view logs:"
echo "  sudo journalctl -u ir-search -f"
echo ""
echo "To restart:"
echo "  sudo systemctl restart ir-search"
echo ""
echo "External access:"
echo "  curl 'http://\$(curl -s ifconfig.me):8080/search?query=test'"
echo "========================================="
