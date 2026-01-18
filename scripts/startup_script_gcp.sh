#!/bin/bash
set -e

APP_USER="rotemazr"
APP_HOME="/home/${APP_USER}"
VENV_DIR="${APP_HOME}/venv"
BUCKET_NAME="bucket_207916263"

echo "=== Starting VM Setup ==="

apt-get update
apt-get install -y python3-venv python3-pip

# Create venv as the normal user (not root)
echo "=== Creating Python virtual environment ==="
sudo -u "${APP_USER}" bash -lc "
python3 -m venv '${VENV_DIR}'
source '${VENV_DIR}/bin/activate'
pip install --upgrade pip
pip install \
  'Flask==2.0.2' \
  'Werkzeug==2.3.8' \
  'flask-restful==0.3.9' \
  'pandas' \
  'google-cloud-storage' \
  'numpy>=1.23.2,<3'
"

# Download required files from GCS
echo "=== Downloading files from GCS ==="
sudo -u "${APP_USER}" bash -lc "
cd ${APP_HOME}

# Download frontend files
gsutil cp gs://${BUCKET_NAME}/frontend/search_frontend.py .
gsutil cp gs://${BUCKET_NAME}/frontend/inverted_index_gcp.py .

# Download metadata files
gsutil cp gs://${BUCKET_NAME}/doc_titles.pkl .
gsutil cp gs://${BUCKET_NAME}/doc_lengths.pkl .
gsutil cp gs://${BUCKET_NAME}/pagerank/pagerank.pkl .
gsutil cp gs://${BUCKET_NAME}/pageviews/pageviews.pkl .

echo 'Files downloaded successfully!'
ls -la
"

echo "=== VM Setup Complete ==="
echo "To start the server, SSH in and run:"
echo "  nohup ~/venv/bin/python ~/search_frontend.py > ~/frontend.log 2>&1 &"
