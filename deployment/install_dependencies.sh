#!/bin/bash
# Dataproc Initialization Script
# Install Python dependencies on all cluster nodes

set -e

echo "Installing Python dependencies for Wikipedia Search Engine..."

# Upgrade pip
pip3 install --upgrade pip

# Install core dependencies
pip3 install \
    numpy==1.21.0 \
    pandas==1.3.0 \
    nltk==3.6.0 \
    google-cloud-storage==2.0.0

# Download NLTK data
python3 -c "import nltk; nltk.download('stopwords')"

echo "Dependencies installed successfully"
