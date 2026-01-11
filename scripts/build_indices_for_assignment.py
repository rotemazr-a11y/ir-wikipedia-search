#!/usr/bin/env python3
"""
Build assignment-compatible indices from full corpus.
Creates body_index.pkl, title_index.pkl, anchor_index.pkl that search_frontend.py expects.

This script reads from postings_gcp/ and creates simplified indices for search_frontend.py.
"""

import sys
import os
import pickle

# Add backend to path
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(SCRIPT_DIR)
BACKEND_DIR = os.path.join(PROJECT_DIR, 'backend')
sys.path.insert(0, BACKEND_DIR)
sys.path.insert(0, PROJECT_DIR)

from inverted_index_gcp import InvertedIndex

print("="*80)
print("Building Assignment-Compatible Indices from Full Corpus")
print("="*80)

# Configuration
GCS_BUCKET = "206969750_bucket"
SOURCE_INDEX_DIR = "postings_gcp"  # Where full corpus is
TARGET_INDEX_DIR = "indices_full"  # Where to save assignment-compatible indices

print(f"\nSource: gs://{GCS_BUCKET}/{SOURCE_INDEX_DIR}/")
print(f"Target: {TARGET_INDEX_DIR}/")
print()

# Create target directory
os.makedirs(TARGET_INDEX_DIR, exist_ok=True)

# Step 1: Load the full corpus body index from GCS
print("Step 1: Loading full corpus body index from GCS...")
print("  (This may take a minute - downloading 18.4 MB index.pkl)")

body_index = InvertedIndex.read_index(SOURCE_INDEX_DIR, 'index', GCS_BUCKET)
print(f"  ✓ Loaded body index: {len(body_index.df):,} terms")

# Step 2: Save as body_index.pkl (assignment format)
print("\nStep 2: Saving body_index.pkl for search_frontend.py...")
output_path = os.path.join(TARGET_INDEX_DIR, 'body_index.pkl')

# IMPORTANT: We need to save this WITHOUT the bucket reference
# so search_frontend.py can load it with read_index(dir, 'body_index', bucket_name=None)

# Save the InvertedIndex object
with open(output_path, 'wb') as f:
    pickle.dump(body_index, f)

print(f"  ✓ Saved: {output_path}")
print(f"  Size: {os.path.getsize(output_path) / (1024*1024):.1f} MB")

# Step 3: Check if title/anchor indices exist in GCS
print("\nStep 3: Checking for title and anchor indices...")

# Note: Your full corpus may not have separate title/anchor indices built
# If they don't exist, we'll create minimal stub indices

try:
    # Try to load title index
    title_index = InvertedIndex.read_index(SOURCE_INDEX_DIR, 'title_index', GCS_BUCKET)
    print(f"  ✓ Found title index: {len(title_index.df)} terms")

    with open(os.path.join(TARGET_INDEX_DIR, 'title_index.pkl'), 'wb') as f:
        pickle.dump(title_index, f)
    print(f"  ✓ Saved title_index.pkl")
except Exception as e:
    print(f"  ⚠ Title index not found in GCS")
    print(f"    Creating empty stub for compatibility...")

    # Create minimal title index
    title_index = InvertedIndex()
    title_index.df = {}
    title_index.term_total = {}
    title_index.posting_locs = {}

    with open(os.path.join(TARGET_INDEX_DIR, 'title_index.pkl'), 'wb') as f:
        pickle.dump(title_index, f)
    print(f"  ✓ Created empty title_index.pkl")

try:
    # Try to load anchor index
    anchor_index = InvertedIndex.read_index(SOURCE_INDEX_DIR, 'anchor_index', GCS_BUCKET)
    print(f"  ✓ Found anchor index: {len(anchor_index.df)} terms")

    with open(os.path.join(TARGET_INDEX_DIR, 'anchor_index.pkl'), 'wb') as f:
        pickle.dump(anchor_index, f)
    print(f"  ✓ Saved anchor_index.pkl")
except Exception as e:
    print(f"  ⚠ Anchor index not found in GCS")
    print(f"    Creating empty stub for compatibility...")

    # Create minimal anchor index
    anchor_index = InvertedIndex()
    anchor_index.df = {}
    anchor_index.term_total = {}
    anchor_index.posting_locs = {}

    with open(os.path.join(TARGET_INDEX_DIR, 'anchor_index.pkl'), 'wb') as f:
        pickle.dump(anchor_index, f)
    print(f"  ✓ Created empty anchor_index.pkl")

# Step 4: Load and save metadata
print("\nStep 4: Loading metadata...")

try:
    from google.cloud import storage

    client = storage.Client()
    bucket = client.bucket(GCS_BUCKET)
    blob = bucket.blob('indices/metadata.pkl')
    metadata_bytes = blob.download_as_bytes()
    metadata = pickle.loads(metadata_bytes)

    print(f"  ✓ Loaded metadata: {metadata['num_docs']:,} documents")

    # Save to target directory
    with open(os.path.join(TARGET_INDEX_DIR, 'metadata.pkl'), 'wb') as f:
        pickle.dump(metadata, f)
    print(f"  ✓ Saved metadata.pkl")

except Exception as e:
    print(f"  ⚠ Could not load metadata: {e}")
    print(f"    Creating minimal metadata...")

    # Create minimal metadata
    metadata = {
        'num_docs': 6000000,  # Approximate
        'doc_titles': {},
        'doc_norms': {},
        'doc_lengths': {}
    }

    with open(os.path.join(TARGET_INDEX_DIR, 'metadata.pkl'), 'wb') as f:
        pickle.dump(metadata, f)
    print(f"  ✓ Created metadata.pkl")

# Step 5: Try to load PageRank and PageViews
print("\nStep 5: Loading PageRank and PageViews...")

# PageRank
try:
    client = storage.Client()
    bucket = client.bucket(GCS_BUCKET)

    # PageRank is in pr/ directory as CSV
    print("  ⚠ PageRank is in CSV format in gs://bucket/pr/")
    print("    You'll need to convert it to .pkl format")
    print("    For now, search will work without it (using zeros)")
except:
    pass

# PageViews
try:
    blob = bucket.blob('pageviews.pkl')
    pv_bytes = blob.download_as_bytes()
    pageviews = pickle.loads(pv_bytes)

    with open(os.path.join(TARGET_INDEX_DIR, 'pageviews.pkl'), 'wb') as f:
        pickle.dump(pageviews, f)
    print(f"  ✓ Saved pageviews.pkl")
except:
    print("  ⚠ PageViews not found")

# Summary
print("\n" + "="*80)
print("SUMMARY")
print("="*80)
print(f"\nIndices created in: {TARGET_INDEX_DIR}/")
print("\nFiles created:")
print(f"  ✓ body_index.pkl - {len(body_index.df):,} terms")
print(f"  ✓ title_index.pkl - {'stub' if not hasattr(title_index, 'df') or len(title_index.df) == 0 else f'{len(title_index.df)} terms'}")
print(f"  ✓ anchor_index.pkl - {'stub' if not hasattr(anchor_index, 'df') or len(anchor_index.df) == 0 else f'{len(anchor_index.df)} terms'}")
print(f"  ✓ metadata.pkl")

print("\n" + "="*80)
print("IMPORTANT: GCS Posting Files")
print("="*80)
print("\nThe body_index.pkl contains references to posting files in GCS.")
print("For search_frontend.py to work, you need EITHER:")
print()
print("Option A: Download all posting files from GCS")
print(f"  gsutil -m cp gs://{GCS_BUCKET}/{SOURCE_INDEX_DIR}/*.bin {TARGET_INDEX_DIR}/")
print()
print("Option B: Keep posting files in GCS and read on-demand")
print("  - search_frontend.py already supports this via bucket_name parameter")
print("  - Pass bucket_name to InvertedIndex.read_a_posting_list()")
print()
print("For the assignment, Option A is recommended (download locally)")
print("="*80)

print("\nNext steps:")
print(f"1. Download posting files: gsutil -m cp gs://{GCS_BUCKET}/{SOURCE_INDEX_DIR}/*.bin {TARGET_INDEX_DIR}/")
print(f"2. Test search_frontend: INDEX_DIR={TARGET_INDEX_DIR} python3 backend/search_frontend.py")
print(f"3. Run evaluation with full corpus")
