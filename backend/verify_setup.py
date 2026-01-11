#!/usr/bin/env python3
"""
Verification script to check if all index files are properly placed.
Run this BEFORE starting the search engine.
"""
import os
import sys
from pathlib import Path
from google.cloud import storage

# Configuration
LOCAL_INDEX_DIR = 'indices_mini'
BUCKET_NAME = '206969750_bucket'

# Required local files
REQUIRED_LOCAL_FILES = [
    'body_index.pkl',
    'title_index.pkl',
    'anchors_index.pkl',
    'metadata.pkl',
    'pagerank.pkl',
    'pageviews.pkl'
]

# Expected GCS file patterns
EXPECTED_GCS_PATTERNS = [
    'body_index',
    'title_index',
    'anchors_index'
]

def check_local_files():
    """Check if all required .pkl files exist in indices_mini/"""
    print("=" * 80)
    print("Checking Local Files (indices_mini/)")
    print("=" * 80)

    all_exist = True
    local_path = Path(LOCAL_INDEX_DIR)

    if not local_path.exists():
        print(f"✗ Directory {LOCAL_INDEX_DIR}/ does not exist!")
        return False

    for filename in REQUIRED_LOCAL_FILES:
        file_path = local_path / filename
        if file_path.exists():
            size_mb = file_path.stat().st_size / (1024 * 1024)
            print(f"✓ {filename} exists ({size_mb:.2f} MB)")
        else:
            print(f"✗ {filename} NOT FOUND")
            all_exist = False

    print()
    return all_exist

def check_gcs_files():
    """Check if .bin files exist in GCS bucket root"""
    print("=" * 80)
    print(f"Checking GCS Bucket (gs://{BUCKET_NAME}/)")
    print("=" * 80)

    try:
        # Initialize GCS client
        client = storage.Client()
        bucket = client.bucket(BUCKET_NAME)

        # List all .bin files in bucket root
        print("Fetching .bin files from bucket root...")
        blobs = list(bucket.list_blobs(max_results=100))

        bin_files = [blob.name for blob in blobs if blob.name.endswith('.bin')]

        if not bin_files:
            print("✗ No .bin files found in bucket!")
            return False

        print(f"\n✓ Found {len(bin_files)} .bin files in bucket root")
        print("\nSample .bin files:")
        for filename in bin_files[:10]:
            print(f"  - {filename}")

        # Check for each index type
        print("\nChecking index types:")
        for pattern in EXPECTED_GCS_PATTERNS:
            matching = [f for f in bin_files if pattern in f]
            if matching:
                print(f"✓ {pattern}: {len(matching)} files")
            else:
                print(f"✗ {pattern}: NO FILES FOUND")

        print()
        return True

    except Exception as e:
        print(f"✗ Error accessing GCS bucket: {e}")
        print("\nPossible issues:")
        print("  1. VM service account lacks Storage Object Viewer role")
        print("  2. Bucket name is incorrect")
        print("  3. google-cloud-storage not installed (pip install google-cloud-storage)")
        print()
        return False

def check_structure():
    """Check for common structural issues"""
    print("=" * 80)
    print("Checking Directory Structure")
    print("=" * 80)

    # Check for incorrect subdirectories
    local_path = Path(LOCAL_INDEX_DIR)
    if local_path.exists():
        subdirs = [d for d in local_path.iterdir() if d.is_dir()]
        if subdirs:
            print(f"⚠ WARNING: Found subdirectories in {LOCAL_INDEX_DIR}/:")
            for subdir in subdirs:
                print(f"  - {subdir.name}/")
            print("\n  These should NOT exist! All .pkl files should be in indices_mini/ directly.")
            print()
            return False
        else:
            print(f"✓ No subdirectories in {LOCAL_INDEX_DIR}/ (correct - flat structure)")

    print()
    return True

def main():
    print("\n" + "=" * 80)
    print("Wikipedia Search Engine - Setup Verification")
    print("=" * 80)
    print()

    # Run checks
    local_ok = check_local_files()
    structure_ok = check_structure()
    gcs_ok = check_gcs_files()

    # Summary
    print("=" * 80)
    print("Summary")
    print("=" * 80)

    if local_ok and structure_ok and gcs_ok:
        print("✓ ALL CHECKS PASSED!")
        print("\nYou can now start the search engine:")
        print("  python3 search_frontend.py")
        print()
        return 0
    else:
        print("✗ SOME CHECKS FAILED")
        print("\nPlease fix the issues above before starting the search engine.")
        print()

        if not local_ok:
            print("Local Files Issue:")
            print("  - Download missing .pkl files to indices_mini/")
            print("  - Do NOT create subdirectories like body/, title/, anchors/")
            print()

        if not structure_ok:
            print("Structure Issue:")
            print("  - Move all .pkl files from subdirectories to indices_mini/")
            print("  - Delete empty subdirectories")
            print()

        if not gcs_ok:
            print("GCS Issue:")
            print("  - Ensure .bin files are in bucket root (not in subdirectories)")
            print("  - Grant VM service account Storage Object Viewer role")
            print()

        return 1

if __name__ == '__main__':
    sys.exit(main())
