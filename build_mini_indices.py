#!/usr/bin/env python3
"""
Mini Corpus Index Builder

Builds inverted indices for the mini corpus (320 documents) for local testing.

Usage:
    python3 build_mini_indices.py

Requirements:
    - data/mini_corpus_doc_ids.json must exist (run create_mini_corpus.py first)
    - Wikipedia dump file must be available (adjust path below)
    - ~5GB free disk space for indices

Output:
    - indices_mini/body_index.pkl + .bin files
    - indices_mini/title_index.pkl + .bin files
    - indices_mini/anchor_index.pkl + .bin files
    - indices_mini/metadata.pkl
"""

import json
import os
import sys
from pathlib import Path

# Add project root to Python path
sys.path.insert(0, str(Path(__file__).parent))

from backend.index_builder import IndexBuilder


# ============================================================================
# CONFIGURATION - ADJUST THESE PATHS FOR YOUR SYSTEM
# ============================================================================

# Path to your Wikipedia dump file
# IMPORTANT: Update this to point to your actual Wikipedia dump
WIKIPEDIA_DUMP_PATH = "data/multistream1_preprocessed.parquet"  # Downloaded from GCP

# Supported formats:
# - JSON: Single large JSON file with array of documents
# - JSONL: One JSON object per line (newline-delimited)
# - Parquet: Apache Parquet format (requires pyarrow)

# Output directory for indices
OUTPUT_DIR = "indices_mini"

# Mini corpus document IDs file
MINI_CORPUS_FILE = "data/mini_corpus_doc_ids.json"

# ============================================================================


def load_wikipedia_dump(dump_path, allowed_doc_ids):
    """
    Load Wikipedia documents from dump file, filtering by allowed doc IDs.

    Parameters:
    -----------
    dump_path : str
        Path to Wikipedia dump file
    allowed_doc_ids : set of str
        Document IDs to include (from mini corpus)

    Yields:
    -------
    dict
        Document with keys: id, title, text, anchor_text (optional)

    Supported Formats:
    ------------------
    1. JSON (single file with array):
       [
         {"id": "123", "title": "...", "text": "...", "anchor_text": ["...", ...]},
         ...
       ]

    2. JSONL (newline-delimited JSON):
       {"id": "123", "title": "...", "text": "..."}
       {"id": "456", "title": "...", "text": "..."}

    3. Parquet (requires pyarrow):
       df = pd.read_parquet(dump_path)
       for row in df.itertuples():
           yield {"id": row.id, "title": row.title, "text": row.text}
    """
    dump_path = Path(dump_path)

    if not dump_path.exists():
        raise FileNotFoundError(f"Wikipedia dump not found: {dump_path}")

    print(f"Loading Wikipedia dump from: {dump_path}")
    print(f"File size: {dump_path.stat().st_size / 1024 / 1024:.2f} MB")

    # Detect format from extension
    if dump_path.suffix == '.parquet':
        # Parquet format
        try:
            import pandas as pd
            df = pd.read_parquet(dump_path)

            count = 0
            for _, row in df.iterrows():
                # Use bracket notation for pandas Series to avoid ambiguous array comparisons
                try:
                    doc_id = str(row['id'])
                except KeyError:
                    doc_id = str(row['doc_id'])

                if doc_id in allowed_doc_ids:
                    # Get anchor_text safely and extract just the text strings
                    anchor_text = row['anchor_text'] if 'anchor_text' in df.columns else []
                    # Convert array of dicts to list of strings
                    if len(anchor_text) > 0 and isinstance(anchor_text[0], dict):
                        anchor_text = [a.get('text', '') for a in anchor_text if isinstance(a, dict)]

                    yield {
                        'id': doc_id,
                        'title': row['title'],
                        'text': row['text'],
                        'anchor_text': anchor_text
                    }
                    count += 1
                    if count % 50 == 0:
                        print(f"  Loaded {count} / {len(allowed_doc_ids)} documents...", end='\r')

            print(f"  Loaded {count} / {len(allowed_doc_ids)} documents... Done!  ")

        except ImportError:
            print("ERROR: pyarrow not installed. Install with: pip install pyarrow")
            sys.exit(1)

    elif dump_path.suffix == '.json':
        # Try JSON array format first
        with open(dump_path, 'r', encoding='utf-8') as f:
            first_char = f.read(1)
            f.seek(0)

            if first_char == '[':
                # JSON array format
                print("Detected JSON array format")
                data = json.load(f)

                count = 0
                for doc in data:
                    doc_id = str(doc.get('id') or doc.get('doc_id'))
                    if doc_id in allowed_doc_ids:
                        yield {
                            'id': doc_id,
                            'title': doc.get('title', ''),
                            'text': doc.get('text', ''),
                            'anchor_text': doc.get('anchor_text', [])
                        }
                        count += 1
                        if count % 50 == 0:
                            print(f"  Loaded {count} / {len(allowed_doc_ids)} documents...", end='\r')

                print(f"  Loaded {count} / {len(allowed_doc_ids)} documents... Done!  ")

            else:
                # JSONL format (one JSON per line)
                print("Detected JSONL format")
                count = 0
                for line in f:
                    line = line.strip()
                    if not line:
                        continue

                    doc = json.loads(line)
                    doc_id = str(doc.get('id') or doc.get('doc_id'))
                    if doc_id in allowed_doc_ids:
                        yield {
                            'id': doc_id,
                            'title': doc.get('title', ''),
                            'text': doc.get('text', ''),
                            'anchor_text': doc.get('anchor_text', [])
                        }
                        count += 1
                        if count % 50 == 0:
                            print(f"  Loaded {count} / {len(allowed_doc_ids)} documents...", end='\r')

                print(f"  Loaded {count} / {len(allowed_doc_ids)} documents... Done!  ")

    else:
        raise ValueError(f"Unsupported format: {dump_path.suffix}. Use .json or .parquet")


def main():
    """Build mini corpus indices."""
    print("\n" + "="*70)
    print("BUILDING MINI CORPUS INDICES")
    print("="*70 + "\n")

    # Step 1: Load mini corpus document IDs
    print("Step 1: Loading mini corpus document IDs...")
    if not os.path.exists(MINI_CORPUS_FILE):
        print(f"ERROR: Mini corpus file not found: {MINI_CORPUS_FILE}")
        print("Please run: python3 scripts/create_mini_corpus.py")
        sys.exit(1)

    with open(MINI_CORPUS_FILE, 'r', encoding='utf-8') as f:
        mini_corpus_ids = json.load(f)

    allowed_doc_ids = set(str(doc_id) for doc_id in mini_corpus_ids)
    print(f"✓ Loaded {len(allowed_doc_ids)} document IDs to index\n")

    # Step 2: Check Wikipedia dump exists
    print("Step 2: Checking Wikipedia dump...")
    if not os.path.exists(WIKIPEDIA_DUMP_PATH):
        print(f"ERROR: Wikipedia dump not found: {WIKIPEDIA_DUMP_PATH}")
        print("\nPlease update WIKIPEDIA_DUMP_PATH in this script to point to your dump file.")
        print("\nSupported formats:")
        print("  - JSON: Single file with array of documents")
        print("  - JSONL: One JSON object per line")
        print("  - Parquet: Apache Parquet format\n")
        sys.exit(1)

    print(f"✓ Found Wikipedia dump: {WIKIPEDIA_DUMP_PATH}\n")

    # Step 3: Create output directory
    print("Step 3: Creating output directory...")
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    print(f"✓ Output directory: {OUTPUT_DIR}\n")

    # Step 4: Build indices
    print("Step 4: Building indices...")
    print("-" * 70)

    builder = IndexBuilder()

    # Load and index documents
    doc_count = 0
    try:
        for doc in load_wikipedia_dump(WIKIPEDIA_DUMP_PATH, allowed_doc_ids):
            builder.add_document(
                doc_id=int(doc['id']),
                title=doc['title'],
                body=doc['text'],
                anchors=doc.get('anchor_text', [])
            )
            doc_count += 1

            # Progress update
            if doc_count % 100 == 0:
                print(f"Indexed {doc_count} / {len(allowed_doc_ids)} documents...", end='\r')

    except Exception as e:
        print(f"\n\nERROR while loading documents: {e}")
        print("\nPlease check:")
        print("  1. WIKIPEDIA_DUMP_PATH points to correct file")
        print("  2. File format matches one of: JSON, JSONL, Parquet")
        print("  3. Documents have required fields: id, title, text")
        sys.exit(1)

    print(f"Indexed {doc_count} / {len(allowed_doc_ids)} documents... Done!  ")

    if doc_count == 0:
        print("\nERROR: No documents were loaded!")
        print("Please check:")
        print("  1. Wikipedia dump file contains the document IDs from mini_corpus_doc_ids.json")
        print("  2. Document ID field name (should be 'id' or 'doc_id')")
        sys.exit(1)

    print(f"\n✓ Indexed {doc_count} documents")

    # Step 5: Build and write indices to disk
    print("\nStep 5: Building and writing indices to disk...")
    print("-" * 70)

    builder.build_indices(OUTPUT_DIR)

    print(f"\n✓ All indices written to: {OUTPUT_DIR}/")

    # Step 6: Verify output
    print("\nStep 6: Verifying output...")
    print("-" * 70)

    expected_files = [
        'body_index.pkl',
        'title_index.pkl',
        'anchor_index.pkl',
        'metadata.pkl'
    ]

    total_size = 0
    for filename in expected_files:
        filepath = os.path.join(OUTPUT_DIR, filename)
        if os.path.exists(filepath):
            size = os.path.getsize(filepath)
            total_size += size
            print(f"  ✓ {filename:25s} ({size / 1024 / 1024:.2f} MB)")
        else:
            print(f"  ✗ {filename:25s} MISSING!")

    # Check for .bin files (posting lists)
    bin_files = list(Path(OUTPUT_DIR).glob('*.bin'))
    for bin_file in bin_files:
        size = bin_file.stat().st_size
        total_size += size
        print(f"  ✓ {bin_file.name:25s} ({size / 1024 / 1024:.2f} MB)")

    print(f"\nTotal index size: {total_size / 1024 / 1024:.2f} MB")

    # Step 7: Summary
    print("\n" + "="*70)
    print("INDEX BUILD COMPLETE!")
    print("="*70)
    print(f"Documents indexed: {doc_count}")
    print(f"Output directory: {OUTPUT_DIR}/")
    print(f"Total size: {total_size / 1024 / 1024:.2f} MB")

    print("\nNext steps:")
    print("  1. Run unit tests: python3 tests/test_search_endpoints.py")
    print("  2. Start Flask server: export INDEX_DIR=indices_mini && python3 backend/search_frontend.py")
    print("  3. Evaluate MAP@10: python3 scripts/evaluate_map.py --server http://localhost:8080")
    print("\n" + "="*70 + "\n")


if __name__ == "__main__":
    main()
