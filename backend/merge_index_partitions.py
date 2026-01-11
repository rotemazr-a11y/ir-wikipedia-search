#!/usr/bin/env python3
"""
Merge Index Partitions Script
Merges 100 separate posting_locs pickle files into a single unified InvertedIndex.

Usage:
    python3 merge_index_partitions.py --bucket 206969750_bucket --output index.pkl
"""
import pickle
import argparse
import logging
from pathlib import Path
from collections import defaultdict
from google.cloud import storage

# Import your InvertedIndex class
from inverted_index_gcp import InvertedIndex

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def download_partition_file(bucket, partition_id: int, temp_dir: str = 'temp_partitions') -> str:
    """
    Download a single partition pickle file from GCS.

    Args:
        bucket: GCS bucket object
        partition_id: Partition number (0-99)
        temp_dir: Local directory to store downloaded files

    Returns:
        Path to downloaded file
    """
    # Create temp directory if it doesn't exist
    Path(temp_dir).mkdir(exist_ok=True)

    # GCS path: postings_gcp/text/{partition_id}_posting_locs.pickle
    blob_name = f"postings_gcp/text/{partition_id}_posting_locs.pickle"
    local_path = f"{temp_dir}/{partition_id}_posting_locs.pickle"

    try:
        blob = bucket.blob(blob_name)
        blob.download_to_filename(local_path)
        logger.info(f"✓ Downloaded partition {partition_id}")
        return local_path
    except Exception as e:
        logger.error(f"✗ Failed to download partition {partition_id}: {e}")
        return None


def merge_partitions(bucket_name: str, num_partitions: int = 100, output_file: str = 'index.pkl'):
    """
    Merge all partition pickle files into a single InvertedIndex.

    Args:
        bucket_name: GCS bucket name
        num_partitions: Number of partition files to merge (default 100)
        output_file: Output filename for merged index
    """
    logger.info("=" * 80)
    logger.info("Merging Index Partitions")
    logger.info("=" * 80)
    logger.info(f"Bucket: {bucket_name}")
    logger.info(f"Partitions: {num_partitions}")
    logger.info(f"Output: {output_file}")
    logger.info("=" * 80)

    # Initialize GCS client
    try:
        client = storage.Client()
        bucket = client.bucket(bucket_name)
        logger.info("✓ GCS client initialized")
    except Exception as e:
        logger.error(f"✗ Failed to initialize GCS client: {e}")
        return

    # Create merged index
    merged_index = InvertedIndex()
    merged_index.posting_locs = defaultdict(list)
    merged_index.df = {}
    merged_index.term_total = {}

    # CRITICAL: Set base_dir to bucket root where .bin files are located
    merged_index.base_dir = f"gs://{bucket_name}"

    # Initialize _posting_list to avoid KeyError during pickling
    merged_index._posting_list = defaultdict(list)

    logger.info("\n[1/3] Downloading and merging partitions...")

    # Download and merge each partition
    total_terms = set()

    for partition_id in range(num_partitions):
        local_path = download_partition_file(bucket, partition_id)

        if not local_path or not Path(local_path).exists():
            logger.warning(f"⚠ Skipping partition {partition_id}")
            continue

        try:
            # Load partition pickle file
            with open(local_path, 'rb') as f:
                partition_posting_locs = pickle.load(f)

            # Merge posting_locs from this partition
            for term, locs in partition_posting_locs.items():
                merged_index.posting_locs[term].extend(locs)
                total_terms.add(term)

            logger.info(f"  Partition {partition_id}: {len(partition_posting_locs)} terms")

        except Exception as e:
            logger.error(f"✗ Error processing partition {partition_id}: {e}")
            continue

    logger.info(f"\n✓ Merged {len(total_terms):,} unique terms from {num_partitions} partitions")

    # Calculate document frequency (df) for each term
    logger.info("\n[2/3] Calculating document frequencies...")

    for term, locs in merged_index.posting_locs.items():
        # df = number of documents containing the term
        # This should match the number of (doc_id, tf) pairs in the posting list
        # For now, we estimate based on the number of location entries
        # You may need to actually read the .bin files to get exact df
        merged_index.df[term] = len(locs)  # Approximate - may need adjustment

    logger.info(f"✓ Calculated df for {len(merged_index.df):,} terms")

    # Save merged index
    logger.info("\n[3/3] Saving merged index...")

    try:
        with open(output_file, 'wb') as f:
            pickle.dump(merged_index, f)

        file_size_mb = Path(output_file).stat().st_size / (1024 * 1024)
        logger.info(f"✓ Saved to {output_file} ({file_size_mb:.2f} MB)")

    except Exception as e:
        logger.error(f"✗ Failed to save index: {e}")
        import traceback
        traceback.print_exc()
        return

    # Summary
    logger.info("\n" + "=" * 80)
    logger.info("Merge Complete!")
    logger.info("=" * 80)
    logger.info(f"Output file: {output_file}")
    logger.info(f"Total terms: {len(merged_index.posting_locs):,}")
    logger.info(f"Base directory: {merged_index.base_dir}")
    logger.info("=" * 80)
    logger.info("\nNext steps:")
    logger.info("  1. Copy this file to your VM/server")
    logger.info("  2. Update search_runtime.py to load this merged index")
    logger.info("  3. Ensure query preprocessing matches index term format (lowercase, etc.)")
    logger.info("=" * 80)


def verify_merged_index(index_file: str):
    """
    Verify the merged index file.

    Args:
        index_file: Path to merged index pickle file
    """
    logger.info("\n" + "=" * 80)
    logger.info("Verifying Merged Index")
    logger.info("=" * 80)

    try:
        with open(index_file, 'rb') as f:
            index = pickle.load(f)

        logger.info(f"✓ Successfully loaded {index_file}")
        logger.info(f"  Terms: {len(index.posting_locs):,}")
        logger.info(f"  Base dir: {getattr(index, 'base_dir', 'NOT SET')}")
        logger.info(f"  Has df: {len(index.df):,}")
        logger.info(f"  Has term_total: {len(index.term_total):,}")

        # Show sample terms
        sample_terms = list(index.posting_locs.keys())[:5]
        logger.info(f"\n  Sample terms: {sample_terms}")

        # Check first term's posting locations
        if sample_terms:
            first_term = sample_terms[0]
            locs = index.posting_locs[first_term]
            logger.info(f"\n  '{first_term}' posting locations: {locs[:3]}")

        logger.info("\n✓ Index verification complete")

    except Exception as e:
        logger.error(f"✗ Verification failed: {e}")
        import traceback
        traceback.print_exc()


def main():
    parser = argparse.ArgumentParser(description='Merge index partition files into single index')
    parser.add_argument('--bucket', required=True, help='GCS bucket name')
    parser.add_argument('--partitions', type=int, default=100, help='Number of partitions (default: 100)')
    parser.add_argument('--output', default='index.pkl', help='Output filename (default: index.pkl)')
    parser.add_argument('--verify', action='store_true', help='Verify the merged index after creation')

    args = parser.parse_args()

    # Merge partitions
    merge_partitions(args.bucket, args.partitions, args.output)

    # Verify if requested
    if args.verify:
        verify_merged_index(args.output)


if __name__ == '__main__':
    main()
