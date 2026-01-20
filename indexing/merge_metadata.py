#!/usr/bin/env python3
"""
Merge partition metadata files into final InvertedIndex objects.
Run this on the GCP cluster master node to fix the indices.

The expensive Spark work (building posting lists) is DONE.
This script just merges the metadata.
"""

import pickle
import io
from google.cloud import storage


def get_bucket(bucket_name):
    return storage.Client().bucket(bucket_name)


class InvertedIndex:
    """Reconstructed InvertedIndex from partition metadata."""
    def __init__(self, index_name=''):
        self.df = {}  # term -> document frequency
        self.term_total = {}  # term -> total occurrences
        self.posting_locs = {}  # term -> [(file, offset, length), ...]
        self._index_name = index_name
        
    def add_term(self, term, locs, df, term_total):
        self.df[term] = df
        self.term_total[term] = term_total
        self.posting_locs[term] = locs


def merge_index(bucket_name: str, index_type: str, num_partitions: int = 200):
    """
    Merge partition metadata files into a single InvertedIndex.
    
    Args:
        bucket_name: GCS bucket name
        index_type: 'body', 'title', or 'anchor'
        num_partitions: Number of partitions (default 200)
    """
    bucket = get_bucket(bucket_name)
    index = InvertedIndex(index_type)
    
    # Try both the correct path and the corrupted path (gs://bucket/gs://bucket/...)
    path_prefixes = [
        f"gs://{bucket_name}/indices/{index_type}_index",  # Corrupted path stored as blob name
        f"indices/{index_type}_index",  # Correct path
    ]
    
    total_terms = 0
    successful_partitions = 0
    
    for i in range(num_partitions):
        found = False
        for prefix in path_prefixes:
            blob_path = f"{prefix}/metadata_{i:04d}.pkl"
            blob = bucket.blob(blob_path)
            
            try:
                if not blob.exists():
                    continue
                    
                # Download and unpickle
                data = blob.download_as_bytes()
                partition_metadata = pickle.loads(data)
                
                # Each item is (term, locs, df, term_total)
                # locs can be [(file_path, offset), ...] or [(file_path, offset, length), ...]
                for item in partition_metadata:
                    term, locs, df, term_total = item
                    # Keep locs as-is (may be 2-tuple or 3-tuple)
                    index.add_term(term, locs, df, term_total)
                    total_terms += 1
                
                successful_partitions += 1
                found = True
                break
                    
            except Exception as e:
                continue
        
        if not found:
            print(f"  Warning: Could not read partition {i}")
        
        if (i + 1) % 20 == 0:
            print(f"  Processed {i + 1}/{num_partitions} partitions, {total_terms} terms so far")
    
    print(f"\n  DONE: {successful_partitions}/{num_partitions} partitions, {total_terms} unique terms")
    return index


def save_index(index: InvertedIndex, bucket_name: str, index_type: str):
    """Save the merged InvertedIndex to GCS."""
    bucket = get_bucket(bucket_name)
    
    # Save to correct path (not corrupted)
    output_path = f"indices/{index_type}_index/{index_type}_index.pkl"
    blob = bucket.blob(output_path)
    
    # Serialize and upload
    data = pickle.dumps(index)
    blob.upload_from_string(data)
    
    print(f"  Saved to gs://{bucket_name}/{output_path} ({len(data) / 1024 / 1024:.1f} MB)")


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--bucket', default='bucket_207916263')
    parser.add_argument('--index_types', default='body,title,anchor')
    parser.add_argument('--partitions', type=int, default=200)
    args = parser.parse_args()
    
    bucket_name = args.bucket
    index_types = args.index_types.split(',')
    
    for index_type in index_types:
        print(f"\n{'='*60}")
        print(f"Merging {index_type.upper()} index metadata")
        print(f"{'='*60}")
        
        try:
            index = merge_index(bucket_name, index_type, args.partitions)
            
            if len(index.df) > 0:
                save_index(index, bucket_name, index_type)
                print(f"  ✅ {index_type.upper()} index complete: {len(index.df)} terms")
            else:
                print(f"  ⚠️ {index_type.upper()} index has 0 terms - metadata may not exist")
                
        except Exception as e:
            print(f"  ❌ Error processing {index_type}: {e}")
    
    print(f"\n{'='*60}")
    print("MERGE COMPLETE!")
    print(f"{'='*60}")


if __name__ == '__main__':
    main()
