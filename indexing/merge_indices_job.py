#!/usr/bin/env python3
"""
PySpark job to merge partition metadata into final InvertedIndex files.
Runs as a Dataproc job for stability and resource management.
"""

import pickle
import argparse
from google.cloud import storage


class InvertedIndex:
    """Minimal InvertedIndex for saving."""
    def __init__(self, index_name=''):
        self.df = {}  # term -> document frequency
        self.term_total = {}  # term -> total occurrences
        self.posting_locs = {}  # term -> [(file, offset), ...]
        self._index_name = index_name


def merge_index(bucket_name: str, index_type: str, num_partitions: int = 200):
    """
    Merge partition metadata files into a single InvertedIndex.
    """
    client = storage.Client()
    bucket = client.bucket(bucket_name)
    
    # Initialize index
    index = InvertedIndex(index_type)
    
    # Path prefix - files are stored with gs:// prefix as blob name
    prefix = f"gs://{bucket_name}/indices/{index_type}_index"
    
    total_terms = 0
    successful_partitions = 0
    
    print(f"Processing {index_type} index...")
    
    for i in range(num_partitions):
        blob_path = f"{prefix}/metadata_{i:04d}.pkl"
        blob = bucket.blob(blob_path)
        
        try:
            if not blob.exists():
                print(f"  Partition {i}: not found")
                continue
            
            # Download and unpickle
            data = blob.download_as_bytes()
            partition_metadata = pickle.loads(data)
            
            # Process each term
            for item in partition_metadata:
                term, locs, df, term_total = item
                index.df[term] = df
                index.term_total[term] = term_total
                index.posting_locs[term] = locs
                total_terms += 1
            
            successful_partitions += 1
            
            # Clear downloaded data immediately
            del data
            del partition_metadata
            
        except Exception as e:
            print(f"  Partition {i}: error - {e}")
        
        if (i + 1) % 20 == 0:
            print(f"  {i + 1}/{num_partitions} partitions, {total_terms} terms, {successful_partitions} successful")
    
    print(f"\n  Total: {total_terms} unique terms from {successful_partitions} partitions")
    
    if total_terms == 0:
        print(f"  ⚠️ No terms found for {index_type}")
        return None
    
    # Serialize and upload
    print(f"  Serializing...")
    index_bytes = pickle.dumps(index)
    size_mb = len(index_bytes) / (1024 * 1024)
    print(f"  Uploading ({size_mb:.1f} MB)...")
    
    # Upload to clean path (not the corrupted gs://bucket/gs://bucket path)
    output_path = f"indices/{index_type}_index/{index_type}_index.pkl"
    output_blob = bucket.blob(output_path)
    output_blob.upload_from_string(index_bytes)
    
    print(f"  ✅ Saved to gs://{bucket_name}/{output_path}")
    
    return index


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--bucket', default='bucket_207916263')
    parser.add_argument('--index_types', default='body,title,anchor')
    parser.add_argument('--partitions', type=int, default=200)
    args = parser.parse_args()
    
    bucket_name = args.bucket
    index_types = args.index_types.split(',')
    
    for index_type in index_types:
        print(f"\n{'='*60}")
        print(f"Processing {index_type.upper()} index")
        print(f"{'='*60}")
        
        try:
            result = merge_index(bucket_name, index_type, args.partitions)
        except Exception as e:
            print(f"  ❌ Error: {e}")
            import traceback
            traceback.print_exc()
    
    print(f"\n{'='*60}")
    print("DONE!")
    print(f"{'='*60}")


if __name__ == '__main__':
    main()
