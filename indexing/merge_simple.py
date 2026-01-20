#!/usr/bin/env python3
"""
Stream partition metadata files and save InvertedIndex in chunks.
Memory-efficient version that processes and saves incrementally.
"""

import pickle
import io
from google.cloud import storage


def get_bucket(bucket_name):
    return storage.Client().bucket(bucket_name)


class InvertedIndex:
    """Minimal InvertedIndex for saving."""
    def __init__(self, index_name=''):
        self.df = {}  # term -> document frequency
        self.term_total = {}  # term -> total occurrences
        self.posting_locs = {}  # term -> [(file, offset), ...]
        self._index_name = index_name


def merge_and_save_index(bucket_name: str, index_type: str, num_partitions: int = 200):
    """
    Process partition metadata files and create final InvertedIndex.
    Uses streaming to avoid memory issues.
    """
    bucket = get_bucket(bucket_name)
    
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
    
    # Save to GCS
    output_path = f"indices/{index_type}_index/{index_type}_index.pkl"
    blob = bucket.blob(output_path)
    
    print(f"  Serializing...")
    data = pickle.dumps(index)
    print(f"  Uploading ({len(data) / 1024 / 1024:.1f} MB)...")
    blob.upload_from_string(data)
    
    print(f"  ✅ Saved to gs://{bucket_name}/{output_path}")
    return output_path


def main():
    import argparse
    import gc
    
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
            result = merge_and_save_index(bucket_name, index_type, args.partitions)
            gc.collect()  # Force garbage collection between indices
        except Exception as e:
            print(f"  ❌ Error: {e}")
    
    print(f"\n{'='*60}")
    print("DONE!")
    print(f"{'='*60}")


if __name__ == '__main__':
    main()
