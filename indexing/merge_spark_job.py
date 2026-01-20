#!/usr/bin/env python3
"""
Two-phase merge:
Phase 1: Create chunks (runs on workers via Spark)
Phase 2: Combine chunks (runs on driver with highmem settings)
"""

import pickle
import argparse
import gc
from google.cloud import storage
from pyspark.sql import SparkSession


class InvertedIndex:
    """Minimal InvertedIndex for saving."""
    def __init__(self, index_name=''):
        self.df = {}
        self.term_total = {}
        self.posting_locs = {}
        self._index_name = index_name


def process_partition_on_worker(partition_info):
    """Process a single partition file - runs on Spark worker."""
    bucket_name, index_type, partition_id = partition_info
    
    from google.cloud import storage
    import pickle
    
    client = storage.Client()
    bucket = client.bucket(bucket_name)
    
    prefix = f"gs://{bucket_name}/indices/{index_type}_index"
    blob_path = f"{prefix}/metadata_{partition_id:04d}.pkl"
    blob = bucket.blob(blob_path)
    
    result = []
    try:
        if blob.exists():
            data = blob.download_as_bytes()
            partition_metadata = pickle.loads(data)
            
            for item in partition_metadata:
                term, locs, df_val, tt_val = item
                result.append((term, locs, df_val, tt_val))
    except Exception as e:
        print(f"Error partition {partition_id}: {e}")
    
    return result


def merge_with_spark(bucket_name: str, index_type: str, num_partitions: int = 200):
    """Use Spark to parallelize reading, then collect and save."""
    
    spark = SparkSession.builder.appName(f"Merge_{index_type}").getOrCreate()
    sc = spark.sparkContext
    
    # Create tasks for each partition
    tasks = [(bucket_name, index_type, i) for i in range(num_partitions)]
    
    print(f"Processing {num_partitions} partitions using Spark...")
    
    # Parallelize and process
    rdd = sc.parallelize(tasks, numSlices=50)
    results = rdd.flatMap(process_partition_on_worker)
    
    # Collect all results
    print("Collecting results...")
    all_terms = results.collect()
    
    print(f"Collected {len(all_terms)} terms")
    
    # Build index
    index = InvertedIndex(index_type)
    for term, locs, df_val, tt_val in all_terms:
        index.df[term] = df_val
        index.term_total[term] = tt_val
        index.posting_locs[term] = locs
    
    del all_terms
    gc.collect()
    
    # Save
    print(f"Serializing {len(index.df)} terms...")
    index_bytes = pickle.dumps(index)
    size_mb = len(index_bytes) / (1024 * 1024)
    print(f"Uploading ({size_mb:.1f} MB)...")
    
    client = storage.Client()
    bucket = client.bucket(bucket_name)
    output_path = f"indices/{index_type}_index/{index_type}_index.pkl"
    bucket.blob(output_path).upload_from_string(index_bytes)
    
    print(f"✅ Saved to gs://{bucket_name}/{output_path}")
    
    spark.stop()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--bucket', default='bucket_207916263')
    parser.add_argument('--index_types', default='body')
    parser.add_argument('--partitions', type=int, default=200)
    args = parser.parse_args()
    
    for index_type in args.index_types.split(','):
        print(f"\n{'='*60}")
        print(f"MERGE: {index_type.upper()}")
        print(f"{'='*60}")
        
        try:
            merge_with_spark(args.bucket, index_type, args.partitions)
        except Exception as e:
            print(f"❌ Error: {e}")
            import traceback
            traceback.print_exc()
    
    print("\n" + "="*60)
    print("DONE!")
    print("="*60)


if __name__ == '__main__':
    main()
