#!/usr/bin/env python3
"""
Memory-efficient merge: Process partitions in chunks and upload intermediate results.
Then combine the chunks at the end.
"""

import pickle
import argparse
import gc
from google.cloud import storage


class InvertedIndex:
    """Minimal InvertedIndex for saving."""
    def __init__(self, index_name=''):
        self.df = {}
        self.term_total = {}
        self.posting_locs = {}
        self._index_name = index_name


def merge_chunk(bucket, bucket_name: str, index_type: str, 
                start_partition: int, end_partition: int, chunk_id: int):
    """Process a chunk of partitions and save intermediate result."""
    
    prefix = f"gs://{bucket_name}/indices/{index_type}_index"
    
    df = {}
    term_total = {}
    posting_locs = {}
    total_terms = 0
    
    for i in range(start_partition, end_partition):
        blob_path = f"{prefix}/metadata_{i:04d}.pkl"
        blob = bucket.blob(blob_path)
        
        try:
            if not blob.exists():
                continue
            
            data = blob.download_as_bytes()
            partition_metadata = pickle.loads(data)
            
            for item in partition_metadata:
                term, locs, df_val, tt_val = item
                df[term] = df_val
                term_total[term] = tt_val
                posting_locs[term] = locs
                total_terms += 1
            
            del data, partition_metadata
            
        except Exception as e:
            print(f"  Partition {i}: error - {e}")
    
    print(f"  Chunk {chunk_id}: partitions {start_partition}-{end_partition-1}, {total_terms} terms")
    
    # Save chunk
    chunk_data = {
        'df': df,
        'term_total': term_total,
        'posting_locs': posting_locs
    }
    
    chunk_bytes = pickle.dumps(chunk_data)
    chunk_path = f"indices/{index_type}_index/chunk_{chunk_id:03d}.pkl"
    bucket.blob(chunk_path).upload_from_string(chunk_bytes)
    
    del df, term_total, posting_locs, chunk_data, chunk_bytes
    gc.collect()
    
    return total_terms


def combine_chunks(bucket, bucket_name: str, index_type: str, num_chunks: int):
    """Combine all chunks into final index."""
    print(f"  Combining {num_chunks} chunks...")
    
    index = InvertedIndex(index_type)
    
    for chunk_id in range(num_chunks):
        chunk_path = f"indices/{index_type}_index/chunk_{chunk_id:03d}.pkl"
        blob = bucket.blob(chunk_path)
        
        if not blob.exists():
            print(f"  Warning: chunk {chunk_id} not found")
            continue
        
        data = blob.download_as_bytes()
        chunk = pickle.loads(data)
        
        index.df.update(chunk['df'])
        index.term_total.update(chunk['term_total'])
        index.posting_locs.update(chunk['posting_locs'])
        
        del data, chunk
        gc.collect()
        
        print(f"  Loaded chunk {chunk_id}, total terms: {len(index.df)}")
    
    # Save final index
    print(f"  Serializing final index with {len(index.df)} terms...")
    index_bytes = pickle.dumps(index)
    size_mb = len(index_bytes) / (1024 * 1024)
    print(f"  Uploading ({size_mb:.1f} MB)...")
    
    output_path = f"indices/{index_type}_index/{index_type}_index.pkl"
    bucket.blob(output_path).upload_from_string(index_bytes)
    
    print(f"  ✅ Saved to gs://{bucket_name}/{output_path}")
    
    # Clean up chunks
    print(f"  Cleaning up chunks...")
    for chunk_id in range(num_chunks):
        chunk_path = f"indices/{index_type}_index/chunk_{chunk_id:03d}.pkl"
        try:
            bucket.blob(chunk_path).delete()
        except:
            pass


def merge_index_chunked(bucket_name: str, index_type: str, 
                        num_partitions: int = 200, chunk_size: int = 50):
    """Process index in chunks to manage memory."""
    
    client = storage.Client()
    bucket = client.bucket(bucket_name)
    
    num_chunks = (num_partitions + chunk_size - 1) // chunk_size
    
    print(f"Processing {index_type} in {num_chunks} chunks of {chunk_size} partitions...")
    
    total_terms = 0
    for chunk_id in range(num_chunks):
        start = chunk_id * chunk_size
        end = min(start + chunk_size, num_partitions)
        terms = merge_chunk(bucket, bucket_name, index_type, start, end, chunk_id)
        total_terms += terms
        gc.collect()
    
    print(f"\n  Total: {total_terms} terms across {num_chunks} chunks")
    
    # Now combine chunks
    combine_chunks(bucket, bucket_name, index_type, num_chunks)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--bucket', default='bucket_207916263')
    parser.add_argument('--index_types', default='body')
    parser.add_argument('--partitions', type=int, default=200)
    parser.add_argument('--chunk_size', type=int, default=50)
    args = parser.parse_args()
    
    for index_type in args.index_types.split(','):
        print(f"\n{'='*60}")
        print(f"Processing {index_type.upper()} index (chunked)")
        print(f"{'='*60}")
        
        try:
            merge_index_chunked(args.bucket, index_type, args.partitions, args.chunk_size)
        except Exception as e:
            print(f"  ❌ Error: {e}")
            import traceback
            traceback.print_exc()
    
    print(f"\n{'='*60}")
    print("DONE!")
    print(f"{'='*60}")


if __name__ == '__main__':
    main()
