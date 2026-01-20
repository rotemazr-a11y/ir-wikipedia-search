# pyspark_index_builder.py
"""
PySpark-based distributed inverted index builder for Wikipedia.

This is the main Spark driver that orchestrates:
1. Reading Wikipedia corpus from GCS Parquet files
2. Building 3 inverted indices (body, title, anchor) using SPIMI
3. Writing final indices in InvertedIndex-compatible format

Usage:
    spark-submit --py-files deps.zip pyspark_index_builder.py \
        --input gs://bucket/corpus/ \
        --output gs://bucket/indices/ \
        --bucket your-bucket-name
"""

import os
import sys
import argparse
import logging
import pickle
from pathlib import Path
from collections import Counter, defaultdict
from typing import Iterator, Tuple, List, Dict, Any

from pyspark.sql import SparkSession, Row
from pyspark import RDD

# Local imports (these will be in the --py-files zip)
from pre_processing import tokenize_and_process, tokenize_no_stem
from spimi_block_builder import (
    SPIMIBlockBuilder,
    create_body_builder,
    create_title_builder,
    create_anchor_builder
)
from inverted_index_gcp import InvertedIndex, MultiFileWriter, TUPLE_SIZE, TF_MASK

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


# ============================================================================
# PARTITION PROCESSING FUNCTION
# ============================================================================

def build_single_index_per_partition(
    doc_partition: Iterator[Tuple[int, Row]],
    index_type: str,
    use_stemming: bool = False
) -> Iterator[Tuple[str, List[Tuple[int, int]]]]:
    """
    Worker function for mapPartitions - builds ONE index type only.
    This is the SAFER approach that processes each index separately.
    
    Args:
        doc_partition: Iterator of (doc_id, row) tuples from the Parquet.
        index_type: Which index to build ('body', 'title', or 'anchor').
        use_stemming: Whether to apply stemming (only for body).
        
    Yields:
        (term, postings_list) tuples for the specified index type.
    """
    # Create only ONE builder - saves memory!
    if index_type == 'body':
        builder = create_body_builder(memory_mb=400)  # More memory since only one index
    elif index_type == 'title':
        builder = create_title_builder(memory_mb=300)
    else:  # anchor
        builder = create_anchor_builder(memory_mb=350)
    
    docs_processed = 0
    
    for doc_id, row in doc_partition:
        try:
            if index_type == 'body':
                # ----------------------------------------------------------------
                # Process BODY text
                # ----------------------------------------------------------------
                body_text = getattr(row, 'text', None) or ''
                if body_text:
                    body_tokens = tokenize_and_process(body_text, use_stemming=use_stemming)
                    body_tf = Counter(body_tokens)
                    for term, tf in body_tf.items():
                        builder.add_posting(term, doc_id, tf)
                        
            elif index_type == 'title':
                # ----------------------------------------------------------------
                # Process TITLE
                # ----------------------------------------------------------------
                title = getattr(row, 'title', None) or ''
                if title:
                    title_tokens = tokenize_no_stem(title)
                    title_tf = Counter(title_tokens)
                    for term, tf in title_tf.items():
                        builder.add_posting(term, doc_id, tf)
                        
            else:  # anchor
                # ----------------------------------------------------------------
                # Process ANCHOR texts
                # ----------------------------------------------------------------
                anchor_text = getattr(row, 'anchor_text', None)
                if anchor_text:
                    anchors_combined = []
                    
                    if isinstance(anchor_text, dict):
                        anchors_combined = list(anchor_text.values())
                    elif isinstance(anchor_text, (list, tuple)):
                        anchors_combined = list(anchor_text)
                    elif isinstance(anchor_text, str):
                        anchors_combined = [anchor_text]
                    
                    for anchor in anchors_combined:
                        if anchor:
                            anchor_tokens = tokenize_no_stem(str(anchor))
                            anchor_tf = Counter(anchor_tokens)
                            for term, tf in anchor_tf.items():
                                builder.add_posting(term, doc_id, tf)
            
            docs_processed += 1
            
            if docs_processed % 10000 == 0:
                logger.info(f"[{index_type}] Partition processed {docs_processed} documents")
                
        except Exception as e:
            logger.warning(f"[{index_type}] Error processing doc_id {doc_id}: {e}")
            continue
    
    logger.info(f"[{index_type}] Partition complete: {docs_processed} documents processed")
    
    # Finalize and yield results
    logger.info(f"Finalizing {index_type} index...")
    for term, postings in builder.finalize():
        yield (term, postings)


# ============================================================================
# INDEX WRITING FUNCTIONS
# ============================================================================

def write_postings_partition(
    partition_id: int,
    partition_data: Iterator[Tuple[str, List[Tuple[int, int]]]],
    base_dir: str,
    index_name: str,
    bucket_name: str = None
) -> Iterator[Tuple[str, List[Tuple[str, int]], int, int]]:
    """
    Write posting lists for a partition using MultiFileWriter.
    
    Args:
        partition_id: The Spark partition ID.
        partition_data: Iterator of (term, postings) tuples.
        base_dir: Base directory for output files.
        index_name: Name of the index (body/title/anchor).
        bucket_name: GCS bucket name (or None for local).
        
    Yields:
        (term, posting_locs, df, term_total) for metadata collection.
    """
    from inverted_index_gcp import MultiFileWriter, TUPLE_SIZE, TF_MASK, get_bucket
    from contextlib import closing
    
    # Create output directory path
    output_dir = f"{base_dir}/{index_name}_index"
    writer_name = f"{index_name}_{partition_id:04d}"
    
    with closing(MultiFileWriter(output_dir, writer_name, bucket_name)) as writer:
        for term, postings in partition_data:
            # Calculate df and term_total
            df = len(postings)
            term_total = sum(tf for _, tf in postings)
            
            # Convert postings to binary format
            # Format: 4 bytes doc_id + 2 bytes tf per posting
            b = b''.join([
                (doc_id << 16 | (tf & TF_MASK)).to_bytes(TUPLE_SIZE, 'big')
                for doc_id, tf in postings
            ])
            
            # Write to binary files
            locs = writer.write(b)
            
            yield (term, locs, df, term_total)


def merge_posting_locs(a: Tuple, b: Tuple) -> Tuple:
    """
    Merge posting location metadata from two partitions.
    
    Args:
        a, b: Tuples of (locs_list, df, term_total)
        
    Returns:
        Merged tuple with combined locs, summed df and term_total.
    """
    locs_a, df_a, total_a = a
    locs_b, df_b, total_b = b
    return (locs_a + locs_b, df_a + df_b, total_a + total_b)


# ============================================================================
# MAIN DRIVER
# ============================================================================

def build_indices(
    spark: SparkSession,
    input_path: str,
    output_path: str,
    bucket_name: str,
    num_partitions: int = 200
) -> Dict[str, str]:
    """
    Main function to build all three inverted indices.
    SAFER APPROACH: Processes each index separately to avoid OOM.
    
    Args:
        spark: SparkSession instance.
        input_path: GCS path to input Parquet files.
        output_path: GCS path for output indices.
        bucket_name: GCS bucket name.
        num_partitions: Number of partitions for processing.
        
    Returns:
        Dictionary mapping index names to their .pkl file paths.
    """
    logger.info(f"Starting index build from {input_path}")
    logger.info("Using SAFE mode: processing each index separately")
    
    index_paths = {}
    
    # ========================================================================
    # Process each index type SEPARATELY (safer, avoids executor OOM)
    # ========================================================================
    for index_type in ['body', 'title', 'anchor']:
        logger.info(f"")
        logger.info(f"{'='*60}")
        logger.info(f"Starting {index_type.upper()} index")
        logger.info(f"{'='*60}")
        
        # ====================================================================
        # Step 1: Read the corpus fresh for each index
        # ====================================================================
        logger.info(f"[{index_type}] Reading corpus from Parquet files...")
        
        docs_df = spark.read.parquet(input_path)
        
        # Convert to RDD of (doc_id, row)
        docs_rdd = docs_df.rdd.map(lambda row: (int(row.id), row))
        
        # Coalesce to target partitions
        current_partitions = docs_rdd.getNumPartitions()
        if num_partitions and num_partitions != current_partitions:
            if num_partitions < current_partitions:
                docs_rdd = docs_rdd.coalesce(num_partitions)
            else:
                docs_rdd = docs_rdd.repartition(num_partitions)
        
        logger.info(f"[{index_type}] Corpus ready with {docs_rdd.getNumPartitions()} partitions")
        
        # ====================================================================
        # Step 2: Build single index using SPIMI
        # ====================================================================
        logger.info(f"[{index_type}] Running SPIMI mapPartitions...")
        
        # IMPORTANT: Use default argument to capture index_type by VALUE, not reference!
        # This avoids the closure bug where all partitions would use the last index_type
        def make_partition_builder(idx_type):
            """Factory function to properly capture index_type in closure."""
            return lambda it: build_single_index_per_partition(it, idx_type)
        
        # Build only THIS index (saves executor memory!)
        index_rdd = docs_rdd.mapPartitions(make_partition_builder(index_type))
        
        # ====================================================================
        # Step 3: Merge postings and sort
        # ====================================================================
        logger.info(f"[{index_type}] Merging postings by term...")
        
        # Merge postings from different partitions
        merged_rdd = index_rdd.reduceByKey(lambda a, b: a + b)
        
        # Sort postings by doc_id within each term
        sorted_rdd = merged_rdd.mapValues(lambda pl: sorted(pl, key=lambda x: x[0]))
        
        # ====================================================================
        # Step 4: Write posting lists to GCS
        # ====================================================================
        output_dir = f"{output_path}/{index_type}_index"
        logger.info(f"[{index_type}] Writing posting lists to {output_dir}...")
        
        # Factory to capture index_type by value for the write function
        def make_partition_writer(idx_type, out_path, bkt_name):
            """Factory function to properly capture values in closure."""
            return lambda idx, it: write_partition_with_index(idx, it, out_path, idx_type, bkt_name)
        
        # Write partitions - metadata goes to GCS pickle files
        metadata_rdd = sorted_rdd.mapPartitionsWithIndex(
            make_partition_writer(index_type, output_path, bucket_name)
        )
        
        # Collect just the counts (tiny! won't cause OOM)
        partition_counts = metadata_rdd.collect()
        total_terms = sum(count for _, count in partition_counts)
        logger.info(f"[{index_type}] Wrote {total_terms} terms across {len(partition_counts)} partitions")
        
        # ====================================================================
        # Step 5: Merge partition metadata
        # ====================================================================
        logger.info(f"[{index_type}] Merging partition metadata...")
        index = merge_partition_metadata(output_dir, index_type, bucket_name, num_partitions)
        
        # Write the final index pickle file
        index_pkl_path = f"{output_dir}/{index_type}_index.pkl"
        index._write_globals(output_dir, f"{index_type}_index", bucket_name)
        
        index_paths[index_type] = index_pkl_path
        logger.info(f"[{index_type}] COMPLETED: {len(index.df)} unique terms")
        
        # ====================================================================
        # Step 6: Clean up before next index (CRITICAL for memory!)
        # ====================================================================
        logger.info(f"[{index_type}] Cleaning up...")
        del index
        del docs_df
        del docs_rdd
        del index_rdd
        del merged_rdd
        del sorted_rdd
        del metadata_rdd
        
        import gc
        gc.collect()
        
        # Force Spark to release cached data
        spark.catalog.clearCache()
        
        logger.info(f"[{index_type}] Memory released, ready for next index")
    
    logger.info("")
    logger.info("="*60)
    logger.info("ALL INDICES BUILT SUCCESSFULLY!")
    logger.info("="*60)
    return index_paths


def write_partition_with_index(
    partition_idx: int,
    partition_iter: Iterator[Tuple[str, List[Tuple[int, int]]]],
    output_path: str,
    index_type: str,
    bucket_name: str
) -> Iterator[Tuple[str, List, int, int]]:
    """
    Write partition data with partition index for unique filenames.
    Also writes partition metadata to a separate pickle file in GCS.
    
    Args:
        partition_idx: Spark partition index.
        partition_iter: Iterator of (term, postings) tuples.
        output_path: Base output path.
        index_type: Type of index (body/title/anchor).
        bucket_name: GCS bucket name.
        
    Yields:
        Just a count of terms written (not all metadata!).
    """
    from inverted_index_gcp import MultiFileWriter, TUPLE_SIZE, TF_MASK, _make_path, _open, get_bucket
    from contextlib import closing
    import pickle
    
    output_dir = f"{output_path}/{index_type}_index"
    writer_name = f"{index_type}_{partition_idx:04d}"
    
    # Collect metadata for this partition
    partition_metadata = []
    
    try:
        with closing(MultiFileWriter(output_dir, writer_name, bucket_name)) as writer:
            for term, postings in partition_iter:
                df = len(postings)
                term_total = sum(tf for _, tf in postings)
                
                # Convert to binary format matching InvertedIndex:
                # Each posting: 4 bytes doc_id (high) + 2 bytes tf (low) = 6 bytes
                # Format: (doc_id << 16 | (tf & TF_MASK)) stored in TUPLE_SIZE bytes
                b = b''.join([
                    ((doc_id << 16) | (tf & TF_MASK)).to_bytes(TUPLE_SIZE, 'big')
                    for doc_id, tf in postings
                ])
                
                locs = writer.write(b)
                partition_metadata.append((term, locs, df, term_total))
        
        # Write partition metadata to GCS (this is the key fix!)
        if partition_metadata:
            metadata_path = _make_path(output_dir, f'metadata_{partition_idx:04d}.pkl', is_gcs=True)
            bucket = get_bucket(bucket_name)
            with _open(metadata_path, 'wb', bucket) as f:
                pickle.dump(partition_metadata, f)
        
        # Return just the count (tiny! won't cause OOM)
        yield (partition_idx, len(partition_metadata))
                
    except Exception as e:
        logger.error(f"Error writing partition {partition_idx}: {e}")
        raise


def merge_partition_metadata(output_dir: str, index_type: str, bucket_name: str, num_partitions: int) -> 'InvertedIndex':
    """
    Merge partition metadata files into a single InvertedIndex.
    Reads one partition at a time to avoid OOM.
    
    Args:
        output_dir: Base output directory for the index.
        index_type: Type of index (body/title/anchor).
        bucket_name: GCS bucket name.
        num_partitions: Number of partitions to merge.
        
    Returns:
        Merged InvertedIndex with all term metadata.
    """
    from inverted_index_gcp import InvertedIndex, _make_path, _open, get_bucket
    import pickle
    import gc
    
    index = InvertedIndex()
    bucket = get_bucket(bucket_name)
    
    for i in range(num_partitions):
        metadata_path = _make_path(output_dir, f'metadata_{i:04d}.pkl', is_gcs=True)
        
        try:
            with _open(metadata_path, 'rb', bucket) as f:
                partition_metadata = pickle.load(f)
            
            # Merge into main index
            for term, locs, df, term_total in partition_metadata:
                if term in index.posting_locs:
                    index.posting_locs[term].extend(locs)
                    index.df[term] += df
                    index.term_total[term] += term_total
                else:
                    index.posting_locs[term] = locs
                    index.df[term] = df
                    index.term_total[term] = term_total
            
            # Free memory from this partition
            del partition_metadata
            
            if i % 20 == 0:
                logger.info(f"  Merged partition {i}/{num_partitions}")
                gc.collect()
                
        except Exception as e:
            logger.warning(f"Could not read metadata partition {i}: {e}")
            continue
    
    return index


def main():
    """Main entry point for the index builder."""
    parser = argparse.ArgumentParser(description='Build Wikipedia inverted indices')
    parser.add_argument('--input', required=True, help='GCS path to input Parquet files')
    parser.add_argument('--output', required=True, help='GCS path for output indices')
    parser.add_argument('--bucket', required=True, help='GCS bucket name')
    parser.add_argument('--partitions', type=int, default=200, help='Number of partitions')
    
    args = parser.parse_args()
    
    # Initialize Spark
    spark = SparkSession.builder \
        .appName("WikipediaIndexBuilder") \
        .config("spark.executor.memory", "8g") \
        .config("spark.driver.memory", "8g") \
        .config("spark.executor.memoryOverhead", "2g") \
        .config("spark.sql.shuffle.partitions", str(args.partitions)) \
        .config("spark.default.parallelism", str(args.partitions)) \
        .config("spark.serializer", "org.apache.spark.serializer.KryoSerializer") \
        .getOrCreate()
    
    try:
        index_paths = build_indices(
            spark=spark,
            input_path=args.input,
            output_path=args.output,
            bucket_name=args.bucket,
            num_partitions=args.partitions
        )
        
        logger.info("Index paths:")
        for name, path in index_paths.items():
            logger.info(f"  {name}: {path}")
            
    finally:
        spark.stop()


if __name__ == '__main__':
    main()
