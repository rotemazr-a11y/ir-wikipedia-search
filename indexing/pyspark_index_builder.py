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

def build_all_indices_per_partition(
    doc_partition: Iterator[Tuple[int, Row]],
    use_stemming: bool = False
) -> Iterator[Tuple[str, str, List[Tuple[int, int]]]]:
    """
    Main worker function for mapPartitions.
    Builds all three indices (body, title, anchor) in a single pass.
    
    Args:
        doc_partition: Iterator of (doc_id, row) tuples from the Parquet.
        use_stemming: Whether to apply stemming (False for title/anchor per requirements).
        
    Yields:
        (index_type, term, postings_list) tuples.
        index_type is one of: 'body', 'title', 'anchor'
    """
    # Create three separate SPIMI builders with allocated memory limits
    body_builder = create_body_builder(memory_mb=250)
    title_builder = create_title_builder(memory_mb=100)
    anchor_builder = create_anchor_builder(memory_mb=150)
    
    docs_processed = 0
    
    for doc_id, row in doc_partition:
        try:
            # ----------------------------------------------------------------
            # Process BODY text
            # ----------------------------------------------------------------
            body_text = getattr(row, 'text', None) or ''
            if body_text:
                # Use stemming for body to improve recall
                body_tokens = tokenize_and_process(body_text, use_stemming=use_stemming)
                body_tf = Counter(body_tokens)
                for term, tf in body_tf.items():
                    body_builder.add_posting(term, doc_id, tf)
            
            # ----------------------------------------------------------------
            # Process TITLE
            # ----------------------------------------------------------------
            title = getattr(row, 'title', None) or ''
            if title:
                # NO stemming for title search (per assignment requirements)
                title_tokens = tokenize_no_stem(title)
                title_tf = Counter(title_tokens)
                for term, tf in title_tf.items():
                    title_builder.add_posting(term, doc_id, tf)
            
            # ----------------------------------------------------------------
            # Process ANCHOR texts
            # ----------------------------------------------------------------
            # Anchor text format may vary - handle both list and dict formats
            anchor_text = getattr(row, 'anchor_text', None)
            if anchor_text:
                anchors_combined = []
                
                if isinstance(anchor_text, dict):
                    # Format: {source_doc_id: "anchor text", ...}
                    anchors_combined = list(anchor_text.values())
                elif isinstance(anchor_text, (list, tuple)):
                    anchors_combined = list(anchor_text)
                elif isinstance(anchor_text, str):
                    anchors_combined = [anchor_text]
                
                # Process all anchor texts for this document
                for anchor in anchors_combined:
                    if anchor:
                        # NO stemming for anchor search
                        anchor_tokens = tokenize_no_stem(str(anchor))
                        anchor_tf = Counter(anchor_tokens)
                        for term, tf in anchor_tf.items():
                            anchor_builder.add_posting(term, doc_id, tf)
            
            docs_processed += 1
            
            # Log progress every 10000 docs
            if docs_processed % 10000 == 0:
                logger.info(f"Partition processed {docs_processed} documents")
                
        except Exception as e:
            logger.warning(f"Error processing doc_id {doc_id}: {e}")
            continue
    
    logger.info(f"Partition complete: {docs_processed} documents processed")
    
    # Finalize each builder and yield results
    logger.info("Finalizing body index...")
    for term, postings in body_builder.finalize():
        yield ('body', term, postings)
    
    logger.info("Finalizing title index...")
    for term, postings in title_builder.finalize():
        yield ('title', term, postings)
    
    logger.info("Finalizing anchor index...")
    for term, postings in anchor_builder.finalize():
        yield ('anchor', term, postings)


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
    
    # ========================================================================
    # Step 1: Read the corpus
    # ========================================================================
    logger.info("Reading corpus from Parquet files...")
    
    docs_df = spark.read.parquet(input_path)
    
    # Convert to RDD of (doc_id, row)
    # Assumes 'id' column contains doc_id
    docs_rdd = docs_df.rdd.map(lambda row: (int(row.id), row))
    
    # Coalesce (not repartition) to target partitions - more efficient
    current_partitions = docs_rdd.getNumPartitions()
    if num_partitions and num_partitions != current_partitions:
        if num_partitions < current_partitions:
            docs_rdd = docs_rdd.coalesce(num_partitions)
        else:
            docs_rdd = docs_rdd.repartition(num_partitions)
    
    # Skip expensive count() - just log partitions
    logger.info(f"Corpus ready with {docs_rdd.getNumPartitions()} partitions")
    
    # ========================================================================
    # Step 2: Run SPIMI pipeline
    # ========================================================================
    logger.info("Running SPIMI mapPartitions...")
    
    # This produces RDD[(index_type, term, postings)]
    all_indices_rdd = docs_rdd.mapPartitions(build_all_indices_per_partition)
    
    # Cache since we'll filter 3 times
    all_indices_rdd.persist()
    
    index_paths = {}
    
    # ========================================================================
    # Step 3: Process each index type
    # ========================================================================
    for index_type in ['body', 'title', 'anchor']:
        logger.info(f"Processing {index_type} index...")
        
        # Filter for this index type
        index_rdd = (all_indices_rdd
            .filter(lambda x: x[0] == index_type)
            .map(lambda x: (x[1], x[2]))  # (term, postings)
        )
        
        # Merge postings from different partitions
        merged_rdd = index_rdd.reduceByKey(lambda a, b: a + b)
        
        # Sort postings by doc_id within each term
        sorted_rdd = merged_rdd.mapValues(lambda pl: sorted(pl, key=lambda x: x[0]))
        
        # Write posting lists to binary files
        output_dir = f"{output_path}/{index_type}_index"
        
        def write_partition(partition_idx_and_data):
            """Wrapper for partition writing."""
            partition_idx, data = partition_idx_and_data
            return write_postings_partition(
                partition_idx, iter(data), output_path, index_type, bucket_name
            )
        
        # Collect metadata from writing
        # Using mapPartitionsWithIndex for partition-aware writing
        metadata_rdd = sorted_rdd.mapPartitionsWithIndex(
            lambda idx, it: write_partition_with_index(idx, it, output_path, index_type, bucket_name)
        )
        
        # Collect all metadata
        all_metadata = metadata_rdd.collect()
        
        # Build final InvertedIndex
        index = InvertedIndex()
        
        for term, locs, df, term_total in all_metadata:
            index.posting_locs[term] = locs
            index.df[term] = df
            index.term_total[term] = term_total
        
        # Write the index pickle file
        index_pkl_path = f"{output_dir}/{index_type}_index.pkl"
        index._write_globals(output_dir, f"{index_type}_index", bucket_name)
        
        index_paths[index_type] = index_pkl_path
        logger.info(f"Completed {index_type} index: {len(index.df)} terms")
    
    # Unpersist cached RDD
    all_indices_rdd.unpersist()
    
    logger.info("All indices built successfully!")
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
    
    Args:
        partition_idx: Spark partition index.
        partition_iter: Iterator of (term, postings) tuples.
        output_path: Base output path.
        index_type: Type of index (body/title/anchor).
        bucket_name: GCS bucket name.
        
    Yields:
        Metadata tuples for each term written.
    """
    from inverted_index_gcp import MultiFileWriter, TUPLE_SIZE, TF_MASK
    from contextlib import closing
    
    output_dir = f"{output_path}/{index_type}_index"
    writer_name = f"{index_type}_{partition_idx:04d}"
    
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
                yield (term, locs, df, term_total)
                
    except Exception as e:
        logger.error(f"Error writing partition {partition_idx}: {e}")
        raise


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
