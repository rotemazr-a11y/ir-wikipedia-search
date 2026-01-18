# compute_pagerank.py
"""
PageRank Calculator for Wikipedia Corpus.

Implements the PageRank algorithm using iterative PySpark DataFrame operations.
No GraphFrames library used - manual implementation for full control and transparency.

Usage:
    spark-submit compute_pagerank.py \
        --input gs://bucket/corpus \
        --output gs://bucket/pagerank \
        --bucket bucket_name \
        --iterations 10

Output:
    Pickled dictionary: {page_id (int): pagerank_norm (float)}
"""

import os
import sys
import argparse
import logging
import pickle
from typing import Dict, Any
from pathlib import Path

from pyspark.sql import SparkSession, DataFrame
from pyspark.sql import functions as F
from pyspark.sql.types import FloatType, LongType

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# PageRank constants
DAMPING_FACTOR = 0.85
DEFAULT_ITERATIONS = 10
CHECKPOINT_INTERVAL = 3  # Checkpoint every N iterations


def extract_graph(spark: SparkSession, input_path: str) -> tuple:
    """
    Extract graph structure from Wikipedia Parquet files.
    
    Args:
        spark: SparkSession instance.
        input_path: GCS path to Parquet files.
        
    Returns:
        Tuple of (edges_df, vertices_df, N) where N is vertex count.
    """
    logger.info(f"Reading corpus from {input_path}")
    
    # Read Parquet files - we need 'id' and 'links' columns
    # 'links' contains array of page IDs this document links to
    docs_df = spark.read.parquet(input_path)
    
    # Show schema for debugging
    logger.info(f"Schema: {docs_df.schema.simpleString()}")
    
    # Check what columns are available
    available_cols = docs_df.columns
    logger.info(f"Available columns: {available_cols}")
    
    # The links column might be named differently
    # Common names: 'links', 'outlinks', 'hyperlinks'
    links_col = None
    for col_name in ['links', 'outlinks', 'hyperlinks', 'link_ids']:
        if col_name in available_cols:
            links_col = col_name
            break
    
    if links_col is None:
        # If no explicit links column, we need to extract from anchor_text
        logger.warning("No explicit links column found. Extracting from anchor_text...")
        # anchor_text is a list of Row(id=target_id, text="anchor text")
        # We need to extract the 'id' field from each Row
        if 'anchor_text' in available_cols:
            # anchor_text is an array of structs: [{id: int, text: string}, ...]
            # Use explode to flatten, then select the 'id' field
            edges_df = docs_df.select(
                F.col('id').alias('src_id'),
                F.explode(F.col('anchor_text')).alias('anchor_row')
            ).select(
                F.col('src_id'),
                F.col('anchor_row.id').alias('dst_id')
            ).filter(F.col('dst_id').isNotNull())
        else:
            raise ValueError("Cannot find links or anchor_text column in dataset")
    else:
        # Explode links array to create edges
        # Each row becomes (src_id, dst_id)
        edges_df = docs_df.select(
            F.col('id').alias('src_id'),
            F.explode(F.col(links_col)).alias('dst_id')
        ).filter(F.col('dst_id').isNotNull())
    
    # Cast to ensure integer types
    edges_df = edges_df.select(
        F.col('src_id').cast(LongType()),
        F.col('dst_id').cast(LongType())
    )
    
    logger.info(f"Edges DataFrame created")
    
    # Get all unique vertices (pages that link OR are linked to)
    src_vertices = edges_df.select(F.col('src_id').alias('page_id'))
    dst_vertices = edges_df.select(F.col('dst_id').alias('page_id'))
    vertices_df = src_vertices.union(dst_vertices).distinct()
    
    # Count total vertices
    N = vertices_df.count()
    logger.info(f"Graph extracted: {N} vertices")
    
    # Cache edges for repeated use
    edges_df.cache()
    edge_count = edges_df.count()
    logger.info(f"Total edges: {edge_count}")
    
    return edges_df, vertices_df, N


def compute_outlink_counts(edges_df: DataFrame) -> DataFrame:
    """
    Calculate the number of outgoing links for each source page.
    
    Args:
        edges_df: DataFrame with (src_id, dst_id) columns.
        
    Returns:
        DataFrame with (src_id, num_outlinks) columns.
    """
    outlinks_df = edges_df.groupBy('src_id').agg(
        F.count('dst_id').alias('num_outlinks')
    )
    return outlinks_df


def run_pagerank_iterations(
    spark: SparkSession,
    edges_df: DataFrame,
    vertices_df: DataFrame,
    N: int,
    iterations: int = DEFAULT_ITERATIONS,
    damping: float = DAMPING_FACTOR
) -> DataFrame:
    """
    Run iterative PageRank algorithm.
    
    Args:
        spark: SparkSession instance.
        edges_df: DataFrame with (src_id, dst_id) edges.
        vertices_df: DataFrame with all page_ids.
        N: Total number of vertices.
        iterations: Number of PageRank iterations.
        damping: Damping factor (typically 0.85).
        
    Returns:
        DataFrame with (page_id, pagerank) columns.
    """
    logger.info(f"Starting PageRank: {iterations} iterations, damping={damping}, N={N}")
    
    # Pre-calculate outlink counts (how many links each page has)
    outlinks_df = compute_outlink_counts(edges_df)
    outlinks_df.cache()
    
    # Initialize PageRank: every page starts with equal rank
    initial_rank = 1.0 / N
    pagerank_df = vertices_df.select(
        F.col('page_id'),
        F.lit(initial_rank).alias('pagerank')
    )
    
    # Precompute the random jump factor (1-d)/N
    random_jump = (1 - damping) / N
    
    # Set checkpoint directory for lineage truncation
    checkpoint_dir = "/tmp/pagerank_checkpoints"
    spark.sparkContext.setCheckpointDir(checkpoint_dir)
    
    for i in range(iterations):
        logger.info(f"Iteration {i + 1}/{iterations}")
        
        # Step 1: Calculate contributions
        # Join edges with pagerank to get rank of source pages
        # Then join with outlinks to calculate contribution per edge
        contributions_df = (
            edges_df
            .join(pagerank_df, edges_df.src_id == pagerank_df.page_id)
            .join(outlinks_df, edges_df.src_id == outlinks_df.src_id)
            .select(
                edges_df.dst_id.alias('page_id'),
                (F.col('pagerank') / F.col('num_outlinks')).alias('contribution')
            )
        )
        
        # Step 2: Aggregate contributions by destination page
        incoming_df = contributions_df.groupBy('page_id').agg(
            F.sum('contribution').alias('incoming_sum')
        )
        
        # Step 3: Update PageRank values
        # new_rank = (1-d)/N + d * sum(contributions)
        # We need to handle pages with no incoming links (dangling nodes)
        pagerank_df = (
            vertices_df
            .join(incoming_df, 'page_id', 'left')
            .select(
                F.col('page_id'),
                (F.lit(random_jump) + F.lit(damping) * F.coalesce(F.col('incoming_sum'), F.lit(0.0))).alias('pagerank')
            )
        )
        
        # Step 4: Checkpoint periodically to truncate lineage
        # This is CRUCIAL for iterative algorithms to prevent
        # lineage explosion which causes memory issues and slow planning
        if (i + 1) % CHECKPOINT_INTERVAL == 0:
            logger.info(f"Checkpointing at iteration {i + 1}")
            pagerank_df = pagerank_df.checkpoint()
            pagerank_df.cache()
            # Force materialization
            pagerank_df.count()
    
    logger.info("PageRank iterations complete")
    return pagerank_df


def normalize_pagerank(pagerank_df: DataFrame) -> DataFrame:
    """
    Normalize PageRank values to [0, 1] range.
    
    Args:
        pagerank_df: DataFrame with (page_id, pagerank) columns.
        
    Returns:
        DataFrame with (page_id, pagerank_norm) columns.
    """
    logger.info("Normalizing PageRank values")
    
    # Find maximum PageRank value
    max_rank = pagerank_df.agg(F.max('pagerank')).collect()[0][0]
    logger.info(f"Max PageRank value: {max_rank}")
    
    if max_rank is None or max_rank == 0:
        logger.warning("Max PageRank is 0 or None, using 1.0 as default")
        max_rank = 1.0
    
    # Normalize
    normalized_df = pagerank_df.select(
        F.col('page_id'),
        (F.col('pagerank') / F.lit(max_rank)).alias('pagerank_norm')
    )
    
    return normalized_df


def save_pagerank_dict(
    pagerank_df: DataFrame,
    output_path: str,
    bucket_name: str = None
) -> Dict[int, float]:
    """
    Collect PageRank results and save as pickled dictionary.
    
    Uses toLocalIterator() to avoid OOM on driver.
    
    Args:
        pagerank_df: DataFrame with (page_id, pagerank_norm) columns.
        output_path: Path to save the pickle file.
        bucket_name: GCS bucket name (or None for local).
        
    Returns:
        The PageRank dictionary.
    """
    logger.info("Collecting PageRank results using toLocalIterator()")
    
    # Build dictionary using iterator to avoid OOM
    # DO NOT use .collect() - it will crash with 6M+ pages
    pagerank_dict = {}
    count = 0
    
    for row in pagerank_df.toLocalIterator():
        page_id = int(row.page_id)
        rank = float(row.pagerank_norm)
        pagerank_dict[page_id] = rank
        count += 1
        
        if count % 500000 == 0:
            logger.info(f"Collected {count} PageRank values")
    
    logger.info(f"Total PageRank entries: {len(pagerank_dict)}")
    
    # Save to file
    if bucket_name:
        # Save to GCS
        from google.cloud import storage
        
        client = storage.Client()
        bucket = client.bucket(bucket_name)
        blob = bucket.blob(output_path)
        
        with blob.open('wb') as f:
            pickle.dump(pagerank_dict, f)
        
        logger.info(f"Saved PageRank to gs://{bucket_name}/{output_path}")
    else:
        # Save locally
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, 'wb') as f:
            pickle.dump(pagerank_dict, f)
        
        logger.info(f"Saved PageRank to {output_path}")
    
    return pagerank_dict


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description='Compute PageRank for Wikipedia')
    parser.add_argument('--input', required=True, help='GCS path to Parquet corpus')
    parser.add_argument('--output', required=True, help='Output path for pagerank.pkl')
    parser.add_argument('--bucket', required=True, help='GCS bucket name')
    parser.add_argument('--iterations', type=int, default=DEFAULT_ITERATIONS,
                        help=f'Number of PageRank iterations (default: {DEFAULT_ITERATIONS})')
    
    args = parser.parse_args()
    
    # Initialize Spark
    spark = SparkSession.builder \
        .appName("WikipediaPageRank") \
        .config("spark.executor.memory", "8g") \
        .config("spark.driver.memory", "8g") \
        .config("spark.sql.shuffle.partitions", "200") \
        .getOrCreate()
    
    try:
        # Step 1: Extract graph
        edges_df, vertices_df, N = extract_graph(spark, args.input)
        
        # Step 2: Run PageRank
        pagerank_df = run_pagerank_iterations(
            spark, edges_df, vertices_df, N,
            iterations=args.iterations
        )
        
        # Step 3: Normalize
        normalized_df = normalize_pagerank(pagerank_df)
        
        # Step 4: Save results
        save_pagerank_dict(normalized_df, args.output, args.bucket)
        
        logger.info("PageRank computation complete!")
        
    finally:
        spark.stop()


if __name__ == '__main__':
    main()
