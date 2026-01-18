# compute_doc_metadata.py
"""
Document Metadata Extractor for Wikipedia.

Extracts doc_titles and doc_lengths from the Wikipedia corpus Parquet files.
These are required by search_frontend.py for:
- doc_titles: Returning article titles in search results
- doc_lengths: Computing TF-IDF normalization (average doc length)

Usage:
    spark-submit compute_doc_metadata.py \
        --input gs://bucket/corpus \
        --output gs://bucket/ \
        --bucket bucket_name

Output:
    - gs://bucket/doc_titles.pkl - Dict[int, str] mapping doc_id to title
    - gs://bucket/doc_lengths.pkl - Dict[int, int] mapping doc_id to token count
"""

import os
import sys
import argparse
import logging
import pickle
from typing import Dict, Tuple

from pyspark.sql import SparkSession

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def extract_doc_metadata(
    spark: SparkSession,
    input_path: str,
    output_path: str,
    bucket_name: str
) -> Tuple[Dict[int, str], Dict[int, int]]:
    """
    Extract document titles and lengths from the Wikipedia corpus.
    
    Args:
        spark: SparkSession instance.
        input_path: GCS path to Parquet corpus files.
        output_path: GCS path for output files.
        bucket_name: GCS bucket name.
        
    Returns:
        Tuple of (doc_titles dict, doc_lengths dict).
    """
    logger.info(f"Reading corpus from {input_path}")
    
    # Read Parquet files
    docs_df = spark.read.parquet(input_path)
    
    logger.info(f"Schema: {docs_df.schema.simpleString()}")
    logger.info(f"Columns: {docs_df.columns}")
    
    # Select only the columns we need
    # Columns expected: id (int), title (str), text (str)
    selected_df = docs_df.select('id', 'title', 'text')
    
    # Count documents
    total_docs = selected_df.count()
    logger.info(f"Processing {total_docs} documents")
    
    # Define function to count tokens in text
    def count_tokens(text: str) -> int:
        """Count tokens in text using simple whitespace split."""
        if not text:
            return 0
        # Use simple word count for length estimation
        # This is faster than full tokenization and sufficient for normalization
        return len(text.split())
    
    # Register UDF for token counting
    from pyspark.sql.functions import udf
    from pyspark.sql.types import IntegerType
    
    count_tokens_udf = udf(count_tokens, IntegerType())
    
    # Add token count column
    with_length_df = selected_df.withColumn('doc_length', count_tokens_udf('text'))
    
    # Collect results using toLocalIterator to avoid OOM
    logger.info("Collecting document metadata...")
    
    doc_titles = {}
    doc_lengths = {}
    count = 0
    
    for row in with_length_df.toLocalIterator():
        doc_id = int(row['id'])
        title = str(row['title']) if row['title'] else f"Document {doc_id}"
        length = int(row['doc_length']) if row['doc_length'] else 0
        
        doc_titles[doc_id] = title
        doc_lengths[doc_id] = length
        
        count += 1
        if count % 500000 == 0:
            logger.info(f"Collected {count} documents")
    
    logger.info(f"Total documents: {len(doc_titles)}")
    
    # Calculate statistics
    if doc_lengths:
        total_length = sum(doc_lengths.values())
        avg_length = total_length / len(doc_lengths)
        max_length = max(doc_lengths.values())
        min_length = min(doc_lengths.values())
        logger.info(f"Length stats - Avg: {avg_length:.2f}, Max: {max_length}, Min: {min_length}")
    
    # Save to GCS
    save_dict_to_gcs(doc_titles, 'doc_titles.pkl', bucket_name)
    save_dict_to_gcs(doc_lengths, 'doc_lengths.pkl', bucket_name)
    
    return doc_titles, doc_lengths


def save_dict_to_gcs(data: Dict, filename: str, bucket_name: str):
    """
    Save a dictionary to GCS as a pickle file.
    
    Args:
        data: Dictionary to save.
        filename: Output filename.
        bucket_name: GCS bucket name.
    """
    from google.cloud import storage
    
    logger.info(f"Saving {filename} to gs://{bucket_name}/{filename}")
    
    client = storage.Client()
    bucket = client.bucket(bucket_name)
    blob = bucket.blob(filename)
    
    with blob.open('wb') as f:
        pickle.dump(data, f)
    
    logger.info(f"Saved {len(data)} entries to gs://{bucket_name}/{filename}")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description='Extract document metadata from Wikipedia corpus')
    parser.add_argument('--input', required=True, help='GCS path to Parquet corpus files')
    parser.add_argument('--output', default='', help='GCS path for output (not used, files go to bucket root)')
    parser.add_argument('--bucket', required=True, help='GCS bucket name')
    
    args = parser.parse_args()
    
    # Initialize Spark
    spark = SparkSession.builder \
        .appName("WikipediaDocMetadata") \
        .config("spark.executor.memory", "4g") \
        .config("spark.driver.memory", "8g") \
        .config("spark.sql.shuffle.partitions", "100") \
        .getOrCreate()
    
    try:
        doc_titles, doc_lengths = extract_doc_metadata(
            spark=spark,
            input_path=args.input,
            output_path=args.output,
            bucket_name=args.bucket
        )
        
        logger.info("Document metadata extraction complete!")
        logger.info(f"  doc_titles.pkl: {len(doc_titles)} entries")
        logger.info(f"  doc_lengths.pkl: {len(doc_lengths)} entries")
        
    finally:
        spark.stop()


if __name__ == '__main__':
    main()
