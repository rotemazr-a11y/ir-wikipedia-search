# compute_pageviews.py
"""
Page View Calculator for Wikipedia.

Processes the Wikipedia page view dump file (August 2021) to calculate
total views per article. Uses PySpark for scalable processing.

Input Data Source:
    https://dumps.wikimedia.org/other/pageview_complete/monthly/2021/2021-08/pageviews-202108-user.bz2

Usage:
    # Option 1: With pre-downloaded file
    spark-submit compute_pageviews.py \
        --input /path/to/pageviews-202108-user.bz2 \
        --output gs://bucket/pageviews.pkl \
        --bucket bucket_name

    # Option 2: Download automatically (not recommended for production)
    spark-submit compute_pageviews.py \
        --download \
        --output gs://bucket/pageviews.pkl \
        --bucket bucket_name

Output:
    Pickled dictionary: {article_id (int): total_views (int)}
"""

import os
import sys
import argparse
import logging
import pickle
import tempfile
import urllib.request
from typing import Dict, Tuple, Optional
from pathlib import Path

from pyspark.sql import SparkSession
from pyspark import RDD

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Page view data URL
PAGEVIEW_URL = "https://dumps.wikimedia.org/other/pageview_complete/monthly/2021/2021-08/pageviews-202108-user.bz2"


def download_pageview_dump(output_path: str = None) -> str:
    """
    Download the Wikipedia pageview dump file.
    
    Args:
        output_path: Where to save the file. If None, uses temp directory.
        
    Returns:
        Path to the downloaded file.
    """
    if output_path is None:
        output_path = os.path.join(tempfile.gettempdir(), "pageviews-202108-user.bz2")
    
    if os.path.exists(output_path):
        logger.info(f"File already exists: {output_path}")
        return output_path
    
    logger.info(f"Downloading pageview dump from {PAGEVIEW_URL}")
    logger.info("This may take a while (file is ~4GB compressed)...")
    
    urllib.request.urlretrieve(PAGEVIEW_URL, output_path)
    
    logger.info(f"Downloaded to: {output_path}")
    return output_path


def parse_pageview_line(line: str) -> Optional[Tuple[int, int]]:
    """
    Parse a single line from the pageview dump.
    
    Format: domain page_title page_id access_type daily_total hourly_totals
    Example: en.wikipedia Article_Title 12345 desktop 100 A1B2C3...
    
    We want: en.wikipedia entries, extract page_id and daily_total (views).
    
    Args:
        line: A single line from the pageview file.
        
    Returns:
        Tuple of (page_id, view_count) or None if invalid.
    """
    try:
        # Filter for English Wikipedia only
        if not line.startswith('en.wikipedia'):
            return None
        
        # Split the line by spaces
        parts = line.split(' ')
        
        # The format is: domain page_title page_id access_type daily_total hourly_data
        # But page_title can contain spaces, making parsing tricky
        
        # Minimum valid parts: domain, title, page_id, access_type, daily_total, hourly
        if len(parts) < 6:
            return None
        
        # The page_id is the third field (index 2) in properly formatted lines
        # But if title has spaces, we need a different approach
        
        # Strategy: work backwards from the end
        # Last field: hourly data (A1B2...)
        # Second to last: daily_total (integer)
        # Third to last: access_type (desktop/mobile-web/mobile-app)
        # Fourth to last: page_id (integer)
        
        # For simple cases where title has no spaces:
        if len(parts) >= 6:
            try:
                # Try the standard format first
                page_id = int(parts[2])
                daily_total = int(parts[4])
                return (page_id, daily_total)
            except (ValueError, IndexError):
                pass
        
        # Fallback: parse from the end
        try:
            # Work backwards
            # parts[-1] = hourly data
            # parts[-2] = daily_total
            # parts[-3] = access_type
            # parts[-4] = page_id
            daily_total = int(parts[-2])
            page_id = int(parts[-4])
            
            # Validate page_id is positive
            if page_id <= 0:
                return None
            
            return (page_id, daily_total)
        except (ValueError, IndexError):
            return None
            
    except Exception:
        return None


def process_pageviews(
    spark: SparkSession,
    input_path: str
) -> RDD:
    """
    Process the pageview dump file into an RDD of (page_id, total_views).
    
    Args:
        spark: SparkSession instance.
        input_path: Path to the .bz2 file (can be local or GCS).
        
    Returns:
        RDD of (page_id, total_views) tuples.
    """
    logger.info(f"Reading pageview file: {input_path}")
    
    # Spark can read .bz2 files directly
    lines_rdd = spark.sparkContext.textFile(input_path)
    
    total_lines = lines_rdd.count()
    logger.info(f"Total lines in file: {total_lines}")
    
    # Parse and filter lines
    parsed_rdd = lines_rdd.map(parse_pageview_line).filter(lambda x: x is not None)
    
    # Aggregate by page_id
    aggregated_rdd = parsed_rdd.reduceByKey(lambda a, b: a + b)
    
    logger.info("Pageview aggregation complete")
    return aggregated_rdd


def process_pageviews_optimized(
    spark: SparkSession,
    input_path: str,
    num_partitions: int = 200
) -> RDD:
    """
    Optimized pageview processing with better parallelism.
    
    Args:
        spark: SparkSession instance.
        input_path: Path to the .bz2 file.
        num_partitions: Number of partitions for aggregation.
        
    Returns:
        RDD of (page_id, total_views) tuples.
    """
    logger.info(f"Reading pageview file: {input_path}")
    
    # Read file
    lines_rdd = spark.sparkContext.textFile(input_path)
    
    # Filter English Wikipedia first (before parsing)
    en_wiki_rdd = lines_rdd.filter(lambda line: line.startswith('en.wikipedia'))
    
    logger.info("Filtered for English Wikipedia")
    
    # Parse remaining lines
    def safe_parse(line: str) -> Optional[Tuple[int, int]]:
        try:
            parts = line.split(' ')
            if len(parts) >= 6:
                # Try standard format
                try:
                    page_id = int(parts[2])
                    daily_total = int(parts[4])
                    if page_id > 0:
                        return (page_id, daily_total)
                except ValueError:
                    pass
                
                # Try backwards parsing
                try:
                    daily_total = int(parts[-2])
                    page_id = int(parts[-4])
                    if page_id > 0:
                        return (page_id, daily_total)
                except ValueError:
                    pass
        except Exception:
            pass
        return None
    
    # Parse and filter
    parsed_rdd = en_wiki_rdd.map(safe_parse).filter(lambda x: x is not None)
    
    # Repartition for better aggregation performance
    parsed_rdd = parsed_rdd.repartition(num_partitions)
    
    # Aggregate views per page
    aggregated_rdd = parsed_rdd.reduceByKey(lambda a, b: a + b)
    
    return aggregated_rdd


def save_pageview_dict(
    pageview_rdd: RDD,
    output_path: str,
    bucket_name: str = None
) -> Dict[int, int]:
    """
    Collect pageview results and save as pickled dictionary.
    
    Uses toLocalIterator() to avoid OOM on driver.
    
    Args:
        pageview_rdd: RDD of (page_id, total_views) tuples.
        output_path: Path to save the pickle file.
        bucket_name: GCS bucket name (or None for local).
        
    Returns:
        The pageview dictionary.
    """
    logger.info("Collecting pageview results using toLocalIterator()")
    
    # Build dictionary using iterator
    pageview_dict = {}
    count = 0
    
    for page_id, views in pageview_rdd.toLocalIterator():
        pageview_dict[int(page_id)] = int(views)
        count += 1
        
        if count % 500000 == 0:
            logger.info(f"Collected {count} pageview entries")
    
    logger.info(f"Total pageview entries: {len(pageview_dict)}")
    
    # Calculate some stats
    if pageview_dict:
        max_views = max(pageview_dict.values())
        min_views = min(pageview_dict.values())
        avg_views = sum(pageview_dict.values()) / len(pageview_dict)
        logger.info(f"View stats - Max: {max_views}, Min: {min_views}, Avg: {avg_views:.2f}")
    
    # Save to file
    if bucket_name:
        # Save to GCS using upload_from_file for compatibility
        from google.cloud import storage
        import io
        
        client = storage.Client()
        bucket = client.bucket(bucket_name)
        blob = bucket.blob(output_path)
        
        # Serialize to bytes buffer first
        buffer = io.BytesIO()
        pickle.dump(pageview_dict, buffer)
        buffer.seek(0)
        
        # Upload from buffer
        blob.upload_from_file(buffer, content_type='application/octet-stream')
        
        logger.info(f"Saved pageviews to gs://{bucket_name}/{output_path}")
    else:
        # Save locally
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, 'wb') as f:
            pickle.dump(pageview_dict, f)
        
        logger.info(f"Saved pageviews to {output_path}")
    
    return pageview_dict


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description='Compute pageviews for Wikipedia')
    parser.add_argument('--input', help='Path to pageviews .bz2 file')
    parser.add_argument('--output', required=True, help='Output path for pageviews.pkl')
    parser.add_argument('--bucket', required=True, help='GCS bucket name')
    parser.add_argument('--download', action='store_true', 
                        help='Download pageview dump automatically')
    parser.add_argument('--partitions', type=int, default=200,
                        help='Number of partitions for processing')
    
    args = parser.parse_args()
    
    # Determine input path
    if args.download:
        input_path = download_pageview_dump()
    elif args.input:
        input_path = args.input
    else:
        parser.error("Either --input or --download must be specified")
    
    # Initialize Spark with optimized memory settings
    spark = SparkSession.builder \
        .appName("WikipediaPageViews") \
        .config("spark.executor.memory", "8g") \
        .config("spark.driver.memory", "10g") \
        .config("spark.executor.memoryOverhead", "2g") \
        .config("spark.driver.memoryOverhead", "2g") \
        .config("spark.default.parallelism", str(args.partitions)) \
        .getOrCreate()
    
    try:
        # Process pageviews
        pageview_rdd = process_pageviews_optimized(
            spark, input_path, args.partitions
        )
        
        # Save results
        save_pageview_dict(pageview_rdd, args.output, args.bucket)
        
        logger.info("Pageview computation complete!")
        
    finally:
        spark.stop()


if __name__ == '__main__':
    main()
