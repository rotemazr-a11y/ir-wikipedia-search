import logging
import argparse
from typing import Dict
import pickle
from pyspark.sql import SparkSession, DataFrame
from pyspark.sql.functions import col, sum as _sum, count, lit, explode
from google.cloud import storage

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class PageRankComputer:
    def __init__(self, spark: SparkSession, bucket_name: str, damping_factor: float = 0.85):
        self.spark = spark
        self.damping_factor = damping_factor
        # הגדרת תיקיית צ'קפוינט למניעת התנפחות ה-Lineage
        if bucket_name:
            self.spark.sparkContext.setCheckpointDir(f"gs://{bucket_name}/checkpoints")
        logger.info(f"PageRankComputer initialized (damping={damping_factor})")

    def extract_link_graph(self, wikipedia_dump_path: str, min_links: int = 1) -> DataFrame:
        logger.info("Extracting link graph...")
        wiki_df = self.spark.read.parquet(wikipedia_dump_path)
        
        if 'doc_id' in wiki_df.columns:
            wiki_df = wiki_df.withColumnRenamed('doc_id', 'id')

        edges_df = wiki_df.select(col('id').alias('src_id'), explode(col('links')).alias('dst_id')) \
                          .filter(col('dst_id').isNotNull())

        outlink_counts = edges_df.groupBy('src_id').agg(count('dst_id').alias('num_outlinks'))
        edges_with_counts = edges_df.join(outlink_counts, on='src_id', how='left')

        if min_links > 1:
            edges_with_counts = edges_with_counts.filter(col('num_outlinks') >= min_links)

        return edges_with_counts.cache()

    def compute_pagerank(self, edges_df: DataFrame, output_path: str, num_iterations: int = 10):
        # זיהוי כל הדפים בגרף
        all_pages = edges_df.select(col('src_id').alias('page_id')) \
                            .union(edges_df.select(col('dst_id').alias('page_id'))) \
                            .distinct().cache()

        N = all_pages.count()
        pagerank_df = all_pages.withColumn('pagerank', lit(1.0 / N))

        for iteration in range(num_iterations):
            logger.info(f"Iteration {iteration + 1}/{num_iterations}")

            # חישוב תרומות
            contributions = edges_df.join(pagerank_df, edges_df.src_id == pagerank_df.page_id) \
                                    .withColumn('contribution', col('pagerank') / col('num_outlinks')) \
                                    .select(col('dst_id').alias('page_id'), 'contribution')

            # סכימה ודאמפינג
            incoming_pr = contributions.groupBy('page_id').agg(_sum('contribution').alias('incoming_sum'))
            
            pagerank_df = all_pages.join(incoming_pr, on='page_id', how='left').fillna(0.0) \
                .withColumn('pagerank', lit((1 - self.damping_factor) / N) + 
                            lit(self.damping_factor) * col('incoming_sum')) \
                .select('page_id', 'pagerank')

            # ביצוע Checkpoint כל 3 איטרציות למניעת קריסה
            if (iteration + 1) % 3 == 0:
                pagerank_df = pagerank_df.checkpoint()

        # נרמול לטווח [0,1]
        max_pr = pagerank_df.agg({"pagerank": "max"}).collect()[0][0]
        final_df = pagerank_df.withColumn('pagerank_norm', col('pagerank') / max_pr)

        # שמירה כ-Parquet (בטוח ויעיל)
        parquet_output = output_path.replace(".pkl", "_results")
        final_df.write.mode("overwrite").parquet(parquet_output)
        
        # אופציונלי: איסוף מילון קטן רק אם הזיכרון מאפשר (לא מומלץ ל-6 מיליון דפים)
        # return {row.page_id: row.pagerank_norm for row in final_df.collect()}
        return parquet_output

    @staticmethod
    def load_pagerank(input_path: str):
        # פונקציה לקריאת התוצאות מה-Parquet
        spark = SparkSession.builder.getOrCreate()
        return spark.read.parquet(input_path)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--bucket', required=True)
    parser.add_argument('--iterations', type=int, default=10)
    args = parser.parse_args()

    spark = SparkSession.builder.appName("Wikipedia PageRank").getOrCreate()

    computer = PageRankComputer(spark, bucket_name=args.bucket)
    edges_df = computer.extract_link_graph(args.input)
    
    # הרצת החישוב ושמירה
    output_dir = computer.compute_pagerank(edges_df, args.output, args.iterations)
    
    logger.info(f"✅ PageRank computation complete. Results saved at: {output_dir}")
    spark.stop()

if __name__ == "__main__":
    main()