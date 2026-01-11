import logging
import pickle
import argparse
from collections import Counter
from pyspark.sql import SparkSession
from google.cloud import storage

# ייבוא מהקבצים שלך (וודאי שהם הועלו לבאקט)
from pre_processing import tokenize_and_process
from inverted_index_gcp import InvertedIndex, get_bucket, _open

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class PySparkIndexBuilder:
    def __init__(self, spark: SparkSession):
        self.spark = spark

    @staticmethod
    def parse_wikipedia_document(row):
        try:
            doc_id = int(row.id)
            anchors_raw = getattr(row, 'anchor_text', [])
            anchors_combined = ' '.join([a['text'] for a in anchors_raw if a and 'text' in a]) if isinstance(anchors_raw, list) else ""
            return (doc_id, {'anchors': anchors_combined, 'title': getattr(row, 'title', '')})
        except: return (None, None)

    @staticmethod
    def emit_term_doc_pairs(doc_tuple, field):
        doc_id, content = doc_tuple
        if doc_id is None or not content: return
        text = content.get(field, '')
        if not text: return
        
        tokens = tokenize_and_process(text, remove_stops=True, stem=(field == 'body'))
        term_counts = Counter(tokens)
        for term, tf in term_counts.items():
            yield (term, (doc_id, tf))

    def build_index_generic(self, docs_rdd, field_name, output_dir, num_partitions, bucket_name):
        # 1. יצירת ה-Postings
        term_postings = docs_rdd.flatMap(lambda doc: PySparkIndexBuilder.emit_term_doc_pairs(doc, field_name)) \
                                .groupByKey(numPartitions=num_partitions) \
                                .mapValues(lambda ps: sorted(list(ps), key=lambda x: x[0]))

        # 2. כתיבת ה-Binaries (Workers)
        def write_partition_to_bin(partition_iterator):
            from pyspark import TaskContext
            ctx = TaskContext.get()
            bucket_id = f"{field_name}_{ctx.partitionId()}"
            partition_data = list(partition_iterator)
            InvertedIndex.write_a_posting_list((bucket_id, partition_data), output_dir, bucket_name)
            return [bucket_id]

        partition_ids = term_postings.mapPartitions(write_partition_to_bin).collect()

        # 3. איסוף המטא-דאטה (Driver) - התיקון הקריטי כאן
        logger.info("Merging temporary metadata files...")
        index = InvertedIndex()
        bucket = get_bucket(bucket_name)
        base_prefix = output_dir.replace(f"gs://{bucket_name}/", "").strip("/")

        for b_id in partition_ids:
            locs_path = f"{base_prefix}/{b_id}_posting_locs.pickle"
            try:
                with _open(locs_path, 'rb', bucket) as f:
                    part_locs = pickle.load(f)
                    for term, locs in part_locs.items():
                        index.posting_locs[term].extend(locs)
                        # DF מדויק: מספר המסמכים הייחודיים
                        index.df[term] = index.df.get(term, 0) + len(locs)
            except Exception as e:
                logger.error(f"Failed to read metadata for {b_id} at {locs_path}: {e}")

        index.write_index(output_dir, f'{field_name}_index', bucket_name)
        return index

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--bucket', required=True)
    parser.add_argument('--partitions', type=int, default=128)
    args = parser.parse_args()

    spark = SparkSession.builder.appName("WikiIndexBuilder").getOrCreate()
    
    try:
        bucket_name = args.bucket
        client = storage.Client()
        bucket = client.get_bucket(bucket_name)
        blobs = bucket.list_blobs()
        parquet_files = [blob.name for blob in blobs if blob.name.endswith('.parquet') and 'postings_gcp' not in blob.name]
        full_paths = [f"gs://{bucket_name}/{f}" for f in parquet_files]
        df = spark.read.parquet(*full_paths)
        
        docs_rdd = df.rdd.map(PySparkIndexBuilder.parse_wikipedia_document) \
                         .filter(lambda x: x[0] is not None).repartition(args.partitions).cache()

        builder = PySparkIndexBuilder(spark)
        builder.build_index_generic(docs_rdd, 'anchors', args.output.rstrip('/'), args.partitions, args.bucket)
        
        logger.info("✅ Done!")
    finally:
        spark.stop()