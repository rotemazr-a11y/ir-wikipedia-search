"""
Search frontend adapted for GCS-based posting files.
This version reads from gs://BUCKET/postings_gcp/ with .bin posting files.
"""

from flask import Flask, request, jsonify
import math
import pickle
from collections import Counter
from pathlib import Path
import sys

# Add backend to path for imports
sys.path.insert(0, str(Path(__file__).parent))

from inverted_index_gcp import InvertedIndex
from pre_processing import tokenize_and_process

class MyFlaskApp(Flask):
    def run(self, host=None, port=None, debug=None, **options):
        super(MyFlaskApp, self).run(host=host, port=port, debug=debug, **options)

app = MyFlaskApp(__name__)
app.config['JSONIFY_PRETTYPRINT_REGULAR'] = False

# Global variables for indices and metadata (loaded once at startup)
BODY_INDEX = None
METADATA = None
BUCKET_NAME = None
INDEX_BASE_DIR = None

def load_indices_gcs(bucket_name="206969750_bucket", index_dir="postings_gcp"):
    """Load indices from GCS bucket with posting files."""
    global BODY_INDEX, METADATA, BUCKET_NAME, INDEX_BASE_DIR

    BUCKET_NAME = 'ir_project_206969750'
    INDEX_PATH = 'postings_gcp/body_index.pkl'

    print(f"Loading indices from gs://{bucket_name}/{index_dir}...")

    try:
        # Load body index (contains df, term_total, posting_locs)
        BODY_INDEX = InvertedIndex.read_index(index_dir, 'index', bucket_name)
        print(f"✓ Body index loaded: {len(BODY_INDEX.df):,} terms")

        # Load metadata (if available)
        try:
            from google.cloud import storage
            client = storage.Client()
            bucket = client.bucket(bucket_name)

            # Try to load metadata.pkl from indices/ directory
            blob = bucket.blob('indices/metadata.pkl')
            metadata_bytes = blob.download_as_bytes()
            METADATA = pickle.loads(metadata_bytes)
            print(f"✓ Metadata loaded: {METADATA['num_docs']:,} documents")
        except Exception as e:
            print(f"⚠ Could not load metadata: {e}")
            # Create minimal metadata
            METADATA = {
                'num_docs': 6000000,  # Approximate Wikipedia size
                'doc_titles': {},
                'doc_norms': {}
            }
            print("  Using default metadata (N=6M docs)")

        print("✓ All indices loaded successfully from GCS!\n")
    except Exception as e:
        print(f"✗ Error loading indices: {e}")
        raise

def compute_tfidf_cosine_gcs(query_tokens, index, N):
    """
    Compute TF-IDF cosine similarity scores using GCS posting files.

    Parameters:
    -----------
    query_tokens : list of str
        Preprocessed query tokens
    index : InvertedIndex
        The inverted index with posting_locs
    N : int
        Total number of documents

    Returns:
    --------
    dict : doc_id -> cosine similarity score
    """
    if not query_tokens:
        return {}

    # 1. Compute query TF-IDF vector and norm
    query_counts = Counter(query_tokens)
    query_tfidf = {}

    for term, count in query_counts.items():
        if term in index.df:
            df = index.df[term]
            idf = math.log(N / df)  # Natural log
            query_tfidf[term] = count * idf

    if not query_tfidf:
        return {}

    # Query L2 norm
    query_norm = math.sqrt(sum(v**2 for v in query_tfidf.values()))

    if query_norm == 0:
        return {}

    # 2. Get candidate documents using read_a_posting_list (reads from GCS)
    candidates = {}

    for term in query_tfidf.keys():
        try:
            # This reads from GCS .bin files
            posting_list = index.read_a_posting_list(INDEX_BASE_DIR, term, BUCKET_NAME)
            for doc_id, tf in posting_list:
                if doc_id not in candidates:
                    candidates[doc_id] = {}
                candidates[doc_id][term] = tf
        except Exception as e:
            # Term might not have postings
            continue

    # 3. Compute cosine similarity for each candidate
    scores = {}

    for doc_id, doc_terms in candidates.items():
        # Dot product
        dot_product = 0.0
        doc_tfidf_squared = 0.0

        for term, tf in doc_terms.items():
            if term in query_tfidf:
                df = index.df[term]
                idf = math.log(N / df)
                doc_tfidf = tf * idf
                dot_product += query_tfidf[term] * doc_tfidf
                doc_tfidf_squared += doc_tfidf ** 2

        # Document norm (use precomputed if available)
        if METADATA and 'doc_norms' in METADATA and doc_id in METADATA['doc_norms']:
            doc_norm = METADATA['doc_norms'][doc_id]
        else:
            # Compute on-the-fly
            doc_norm = math.sqrt(doc_tfidf_squared)

        # Cosine similarity
        if doc_norm > 0:
            scores[doc_id] = dot_product / (query_norm * doc_norm)

    return scores

@app.route("/search")
def search():
    '''
    Best search using full Wikipedia corpus from GCS.
    Returns up to 100 search results for the query.
    '''
    res = []
    query = request.args.get('query', '')
    if len(query) == 0:
        return jsonify(res)

    # Tokenize query
    query_tokens = tokenize_and_process(query, remove_stops=True, stem=True)

    if not query_tokens:
        return jsonify(res)

    # Use TF-IDF on body index
    N = METADATA['num_docs']
    scores = compute_tfidf_cosine_gcs(query_tokens, BODY_INDEX, N)

    # Sort by score (descending)
    sorted_docs = sorted(scores.items(), key=lambda x: x[1], reverse=True)

    # Return top 100 as (doc_id, title) tuples
    for doc_id, score in sorted_docs[:100]:
        title = METADATA['doc_titles'].get(doc_id, "")
        res.append((str(doc_id), title))

    return jsonify(res)

@app.route("/search_body")
def search_body():
    '''Search using body index only (same as /search for GCS version).'''
    return search()

def run(**options):
    app.run(**options)

if __name__ == '__main__':
    # Load indices before starting server
    import os

    bucket_name = os.environ.get('GCS_BUCKET', '206969750_bucket')
    index_dir = os.environ.get('INDEX_DIR', 'postings_gcp')

    load_indices_gcs(bucket_name, index_dir)

    # Run Flask server
    print(f"\nStarting Flask server on http://0.0.0.0:8080")
    print(f"Endpoints available:")
    print(f"  GET  /search?query=...")
    print(f"  GET  /search_body?query=...")
    print()
    app.run(host='0.0.0.0', port=8080, debug=False)
