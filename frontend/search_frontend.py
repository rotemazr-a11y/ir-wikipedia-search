# search_frontend.py
"""
Flask-based Search Engine Frontend for Wikipedia.

Implements 6 search endpoints:
1. /search - Best combined search (uses TF-IDF + PageRank + PageView)
2. /search_body - TF-IDF cosine similarity on body text
3. /search_title - Binary ranking on title matches
4. /search_anchor - Binary ranking on anchor text matches
5. /get_pagerank - Returns PageRank scores for article IDs
6. /get_pageview - Returns page view counts for article IDs
"""

import os
import sys
import re
import math
import pickle
import logging
from pathlib import Path
from collections import Counter, defaultdict
from typing import List, Tuple, Dict, Any, Optional

from flask import Flask, request, jsonify

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ============================================================================
# CONFIGURATION
# ============================================================================

# GCS bucket name
BUCKET_NAME = os.environ.get('BUCKET_NAME', 'bucket_207916263')

# Index paths (relative to bucket or local)
# The pyspark_index_builder outputs to indices/<index_type>_index/
INDEX_BASE_PATH = os.environ.get('INDEX_PATH', 'indices')

# Paths for indices
BODY_INDEX_PATH = os.path.join(INDEX_BASE_PATH, 'body_index')
TITLE_INDEX_PATH = os.path.join(INDEX_BASE_PATH, 'title_index')
ANCHOR_INDEX_PATH = os.path.join(INDEX_BASE_PATH, 'anchor_index')

# Paths for PageRank and PageView
PAGERANK_PATH = os.environ.get('PAGERANK_PATH', 'pagerank/pagerank.pkl')
PAGEVIEW_PATH = os.environ.get('PAGEVIEW_PATH', 'pageviews/pageviews.pkl')

# Document metadata paths
DOC_TITLES_PATH = os.environ.get('DOC_TITLES_PATH', 'doc_titles.pkl')
DOC_LENGTHS_PATH = os.environ.get('DOC_LENGTHS_PATH', 'doc_lengths.pkl')

# Use GCS or local files
USE_GCS = os.environ.get('USE_GCS', 'true').lower() == 'true'

# ============================================================================
# TOKENIZER (From Assignment 3 - staff-provided)
# ============================================================================

# Stopwords from NLTK + corpus-specific
ENGLISH_STOPWORDS = frozenset([
    'a', 'about', 'above', 'after', 'again', 'against', 'ain', 'all', 'am', 'an', 'and', 
    'any', 'are', 'aren', "aren't", 'as', 'at', 'be', 'because', 'been', 'before', 'being', 
    'below', 'between', 'both', 'but', 'by', 'can', 'couldn', "couldn't", 'd', 'did', 
    'didn', "didn't", 'do', 'does', 'doesn', "doesn't", 'doing', 'don', "don't", 'down', 
    'during', 'each', 'few', 'for', 'from', 'further', 'had', 'hadn', "hadn't", 'has', 
    'hasn', "hasn't", 'have', 'haven', "haven't", 'having', 'he', 'her', 'here', 'hers', 
    'herself', 'him', 'himself', 'his', 'how', 'i', 'if', 'in', 'into', 'is', 'isn', 
    "isn't", 'it', "it's", 'its', 'itself', 'just', 'll', 'm', 'ma', 'me', 'mightn', 
    "mightn't", 'more', 'most', 'mustn', "mustn't", 'my', 'myself', 'needn', "needn't", 
    'no', 'nor', 'not', 'now', 'o', 'of', 'off', 'on', 'once', 'only', 'or', 'other', 
    'our', 'ours', 'ourselves', 'out', 'over', 'own', 're', 's', 'same', 'shan', "shan't", 
    'she', "she's", 'should', "should've", 'shouldn', "shouldn't", 'so', 'some', 'such', 
    't', 'than', 'that', "that'll", 'the', 'their', 'theirs', 'them', 'themselves', 'then', 
    'there', 'these', 'they', 'this', 'those', 'through', 'to', 'too', 'under', 'until', 
    'up', 've', 'very', 'was', 'wasn', "wasn't", 'we', 'were', 'weren', "weren't", 'what', 
    'when', 'where', 'which', 'while', 'who', 'whom', 'why', 'will', 'with', 'won', 
    "won't", 'wouldn', "wouldn't", 'y', 'you', "you'd", "you'll", "you're", "you've", 
    'your', 'yours', 'yourself', 'yourselves'
])

# Corpus-specific stopwords
CORPUS_STOPWORDS = frozenset([
    'category', 'references', 'also', 'links', 'see', 'first', 'one', 'two', 'new',
    'may', 'would', 'could', 'time', 'year', 'years', 'use', 'used', 'using', 'make',
    'like', 'known', 'made', 'many', 'people', 'world', 'including', 'part', 'number',
    'well', 'however', 'since', 'called', 'often', 'became', 'later', 'though', 'although'
])

ALL_STOPWORDS = ENGLISH_STOPWORDS | CORPUS_STOPWORDS

# Regex for tokenization (matches words with optional apostrophes)
RE_WORD = re.compile(r"""[\#\@\w](['\-]?\w){2,24}""", re.UNICODE)


def tokenize_no_stem(text: str) -> List[str]:
    """
    Tokenize text, remove stopwords, lowercase. No stemming.
    Uses the staff-provided tokenizer pattern.
    """
    if not text:
        return []
    
    tokens = [match.group().lower() for match in RE_WORD.finditer(text)]
    tokens = [t for t in tokens if t not in ALL_STOPWORDS and len(t) >= 2]
    return tokens


# ============================================================================
# INVERTED INDEX LOADING
# ============================================================================

# Try multiple import paths for InvertedIndex
InvertedIndex = None
try:
    # First try: same directory (for Colab/GCP deployment)
    from inverted_index_gcp import InvertedIndex
    logger.info("Imported InvertedIndex from current directory")
except ImportError:
    try:
        # Second try: parent indexing directory (for local dev)
        sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'indexing'))
        from inverted_index_gcp import InvertedIndex
        logger.info("Imported InvertedIndex from indexing directory")
    except ImportError:
        logger.warning("Could not import InvertedIndex - index loading will fail")


# ============================================================================
# SEARCH ENGINE CLASS
# ============================================================================

class SearchEngine:
    """
    Singleton class to hold all loaded indices and data.
    """
    
    _instance = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialized = False
        return cls._instance
    
    def __init__(self):
        if self._initialized:
            return
        
        self.body_index: Optional[Any] = None
        self.title_index: Optional[Any] = None
        self.anchor_index: Optional[Any] = None
        
        self.pagerank: Dict[int, float] = {}
        self.pageviews: Dict[int, int] = {}
        self.doc_titles: Dict[int, str] = {}
        self.doc_lengths: Dict[int, int] = {}
        
        # Corpus statistics
        self.avg_doc_length: float = 0.0
        self.total_docs: int = 0
        
        self._initialized = True
    
    def load_all(self):
        """Load all indices and data files."""
        logger.info("Loading search engine data...")
        
        self._load_indices()
        self._load_pagerank()
        self._load_pageviews()
        self._load_doc_metadata()
        self._compute_stats()
        
        logger.info("Search engine loaded successfully!")
    
    def _load_indices(self):
        """Load inverted indices."""
        if InvertedIndex is None:
            logger.error("InvertedIndex class not available")
            return
            
        try:
            if USE_GCS:
                self.body_index = InvertedIndex.read_index(BODY_INDEX_PATH, 'body_index', BUCKET_NAME)
            else:
                self.body_index = InvertedIndex.read_index(BODY_INDEX_PATH, 'body_index')
            logger.info(f"Loaded body index: {len(self.body_index.df)} terms")
        except Exception as e:
            logger.error(f"Failed to load body index: {e}")
        
        try:
            if USE_GCS:
                self.title_index = InvertedIndex.read_index(TITLE_INDEX_PATH, 'title_index', BUCKET_NAME)
            else:
                self.title_index = InvertedIndex.read_index(TITLE_INDEX_PATH, 'title_index')
            logger.info(f"Loaded title index: {len(self.title_index.df)} terms")
        except Exception as e:
            logger.error(f"Failed to load title index: {e}")
        
        try:
            if USE_GCS:
                self.anchor_index = InvertedIndex.read_index(ANCHOR_INDEX_PATH, 'anchor_index', BUCKET_NAME)
            else:
                self.anchor_index = InvertedIndex.read_index(ANCHOR_INDEX_PATH, 'anchor_index')
            logger.info(f"Loaded anchor index: {len(self.anchor_index.df)} terms")
        except Exception as e:
            logger.error(f"Failed to load anchor index: {e}")
    
    def _load_pagerank(self):
        """Load PageRank scores."""
        try:
            if USE_GCS:
                from google.cloud import storage
                client = storage.Client()
                bucket = client.bucket(BUCKET_NAME)
                blob = bucket.blob(PAGERANK_PATH)
                with blob.open('rb') as f:
                    self.pagerank = pickle.load(f)
            else:
                with open(PAGERANK_PATH, 'rb') as f:
                    self.pagerank = pickle.load(f)
            logger.info(f"Loaded PageRank: {len(self.pagerank)} entries")
        except Exception as e:
            logger.warning(f"Failed to load PageRank: {e}")
            self.pagerank = {}
    
    def _load_pageviews(self):
        """Load page view counts."""
        try:
            if USE_GCS:
                from google.cloud import storage
                client = storage.Client()
                bucket = client.bucket(BUCKET_NAME)
                blob = bucket.blob(PAGEVIEW_PATH)
                with blob.open('rb') as f:
                    self.pageviews = pickle.load(f)
            else:
                with open(PAGEVIEW_PATH, 'rb') as f:
                    self.pageviews = pickle.load(f)
            logger.info(f"Loaded PageViews: {len(self.pageviews)} entries")
        except Exception as e:
            logger.warning(f"Failed to load PageViews: {e}")
            self.pageviews = {}
    
    def _load_doc_metadata(self):
        """Load document titles and lengths."""
        try:
            if USE_GCS:
                from google.cloud import storage
                client = storage.Client()
                bucket = client.bucket(BUCKET_NAME)
                
                blob = bucket.blob(DOC_TITLES_PATH)
                with blob.open('rb') as f:
                    self.doc_titles = pickle.load(f)
                
                blob = bucket.blob(DOC_LENGTHS_PATH)
                with blob.open('rb') as f:
                    self.doc_lengths = pickle.load(f)
            else:
                with open(DOC_TITLES_PATH, 'rb') as f:
                    self.doc_titles = pickle.load(f)
                with open(DOC_LENGTHS_PATH, 'rb') as f:
                    self.doc_lengths = pickle.load(f)
            
            logger.info(f"Loaded doc metadata: {len(self.doc_titles)} titles")
        except Exception as e:
            logger.warning(f"Failed to load doc metadata: {e}")
    
    def _compute_stats(self):
        """Compute corpus statistics."""
        if self.doc_lengths:
            self.total_docs = len(self.doc_lengths)
            self.avg_doc_length = sum(self.doc_lengths.values()) / self.total_docs
        elif self.body_index and hasattr(self.body_index, 'df'):
            self.total_docs = max(self.body_index.df.values()) if self.body_index.df else 0
        
        logger.info(f"Corpus stats: {self.total_docs} docs, avg length: {self.avg_doc_length:.2f}")


# Initialize search engine singleton
engine = SearchEngine()


# ============================================================================
# SEARCH ALGORITHMS
# ============================================================================

def get_posting_list(index, term: str, base_path: str) -> List[Tuple[int, int]]:
    """Retrieve posting list for a term from an index."""
    if index is None or term not in index.df:
        return []
    
    try:
        if USE_GCS:
            return index.read_a_posting_list(base_path, term, BUCKET_NAME)
        else:
            return index.read_a_posting_list(base_path, term)
    except Exception as e:
        logger.error(f"Error reading posting list for '{term}': {e}")
        return []


def compute_tfidf_cosine(
    query_tokens: List[str],
    index,
    base_path: str,
    top_k: int = 100
) -> List[Tuple[int, float]]:
    """
    Compute TF-IDF cosine similarity scores.
    """
    if index is None:
        return []
    
    N = engine.total_docs or 1
    query_tf = Counter(query_tokens)
    
    # Compute query vector weights
    query_weights = {}
    for term, tf in query_tf.items():
        if term in index.df:
            df = index.df[term]
            idf = math.log10(N / df) if df > 0 else 0
            query_weights[term] = tf * idf
    
    if not query_weights:
        return []
    
    # Normalize query vector
    query_norm = math.sqrt(sum(w**2 for w in query_weights.values()))
    if query_norm == 0:
        return []
    
    # Accumulate document scores
    doc_scores = defaultdict(float)
    doc_norms = defaultdict(float)
    
    for term, query_weight in query_weights.items():
        postings = get_posting_list(index, term, base_path)
        idf = math.log10(N / index.df[term])
        
        for doc_id, tf in postings:
            # Document TF-IDF weight (log-weighted TF)
            doc_weight = (1 + math.log10(tf)) * idf if tf > 0 else 0
            
            # Accumulate dot product
            doc_scores[doc_id] += (query_weight / query_norm) * doc_weight
            doc_norms[doc_id] += doc_weight ** 2
    
    # Normalize by document length (cosine similarity)
    results = []
    for doc_id, score in doc_scores.items():
        if doc_norms[doc_id] > 0:
            normalized_score = score / math.sqrt(doc_norms[doc_id])
            results.append((doc_id, normalized_score))
    
    # Sort by score descending
    results.sort(key=lambda x: x[1], reverse=True)
    
    return results[:top_k]


def binary_ranking(
    query_tokens: List[str],
    index,
    base_path: str
) -> List[Tuple[int, int]]:
    """
    Binary ranking: count distinct query terms matching each document.
    Returns ALL matching documents.
    """
    if index is None:
        return []
    
    unique_terms = set(query_tokens)
    doc_matches = defaultdict(set)
    
    for term in unique_terms:
        postings = get_posting_list(index, term, base_path)
        for doc_id, tf in postings:
            doc_matches[doc_id].add(term)
    
    # Convert to (doc_id, count) and sort by count descending
    results = [(doc_id, len(terms)) for doc_id, terms in doc_matches.items()]
    results.sort(key=lambda x: x[1], reverse=True)
    
    return results


def combined_search(
    query: str,
    top_k: int = 100,
    body_weight: float = 0.5,
    title_weight: float = 0.25,
    anchor_weight: float = 0.1,
    pagerank_weight: float = 0.1,
    pageview_weight: float = 0.05
) -> List[Tuple[int, str]]:
    """
    Combined search using multiple signals.
    """
    # Tokenize query
    query_tokens = tokenize_no_stem(query)
    if not query_tokens:
        return []
    
    # Get candidates from all sources
    body_results = compute_tfidf_cosine(query_tokens, engine.body_index, BODY_INDEX_PATH, top_k * 2)
    title_results = binary_ranking(query_tokens, engine.title_index, TITLE_INDEX_PATH)
    anchor_results = binary_ranking(query_tokens, engine.anchor_index, ANCHOR_INDEX_PATH)
    
    # Normalize scores to 0-1 range
    def normalize_scores(results: List[Tuple[int, float]]) -> Dict[int, float]:
        if not results:
            return {}
        max_score = max(s for _, s in results) or 1
        return {doc_id: score / max_score for doc_id, score in results}
    
    body_scores = normalize_scores(body_results)
    title_scores = normalize_scores(title_results)
    anchor_scores = normalize_scores(anchor_results)
    
    # Normalize PageViews
    max_pageview = max(engine.pageviews.values()) if engine.pageviews else 1
    
    # Collect all candidate documents
    all_docs = set(body_scores.keys()) | set(title_scores.keys()) | set(anchor_scores.keys())
    
    # Compute combined scores
    final_scores = []
    for doc_id in all_docs:
        score = 0.0
        score += body_weight * body_scores.get(doc_id, 0)
        score += title_weight * title_scores.get(doc_id, 0)
        score += anchor_weight * anchor_scores.get(doc_id, 0)
        score += pagerank_weight * engine.pagerank.get(doc_id, 0)
        score += pageview_weight * (engine.pageviews.get(doc_id, 0) / max_pageview)
        
        final_scores.append((doc_id, score))
    
    # Sort and get top-k
    final_scores.sort(key=lambda x: x[1], reverse=True)
    top_results = final_scores[:top_k]
    
    # Add titles
    results_with_titles = [
        (doc_id, engine.doc_titles.get(doc_id, f"Document {doc_id}"))
        for doc_id, score in top_results
    ]
    
    return results_with_titles


# ============================================================================
# FLASK APPLICATION
# ============================================================================

class MyFlaskApp(Flask):
    def run(self, host=None, port=None, debug=None, **options):
        # Load engine data before starting
        engine.load_all()
        super(MyFlaskApp, self).run(host=host, port=port, debug=debug, **options)


app = MyFlaskApp(__name__)
app.config['JSONIFY_PRETTYPRINT_REGULAR'] = False


@app.route("/search")
def search():
    ''' Returns up to a 100 of your best search results for the query.
        Uses combined TF-IDF + PageRank + PageView ranking.
    '''
    res = []
    query = request.args.get('query', '')
    if len(query) == 0:
        return jsonify(res)
    
    try:
        res = combined_search(query, top_k=100)
    except Exception as e:
        logger.error(f"Search error: {e}")
    
    return jsonify(res)


@app.route("/search_body")
def search_body():
    ''' Returns up to a 100 search results for the query using TFIDF AND COSINE
        SIMILARITY OF THE BODY OF ARTICLES ONLY. No stemming.
    '''
    res = []
    query = request.args.get('query', '')
    if len(query) == 0:
        return jsonify(res)
    
    try:
        query_tokens = tokenize_no_stem(query)
        results = compute_tfidf_cosine(query_tokens, engine.body_index, BODY_INDEX_PATH, 100)
        res = [
            (doc_id, engine.doc_titles.get(doc_id, f"Document {doc_id}"))
            for doc_id, score in results
        ]
    except Exception as e:
        logger.error(f"Search body error: {e}")
    
    return jsonify(res)


@app.route("/search_title")
def search_title():
    ''' Returns ALL search results that contain A QUERY WORD IN THE TITLE,
        ordered by NUMBER OF DISTINCT QUERY WORDS in title. No stemming.
    '''
    res = []
    query = request.args.get('query', '')
    if len(query) == 0:
        return jsonify(res)
    
    try:
        query_tokens = tokenize_no_stem(query)
        results = binary_ranking(query_tokens, engine.title_index, TITLE_INDEX_PATH)
        res = [
            (doc_id, engine.doc_titles.get(doc_id, f"Document {doc_id}"))
            for doc_id, count in results
        ]
    except Exception as e:
        logger.error(f"Search title error: {e}")
    
    return jsonify(res)


@app.route("/search_anchor")
def search_anchor():
    ''' Returns ALL search results that contain A QUERY WORD IN THE ANCHOR TEXT,
        ordered by NUMBER OF QUERY WORDS in anchor text. No stemming.
    '''
    res = []
    query = request.args.get('query', '')
    if len(query) == 0:
        return jsonify(res)
    
    try:
        query_tokens = tokenize_no_stem(query)
        results = binary_ranking(query_tokens, engine.anchor_index, ANCHOR_INDEX_PATH)
        res = [
            (doc_id, engine.doc_titles.get(doc_id, f"Document {doc_id}"))
            for doc_id, count in results
        ]
    except Exception as e:
        logger.error(f"Search anchor error: {e}")
    
    return jsonify(res)


@app.route("/get_pagerank", methods=['POST'])
def get_pagerank():
    ''' Returns PageRank values for a list of provided wiki article IDs. '''
    res = []
    wiki_ids = request.get_json()
    if not wiki_ids or len(wiki_ids) == 0:
        return jsonify(res)
    
    try:
        res = [engine.pagerank.get(int(wiki_id), 0.0) for wiki_id in wiki_ids]
    except Exception as e:
        logger.error(f"Get pagerank error: {e}")
    
    return jsonify(res)


@app.route("/get_pageview", methods=['POST'])
def get_pageview():
    ''' Returns page view counts for a list of provided wiki article IDs. '''
    res = []
    wiki_ids = request.get_json()
    if not wiki_ids or len(wiki_ids) == 0:
        return jsonify(res)
    
    try:
        res = [engine.pageviews.get(int(wiki_id), 0) for wiki_id in wiki_ids]
    except Exception as e:
        logger.error(f"Get pageview error: {e}")
    
    return jsonify(res)


# Health check endpoint
@app.route("/health")
def health():
    """Health check endpoint."""
    return jsonify({
        "status": "healthy",
        "body_index": engine.body_index is not None,
        "title_index": engine.title_index is not None,
        "anchor_index": engine.anchor_index is not None,
        "pagerank_loaded": len(engine.pagerank) > 0,
        "pageview_loaded": len(engine.pageviews) > 0,
        "total_docs": engine.total_docs
    })


def run(**options):
    app.run(**options)


if __name__ == '__main__':
    # Run the Flask RESTful API
    app.run(host='0.0.0.0', port=8080, debug=True)
