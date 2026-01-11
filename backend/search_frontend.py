"""
Wikipedia Search Engine - Flask Frontend
Production version with comprehensive error handling and logging.
"""
from flask import Flask, request, jsonify
import logging
import sys
from pathlib import Path

# Configure system logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Ensure the backend directory is in the Python path for imports
sys.path.insert(0, str(Path(__file__).parent))

BUCKET_NAME = '206969750_bucket'

# Import search logic modules from the backend
try:
    from search_runtime import initialize_engine, get_engine
    from pre_processing import tokenize_and_process
    SEARCH_AVAILABLE = True
    logger.info("✓ Search modules imported successfully")
except ImportError as e:
    logger.error(f"✗ Failed to import search modules: {e}")
    SEARCH_AVAILABLE = False

# Initialize Flask application
app = Flask(__name__)
app.config['JSONIFY_PRETTYPRINT_REGULAR'] = False


@app.route("/")
def home():
    """Health check endpoint to verify server status and component availability."""
    return jsonify({
        "status": "running",
        "search_available": SEARCH_AVAILABLE,
        "endpoints": [
            "/search",
            "/search_body",
            "/search_title",
            "/search_anchor",
            "/get_pagerank",
            "/get_pageview"
        ]
    })

@app.route("/search")
def search():
    """
    Main search endpoint combining Title, Body, and Anchor indices.
    Weights are applied within the engine.search() method.
    """
    try:
        raw_query = request.args.get('query', '').strip()
        if not raw_query: 
            return jsonify([])
        
        engine = get_engine()
        results = engine.search(raw_query, top_n=100)
        
        # Logging search performance for debugging
        print(f"\n🔍 DEBUG: Query '{raw_query}' returned {len(results)} results.")
        if results:
            print(f"🔝 Top 3 IDs: {[str(r[0]) for r in results[:3]]}")

        # Format output as a list of (doc_id, title) tuples
        response = [(str(doc_id), title) for doc_id, title, score in results]
        return jsonify(response)
    except Exception as e:
        logger.error(f"❌ ERROR in search: {e}")
        return jsonify([])

@app.route("/search_body")
def search_body():
    """Returns BM25 results from Body index only (Stemmed)."""
    try:
        raw_query = request.args.get('query', '').strip()
        if not raw_query: 
            return jsonify([])
        
        engine = get_engine()
        # Body index requires stemmed tokens
        query_tokens = tokenize_and_process(raw_query, remove_stops=True, stem=True)
        
        # Retrieve BM25 scores from backend
        scores = engine.search_body_bm25(query_tokens)
        sorted_docs = sorted(scores.items(), key=lambda x: x[1], reverse=True)[:100]

        response = [
            (str(doc_id), engine.doc_titles.get(int(doc_id), f"Document {doc_id}"))
            for doc_id, score in sorted_docs
        ]
        return jsonify(response)
    except Exception as e:
        logger.error(f"[BODY] Error: {e}")
        return jsonify([])


@app.route("/search_title")
def search_title():
    """Search using Title index only (No stemming)."""
    try:
        raw_query = request.args.get('query', '').strip()

        if not raw_query or not SEARCH_AVAILABLE:
            return jsonify([])

        engine = get_engine()

        # Title index utilizes exact matching (No stemming)
        query_tokens = tokenize_and_process(
            raw_query.lower(),
            remove_stops=True,
            stem=False
        )

        if not query_tokens:
            return jsonify([])

        scores = engine.search_title(query_tokens)
        sorted_docs = sorted(scores.items(), key=lambda x: x[1], reverse=True)[:100]

        response = [
            (str(doc_id), engine.doc_titles.get(int(doc_id), f"Document {doc_id}"))
            for doc_id, score in sorted_docs
        ]

        return jsonify(response)

    except Exception as e:
        logger.error(f"[TITLE] Error: {e}", exc_info=True)
        return jsonify([])


@app.route("/search_anchor")
def search_anchor():
    """Search using Anchor text index only (No stemming)."""
    try:
        raw_query = request.args.get('query', '').strip()

        if not raw_query or not SEARCH_AVAILABLE:
            return jsonify([])

        engine = get_engine()

        # Anchor index utilizes exact matching (No stemming)
        query_tokens = tokenize_and_process(
            raw_query.lower(),
            remove_stops=True,
            stem=False
        )

        if not query_tokens:
            return jsonify([])

        scores = engine.search_anchor(query_tokens)
        sorted_docs = sorted(scores.items(), key=lambda x: x[1], reverse=True)[:100]

        response = [
            (str(doc_id), engine.doc_titles.get(int(doc_id), f"Document {doc_id}"))
            for doc_id, score in sorted_docs
        ]

        return jsonify(response)

    except Exception as e:
        logger.error(f"[ANCHOR] Error: {e}", exc_info=True)
        return jsonify([])


@app.route("/get_pagerank", methods=['POST'])
def get_pagerank():
    """Retrieve PageRank values for a given list of document IDs."""
    try:
        data = request.get_json()
        if not data or not isinstance(data, list):
            return jsonify([])

        if not SEARCH_AVAILABLE:
            return jsonify([0.0] * len(data))

        engine = get_engine()
        # Convert incoming IDs to integers for dictionary lookup
        doc_ids_int = []
        for d in data:
            try: 
                doc_ids_int.append(int(d))
            except (ValueError, TypeError): 
                doc_ids_int.append(0)

        scores = engine.get_pagerank(doc_ids_int)
        return jsonify(scores)
    except Exception as e:
        logger.error(f"[PAGERANK] Error: {e}")
        return jsonify([])


@app.route("/get_pageview", methods=['POST'])
def get_pageview():
    """Retrieve PageView counts for a given list of document IDs."""
    try:
        doc_ids = request.get_json() or []
        if not doc_ids:
            return jsonify([])

        if not SEARCH_AVAILABLE:
            return jsonify([0] * len(doc_ids))

        engine = get_engine()
        # Standardize doc IDs to integers
        doc_ids_int = []
        for doc_id in doc_ids:
            try:
                doc_ids_int.append(int(doc_id))
            except (ValueError, TypeError):
                doc_ids_int.append(0)

        counts = engine.get_pageviews(doc_ids_int)
        return jsonify(counts)

    except Exception as e:
        logger.error(f"[PAGEVIEW] Error: {e}", exc_info=True)
        return jsonify([])


if __name__ == '__main__':
    # Configuration for the Flask server
    HOST = '0.0.0.0'
    PORT = 8080

    # Initialize the search engine before starting the server
    if SEARCH_AVAILABLE:
        try:
            logger.info("Initializing search engine...")
            initialize_engine()
            logger.info("✓ Search Engine Initialized Successfully")
        except Exception as e:
            logger.error(f"✗ Failed search engine initialization: {e}")
            SEARCH_AVAILABLE = False

    # Start the production server
    app.run(host=HOST, port=PORT, debug=False, threaded=True)