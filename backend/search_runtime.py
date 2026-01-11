from contextlib import closing
import os
from pathlib import Path
import pickle
import math
import logging
import re
import struct
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from google.cloud import storage

# External imports assumed to be in your environment
from inverted_index_gcp import MultiFileReader
from pre_processing import tokenize_and_process

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ============================================================================
# INVERTED INDEX CLASS
# ============================================================================
class InvertedIndex:
    def __init__(self):
        self.df = Counter()
        self.term_total = Counter()
        self.posting_locs = defaultdict(list)

# ============================================================================
# CONFIGURATION
# ============================================================================
BUCKET_NAME = '206969750_bucket'

# ============================================================================
# SEARCH ENGINE CORE
# ============================================================================
class SearchEngine:
    def __init__(self):
        self.client = storage.Client()
        self.bucket = self.client.bucket(BUCKET_NAME)
        self._load_indices()
        self._load_metadata()

    def _load_indices(self):
        """Load pickled index objects and set binary directories from GCS."""
        logger.info("Loading indices from GCS...")
        
        self.title_idx = self._load_single_index('postings_gcp/title/pkl/index.pkl')
        self.anchor_idx = self._load_single_index('postings_gcp/anchor_manual/pkl/index.pkl')
        self.body_idx = self._load_single_index('postings_gcp/body/pkl/index.pkl')  

        # Set binary storage directories
        self.body_idx.base_dir = 'postings_gcp/body/bin_new' 
        self.title_idx.base_dir = 'postings_gcp/title/bin'
        self.anchor_idx.base_dir = 'postings_gcp/anchor_manual/bin'
        
        # Mapping for fast access
        self.body_posting_locs = self.body_idx.posting_locs
        self.body_df = self.body_idx.df
        self.title_posting_locs = self.title_idx.posting_locs
        self.title_df = self.title_idx.df
        self.anchor_posting_locs = self.anchor_idx.posting_locs
        self.anchor_df = self.anchor_idx.df
        
        logger.info("Indices loaded successfully.")
        
    def _load_single_index(self, gcs_path):
        """Helper to load a single index pickle from GCS."""
        try:
            blob = self.bucket.blob(gcs_path)
            data = blob.download_as_bytes()
            index_data = pickle.loads(data)
            
            if isinstance(index_data, dict):
                index_obj = InvertedIndex()
                for key, value in index_data.items():
                    setattr(index_obj, key, value)
                return index_obj
            return index_data
        except Exception as e:
            logger.error(f"Failed to load index from {gcs_path}: {e}")
            raise

    def _load_metadata(self):
        """Load document metadata and popularity signals (PageRank and PageViews)."""
        logger.info("Loading metadata and importance signals...")
        try:
            # Load titles and document lengths
            blob = self.bucket.blob('postings_gcp/metadata.pkl')   
            metadata = pickle.loads(blob.download_as_bytes())
            self.doc_titles = metadata.get('doc_titles') or {}
            self.doc_lengths = metadata.get('doc_lengths') or {}
            
            # Load PageRank
            pr_blob = self.bucket.blob('pagerank_dict.pkl')
            self.pagerank = pickle.loads(pr_blob.download_as_bytes()) if pr_blob.exists() else {}
            
            # Load PageViews 
            pv_blob = self.bucket.blob('pageviews_dict.pkl')
            self.pageviews = pickle.loads(pv_blob.download_as_bytes()) if pv_blob.exists() else {}
            
            # Set corpus statistics
            self.num_docs = len(self.doc_titles) if self.doc_titles else 6831681
            self.avg_doc_length = sum(self.doc_lengths.values()) / len(self.doc_lengths) if self.doc_lengths else 450.0
            logger.info("Metadata loaded successfully.")
        except Exception as e:
            logger.error(f"Failed to load metadata: {e}")
            self.doc_titles, self.doc_lengths, self.pagerank, self.pageviews = {}, {}, {}, {}

    def read_posting_list(self, term, posting_locs, df, base_dir, tuple_size=6):
        """Fetch binary posting list from GCS with a limit on candidates for performance."""
        MAX_CANDIDATES = 50000 
        actual_df = min(df, MAX_CANDIDATES)
        posting_list = []
        
        if not posting_locs or actual_df == 0:
            return posting_list
            
        fmt = ">II" if tuple_size == 8 else ">IH"
        clean_base_dir = str(base_dir).strip('/')
        # Calculate size to read based on document frequency
        read_size = df * tuple_size
        
        for file_name, offset in posting_locs:
            actual_filename = os.path.basename(str(file_name))
            blob_path = f"{clean_base_dir}/{actual_filename}"
            
            try:
                blob = self.bucket.blob(blob_path)
                content = blob.download_as_bytes(start=offset, end=offset + read_size - 1)
                
                if not content:
                    continue

                for i in range(0, len(content), tuple_size):
                    chunk = content[i : i + tuple_size]
                    if len(chunk) == tuple_size:
                        d_id, tf_val = struct.unpack(fmt, chunk)
                        posting_list.append((d_id, tf_val))
            except Exception as e:
                logger.error(f"Error parsing term {term}: {e}")
        return posting_list

    def search_body_bm25(self, query_tokens):
        """Score documents using BM25 ranking on the body index."""
        if not query_tokens: return {}
        scores = defaultdict(float)
        k1, b = 1.2, 0.75 # Standard BM25 parameters
        
        for term in set(query_tokens):
            if term not in self.body_df: continue
            df = self.body_df[term]
            
            # Skip extremely common terms (stop-word like behavior)
            if df > (self.num_docs * 0.9): continue 
            
            plist = self.read_posting_list(term, self.body_posting_locs.get(term, []), df, self.body_idx.base_dir, 8)
            idf = math.log((self.num_docs - df + 0.5) / (df + 0.5) + 1.0)
            
            for doc_id, tf in plist:
                doc_len = self.doc_lengths.get(doc_id, self.avg_doc_length)
                # BM25 Formula
                scores[doc_id] += idf * (tf * (k1 + 1) / (tf + k1 * (1 - b + b * doc_len / self.avg_doc_length)))
        
        # Return top 1000 candidates for further re-ranking
        return dict(sorted(scores.items(), key=lambda x: x[1], reverse=True)[:1000])
    
    def search_title(self, query_tokens):
        """Score documents based on binary Title match (Precision focus)."""
        if not query_tokens: return {}
        scores = defaultdict(float)
        for term in set(query_tokens):
            if term not in self.title_df: continue
            df = self.title_df[term]
            
            posting_list = self.read_posting_list(
                term, self.title_posting_locs.get(term, []), 
                df, self.title_idx.base_dir, tuple_size=6
            )
            for doc_id, tf in posting_list:
                scores[doc_id] += 1.0
        return dict(scores)

    def search_anchor(self, query_tokens):
        """Score documents based on Anchor text frequency."""
        if not query_tokens: return {}
        scores = defaultdict(float)
        for term in set(query_tokens):
            if term not in self.anchor_df: continue
            df = self.anchor_df[term]
            
            posting_list = self.read_posting_list(
                term, self.anchor_posting_locs.get(term, []), 
                df, self.anchor_idx.base_dir, tuple_size=6
            )
            for doc_id, tf in posting_list:
                scores[doc_id] += tf
        return dict(scores)

    def search(self, query, top_n=100):
        """Main search method combining Body, Title, Anchor, PageRank, and PageViews."""
        from pre_processing import tokenize_and_process
        tokens = tokenize_and_process(query, remove_stops=True, stem=False)
        if not tokens: return []

        # 1. Retrieve raw scores from different indices
        res_body = self.search_body_bm25(tokens)
        res_title = self.search_title(tokens)
        res_anchor = self.search_anchor(tokens)
        
        # Union of all candidate documents
        candidates = set(res_body.keys()) | set(res_title.keys()) | set(res_anchor.keys())
        
        # Max scores for normalization
        max_b = max(res_body.values()) if res_body else 1.0
        max_t = max(res_title.values()) if res_title else 1.0
        max_a = max(res_anchor.values()) if res_anchor else 1.0
        
        final_scores = {}
        for d in candidates:
            # Textual Relevance Scoring (Weighted Linear Combination)
            norm_b = res_body.get(d, 0.0) / max_b
            norm_t = res_title.get(d, 0.0) / max_t
            norm_a = res_anchor.get(d, 0.0) / max_a
            
            # Weighting: Title (70%), Body (20%), Anchor (10%)
            text_score = (norm_t * 0.7) + (norm_a * 0.1) + (norm_b * 0.2)
            
            # Precision Bonus for exact Title match
            if res_title.get(d, 0.0) > 0:
                text_score += 0.5
            
            # 2. PageRank Component (Logarithmic Normalization)
            pr = self.pagerank.get(d, 0)
            pr_boost = math.log10(pr * 1_000_000 + 1) if pr > 0 else 0
            
            # 3. PageViews Component (Logarithmic Normalization)
            pv = self.pageviews.get(d, 0)
            pv_boost = math.log10(pv + 1) if pv > 0 else 0
            
            # 4. Final Score Fusion
            # Combine textual score with importance signals
            final_boost = 1 + (0.1 * pr_boost) + (0.05 * pv_boost)
            final_scores[d] = text_score * final_boost
            
        # Sort and return top results
        results = sorted(final_scores.items(), key=lambda x: x[1], reverse=True)[:top_n]
        return [(str(doc_id), self.doc_titles.get(int(doc_id), f"Doc {doc_id}")) for doc_id, score in results]

# ============================================================================
# SINGLETON ENGINE MANAGEMENT
# ============================================================================
_ENGINE = None

def initialize_engine():
    global _ENGINE
    if _ENGINE is None:
        _ENGINE = SearchEngine()

def get_engine():
    if _ENGINE is None:
        raise RuntimeError("Search Engine not initialized.")
    return _ENGINE