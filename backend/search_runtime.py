from contextlib import closing
import os
from pathlib import Path
import pickle
import math
import logging
import re
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from google.cloud import storage
from inverted_index_gcp import MultiFileReader
from pre_processing import tokenize_and_process

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ============================================================================
# INVERTED INDEX CLASSES
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
TUPLE_SIZE = 6
BLOCK_SIZE = 1999998 

# ============================================================================
# SEARCH ENGINE
# ============================================================================
class SearchEngine:
    def __init__(self):
        self.client = storage.Client()
        self.bucket = self.client.bucket(BUCKET_NAME)
        self._load_indices()
        self._load_metadata()

    def _load_indices(self):
        """Load indices from GCS."""
        logger.info("Loading indices from GCS...")
        
        # Load all three indices

        self.title_idx = self._load_single_index('postings_gcp/title/pkl/index.pkl')
        self.anchor_idx = self._load_single_index('postings_gcp/anchor_manual/pkl/index.pkl')
        self.body_idx = self._load_single_index('postings_gcp/body/pkl/index.pkl')  

        # Set base directories

        self.body_idx.base_dir = 'postings_gcp/body/bin_new' 
        self.body_idx.bucket_name = BUCKET_NAME
        self.title_idx.base_dir = 'postings_gcp/title/bin'
        self.title_idx.bucket_name = BUCKET_NAME
        self.anchor_idx.base_dir = 'postings_gcp/anchor_manual/bin'
        self.anchor_idx.bucket_name = BUCKET_NAME
        # Quick access references
        self.body_posting_locs = self.body_idx.posting_locs
        self.body_df = self.body_idx.df
        self.title_posting_locs = self.title_idx.posting_locs
        self.title_df = self.title_idx.df
        self.anchor_posting_locs = self.anchor_idx.posting_locs
        self.anchor_df = self.anchor_idx.df
        
        logger.info(f"✓ Indices loaded: Body({len(self.body_df)}), Title({len(self.title_df)}), Anchor({len(self.anchor_df)})")
        
    def _load_single_index(self, gcs_path):
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
            logger.error(f"✗ Failed to load index from {gcs_path}: {e}")
            raise

    def _load_metadata(self):
        logger.info("Loading metadata...")
        try:
            blob = self.bucket.blob('postings_gcp/metadata.pkl')   
            metadata = pickle.loads(blob.download_as_bytes())
            self.doc_titles = metadata.get('doc_titles') or metadata.get('titles') or {}
            self.doc_lengths = metadata.get('doc_lengths') or metadata.get('lengths') or {}
            self.num_docs = metadata.get('num_docs') or len(self.doc_titles) or 6831681
            self.avg_doc_length = sum(self.doc_lengths.values()) / len(self.doc_lengths) if self.doc_lengths else 450.0
        except Exception as e:
            logger.error(f"✗ Failed to load metadata: {e}")
            self.doc_titles, self.doc_lengths = {}, {}
            self.num_docs, self.avg_doc_length = 6831681, 450.0
    def read_posting_list(self, term, posting_locs, df, base_dir, tuple_size=6):
        """Read posting list with dynamic tuple size support."""
        posting_list = []
        if not posting_locs or df == 0:
            return posting_list

        clean_base_dir = str(base_dir).strip('/')
        # Calculate size based on the specific index requirements
        read_size = df * tuple_size
        
        for file_name, offset in posting_locs:
            fname_str = str(file_name)
            actual_filename = os.path.basename(fname_str)
            
            # Robust path patterns
            paths_to_try = [f"{clean_base_dir}/{actual_filename}"]
            parts = actual_filename.replace('.bin', '').split('_')
            if len(parts) >= 3:
                paths_to_try.append(f"{clean_base_dir}/{parts[0]}_{parts[-1]}.bin")
                paths_to_try.append(f"{clean_base_dir}/{parts[0]}_{parts[1]}_{parts[-1]}.bin")
            
            found = False
            for blob_path in set(paths_to_try):
                try:
                    blob = self.bucket.blob(blob_path)
                    content = blob.download_as_bytes(start=offset, end=offset + read_size - 1)
                    if not content: continue
                    
                    for i in range(0, len(content), tuple_size):
                        chunk = content[i : i + tuple_size]
                        if len(chunk) == tuple_size:
                            doc_id = int.from_bytes(chunk[:4], 'big')
                            tf = int.from_bytes(chunk[4:], 'big')
                            if doc_id > 0:
                                posting_list.append((doc_id, tf))
                    if posting_list:
                        found = True
                        break
                except: continue
            if found: break
        return posting_list

    def search_body_bm25(self, query_tokens):
        if not query_tokens: return {}
        scores = defaultdict(float)
        k1, b = 1.2, 0.75 # Reverted to standard BM25 params for stability
        avg_len = self.avg_doc_length
        
        for term in set(query_tokens):
            df = self.body_df.get(term, 0)
            if df == 0 or df > (self.num_docs * 0.1): continue
            
            # CRITICAL: Body index uses 8 bytes (4 ID + 4 TF)
            posting_list = self.read_posting_list(term, self.body_posting_locs[term], df, self.body_idx.base_dir, tuple_size=8)
            if not posting_list: continue
            
            idf = math.log((self.num_docs - df + 0.5) / (df + 0.5) + 1.0)
            for doc_id, tf in posting_list:
                doc_len = self.doc_lengths.get(doc_id, avg_len)
                denominator = tf + k1 * (1 - b + b * doc_len / avg_len)
                scores[doc_id] += idf * (tf * (k1 + 1) / denominator)
        return dict(scores)

    def search_title(self, query_tokens):
        """Enhanced Title search with your multi-term matching bonuses."""
        if not query_tokens: return {}
        scores = defaultdict(float)
        term_hits = defaultdict(set)
        
        for term in set(query_tokens):
            df = self.title_df.get(term, 0)
            if df == 0: continue
            
            # Title uses 6 bytes (4 ID + 2 TF)
            idf = math.log(self.num_docs / df)
            posting_list = self.read_posting_list(term, self.title_posting_locs[term], df, self.title_idx.base_dir, tuple_size=6)
            
            for doc_id, tf in posting_list:
                term_hits[doc_id].add(term)
                scores[doc_id] += idf * math.log(1 + tf)
        
        # Apply your coverage bonuses (kept from your code)
        num_query_terms = len(set(query_tokens))
        for doc_id, matched_terms in term_hits.items():
            coverage = len(matched_terms) / num_query_terms
            if coverage == 1.0: scores[doc_id] *= 15.0
            elif coverage >= 0.5: scores[doc_id] *= 5.0
            
            if len(matched_terms) > 1:
                scores[doc_id] += len(matched_terms) * 2.0
                
        return dict(scores)

    def search_anchor(self, query_tokens):
        if not query_tokens: return {}
        scores = defaultdict(float)
        for term in set(query_tokens):
            df = self.anchor_df.get(term, 0)
            if df == 0: continue
            
            # Anchor uses 6 bytes
            posting_list = self.read_posting_list(term, self.anchor_posting_locs[term], df, self.anchor_idx.base_dir, tuple_size=6)
            for doc_id, tf in posting_list:
                scores[doc_id] += tf
        return dict(scores)

    def search(self, query, top_n=100):
        if not query: return []
        
        tokens_body = tokenize_and_process(query, remove_stops=True, stem=True)
        tokens_title = tokenize_and_process(query, remove_stops=True, stem=False)
        
        with ThreadPoolExecutor(max_workers=3) as executor:
            f_body = executor.submit(self.search_body_bm25, tokens_body)
            f_title = executor.submit(self.search_title, tokens_title)
            f_anchor = executor.submit(self.search_anchor, tokens_title)
            
            body_scores = f_body.result()
            title_scores = f_title.result()
            anchor_scores = f_anchor.result()
        
        all_docs = set(body_scores.keys()) | set(title_scores.keys()) | set(anchor_scores.keys())
        if not all_docs: return []
        
        max_b = max(body_scores.values()) if body_scores else 1.0
        max_t = max(title_scores.values()) if title_scores else 1.0
        max_a = max(anchor_scores.values()) if anchor_scores else 1.0

        final_scores = {}
        for doc_id in all_docs:
            norm_b = body_scores.get(doc_id, 0.0) / max_b
            norm_t = title_scores.get(doc_id, 0.0) / max_t
            norm_a = anchor_scores.get(doc_id, 0.0) / max_a
            
            # Balanced fusion (Title is still king, but Body counts)
            # 50% Title, 25% Anchor, 25% Body
            score = (norm_t * 0.50) + (norm_a * 0.25) + (norm_b * 0.25)
            
            # Small co-occurrence bonus
            if norm_t > 0 and (norm_a > 0 or norm_b > 0):
                score *= 1.1
                
            final_scores[doc_id] = score

        results = sorted(final_scores.items(), key=lambda x: x[1], reverse=True)[:top_n]
        return [(doc_id, self.doc_titles.get(int(doc_id), f"Doc {doc_id}"), score) for doc_id, score in results]
    
# SINGLETON ENGINE MANAGEMENT
_ENGINE = None

def initialize_engine():
    global _ENGINE
    if _ENGINE is None:
        _ENGINE = SearchEngine()

def get_engine():
    if _ENGINE is None:
        raise RuntimeError("Engine not initialized.")
    return _ENGINE