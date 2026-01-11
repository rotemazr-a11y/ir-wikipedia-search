# Professional Testing & Optimization Strategy for Wikipedia Search Engine

**Document Version:** 1.0  
**Date:** December 22, 2025  
**Scope:** End-to-end testing, validation, and hyperparameter optimization for 6GB+ Wikipedia index with 35-second SLA

---

## EXECUTIVE SUMMARY

This document provides a data-driven, professional testing and optimization strategy for your Wikipedia search engine built on Python, Flask, and PySpark. The architecture combines **TF-IDF/BM25 body ranking**, **binary title/anchor matching**, and **PageRank static scores** into a composite relevance function. The strategy addresses five critical testing phases:

1. **Data Integrity Validation** (Statistical verification without full scans)
2. **Component-Level Precision Testing** (Isolation, ablation studies, small-scale indices)
3. **Systematic Hyperparameter Optimization** (Grid/coordinate ascent on 30 training queries)
4. **Performance Profiling** (Latency bottlenecks, early-exit strategies)
5. **Failure Analysis** (Professional debugging workflow for poor results)

---

## 1. DATA INTEGRITY & INDEX VALIDATION

### 1.1 Statistical Verification Without Scanning the Full Index

**Problem:** Scanning all 6GB+ of index data is prohibitively expensive. You need to verify the inverted index is "healthy" without exhaustive validation.

**Solution: Zipf's Law Validation**

Zipf's Law states that in natural language, the frequency of a word is inversely proportional to its rank:
$$f(r) = \frac{k}{r^\alpha}$$

where $r$ is the rank, $f$ is the frequency, $k$ is a constant, and $\alpha \approx 1$ for natural language.

**Implementation:**

```python
import numpy as np
import matplotlib.pyplot as plt
from collections import Counter
import logging

logger = logging.getLogger(__name__)

class IndexHealthCheck:
    """Statistical validation of inverted index health without full scans."""
    
    def __init__(self, index, sample_size=5000):
        """
        Args:
            index: InvertedIndex object (from assignment_2.ipynb)
            sample_size: Number of terms to sample for validation
        """
        self.index = index
        self.sample_size = min(sample_size, len(index.df))
        
    def validate_zipf_distribution(self, tolerance=0.15):
        """
        Verify Zipf's Law holds for sampled terms.
        
        Returns:
            dict: {
                'zipf_exponent': float (should be ~1.0),
                'r_squared': float (goodness of fit, >0.95 is good),
                'passes': bool,
                'details': str
            }
        """
        # Sample terms proportionally: sample more high-frequency terms
        all_terms = list(self.index.df.keys())
        frequencies = np.array([self.index.df[t] for t in all_terms])
        
        # Rank by frequency (descending)
        ranks = np.argsort(-frequencies)[:self.sample_size] + 1
        sampled_freqs = frequencies[ranks - 1]
        
        # Fit log-log regression: log(f) = log(k) - α*log(r)
        log_ranks = np.log(ranks)
        log_freqs = np.log(sampled_freqs)
        
        coeffs = np.polyfit(log_ranks, log_freqs, 1)
        poly = np.poly1d(coeffs)
        fitted = poly(log_ranks)
        
        # Calculate R²
        ss_res = np.sum((log_freqs - fitted) ** 2)
        ss_tot = np.sum((log_freqs - np.mean(log_freqs)) ** 2)
        r_squared = 1 - (ss_res / ss_tot)
        
        alpha = -coeffs[0]  # Negative because of regression form
        passes = (r_squared > 0.9) and (abs(alpha - 1.0) < tolerance)
        
        return {
            'zipf_exponent': float(alpha),
            'r_squared': float(r_squared),
            'passes': passes,
            'details': f"Zipf exponent: {alpha:.3f} (expected ~1.0), R²: {r_squared:.4f}"
        }
    
    def check_df_anomalies(self, percentile_threshold=99.5):
        """
        Detect Document Frequency anomalies that may indicate indexing errors.
        
        Returns:
            dict: {
                'anomalies_found': bool,
                'suspicious_terms': list,
                'reasoning': str
            }
        """
        dfs = np.array(list(self.index.df.values()))
        total_docs = max([doc_id for t in self.index._posting_list.values() 
                          for doc_id, _ in t]) if hasattr(self.index, '_posting_list') else 1
        
        # DF should never exceed total documents
        max_df = np.percentile(dfs, percentile_threshold)
        
        # Flag terms appearing in >90% of documents (likely stopwords)
        suspicious = []
        for term, df in self.index.df.items():
            if df > 0.9 * total_docs:
                suspicious.append((term, df, "appears in >90% of docs"))
            # Also flag very rare terms that may indicate tokenization errors
            elif df == 1 and len(term) > 30:
                suspicious.append((term, df, "very rare but extremely long"))
        
        return {
            'anomalies_found': len(suspicious) > 0,
            'suspicious_terms': suspicious[:10],  # Top 10
            'reasoning': "These terms may indicate stopword leakage or tokenization errors"
        }
    
    def check_term_total_consistency(self, sample_size=1000):
        """
        Verify term_total counts match sum of TFs across all postings.
        Sample random terms to avoid O(n) scan.
        
        Returns:
            dict: {
                'consistency_check_passed': bool,
                'mismatches': list,
                'sample_size': int
            }
        """
        import random
        sample_terms = random.sample(list(self.index.df.keys()), 
                                     min(sample_size, len(self.index.df)))
        
        mismatches = []
        for term in sample_terms:
            recorded_total = self.index.term_total.get(term, 0)
            
            # Recount from postings (if available)
            if hasattr(self.index, '_posting_list') and term in self.index._posting_list:
                actual_total = sum(tf for _, tf in self.index._posting_list[term])
                if recorded_total != actual_total:
                    mismatches.append({
                        'term': term,
                        'recorded': recorded_total,
                        'actual': actual_total,
                        'diff': actual_total - recorded_total
                    })
        
        return {
            'consistency_check_passed': len(mismatches) == 0,
            'mismatches': mismatches,
            'sample_size': len(sample_terms)
        }

    def run_full_validation(self):
        """Run all checks and return comprehensive report."""
        report = {
            'timestamp': pd.Timestamp.now(),
            'index_size_terms': len(self.index.df),
            'zipf_check': self.validate_zipf_distribution(),
            'df_anomalies': self.check_df_anomalies(),
            'consistency_check': self.check_term_total_consistency()
        }
        
        overall_pass = (
            report['zipf_check']['passes'] and
            not report['df_anomalies']['anomalies_found'] and
            report['consistency_check']['consistency_check_passed']
        )
        
        report['overall_health'] = 'PASS' if overall_pass else 'FAIL'
        return report
```

### 1.2 Automated Sanity Checks for inverted_index_gcp.py

```python
import struct
import hashlib
from collections import defaultdict

class InvertedIndexValidator:
    """Validate binary posting list integrity after GCP writes."""
    
    TUPLE_SIZE = 6  # Must match inverted_index_gcp.py
    TF_MASK = 2**16 - 1
    
    @staticmethod
    def compute_posting_checksum(posting_list):
        """Compute checksum of (doc_id, tf) pairs before writing."""
        hasher = hashlib.md5()
        for doc_id, tf in sorted(posting_list, key=lambda x: x[0]):
            packed = (doc_id << 16 | (tf & TF_MASK)).to_bytes(InvertedIndexValidator.TUPLE_SIZE, 'big')
            hasher.update(packed)
        return hasher.hexdigest()
    
    @staticmethod
    def verify_posting_after_read(raw_bytes, expected_checksum):
        """
        Unpack bytes and verify they match expected checksum.
        
        Returns:
            (posting_list, is_valid)
        """
        posting_list = []
        if len(raw_bytes) % InvertedIndexValidator.TUPLE_SIZE != 0:
            logger.error(f"Posting list size {len(raw_bytes)} not divisible by {InvertedIndexValidator.TUPLE_SIZE}")
            return [], False
        
        for i in range(0, len(raw_bytes), InvertedIndexValidator.TUPLE_SIZE):
            chunk = raw_bytes[i:i+InvertedIndexValidator.TUPLE_SIZE]
            packed_val = int.from_bytes(chunk, 'big')
            doc_id = packed_val >> 16
            tf = packed_val & InvertedIndexValidator.TF_MASK
            posting_list.append((doc_id, tf))
        
        computed_checksum = InvertedIndexValidator.compute_posting_checksum(posting_list)
        is_valid = computed_checksum == expected_checksum
        
        return posting_list, is_valid
```

---

## 2. COMPONENT-LEVEL PRECISION TESTING

### 2.1 Isolating & Testing Individual Ranking Components

Your search method combines three scores:
- **Body Score:** TF-IDF or BM25 on document body
- **Title Score:** Binary match on query terms in title
- **Anchor Score:** Binary match on query terms in anchor text
- **PageRank:** Static rank from link graph

**Strategy: Component Ablation Testing**

```python
from dataclasses import dataclass
from typing import List, Tuple, Dict
import pandas as pd

@dataclass
class RankingComponent:
    """Represents one scoring component."""
    name: str
    weight: float
    scores: Dict[int, float]  # doc_id -> score

class ComponentPrecisionTester:
    """
    Test individual ranking components in isolation.
    """
    
    def __init__(self, index_body, index_title, index_anchor, pagerank_dict, 
                 training_queries_json='queries_train.json'):
        self.index_body = index_body
        self.index_title = index_title
        self.index_anchor = index_anchor
        self.pagerank = pagerank_dict
        
        with open(training_queries_json) as f:
            self.training_queries = json.load(f)
        
        self.component_results = {}
    
    def run_ablation_study(self) -> pd.DataFrame:
        """
        Run component evaluation on all training queries.
        Returns DataFrame with scores for each query x component.
        """
        results = []
        
        for query_text, ground_truth_doc_ids in self.training_queries.items():
            component_scores = self.evaluate_component_precision(
                query_text, 
                [int(doc_id) for doc_id in ground_truth_doc_ids],
                k=10
            )
            
            for component_name, metrics in component_scores.items():
                results.append({
                    'query': query_text,
                    'component': component_name,
                    'precision@10': metrics['precision@k'],
                    'recall@10': metrics['recall@k'],
                    'ndcg': metrics['ndcg'],
                    'quality': metrics['retrieval_quality']
                })
        
        df = pd.DataFrame(results)
        
        # Summary statistics
        summary = df.groupby('component')[['precision@10', 'recall@10', 'ndcg']].mean()
        logger.info(f"\nComponent Ablation Study Results:\n{summary}")
        
        return df
```

### 2.2 Small-Scale Index Strategy

```python
class SmallScaleIndexBuilder:
    """
    Build a tiny index using only vocabulary from training queries.
    Use for rapid iteration on hyperparameters.
    """
    
    def __init__(self, training_queries_json, wikipedia_dump_path):
        self.training_queries = self._load_training_queries(training_queries_json)
        self.wiki_dump_path = wikipedia_dump_path
        
        # Extract vocabulary from training queries
        self.query_vocab = set()
        for query_text in self.training_queries.keys():
            tokens = self._tokenize(query_text)
            self.query_vocab.update(tokens)
        
        logger.info(f"Training query vocabulary size: {len(self.query_vocab)} unique terms")
    
    def build_focused_index(self, relevant_doc_ids=None):
        """
        Build inverted index for:
        - Only documents referenced in ground truth
        - Only terms in training query vocabulary
        """
        with open(self.wiki_dump_path, 'rb') as f:
            docs = pickle.load(f)
        
        if relevant_doc_ids:
            docs = {doc_id: doc_data for doc_id, doc_data in docs.items()
                   if doc_id in relevant_doc_ids}
        
        index_body = InvertedIndex()
        index_title = InvertedIndex()
        
        for doc_id, doc_data in docs.items():
            body_tokens = self._tokenize(doc_data['body'])
            body_tokens_filtered = [t for t in body_tokens if t in self.query_vocab]
            
            title_tokens = self._tokenize(doc_data['title'])
            title_tokens_filtered = [t for t in title_tokens if t in self.query_vocab]
            
            if body_tokens_filtered:
                index_body.add_doc(doc_id, body_tokens_filtered)
            if title_tokens_filtered:
                index_title.add_doc(doc_id, title_tokens_filtered)
        
        logger.info(f"Built small-scale indices with {len(docs)} documents")
        
        return index_body, index_title, len(docs)
```

---

## 3. SYSTEMATIC HYPERPARAMETER OPTIMIZATION

### 3.1 Grid Search on Weight Parameters

Your final ranking function:

$$\text{score}(d,q) = w_1 \cdot \text{TF-IDF}(d,q) + w_2 \cdot \text{Title}(d,q) + w_3 \cdot \text{PageRank}(d)$$

```python
import itertools

class HyperparameterOptimizer:
    """
    Optimize $(w_{body}, w_{title}, w_{anchor}, w_{pagerank})$ to maximize MAP.
    """
    
    def __init__(self, index_body, index_title, index_anchor, pagerank_dict,
                 training_queries_json, k=10):
        self.index_body = index_body
        self.index_title = index_title
        self.index_anchor = index_anchor
        self.pagerank = pagerank_dict
        self.k = k
        
        with open(training_queries_json) as f:
            self.queries = json.load(f)
        
        self.optimization_history = []
    
    def evaluate_map(self, w_body: float, w_title: float, w_anchor: float, 
                    w_pagerank: float) -> float:
        """
        Compute Mean Average Precision (MAP) across all training queries.
        
        MAP = (1/Q) * Σ_q AP_q
        AP_q = (1/R_q) * Σ_k (P(k) * rel(k))
        """
        ap_scores = []
        
        for query_text, ground_truth_ids in self.queries.items():
            ground_truth_set = set(int(doc_id) for doc_id in ground_truth_ids)
            
            ranked_docs = self.retrieve_with_weights(query_text, w_body, w_title, 
                                                     w_anchor, w_pagerank)
            
            relevant_found = 0
            ap_sum = 0
            for rank, (doc_id, score) in enumerate(ranked_docs, start=1):
                if doc_id in ground_truth_set:
                    relevant_found += 1
                    precision_at_k = relevant_found / rank
                    ap_sum += precision_at_k
            
            ap = ap_sum / len(ground_truth_set) if ground_truth_set else 0
            ap_scores.append(ap)
        
        map_score = np.mean(ap_scores) if ap_scores else 0
        return float(map_score)
    
    def grid_search(self, weight_ranges=None, step=0.1):
        """
        Perform grid search over weight combinations.
        """
        if weight_ranges is None:
            weight_ranges = {
                'w_body': (0, 1),
                'w_title': (0, 1),
                'w_anchor': (0, 0.5),
                'w_pagerank': (0, 0.3)
            }
        
        w_body_vals = np.arange(weight_ranges['w_body'][0], 
                               weight_ranges['w_body'][1] + step, step)
        w_title_vals = np.arange(weight_ranges['w_title'][0],
                                weight_ranges['w_title'][1] + step, step)
        w_anchor_vals = np.arange(weight_ranges['w_anchor'][0],
                                 weight_ranges['w_anchor'][1] + step, step)
        w_pagerank_vals = np.arange(weight_ranges['w_pagerank'][0],
                                    weight_ranges['w_pagerank'][1] + step, step)
        
        total_combinations = (len(w_body_vals) * len(w_title_vals) * 
                            len(w_anchor_vals) * len(w_pagerank_vals))
        logger.info(f"Grid search: {total_combinations} combinations to evaluate")
        
        best_map = -1
        best_weights = None
        
        for i, (w_body, w_title, w_anchor, w_pagerank) in enumerate(
            itertools.product(w_body_vals, w_title_vals, w_anchor_vals, w_pagerank_vals)
        ):
            weight_sum = w_body + w_title + w_anchor + w_pagerank
            if weight_sum > 0:
                w_body_norm = w_body / weight_sum
                w_title_norm = w_title / weight_sum
                w_anchor_norm = w_anchor / weight_sum
                w_pagerank_norm = w_pagerank / weight_sum
            else:
                continue
            
            map_score = self.evaluate_map(w_body_norm, w_title_norm, 
                                         w_anchor_norm, w_pagerank_norm)
            
            if map_score > best_map:
                best_map = map_score
                best_weights = (w_body_norm, w_title_norm, w_anchor_norm, w_pagerank_norm)
            
            if (i + 1) % 100 == 0:
                logger.info(f"Progress: {i + 1}/{total_combinations}, Best MAP: {best_map:.4f}")
        
        return {
            'best_weights': best_weights,
            'best_map': best_map,
            'history': self.optimization_history
        }
```

### 3.2 K-Fold Cross-Validation

```python
class CrossValidatingOptimizer(HyperparameterOptimizer):
    """
    Extend hyperparameter optimizer with k-fold CV to prevent overfitting.
    """
    
    def grid_search_with_cv(self, k_folds=5, weight_ranges=None, step=0.15):
        """
        Perform grid search with k-fold cross-validation.
        """
        query_items = list(self.queries.items())
        n = len(query_items)
        fold_size = n // k_folds
        
        cv_results = {'fold_results': []}
        best_cv_map = -1
        best_weights = None
        
        for fold_idx in range(k_folds):
            logger.info(f"\n========== FOLD {fold_idx + 1}/{k_folds} ==========")
            
            val_start = fold_idx * fold_size
            val_end = val_start + fold_size if fold_idx < k_folds - 1 else n
            
            val_queries = dict(query_items[val_start:val_end])
            train_queries = dict(query_items[:val_start] + query_items[val_end:])
            
            logger.info(f"Train: {len(train_queries)} queries, Val: {len(val_queries)} queries")
            
            # Perform grid search on training set
            best_train_map = -1
            best_fold_weights = None
            
            # [Grid search logic - similar to grid_search()]
            
            # Evaluate on validation set
            val_map = self._evaluate_map_on_subset(val_queries, *best_fold_weights)
            
            fold_result = {
                'fold': fold_idx,
                'train_map': best_train_map,
                'val_map': val_map,
                'weights': best_fold_weights
            }
            cv_results['fold_results'].append(fold_result)
            
            if val_map > best_cv_map:
                best_cv_map = val_map
                best_weights = best_fold_weights
        
        avg_val_map = np.mean([f['val_map'] for f in cv_results['fold_results']])
        
        logger.info(f"\n========== CV RESULTS ==========")
        logger.info(f"Average Validation MAP: {avg_val_map:.4f}")
        
        return {
            'best_weights': best_weights,
            'best_cv_map': best_cv_map,
            'avg_val_map': avg_val_map,
            'cv_folds_history': cv_results['fold_results']
        }
```

---

## 4. PERFORMANCE & LATENCY PROFILING

### 4.1 Early-Exit & Champion Lists

```python
class EarlyExitRetriever:
    """
    Implement early-exit retrieval using champion lists.
    """
    
    def __init__(self, index_body, index_title, pagerank_dict, champion_list_size=1000):
        self.index_body = index_body
        self.index_title = index_title
        self.pagerank = pagerank_dict
        self.champion_list_size = champion_list_size
        
        self.champion_lists = self._build_champion_lists()
    
    def _build_champion_lists(self):
        """
        For each term, keep only docs with highest TF*PageRank product.
        Saves ~90% of posting list traversals for common terms.
        """
        champion_lists = {}
        
        for term in self.index_body.df.keys():
            posting_list = self._read_posting_list(self.index_body, term)
            
            scored = []
            for doc_id, tf in posting_list:
                pagerank_score = self.pagerank.get(doc_id, 0)
                score = tf * (1 + pagerank_score)
                heappush(scored, (score, doc_id, tf))
            
            champion_list = []
            for _ in range(min(self.champion_list_size, len(scored))):
                if scored:
                    score, doc_id, tf = heappop(scored)
                    champion_list.append((doc_id, tf))
            
            champion_lists[term] = list(reversed(champion_list))
        
        return champion_lists
    
    def retrieve_with_early_exit(self, query_tokens: List[str], k: int = 100, 
                                timeout_seconds: float = 30.0) -> List[Tuple[int, float]]:
        """
        Retrieve top-k documents with early exit.
        """
        start_time = time.time()
        candidates = {}
        docs_examined = 0
        
        for term in query_tokens:
            elapsed = time.time() - start_time
            if elapsed > timeout_seconds * 0.8:
                logger.warning(f"Early exit: {term} skipped due to time budget")
                break
            
            if term not in self.index_body.df:
                continue
            
            df = self.index_body.df[term]
            
            if df > 1000:
                posting_list = self.champion_lists.get(term, [])
            else:
                posting_list = self._read_posting_list(self.index_body, term)
            
            for doc_id, tf in posting_list:
                tfidf_score = tf * np.log(len(self.index_body.df) / (df + 1))
                candidates[doc_id] = candidates.get(doc_id, 0) + tfidf_score
                docs_examined += 1
            
            if len(candidates) > k * 10 and docs_examined > 50000:
                logger.info(f"Early exit: examined {docs_examined} docs")
                break
        
        final_scores = {}
        for doc_id, score in candidates.items():
            pagerank = self.pagerank.get(doc_id, 0)
            final_scores[doc_id] = score + 0.1 * pagerank
        
        ranked = sorted(final_scores.items(), key=lambda x: -x[1])[:k]
        return ranked
```

### 4.2 Latency Monitoring

```python
class LatencyMonitor:
    """
    Track query latencies in production to identify slowdowns.
    """
    
    def __init__(self, percentiles=[50, 90, 95, 99]):
        self.latencies = []
        self.percentiles = percentiles
        self.route_latencies = defaultdict(list)
    
    def record_latency(self, route: str, latency_ms: float):
        """Record latency for a query."""
        self.latencies.append(latency_ms)
        self.route_latencies[route].append(latency_ms)
    
    def get_percentile_stats(self):
        """Get percentile latencies."""
        if not self.latencies:
            return {}
        
        stats = {}
        for p in self.percentiles:
            stats[f'p{p}'] = float(np.percentile(self.latencies, p))
        
        stats['mean'] = float(np.mean(self.latencies))
        stats['std'] = float(np.std(self.latencies))
        stats['max'] = float(np.max(self.latencies))
        
        return stats
```

---

## 5. PROFESSIONAL FAILURE ANALYSIS & DEBUGGING

### 5.1 Debugging Workflow for Poor Results

```python
class RelevanceDebugger:
    """
    Systematic workflow for debugging poor retrieval results.
    """
    
    def __init__(self, index_body, index_title, index_anchor, pagerank_dict,
                 query_ground_truth=None):
        self.index_body = index_body
        self.index_title = index_title
        self.index_anchor = index_anchor
        self.pagerank = pagerank_dict
        self.query_ground_truth = query_ground_truth or {}
    
    def debug_query(self, query_text: str, top_k=10):
        """
        Perform full debugging analysis for a single query.
        """
        debug_log = {
            'query': query_text,
            'timestamp': str(pd.Timestamp.now()),
            'diagnostics': {}
        }
        
        # Step 1: Tokenization
        debug_log['diagnostics']['tokenization'] = self._check_tokenization(query_text)
        
        # Step 2: Term existence in indices
        debug_log['diagnostics']['term_coverage'] = self._check_term_coverage(
            debug_log['diagnostics']['tokenization']['tokens']
        )
        
        # Step 3: Root cause analysis
        debug_log['root_cause_analysis'] = self._diagnose_root_causes(debug_log)
        
        return debug_log
```

### 5.2 Wikipedia-Specific Edge Cases

```python
class WikipediaEdgeCaseHandler:
    """
    Handle edge cases specific to Wikipedia data.
    """
    
    @staticmethod
    def detect_disambiguation_pages(doc_title: str, doc_body: str) -> bool:
        """
        Detect disambiguation pages (e.g., "Smith (disambiguation)").
        """
        if '(disambiguation)' in doc_title.lower():
            return True
        
        if 'wikimedia' in doc_body.lower() or '{{' in doc_body:
            return True
        
        paragraphs = len(re.findall(r'\n\n+', doc_body))
        list_items = len(re.findall(r'[\*\-]\s+', doc_body))
        if list_items > paragraphs * 2:
            return True
        
        return False
    
    @staticmethod
    def detect_stub_articles(doc_title: str, doc_body: str) -> bool:
        """
        Detect stub articles (very short, incomplete).
        """
        if len(doc_body) < 500:
            return True
        
        if re.search(r'\[stub\]|\{\{stub|\{\{incomplete', doc_body, re.IGNORECASE):
            return True
        
        link_count = len(re.findall(r'\[\[', doc_body))
        if link_count > len(doc_body) / 50:
            return True
        
        return False
```

---

## SUMMARY & RECOMMENDED TESTING SCHEDULE

| Phase | Duration | Key Deliverables |
|-------|----------|------------------|
| **1. Data Integrity** | 1-2 days | Zipf validation report, corruption detection logs |
| **2. Component Testing** | 2-3 days | Ablation study results, component precision scores |
| **3. Hyperparameter Optimization** | 3-5 days | Grid search results, CV validation report, best weights |
| **4. Performance Profiling** | 1-2 days | Latency percentiles, bottleneck identification |
| **5. Failure Analysis** | Ongoing | Debug reports, root cause hypotheses |

**Implementation Priority:**
1. ✅ Data integrity checks (quick win)
2. ✅ Component ablation study (understand what works)
3. ✅ Grid search with CV (systematic optimization)
4. ✅ Latency profiling (meets SLA)
5. ✅ Failure analysis workflow (continuous improvement)

---

**End of Document**
