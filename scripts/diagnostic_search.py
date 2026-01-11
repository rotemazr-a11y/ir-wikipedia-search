"""
Diagnostic Search Backend - Instrumented Search with Term-Level Analytics

This module provides instrumented versions of the search functions that track
detailed diagnostic information at the query, term, and document level.

Key Functions:
- compute_tfidf_with_diagnostics(): Enhanced TF-IDF computation with tracking
- multi_field_fusion_with_diagnostics(): Multi-field fusion with field contribution tracking
- search_with_full_diagnostics(): Complete diagnostic search pipeline

Author: IR Project Team
Date: 2025-12-29
"""

import math
import sys
from collections import Counter, defaultdict
from pathlib import Path

# Add backend to path for imports
backend_path = Path(__file__).parent.parent / 'backend'
sys.path.insert(0, str(backend_path))

from inverted_index_gcp import InvertedIndex
from pre_processing import tokenize_and_process


def compute_tfidf_with_diagnostics(query_tokens, index, index_dir, N, use_doc_norms=True, doc_norms=None):
    """
    Compute TF-IDF cosine similarity with detailed diagnostic tracking.

    This function extends the standard TF-IDF computation by tracking:
    - Query-level statistics (token counts, TF-IDF values, norm)
    - Term-level statistics (DF, IDF, contribution to matches)
    - Document-level details (matched terms, score contributions, norms)

    Parameters:
    -----------
    query_tokens : list of str
        Preprocessed query tokens
    index : InvertedIndex
        The inverted index to search
    index_dir : str
        Directory containing index files
    N : int
        Total number of documents
    use_doc_norms : bool
        Whether to use precomputed document norms
    doc_norms : dict
        Precomputed document norms (if use_doc_norms=True)

    Returns:
    --------
    tuple: (scores, diagnostics)
        scores: dict of doc_id -> cosine similarity score
        diagnostics: dict with comprehensive diagnostic information
    """
    diagnostics = {
        'query_analysis': {},
        'term_stats': {},
        'document_details': {},
        'summary': {}
    }

    if not query_tokens:
        diagnostics['summary']['empty_query'] = True
        return {}, diagnostics

    # Track original query tokens
    diagnostics['query_analysis']['original_tokens'] = list(query_tokens)
    diagnostics['query_analysis']['token_count'] = len(query_tokens)
    diagnostics['query_analysis']['unique_terms'] = list(set(query_tokens))
    diagnostics['query_analysis']['unique_term_count'] = len(set(query_tokens))

    # 1. Compute query TF-IDF vector and norm
    query_counts = Counter(query_tokens)
    query_tfidf = {}
    oov_terms = []  # Out-of-vocabulary terms

    for term, count in query_counts.items():
        if term in index.df:
            df = index.df[term]
            idf = math.log(N / df)  # Natural log
            query_tfidf[term] = count * idf

            # Track term-level statistics
            diagnostics['term_stats'][term] = {
                'df': df,
                'idf': idf,
                'query_tf': count,
                'query_tfidf': query_tfidf[term],
                'in_vocabulary': True,
                'num_matched_docs': 0  # Will be updated later
            }
        else:
            oov_terms.append(term)
            diagnostics['term_stats'][term] = {
                'df': 0,
                'idf': 0.0,
                'query_tf': count,
                'query_tfidf': 0.0,
                'in_vocabulary': False,
                'num_matched_docs': 0
            }

    diagnostics['query_analysis']['oov_terms'] = oov_terms
    diagnostics['query_analysis']['oov_count'] = len(oov_terms)
    diagnostics['query_analysis']['query_tfidf'] = dict(query_tfidf)

    if not query_tfidf:
        diagnostics['summary']['no_matching_terms'] = True
        return {}, diagnostics

    # Query L2 norm
    query_norm = math.sqrt(sum(v**2 for v in query_tfidf.values()))
    diagnostics['query_analysis']['query_norm'] = query_norm

    if query_norm == 0:
        diagnostics['summary']['zero_query_norm'] = True
        return {}, diagnostics

    # 2. Get candidate documents and track term occurrence
    candidates = {}
    term_doc_counts = defaultdict(int)  # Count docs per term

    for term in query_tfidf.keys():
        posting_list = index.read_a_posting_list(index_dir, term)
        term_doc_counts[term] = len(posting_list)

        for doc_id, tf in posting_list:
            if doc_id not in candidates:
                candidates[doc_id] = {}
            candidates[doc_id][term] = tf

    # Update term statistics with matched doc counts
    for term, count in term_doc_counts.items():
        if term in diagnostics['term_stats']:
            diagnostics['term_stats'][term]['num_matched_docs'] = count

    diagnostics['summary']['num_candidate_docs'] = len(candidates)

    # 3. Compute cosine similarity for each candidate with detailed tracking
    scores = {}

    for doc_id, doc_terms in candidates.items():
        # Track document-level details
        doc_detail = {
            'matched_terms': list(doc_terms.keys()),
            'matched_term_count': len(doc_terms),
            'term_contributions': {}
        }

        # Dot product and doc norm computation
        dot_product = 0.0
        doc_tfidf_squared = 0.0

        for term, tf in doc_terms.items():
            if term in query_tfidf:
                df = index.df[term]
                idf = math.log(N / df)
                doc_tfidf = tf * idf

                # Contribution to dot product
                contribution = query_tfidf[term] * doc_tfidf
                dot_product += contribution
                doc_tfidf_squared += doc_tfidf ** 2

                # Track term-level contribution
                doc_detail['term_contributions'][term] = {
                    'doc_tf': tf,
                    'doc_tfidf': doc_tfidf,
                    'contribution_to_dot_product': contribution
                }

        # Document norm
        if use_doc_norms and doc_norms and doc_id in doc_norms:
            doc_norm = doc_norms[doc_id]
        else:
            doc_norm = math.sqrt(doc_tfidf_squared)

        doc_detail['dot_product'] = dot_product
        doc_detail['doc_norm'] = doc_norm

        # Cosine similarity
        if doc_norm > 0:
            cosine_score = dot_product / (query_norm * doc_norm)
            scores[doc_id] = cosine_score
            doc_detail['cosine_score'] = cosine_score

            # Calculate percentage contribution per term
            if dot_product > 0:
                for term in doc_detail['term_contributions']:
                    contrib = doc_detail['term_contributions'][term]['contribution_to_dot_product']
                    doc_detail['term_contributions'][term]['percentage_of_total_score'] = \
                        (contrib / dot_product) * 100
            else:
                for term in doc_detail['term_contributions']:
                    doc_detail['term_contributions'][term]['percentage_of_total_score'] = 0.0
        else:
            doc_detail['cosine_score'] = 0.0

        diagnostics['document_details'][doc_id] = doc_detail

    diagnostics['summary']['num_scored_docs'] = len(scores)
    diagnostics['summary']['max_score'] = max(scores.values()) if scores else 0.0
    diagnostics['summary']['min_score'] = min(scores.values()) if scores else 0.0
    diagnostics['summary']['avg_score'] = sum(scores.values()) / len(scores) if scores else 0.0

    return scores, diagnostics


def binary_ranking_with_diagnostics(query_tokens, index, index_dir):
    """
    Binary ranking with diagnostic tracking.

    Parameters:
    -----------
    query_tokens : list of str
        Preprocessed query tokens
    index : InvertedIndex
        The inverted index to search
    index_dir : str
        Directory containing index files

    Returns:
    --------
    tuple: (scores, diagnostics)
        scores: dict of doc_id -> count of distinct query words
        diagnostics: dict with diagnostic information
    """
    diagnostics = {
        'query_terms': list(set(query_tokens)),
        'term_matches': {},
        'document_matches': {}
    }

    if not query_tokens:
        return {}, diagnostics

    query_set = set(query_tokens)

    # Track which query words appear in each document
    doc_matches = {}

    for term in query_set:
        if term in index.df:
            posting_list = index.read_a_posting_list(index_dir, term)
            diagnostics['term_matches'][term] = {
                'df': index.df[term],
                'num_docs_matched': len(posting_list),
                'in_vocabulary': True
            }

            for doc_id, tf in posting_list:
                if doc_id not in doc_matches:
                    doc_matches[doc_id] = set()
                doc_matches[doc_id].add(term)
        else:
            diagnostics['term_matches'][term] = {
                'df': 0,
                'num_docs_matched': 0,
                'in_vocabulary': False
            }

    # Score = number of distinct query words matched
    scores = {doc_id: len(matched_terms)
              for doc_id, matched_terms in doc_matches.items()}

    # Document-level diagnostics
    for doc_id, matched_terms in doc_matches.items():
        diagnostics['document_matches'][doc_id] = {
            'matched_terms': list(matched_terms),
            'match_count': len(matched_terms),
            'match_percentage': (len(matched_terms) / len(query_set)) * 100 if query_set else 0.0
        }

    diagnostics['summary'] = {
        'num_query_terms': len(query_set),
        'num_matched_docs': len(scores),
        'max_matches': max(scores.values()) if scores else 0,
        'avg_matches': sum(scores.values()) / len(scores) if scores else 0.0
    }

    return scores, diagnostics


def multi_field_fusion_with_diagnostics(query_tokens, body_index, title_index, anchor_index,
                                        index_dir, N, metadata, pagerank, pageviews,
                                        w_body=0.40, w_title=0.25, w_anchor=0.15,
                                        w_pagerank=0.15, w_pageviews=0.05):
    """
    Multi-field fusion scoring with field-level contribution tracking.

    Parameters:
    -----------
    query_tokens : list
        Preprocessed query tokens
    body_index, title_index, anchor_index : InvertedIndex
        The inverted indices for each field
    index_dir : str
        Index directory path
    N : int
        Total number of documents
    metadata : dict
        Metadata with doc_titles, num_docs, doc_norms
    pagerank : dict
        PageRank scores
    pageviews : dict
        Page view counts
    w_* : float
        Weights for each signal

    Returns:
    --------
    tuple: (final_scores, field_diagnostics)
        final_scores: dict of doc_id -> final weighted score
        field_diagnostics: dict with per-field contribution details
    """
    field_diagnostics = {
        'weights': {
            'body': w_body,
            'title': w_title,
            'anchor': w_anchor,
            'pagerank': w_pagerank,
            'pageviews': w_pageviews
        },
        'field_scores': {
            'body': {},
            'title': {},
            'anchor': {},
            'pagerank': {},
            'pageviews': {}
        },
        'document_contributions': {},
        'summary': {}
    }

    # 1. Get body TF-IDF scores with diagnostics
    doc_norms = metadata.get('doc_norms', {}) if metadata else {}
    body_scores, body_diag = compute_tfidf_with_diagnostics(
        query_tokens, body_index, index_dir, N, use_doc_norms=True, doc_norms=doc_norms
    )
    field_diagnostics['field_scores']['body'] = body_scores
    field_diagnostics['body_diagnostics'] = body_diag

    # 2. Get title matches with diagnostics
    title_scores, title_diag = binary_ranking_with_diagnostics(
        query_tokens, title_index, index_dir
    )
    field_diagnostics['field_scores']['title'] = title_scores
    field_diagnostics['title_diagnostics'] = title_diag

    # 3. Get anchor matches with diagnostics
    anchor_scores, anchor_diag = binary_ranking_with_diagnostics(
        query_tokens, anchor_index, index_dir
    )
    field_diagnostics['field_scores']['anchor'] = anchor_scores
    field_diagnostics['anchor_diagnostics'] = anchor_diag

    # 4. Normalize binary scores to [0, 1]
    query_terms_count = len(set(query_tokens))
    if query_terms_count > 0:
        title_scores_norm = {doc_id: score / query_terms_count
                            for doc_id, score in title_scores.items()}
        anchor_scores_norm = {doc_id: score / query_terms_count
                             for doc_id, score in anchor_scores.items()}
    else:
        title_scores_norm = {}
        anchor_scores_norm = {}

    # 5. Normalize PageRank to [0, 1]
    if pagerank:
        max_pr = max(pagerank.values()) if pagerank else 1.0
        pagerank_norm = {doc_id: pr / max_pr for doc_id, pr in pagerank.items()}
    else:
        pagerank_norm = {}

    field_diagnostics['field_scores']['pagerank'] = pagerank_norm
    field_diagnostics['summary']['max_pagerank'] = max(pagerank.values()) if pagerank else 0.0

    # 6. Normalize page views to [0, 1]
    if pageviews:
        max_pv = max(pageviews.values()) if pageviews else 1.0
        pageviews_norm = {doc_id: pv / max_pv for doc_id, pv in pageviews.items()}
    else:
        pageviews_norm = {}

    field_diagnostics['field_scores']['pageviews'] = pageviews_norm
    field_diagnostics['summary']['max_pageviews'] = max(pageviews.values()) if pageviews else 0

    # 7. Get all candidate documents (union of all signals)
    all_docs = set(body_scores.keys()) | set(title_scores.keys()) | set(anchor_scores.keys())

    # 8. Compute weighted fusion score with per-document tracking
    final_scores = {}

    for doc_id in all_docs:
        # Individual field contributions
        body_contrib = w_body * body_scores.get(doc_id, 0.0)
        title_contrib = w_title * title_scores_norm.get(doc_id, 0.0)
        anchor_contrib = w_anchor * anchor_scores_norm.get(doc_id, 0.0)
        pagerank_contrib = w_pagerank * pagerank_norm.get(doc_id, 0.0)
        pageviews_contrib = w_pageviews * pageviews_norm.get(doc_id, 0.0)

        # Final score
        final_score = body_contrib + title_contrib + anchor_contrib + pagerank_contrib + pageviews_contrib
        final_scores[doc_id] = final_score

        # Track per-document contributions
        field_diagnostics['document_contributions'][doc_id] = {
            'body_score': body_scores.get(doc_id, 0.0),
            'title_score': title_scores_norm.get(doc_id, 0.0),
            'anchor_score': anchor_scores_norm.get(doc_id, 0.0),
            'pagerank_score': pagerank_norm.get(doc_id, 0.0),
            'pageviews_score': pageviews_norm.get(doc_id, 0.0),
            'body_contribution': body_contrib,
            'title_contribution': title_contrib,
            'anchor_contribution': anchor_contrib,
            'pagerank_contribution': pagerank_contrib,
            'pageviews_contribution': pageviews_contrib,
            'final_score': final_score
        }

        # Calculate percentage contributions
        if final_score > 0:
            field_diagnostics['document_contributions'][doc_id]['field_percentages'] = {
                'body': (body_contrib / final_score) * 100,
                'title': (title_contrib / final_score) * 100,
                'anchor': (anchor_contrib / final_score) * 100,
                'pagerank': (pagerank_contrib / final_score) * 100,
                'pageviews': (pageviews_contrib / final_score) * 100
            }
        else:
            field_diagnostics['document_contributions'][doc_id]['field_percentages'] = {
                'body': 0.0, 'title': 0.0, 'anchor': 0.0, 'pagerank': 0.0, 'pageviews': 0.0
            }

    # Summary statistics
    field_diagnostics['summary']['num_candidates'] = len(all_docs)
    field_diagnostics['summary']['num_scored'] = len(final_scores)
    field_diagnostics['summary']['num_body_matches'] = len(body_scores)
    field_diagnostics['summary']['num_title_matches'] = len(title_scores)
    field_diagnostics['summary']['num_anchor_matches'] = len(anchor_scores)

    return final_scores, field_diagnostics


def search_with_full_diagnostics(query, body_index, title_index, anchor_index,
                                 index_dir, metadata, pagerank, pageviews,
                                 w_body=0.40, w_title=0.25, w_anchor=0.15,
                                 w_pagerank=0.15, w_pageviews=0.05,
                                 remove_stops=True, stem=True):
    """
    Complete diagnostic search pipeline.

    Parameters:
    -----------
    query : str
        Raw query string
    body_index, title_index, anchor_index : InvertedIndex
        Inverted indices for each field
    index_dir : str
        Index directory path
    metadata : dict
        Document metadata
    pagerank : dict
        PageRank scores
    pageviews : dict
        Page view counts
    w_* : float
        Field weights
    remove_stops : bool
        Whether to remove stopwords
    stem : bool
        Whether to apply stemming

    Returns:
    --------
    dict: Complete diagnostic report including:
        - query_info: Original query and preprocessing details
        - field_diagnostics: Per-field scoring details
        - ranking: Top-K results with scores
        - summary: High-level statistics
    """
    report = {
        'query_info': {
            'original_query': query,
            'remove_stops': remove_stops,
            'stem': stem
        }
    }

    # Preprocess query
    query_tokens = tokenize_and_process(query, remove_stops=remove_stops, stem=stem)
    report['query_info']['tokens'] = query_tokens
    report['query_info']['token_count'] = len(query_tokens)
    report['query_info']['unique_terms'] = list(set(query_tokens))

    if not query_tokens:
        report['error'] = 'Empty query after preprocessing'
        return report

    # Multi-field fusion with diagnostics
    N = metadata['num_docs']
    final_scores, field_diag = multi_field_fusion_with_diagnostics(
        query_tokens,
        body_index, title_index, anchor_index,
        index_dir, N, metadata, pagerank, pageviews,
        w_body, w_title, w_anchor, w_pagerank, w_pageviews
    )

    report['field_diagnostics'] = field_diag

    # Sort and create ranking
    sorted_docs = sorted(final_scores.items(), key=lambda x: x[1], reverse=True)

    report['ranking'] = {
        'top_100': [
            {
                'rank': i + 1,
                'doc_id': doc_id,
                'score': score,
                'title': metadata.get('doc_titles', {}).get(doc_id, "")
            }
            for i, (doc_id, score) in enumerate(sorted_docs[:100])
        ],
        'total_results': len(sorted_docs)
    }

    # High-level summary
    report['summary'] = {
        'num_results': len(final_scores),
        'max_score': max(final_scores.values()) if final_scores else 0.0,
        'min_score': min(final_scores.values()) if final_scores else 0.0,
        'avg_score': sum(final_scores.values()) / len(final_scores) if final_scores else 0.0,
        'weights_used': field_diag['weights']
    }

    return report
