"""
Term Contribution Analyzer - Ablation Testing & Term Effectiveness

This module provides functions to analyze individual term impact on retrieval
through ablation testing and effectiveness scoring.

Key Functions:
- compute_term_effectiveness_scores(): Measure term retrieval effectiveness
- ablation_analysis(): Systematically remove terms to measure impact
- analyze_dominant_terms(): Identify terms overwhelming the score

Author: IR Project Team
Date: 2025-12-29
"""

import sys
from pathlib import Path
from collections import defaultdict
import numpy as np

# Add backend to path
parent_dir = Path(__file__).parent.parent
backend_path = parent_dir / 'backend'
sys.path.insert(0, str(parent_dir))
sys.path.insert(0, str(backend_path))

from diagnostic_search import multi_field_fusion_with_diagnostics


def compute_term_effectiveness_scores(term, diagnostic_data, relevant_docs):
    """
    Compute retrieval effectiveness scores for a single term.

    Parameters:
    -----------
    term : str
        The term to analyze
    diagnostic_data : dict
        Full diagnostic data from search_with_full_diagnostics()
    relevant_docs : list
        List of relevant document IDs for the query

    Returns:
    --------
    dict: Term effectiveness metrics including:
        - recall: % of relevant docs containing the term
        - precision: % of docs with term that are relevant
        - avg_contribution: Average score contribution across relevant docs
        - dominance_ratio: How much this term dominates scoring
        - field_distribution: Where the term appears
    """
    effectiveness = {
        'term': term,
        'recall': 0.0,
        'precision': 0.0,
        'avg_contribution': 0.0,
        'dominance_ratio': 0.0,
        'field_distribution': {
            'body_count': 0,
            'title_count': 0,
            'anchor_count': 0
        },
        'statistics': {}
    }

    field_diag = diagnostic_data.get('field_diagnostics', {})

    # Get term statistics from body diagnostics
    body_diag = field_diag.get('body_diagnostics', {})
    term_stats = body_diag.get('term_stats', {}).get(term, {})

    if not term_stats or not term_stats.get('in_vocabulary', False):
        effectiveness['statistics']['in_vocabulary'] = False
        return effectiveness

    effectiveness['statistics'] = {
        'in_vocabulary': True,
        'df': term_stats.get('df', 0),
        'idf': term_stats.get('idf', 0.0),
        'num_matched_docs': term_stats.get('num_matched_docs', 0)
    }

    # Get documents containing this term from body diagnostics
    doc_details = body_diag.get('document_details', {})
    docs_with_term = []

    for doc_id, details in doc_details.items():
        if term in details.get('matched_terms', []):
            docs_with_term.append(doc_id)

    # Calculate recall: % of relevant docs containing term
    if relevant_docs:
        relevant_with_term = [doc for doc in relevant_docs if doc in docs_with_term]
        effectiveness['recall'] = len(relevant_with_term) / len(relevant_docs) * 100
    else:
        effectiveness['recall'] = 0.0

    # Calculate precision: % of docs with term that are relevant
    if docs_with_term:
        relevant_with_term = [doc for doc in docs_with_term if doc in relevant_docs]
        effectiveness['precision'] = len(relevant_with_term) / len(docs_with_term) * 100
    else:
        effectiveness['precision'] = 0.0

    # Calculate average contribution to score in relevant docs
    contributions = []
    for doc_id in relevant_docs:
        if doc_id in doc_details:
            term_contrib = doc_details[doc_id].get('term_contributions', {}).get(term, {})
            contrib_score = term_contrib.get('contribution_to_dot_product', 0.0)
            contributions.append(contrib_score)

    if contributions:
        effectiveness['avg_contribution'] = float(np.mean(contributions))
        effectiveness['max_contribution'] = float(np.max(contributions))
        effectiveness['min_contribution'] = float(np.min(contributions))

        # Dominance ratio: how much does max exceed mean?
        mean_contrib = np.mean(contributions)
        if mean_contrib > 0:
            effectiveness['dominance_ratio'] = float(np.max(contributions) / mean_contrib)
        else:
            effectiveness['dominance_ratio'] = 0.0
    else:
        effectiveness['avg_contribution'] = 0.0
        effectiveness['max_contribution'] = 0.0
        effectiveness['min_contribution'] = 0.0
        effectiveness['dominance_ratio'] = 0.0

    # Field distribution
    title_diag = field_diag.get('title_diagnostics', {})
    anchor_diag = field_diag.get('anchor_diagnostics', {})

    title_term_matches = title_diag.get('term_matches', {}).get(term, {})
    anchor_term_matches = anchor_diag.get('term_matches', {}).get(term, {})

    effectiveness['field_distribution'] = {
        'body_count': term_stats.get('num_matched_docs', 0),
        'title_count': title_term_matches.get('num_docs_matched', 0),
        'anchor_count': anchor_term_matches.get('num_docs_matched', 0)
    }

    return effectiveness


def ablation_analysis(query_tokens, body_index, title_index, anchor_index,
                     index_dir, metadata, pagerank, pageviews, relevant_docs,
                     weights=None):
    """
    Systematically remove each term and measure impact on retrieval.

    For each term:
    1. Run search without that term
    2. Compute ranking of relevant documents
    3. Calculate delta from full query

    Parameters:
    -----------
    query_tokens : list
        Preprocessed query tokens
    body_index, title_index, anchor_index : InvertedIndex
        Inverted indices
    index_dir : str
        Index directory
    metadata : dict
        Document metadata
    pagerank, pageviews : dict
        Ranking signals
    relevant_docs : list
        List of relevant document IDs
    weights : dict, optional
        Field weights (uses defaults if not provided)

    Returns:
    --------
    dict: Ablation results for each term including:
        - map_delta: Change in MAP when term removed
        - rank_changes: How relevant docs' ranks changed
        - is_critical: Whether term is critical for retrieval
    """
    if weights is None:
        weights = {
            'w_body': 0.40,
            'w_title': 0.25,
            'w_anchor': 0.15,
            'w_pagerank': 0.15,
            'w_pageviews': 0.05
        }

    N = metadata['num_docs']

    # Get baseline (full query) scores
    baseline_scores, _ = multi_field_fusion_with_diagnostics(
        query_tokens,
        body_index, title_index, anchor_index,
        index_dir, N, metadata, pagerank, pageviews,
        **weights
    )

    # Get baseline ranking of relevant docs
    baseline_ranking = _rank_relevant_docs(baseline_scores, relevant_docs)
    baseline_map = _compute_map_at_k(baseline_scores, relevant_docs, k=10)

    # Results storage
    ablation_results = {
        'baseline_map': baseline_map,
        'term_ablations': {},
        'summary': {}
    }

    # Get unique terms
    unique_terms = list(set(query_tokens))

    # Ablation: remove each term one at a time
    for term_to_remove in unique_terms:
        # Create reduced query (without this term)
        reduced_tokens = [t for t in query_tokens if t != term_to_remove]

        if not reduced_tokens:
            # Can't remove all terms
            ablation_results['term_ablations'][term_to_remove] = {
                'error': 'Cannot remove all terms',
                'map_delta': 0.0,
                'is_critical': False
            }
            continue

        # Run search with reduced query
        reduced_scores, _ = multi_field_fusion_with_diagnostics(
            reduced_tokens,
            body_index, title_index, anchor_index,
            index_dir, N, metadata, pagerank, pageviews,
            **weights
        )

        # Get ranking of relevant docs with reduced query
        reduced_ranking = _rank_relevant_docs(reduced_scores, relevant_docs)
        reduced_map = _compute_map_at_k(reduced_scores, relevant_docs, k=10)

        # Calculate impact
        map_delta = reduced_map - baseline_map

        # Analyze rank changes
        rank_changes = []
        for doc_id in relevant_docs:
            baseline_rank = baseline_ranking.get(doc_id, float('inf'))
            reduced_rank = reduced_ranking.get(doc_id, float('inf'))
            rank_change = reduced_rank - baseline_rank  # Positive = worse rank

            rank_changes.append({
                'doc_id': doc_id,
                'baseline_rank': baseline_rank if baseline_rank != float('inf') else None,
                'reduced_rank': reduced_rank if reduced_rank != float('inf') else None,
                'rank_change': rank_change if rank_change != float('inf') else None
            })

        # Store results
        ablation_results['term_ablations'][term_to_remove] = {
            'baseline_map': baseline_map,
            'reduced_map': reduced_map,
            'map_delta': map_delta,
            'map_delta_percentage': (map_delta / baseline_map * 100) if baseline_map > 0 else 0.0,
            'is_critical': map_delta < -0.05,  # Critical if MAP drops by > 0.05
            'is_redundant': abs(map_delta) < 0.01,  # Redundant if no significant impact
            'rank_changes': rank_changes,
            'avg_rank_change': float(np.mean([
                rc['rank_change'] for rc in rank_changes
                if rc['rank_change'] is not None and rc['rank_change'] != float('inf')
            ])) if any(rc['rank_change'] is not None and rc['rank_change'] != float('inf') for rc in rank_changes) else 0.0
        }

    # Summary statistics
    term_deltas = [
        (term, data['map_delta'])
        for term, data in ablation_results['term_ablations'].items()
        if 'error' not in data
    ]

    if term_deltas:
        # Sort by impact (most negative = most critical)
        term_deltas_sorted = sorted(term_deltas, key=lambda x: x[1])

        ablation_results['summary'] = {
            'most_critical_term': term_deltas_sorted[0][0],
            'most_critical_delta': term_deltas_sorted[0][1],
            'least_critical_term': term_deltas_sorted[-1][0],
            'least_critical_delta': term_deltas_sorted[-1][1],
            'avg_map_delta': float(np.mean([d for _, d in term_deltas])),
            'num_critical_terms': sum(1 for term, _ in term_deltas
                                     if ablation_results['term_ablations'][term]['is_critical']),
            'num_redundant_terms': sum(1 for term, _ in term_deltas
                                      if ablation_results['term_ablations'][term]['is_redundant'])
        }

    return ablation_results


def analyze_dominant_terms(diagnostic_data):
    """
    Identify if one or more terms are dominating the scoring.

    Parameters:
    -----------
    diagnostic_data : dict
        Full diagnostic data from search_with_full_diagnostics()

    Returns:
    --------
    dict: Analysis of term dominance including:
        - dominant_term: The most dominant term (if any)
        - dominance_score: Measure of dominance
        - is_dominated: Whether scoring is dominated by few terms
    """
    field_diag = diagnostic_data.get('field_diagnostics', {})
    body_diag = field_diag.get('body_diagnostics', {})
    term_stats = body_diag.get('term_stats', {})

    if not term_stats:
        return {
            'is_dominated': False,
            'dominant_term': None,
            'analysis': 'No terms found'
        }

    # Calculate dominance metrics
    term_metrics = []

    for term, stats in term_stats.items():
        if stats.get('in_vocabulary', False):
            term_metrics.append({
                'term': term,
                'num_matched_docs': stats.get('num_matched_docs', 0),
                'idf': stats.get('idf', 0.0),
                'df': stats.get('df', 0)
            })

    if not term_metrics:
        return {
            'is_dominated': False,
            'dominant_term': None,
            'analysis': 'No in-vocabulary terms'
        }

    # Sort by number of matched docs
    term_metrics_sorted = sorted(term_metrics, key=lambda x: x['num_matched_docs'], reverse=True)

    # Check for dominance
    max_matches = term_metrics_sorted[0]['num_matched_docs']
    avg_matches = np.mean([t['num_matched_docs'] for t in term_metrics])

    # Dominance ratio: max / mean
    dominance_ratio = max_matches / avg_matches if avg_matches > 0 else 0.0

    # Check IDF dominance (one term much more common)
    idf_values = [t['idf'] for t in term_metrics]
    min_idf = min(idf_values)
    avg_idf = np.mean(idf_values)

    idf_dominance_ratio = avg_idf / min_idf if min_idf > 0 else 0.0

    # Determine if dominated
    is_dominated = dominance_ratio > 2.0 or idf_dominance_ratio > 2.0

    return {
        'is_dominated': is_dominated,
        'dominant_term': term_metrics_sorted[0]['term'] if is_dominated else None,
        'dominance_ratio': float(dominance_ratio),
        'idf_dominance_ratio': float(idf_dominance_ratio),
        'term_rankings': term_metrics_sorted,
        'analysis': {
            'max_matches': max_matches,
            'avg_matches': float(avg_matches),
            'min_idf': float(min_idf),
            'avg_idf': float(avg_idf),
            'interpretation': _interpret_dominance(dominance_ratio, idf_dominance_ratio)
        }
    }


def _rank_relevant_docs(scores, relevant_docs):
    """
    Get ranking positions of relevant documents.

    Parameters:
    -----------
    scores : dict
        Document scores
    relevant_docs : list
        List of relevant doc IDs

    Returns:
    --------
    dict: doc_id -> rank (1-indexed)
    """
    # Sort all docs by score
    sorted_docs = sorted(scores.items(), key=lambda x: x[1], reverse=True)

    # Find rank of each relevant doc
    ranking = {}
    for rank, (doc_id, score) in enumerate(sorted_docs, start=1):
        if doc_id in relevant_docs:
            ranking[doc_id] = rank

    return ranking


def _compute_map_at_k(scores, relevant_docs, k=10):
    """
    Compute Mean Average Precision at K.

    Parameters:
    -----------
    scores : dict
        Document scores
    relevant_docs : list
        List of relevant doc IDs
    k : int
        Cutoff rank

    Returns:
    --------
    float: MAP@K score
    """
    if not relevant_docs:
        return 0.0

    # Sort docs by score
    sorted_docs = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    top_k = sorted_docs[:k]

    # Calculate average precision
    precisions = []
    num_relevant_seen = 0

    for rank, (doc_id, score) in enumerate(top_k, start=1):
        if doc_id in relevant_docs:
            num_relevant_seen += 1
            precision_at_rank = num_relevant_seen / rank
            precisions.append(precision_at_rank)

    if not precisions:
        return 0.0

    # Average precision
    return sum(precisions) / len(relevant_docs)


def _interpret_dominance(dominance_ratio, idf_dominance_ratio):
    """Interpret dominance ratios."""
    if dominance_ratio > 3.0 or idf_dominance_ratio > 3.0:
        return "Strong dominance: One term is overwhelming other terms in scoring"
    elif dominance_ratio > 2.0 or idf_dominance_ratio > 2.0:
        return "Moderate dominance: One term has disproportionate influence"
    else:
        return "Balanced: Terms contribute relatively equally"
