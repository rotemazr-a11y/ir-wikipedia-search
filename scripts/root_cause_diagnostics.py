"""
Root Cause Diagnostics - Automated Failure Mode Detection

This module provides automated diagnosis of query failure modes to identify
specific reasons why queries fail to retrieve relevant documents.

Failure Modes Detected:
1. OOV_TERMS: Query terms not in index
2. SEMANTIC_DILUTION: Multiple unrelated concepts diluting scores
3. DOMINANT_TERM: One common term overwhelming others
4. FIELD_MISMATCH: Terms in wrong field (e.g., title vs. body)
5. PAGERANK_BIAS: Popular docs displacing relevant ones
6. COMPOUND_PHRASE: Phrase should be treated as single unit

Author: IR Project Team
Date: 2025-12-29
"""

import numpy as np
from collections import Counter


class FailureMode:
    """Enumeration of failure modes."""
    OOV_TERMS = "OOV_TERMS"
    SEMANTIC_DILUTION = "SEMANTIC_DILUTION"
    DOMINANT_TERM = "DOMINANT_TERM"
    FIELD_MISMATCH = "FIELD_MISMATCH"
    PAGERANK_BIAS = "PAGERANK_BIAS"
    COMPOUND_PHRASE = "COMPOUND_PHRASE"
    LOW_TERM_SPECIFICITY = "LOW_TERM_SPECIFICITY"
    INSUFFICIENT_MATCHES = "INSUFFICIENT_MATCHES"


def diagnose_failure_modes(query, diagnostic_data, evaluation_result, query_characteristics=None):
    """
    Identify specific failure reasons for a query.

    Parameters:
    -----------
    query : str
        Original query text
    diagnostic_data : dict
        Full diagnostic data from search_with_full_diagnostics()
    evaluation_result : dict
        Evaluation metrics (MAP, relevant docs, etc.)
    query_characteristics : dict, optional
        Pre-computed query characteristics

    Returns:
    --------
    dict: Diagnosis including:
        - primary_issue: Main failure mode
        - contributing_factors: List of contributing issues
        - evidence: Supporting data
        - recommended_fixes: Actionable recommendations
    """
    diagnosis = {
        'query': query,
        'map_score': evaluation_result.get('map', 0.0),
        'primary_issue': None,
        'contributing_factors': [],
        'evidence': {},
        'recommended_fixes': [],
        'failure_modes_detected': []
    }

    # Run all diagnostic checks
    failure_modes = []

    # 1. Check for OOV terms
    oov_result = detect_oov_terms(diagnostic_data)
    if oov_result['is_issue']:
        failure_modes.append({
            'mode': FailureMode.OOV_TERMS,
            'severity': oov_result['severity'],
            'evidence': oov_result,
            'fixes': oov_result['recommendations']
        })

    # 2. Check for semantic dilution
    dilution_result = detect_semantic_dilution(diagnostic_data, query_characteristics)
    if dilution_result['is_issue']:
        failure_modes.append({
            'mode': FailureMode.SEMANTIC_DILUTION,
            'severity': dilution_result['severity'],
            'evidence': dilution_result,
            'fixes': dilution_result['recommendations']
        })

    # 3. Check for dominant term
    dominant_result = detect_dominant_term_problem(diagnostic_data)
    if dominant_result['is_issue']:
        failure_modes.append({
            'mode': FailureMode.DOMINANT_TERM,
            'severity': dominant_result['severity'],
            'evidence': dominant_result,
            'fixes': dominant_result['recommendations']
        })

    # 4. Check for field mismatch
    field_result = detect_field_mismatch(diagnostic_data, evaluation_result)
    if field_result['is_issue']:
        failure_modes.append({
            'mode': FailureMode.FIELD_MISMATCH,
            'severity': field_result['severity'],
            'evidence': field_result,
            'fixes': field_result['recommendations']
        })

    # 5. Check for PageRank bias
    pagerank_result = detect_pagerank_bias(diagnostic_data, evaluation_result)
    if pagerank_result['is_issue']:
        failure_modes.append({
            'mode': FailureMode.PAGERANK_BIAS,
            'severity': pagerank_result['severity'],
            'evidence': pagerank_result,
            'fixes': pagerank_result['recommendations']
        })

    # 6. Check for compound phrase issues
    compound_result = detect_compound_phrase_issue(query, diagnostic_data)
    if compound_result['is_issue']:
        failure_modes.append({
            'mode': FailureMode.COMPOUND_PHRASE,
            'severity': compound_result['severity'],
            'evidence': compound_result,
            'fixes': compound_result['recommendations']
        })

    # 7. Check for low term specificity
    specificity_result = detect_low_term_specificity(diagnostic_data)
    if specificity_result['is_issue']:
        failure_modes.append({
            'mode': FailureMode.LOW_TERM_SPECIFICITY,
            'severity': specificity_result['severity'],
            'evidence': specificity_result,
            'fixes': specificity_result['recommendations']
        })

    # 8. Check for insufficient matches
    matches_result = detect_insufficient_matches(diagnostic_data, evaluation_result)
    if matches_result['is_issue']:
        failure_modes.append({
            'mode': FailureMode.INSUFFICIENT_MATCHES,
            'severity': matches_result['severity'],
            'evidence': matches_result,
            'fixes': matches_result['recommendations']
        })

    # Sort by severity
    failure_modes_sorted = sorted(failure_modes, key=lambda x: x['severity'], reverse=True)

    if failure_modes_sorted:
        # Primary issue is highest severity
        primary = failure_modes_sorted[0]
        diagnosis['primary_issue'] = primary['mode']
        diagnosis['evidence'] = primary['evidence']
        diagnosis['recommended_fixes'] = primary['fixes']

        # Contributing factors are others
        diagnosis['contributing_factors'] = [
            fm['mode'] for fm in failure_modes_sorted[1:]
        ]

        diagnosis['failure_modes_detected'] = [
            {
                'mode': fm['mode'],
                'severity': fm['severity'],
                'fixes': fm['fixes']
            }
            for fm in failure_modes_sorted
        ]
    else:
        diagnosis['primary_issue'] = 'UNKNOWN'
        diagnosis['evidence'] = {'note': 'No specific failure mode detected'}
        diagnosis['recommended_fixes'] = ['Manual investigation required']

    return diagnosis


def detect_oov_terms(diagnostic_data):
    """
    Detect if query has out-of-vocabulary terms.

    Indicators:
    - One or more terms not in index
    - High percentage of OOV terms
    """
    field_diag = diagnostic_data.get('field_diagnostics', {})
    body_diag = field_diag.get('body_diagnostics', {})
    query_analysis = body_diag.get('query_analysis', {})

    oov_terms = query_analysis.get('oov_terms', [])
    total_terms = query_analysis.get('unique_term_count', 0)

    oov_count = len(oov_terms)
    oov_percentage = (oov_count / total_terms * 100) if total_terms > 0 else 0.0

    is_issue = oov_count > 0
    severity = min(oov_percentage / 10, 10)  # Scale 0-10

    result = {
        'is_issue': is_issue,
        'severity': severity,
        'oov_count': oov_count,
        'oov_terms': oov_terms,
        'oov_percentage': oov_percentage,
        'recommendations': []
    }

    if is_issue:
        result['recommendations'] = [
            f"Check stemming for OOV terms: {', '.join(oov_terms)}",
            "Consider query expansion or synonym matching",
            "Verify tokenization is consistent with index"
        ]

    return result


def detect_semantic_dilution(diagnostic_data, query_characteristics=None):
    """
    Detect if multiple unrelated concepts are diluting scores.

    Indicators:
    - High IDF variance (CV > 0.5)
    - Multiple terms with very different specificity
    - Low overlap in document sets per term
    """
    if query_characteristics:
        coherence = query_characteristics.get('coherence', {})
        idf_cv = coherence.get('idf_coefficient_of_variation', 0.0)
        is_multi_concept = coherence.get('is_multi_concept', False)
    else:
        # Compute from diagnostic data
        field_diag = diagnostic_data.get('field_diagnostics', {})
        body_diag = field_diag.get('body_diagnostics', {})
        term_stats = body_diag.get('term_stats', {})

        idf_values = [
            stats['idf'] for stats in term_stats.values()
            if stats.get('in_vocabulary', False)
        ]

        if len(idf_values) > 1:
            mean_idf = np.mean(idf_values)
            std_idf = np.std(idf_values)
            idf_cv = (std_idf / mean_idf) if mean_idf > 0 else 0.0
            is_multi_concept = idf_cv > 0.5
        else:
            idf_cv = 0.0
            is_multi_concept = False

    is_issue = is_multi_concept
    severity = min(idf_cv * 10, 10)  # Scale 0-10

    result = {
        'is_issue': is_issue,
        'severity': severity,
        'idf_cv': idf_cv,
        'is_multi_concept': is_multi_concept,
        'recommendations': []
    }

    if is_issue:
        result['recommendations'] = [
            "Query contains multiple unrelated concepts",
            "Consider splitting into separate queries",
            "Try boosting weights of rarer, more specific terms",
            "Use query segmentation or concept extraction"
        ]

    return result


def detect_dominant_term_problem(diagnostic_data):
    """
    Identify if one term is overwhelming others.

    Indicators:
    - One term matches >> more docs than others
    - One term's IDF << other terms' IDFs
    - Score distribution highly skewed toward one term
    """
    field_diag = diagnostic_data.get('field_diagnostics', {})
    body_diag = field_diag.get('body_diagnostics', {})
    term_stats = body_diag.get('term_stats', {})

    if not term_stats:
        return {'is_issue': False, 'severity': 0}

    # Get match counts per term
    match_counts = []
    idf_values = []
    terms_data = []

    for term, stats in term_stats.items():
        if stats.get('in_vocabulary', False):
            num_matches = stats.get('num_matched_docs', 0)
            idf = stats.get('idf', 0.0)
            match_counts.append(num_matches)
            idf_values.append(idf)
            terms_data.append({'term': term, 'matches': num_matches, 'idf': idf})

    if not match_counts:
        return {'is_issue': False, 'severity': 0}

    max_matches = max(match_counts)
    avg_matches = np.mean(match_counts)
    min_idf = min(idf_values)
    avg_idf = np.mean(idf_values)

    # Dominance ratios
    match_dominance = max_matches / avg_matches if avg_matches > 0 else 0.0
    idf_dominance = avg_idf / min_idf if min_idf > 0 else 0.0

    # Issue if one term dominates
    is_issue = match_dominance > 3.0 or idf_dominance > 3.0
    severity = min((match_dominance + idf_dominance) / 2, 10)

    # Find dominant term
    dominant_term = max(terms_data, key=lambda x: x['matches'])

    result = {
        'is_issue': is_issue,
        'severity': severity,
        'dominant_term': dominant_term['term'],
        'match_dominance_ratio': match_dominance,
        'idf_dominance_ratio': idf_dominance,
        'dominant_term_matches': dominant_term['matches'],
        'dominant_term_idf': dominant_term['idf'],
        'avg_matches': avg_matches,
        'recommendations': []
    }

    if is_issue:
        result['recommendations'] = [
            f"Term '{dominant_term['term']}' is too common and overwhelming other terms",
            f"Consider adding '{dominant_term['term']}' to stopword list",
            "Increase weight on rarer terms (higher IDF)",
            "Use sublinear TF scaling to reduce common term impact"
        ]

    return result


def detect_field_mismatch(diagnostic_data, evaluation_result):
    """
    Detect if terms are in the wrong field.

    Indicators:
    - Many title/anchor matches but few body matches
    - Relevant docs exist but not being retrieved
    - High field imbalance
    """
    field_diag = diagnostic_data.get('field_diagnostics', {})
    field_summary = field_diag.get('summary', {})

    body_matches = field_summary.get('num_body_matches', 0)
    title_matches = field_summary.get('num_title_matches', 0)
    anchor_matches = field_summary.get('num_anchor_matches', 0)

    num_relevant = evaluation_result.get('num_relevant', 0)
    map_score = evaluation_result.get('map', 0.0)

    # Check for field imbalance
    total_matches = body_matches + title_matches + anchor_matches
    if total_matches == 0:
        return {'is_issue': False, 'severity': 0}

    body_pct = body_matches / total_matches * 100
    title_pct = title_matches / total_matches * 100
    anchor_pct = anchor_matches / total_matches * 100

    # Issue if: low body matches but high title/anchor, and low MAP
    is_issue = (body_pct < 30 and (title_pct > 40 or anchor_pct > 40)) and map_score < 0.1

    # Severity based on imbalance
    imbalance = max(abs(body_pct - 40), abs(title_pct - 25), abs(anchor_pct - 15))
    severity = min(imbalance / 10, 10)

    result = {
        'is_issue': is_issue,
        'severity': severity,
        'body_matches': body_matches,
        'title_matches': title_matches,
        'anchor_matches': anchor_matches,
        'body_percentage': body_pct,
        'title_percentage': title_pct,
        'anchor_percentage': anchor_pct,
        'recommendations': []
    }

    if is_issue:
        if title_pct > body_pct:
            result['recommendations'].append(
                f"Many title matches ({title_matches}) but few body matches ({body_matches}). "
                f"Consider increasing title weight from 0.25 to 0.40"
            )
        if anchor_pct > body_pct:
            result['recommendations'].append(
                f"Many anchor matches ({anchor_matches}) but few body matches ({body_matches}). "
                f"Consider increasing anchor weight from 0.15 to 0.25"
            )

    return result


def detect_pagerank_bias(diagnostic_data, evaluation_result):
    """
    Detect if PageRank is displacing relevant documents.

    Indicators:
    - Relevant docs exist but not in top-K
    - High PageRank scores dominating
    - Low content similarity but high overall scores
    """
    field_diag = diagnostic_data.get('field_diagnostics', {})
    ranking = diagnostic_data.get('ranking', {})
    top_docs = ranking.get('top_100', [])[:10]

    map_score = evaluation_result.get('map', 0.0)
    num_relevant = evaluation_result.get('num_relevant', 0)

    # Check if PageRank contributing significantly
    doc_contributions = field_diag.get('document_contributions', {})

    pagerank_heavy_docs = 0
    for doc_id in [doc['doc_id'] for doc in top_docs]:
        contrib = doc_contributions.get(doc_id, {})
        field_pcts = contrib.get('field_percentages', {})
        pr_pct = field_pcts.get('pagerank', 0.0)

        if pr_pct > 40:  # PageRank contributes > 40%
            pagerank_heavy_docs += 1

    # Issue if: low MAP, relevant docs exist, and PageRank dominating
    is_issue = (map_score < 0.1 and num_relevant > 10 and pagerank_heavy_docs > 5)

    severity = min(pagerank_heavy_docs * 2, 10)

    result = {
        'is_issue': is_issue,
        'severity': severity,
        'pagerank_heavy_docs': pagerank_heavy_docs,
        'recommendations': []
    }

    if is_issue:
        result['recommendations'] = [
            f"{pagerank_heavy_docs} of top-10 docs are PageRank-heavy",
            "Consider reducing PageRank weight from 0.15 to 0.10",
            "Increase body/title weights to prioritize content relevance",
            "Popular documents may be displacing relevant but less-popular ones"
        ]

    return result


def detect_compound_phrase_issue(query, diagnostic_data):
    """
    Detect if query contains compound phrases that should be treated as units.

    Indicators:
    - Query has capitalized multi-word terms (e.g., "Mount Everest")
    - Adjacent terms with high co-occurrence
    - Named entities or proper nouns
    """
    query_info = diagnostic_data.get('query_info', {})
    original_query = query_info.get('original_query', query)

    # Simple heuristic: check for capitalized words
    words = original_query.split()
    capitalized_sequences = []

    i = 0
    while i < len(words):
        if words[i][0].isupper():
            # Start of capitalized sequence
            sequence = [words[i]]
            j = i + 1
            while j < len(words) and words[j][0].isupper():
                sequence.append(words[j])
                j += 1

            if len(sequence) >= 2:
                capitalized_sequences.append(' '.join(sequence))

            i = j
        else:
            i += 1

    is_issue = len(capitalized_sequences) > 0
    severity = min(len(capitalized_sequences) * 3, 10)

    result = {
        'is_issue': is_issue,
        'severity': severity,
        'compound_phrases': capitalized_sequences,
        'recommendations': []
    }

    if is_issue:
        result['recommendations'] = [
            f"Query contains compound phrases: {', '.join(capitalized_sequences)}",
            "Consider treating these as bigrams or phrases",
            "Use proximity scoring to boost documents with adjacent terms",
            "Implement named entity recognition for proper handling"
        ]

    return result


def detect_low_term_specificity(diagnostic_data):
    """
    Detect if query terms are too common (low IDF).

    Indicators:
    - All terms have low IDF (< 1.0)
    - Many very common terms
    """
    field_diag = diagnostic_data.get('field_diagnostics', {})
    body_diag = field_diag.get('body_diagnostics', {})
    term_stats = body_diag.get('term_stats', {})

    idf_values = [
        stats['idf'] for stats in term_stats.values()
        if stats.get('in_vocabulary', False)
    ]

    if not idf_values:
        return {'is_issue': False, 'severity': 0}

    mean_idf = np.mean(idf_values)
    low_idf_count = sum(1 for idf in idf_values if idf < 1.0)

    is_issue = mean_idf < 1.0 and low_idf_count == len(idf_values)
    severity = min((1.0 - mean_idf) * 10, 10)

    result = {
        'is_issue': is_issue,
        'severity': severity,
        'mean_idf': mean_idf,
        'low_idf_count': low_idf_count,
        'total_terms': len(idf_values),
        'recommendations': []
    }

    if is_issue:
        result['recommendations'] = [
            f"All query terms are very common (mean IDF: {mean_idf:.2f})",
            "Add more specific, rare terms to the query",
            "Consider query expansion with synonyms or related terms",
            "Increase minimum IDF threshold for term selection"
        ]

    return result


def detect_insufficient_matches(diagnostic_data, evaluation_result):
    """
    Detect if query is retrieving too few documents.

    Indicators:
    - Very few candidates retrieved
    - Many relevant docs exist but not matched
    """
    field_diag = diagnostic_data.get('field_diagnostics', {})
    field_summary = field_diag.get('summary', {})

    num_candidates = field_summary.get('num_candidates', 0)
    num_relevant = evaluation_result.get('num_relevant', 0)
    map_score = evaluation_result.get('map', 0.0)

    # Issue if: few candidates but many relevant docs exist
    is_issue = num_candidates < 10 and num_relevant > 20 and map_score < 0.1

    severity = min((num_relevant - num_candidates) / 5, 10)

    result = {
        'is_issue': is_issue,
        'severity': severity,
        'num_candidates': num_candidates,
        'num_relevant': num_relevant,
        'recommendations': []
    }

    if is_issue:
        result['recommendations'] = [
            f"Only {num_candidates} candidates retrieved, but {num_relevant} relevant docs exist",
            "Query may be too restrictive or specific",
            "Consider query relaxation or expansion",
            "Check if term combinations are too narrow"
        ]

    return result
