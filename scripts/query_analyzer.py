"""
Query Analyzer - Statistical Analysis of Query Characteristics

This module provides functions to analyze query characteristics and compare
successful vs. failing queries to identify patterns.

Key Functions:
- analyze_query_characteristics(): Compute query-level statistics
- compare_successful_vs_failing_queries(): Statistical comparison
- compute_confidence_intervals(): Statistical significance testing

Author: IR Project Team
Date: 2025-12-29
"""

import numpy as np
from collections import defaultdict

# Try to import scipy, use simpler stats if not available
try:
    from scipy import stats
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False
    print("⚠ scipy not available. Using simplified statistical tests.")


def analyze_query_characteristics(query_text, diagnostic_data, evaluation_result=None):
    """
    Compute comprehensive query-level statistics.

    Parameters:
    -----------
    query_text : str
        Original query text
    diagnostic_data : dict
        Diagnostic data from search_with_full_diagnostics()
    evaluation_result : dict, optional
        Evaluation metrics (MAP, precision, etc.)

    Returns:
    --------
    dict: Query characteristics including:
        - length: Token counts
        - term_specificity: IDF statistics
        - coverage: Vocabulary coverage
        - coherence: Term relationship metrics
    """
    characteristics = {
        'query_text': query_text,
        'length': {},
        'term_specificity': {},
        'coverage': {},
        'coherence': {},
        'field_matching': {},
        'performance': {}
    }

    # Extract query info
    query_info = diagnostic_data.get('query_info', {})
    field_diag = diagnostic_data.get('field_diagnostics', {})
    body_diag = field_diag.get('body_diagnostics', {})
    term_stats = body_diag.get('term_stats', {})

    # 1. Length characteristics
    characteristics['length'] = {
        'original': len(query_info.get('tokens', [])),
        'after_stopwords': len(query_info.get('tokens', [])),
        'unique_terms': len(query_info.get('unique_terms', []))
    }

    # 2. Term specificity (IDF analysis)
    idf_values = []
    rare_terms = []  # High IDF (> 2.0)
    common_terms = []  # Low IDF (< 0.5)
    oov_terms = []

    for term, stats in term_stats.items():
        if stats.get('in_vocabulary', False):
            idf = stats['idf']
            idf_values.append(idf)

            if idf > 2.0:
                rare_terms.append({'term': term, 'idf': idf, 'df': stats['df']})
            elif idf < 0.5:
                common_terms.append({'term': term, 'idf': idf, 'df': stats['df']})
        else:
            oov_terms.append(term)

    if idf_values:
        characteristics['term_specificity'] = {
            'mean_idf': float(np.mean(idf_values)),
            'median_idf': float(np.median(idf_values)),
            'std_idf': float(np.std(idf_values)),
            'min_idf': float(np.min(idf_values)),
            'max_idf': float(np.max(idf_values)),
            'idf_variance': float(np.var(idf_values)),
            'rare_terms': rare_terms,  # IDF > 2.0
            'common_terms': common_terms,  # IDF < 0.5
            'num_rare': len(rare_terms),
            'num_common': len(common_terms)
        }
    else:
        characteristics['term_specificity'] = {
            'mean_idf': 0.0,
            'median_idf': 0.0,
            'std_idf': 0.0,
            'min_idf': 0.0,
            'max_idf': 0.0,
            'idf_variance': 0.0,
            'rare_terms': [],
            'common_terms': [],
            'num_rare': 0,
            'num_common': 0
        }

    # 3. Coverage (vocabulary matching)
    total_terms = len(term_stats)
    in_vocab_count = sum(1 for stats in term_stats.values() if stats.get('in_vocabulary', False))

    characteristics['coverage'] = {
        'total_terms': total_terms,
        'in_vocabulary': in_vocab_count,
        'oov_count': len(oov_terms),
        'oov_percentage': (len(oov_terms) / total_terms * 100) if total_terms > 0 else 0.0,
        'oov_terms': oov_terms,
        'matched_terms': in_vocab_count
    }

    # 4. Coherence (term relationship)
    # High coefficient of variation suggests multiple unrelated concepts
    if idf_values and len(idf_values) > 1:
        mean_idf = np.mean(idf_values)
        std_idf = np.std(idf_values)
        cv = (std_idf / mean_idf) if mean_idf > 0 else 0.0

        characteristics['coherence'] = {
            'idf_coefficient_of_variation': float(cv),
            'is_multi_concept': cv > 0.5,  # High variance = multiple concepts
            'is_coherent': cv < 0.3  # Low variance = coherent query
        }
    else:
        characteristics['coherence'] = {
            'idf_coefficient_of_variation': 0.0,
            'is_multi_concept': False,
            'is_coherent': True
        }

    # 5. Field matching patterns
    field_summary = field_diag.get('summary', {})
    characteristics['field_matching'] = {
        'num_body_matches': field_summary.get('num_body_matches', 0),
        'num_title_matches': field_summary.get('num_title_matches', 0),
        'num_anchor_matches': field_summary.get('num_anchor_matches', 0),
        'total_candidates': field_summary.get('num_candidates', 0),
        'has_title_match': field_summary.get('num_title_matches', 0) > 0,
        'has_anchor_match': field_summary.get('num_anchor_matches', 0) > 0
    }

    # 6. Performance metrics (if provided)
    if evaluation_result:
        characteristics['performance'] = {
            'map_score': evaluation_result.get('map', 0.0),
            'has_zero_map': evaluation_result.get('map', 0.0) == 0.0,
            'num_relevant_docs': evaluation_result.get('num_relevant', 0),
            'num_retrieved_relevant': evaluation_result.get('num_retrieved_relevant', 0)
        }

    return characteristics


def compare_successful_vs_failing_queries(all_characteristics, threshold=0.0):
    """
    Statistical comparison between successful and failing queries.

    Parameters:
    -----------
    all_characteristics : list of dict
        List of query characteristics from analyze_query_characteristics()
    threshold : float
        MAP threshold to distinguish success (default: 0.0 for zero vs. non-zero)

    Returns:
    --------
    dict: Statistical comparison with:
        - Group statistics (mean, std, median)
        - Statistical tests (t-test, Mann-Whitney U)
        - Effect sizes
        - Recommendations
    """
    # Separate into successful and failing groups
    successful = []
    failing = []

    for char in all_characteristics:
        map_score = char.get('performance', {}).get('map_score', 0.0)
        if map_score > threshold:
            successful.append(char)
        else:
            failing.append(char)

    comparison = {
        'threshold': threshold,
        'num_successful': len(successful),
        'num_failing': len(failing),
        'metrics': {}
    }

    # Compare various metrics
    metrics_to_compare = [
        ('query_length', lambda c: c['length']['unique_terms']),
        ('mean_idf', lambda c: c['term_specificity']['mean_idf']),
        ('idf_variance', lambda c: c['term_specificity']['idf_variance']),
        ('oov_percentage', lambda c: c['coverage']['oov_percentage']),
        ('num_rare_terms', lambda c: c['term_specificity']['num_rare']),
        ('num_common_terms', lambda c: c['term_specificity']['num_common']),
        ('idf_cv', lambda c: c['coherence']['idf_coefficient_of_variation']),
        ('body_matches', lambda c: c['field_matching']['num_body_matches']),
        ('title_matches', lambda c: c['field_matching']['num_title_matches']),
        ('anchor_matches', lambda c: c['field_matching']['num_anchor_matches'])
    ]

    for metric_name, extractor in metrics_to_compare:
        try:
            successful_values = [extractor(c) for c in successful]
            failing_values = [extractor(c) for c in failing]

            if not successful_values or not failing_values:
                continue

            # Descriptive statistics
            comparison['metrics'][metric_name] = {
                'successful': {
                    'mean': float(np.mean(successful_values)),
                    'median': float(np.median(successful_values)),
                    'std': float(np.std(successful_values)),
                    'min': float(np.min(successful_values)),
                    'max': float(np.max(successful_values))
                },
                'failing': {
                    'mean': float(np.mean(failing_values)),
                    'median': float(np.median(failing_values)),
                    'std': float(np.std(failing_values)),
                    'min': float(np.min(failing_values)),
                    'max': float(np.max(failing_values))
                }
            }

            # Statistical tests
            # T-test (assumes normality)
            if len(successful_values) > 1 and len(failing_values) > 1:
                if SCIPY_AVAILABLE:
                    t_stat, p_value = stats.ttest_ind(successful_values, failing_values)
                    comparison['metrics'][metric_name]['t_test'] = {
                        't_statistic': float(t_stat),
                        'p_value': float(p_value),
                        'significant': p_value < 0.05
                    }

                    # Mann-Whitney U test (non-parametric)
                    u_stat, u_p_value = stats.mannwhitneyu(successful_values, failing_values, alternative='two-sided')
                    comparison['metrics'][metric_name]['mann_whitney'] = {
                        'u_statistic': float(u_stat),
                        'p_value': float(u_p_value),
                        'significant': u_p_value < 0.05
                    }
                else:
                    # Simplified test: check if means differ by > 1 std
                    mean_diff = abs(np.mean(successful_values) - np.mean(failing_values))
                    pooled_std = np.sqrt((np.var(successful_values) + np.var(failing_values)) / 2)
                    comparison['metrics'][metric_name]['t_test'] = {
                        't_statistic': mean_diff / pooled_std if pooled_std > 0 else 0.0,
                        'p_value': 0.05 if mean_diff > pooled_std else 0.5,  # Rough approximation
                        'significant': mean_diff > pooled_std
                    }

                # Effect size (Cohen's d)
                pooled_std = np.sqrt(
                    ((len(successful_values) - 1) * np.var(successful_values) +
                     (len(failing_values) - 1) * np.var(failing_values)) /
                    (len(successful_values) + len(failing_values) - 2)
                )
                if pooled_std > 0:
                    cohens_d = (np.mean(successful_values) - np.mean(failing_values)) / pooled_std
                    comparison['metrics'][metric_name]['effect_size'] = {
                        'cohens_d': float(cohens_d),
                        'interpretation': _interpret_cohens_d(cohens_d)
                    }

        except Exception as e:
            comparison['metrics'][metric_name] = {'error': str(e)}

    # Generate insights and recommendations
    comparison['insights'] = _generate_insights(comparison)

    return comparison


def _interpret_cohens_d(d):
    """Interpret Cohen's d effect size."""
    abs_d = abs(d)
    if abs_d < 0.2:
        return 'negligible'
    elif abs_d < 0.5:
        return 'small'
    elif abs_d < 0.8:
        return 'medium'
    else:
        return 'large'


def _generate_insights(comparison):
    """Generate actionable insights from statistical comparison."""
    insights = []

    for metric_name, data in comparison['metrics'].items():
        if 'error' in data:
            continue

        # Check if difference is statistically significant
        t_test = data.get('t_test', {})
        mw_test = data.get('mann_whitney', {})
        effect = data.get('effect_size', {})

        if t_test.get('significant', False) or mw_test.get('significant', False):
            successful_mean = data['successful']['mean']
            failing_mean = data['failing']['mean']
            difference = successful_mean - failing_mean
            effect_size = effect.get('cohens_d', 0.0)

            insight = {
                'metric': metric_name,
                'finding': f"Successful queries have {'higher' if difference > 0 else 'lower'} {metric_name}",
                'successful_mean': successful_mean,
                'failing_mean': failing_mean,
                'difference': difference,
                'p_value': min(t_test.get('p_value', 1.0), mw_test.get('p_value', 1.0)),
                'effect_size': effect_size,
                'effect_interpretation': effect.get('interpretation', 'unknown'),
                'actionable': True
            }

            # Add specific recommendations
            if metric_name == 'mean_idf' and difference > 0:
                insight['recommendation'] = "Failing queries have lower-IDF terms. Consider boosting rare term weights."
            elif metric_name == 'num_common_terms' and difference < 0:
                insight['recommendation'] = "Failing queries have too many common terms. Consider increasing stopword filtering."
            elif metric_name == 'idf_cv' and difference < 0:
                insight['recommendation'] = "Failing queries have higher IDF variance (multi-concept). Consider query segmentation."
            elif metric_name == 'title_matches' and difference > 0:
                insight['recommendation'] = "Title matches correlate with success. Consider increasing title weight."
            elif metric_name == 'oov_percentage' and difference < 0:
                insight['recommendation'] = "Failing queries have more OOV terms. Check stemming/tokenization."

            insights.append(insight)

    return insights


def analyze_zero_map_queries(all_characteristics):
    """
    Specific analysis of queries with zero MAP.

    Parameters:
    -----------
    all_characteristics : list of dict
        List of query characteristics

    Returns:
    --------
    dict: Analysis of zero-MAP queries including common patterns
    """
    zero_map_queries = [
        c for c in all_characteristics
        if c.get('performance', {}).get('map_score', 0.0) == 0.0
    ]

    analysis = {
        'num_zero_map': len(zero_map_queries),
        'queries': [],
        'common_patterns': {}
    }

    # Analyze each zero-MAP query
    for char in zero_map_queries:
        query_analysis = {
            'query': char['query_text'],
            'unique_terms': char['length']['unique_terms'],
            'mean_idf': char['term_specificity']['mean_idf'],
            'oov_count': char['coverage']['oov_count'],
            'num_rare_terms': char['term_specificity']['num_rare'],
            'is_multi_concept': char['coherence']['is_multi_concept'],
            'body_matches': char['field_matching']['num_body_matches'],
            'title_matches': char['field_matching']['num_title_matches'],
            'num_relevant': char.get('performance', {}).get('num_relevant_docs', 0)
        }
        analysis['queries'].append(query_analysis)

    # Identify common patterns
    if zero_map_queries:
        analysis['common_patterns'] = {
            'avg_query_length': np.mean([q['unique_terms'] for q in analysis['queries']]),
            'avg_mean_idf': np.mean([q['mean_idf'] for q in analysis['queries']]),
            'avg_oov_count': np.mean([q['oov_count'] for q in analysis['queries']]),
            'pct_with_multi_concept': sum(1 for q in analysis['queries'] if q['is_multi_concept']) / len(zero_map_queries) * 100,
            'pct_with_no_title_match': sum(1 for q in analysis['queries'] if q['title_matches'] == 0) / len(zero_map_queries) * 100,
            'avg_body_matches': np.mean([q['body_matches'] for q in analysis['queries']]),
            'avg_relevant_docs': np.mean([q['num_relevant'] for q in analysis['queries']])
        }

    return analysis


def compute_confidence_intervals(values, confidence=0.95):
    """
    Compute confidence intervals for a list of values.

    Parameters:
    -----------
    values : list of float
        Numeric values
    confidence : float
        Confidence level (default: 0.95)

    Returns:
    --------
    dict: Mean, std, and confidence interval
    """
    if not values or len(values) < 2:
        return {
            'mean': 0.0,
            'std': 0.0,
            'ci_lower': 0.0,
            'ci_upper': 0.0,
            'confidence': confidence
        }

    mean = np.mean(values)
    std = np.std(values, ddof=1)
    n = len(values)

    # Standard error
    se = std / np.sqrt(n)

    # Critical value from t-distribution
    if SCIPY_AVAILABLE:
        t_critical = stats.t.ppf((1 + confidence) / 2, df=n - 1)
    else:
        # Use normal approximation (z-score) for large n, or conservative value for small n
        if n > 30:
            t_critical = 1.96 if confidence == 0.95 else 2.576  # z-scores for 95% and 99%
        else:
            t_critical = 2.5  # Conservative estimate

    # Confidence interval
    margin = t_critical * se

    return {
        'mean': float(mean),
        'std': float(std),
        'se': float(se),
        'ci_lower': float(mean - margin),
        'ci_upper': float(mean + margin),
        'confidence': confidence,
        'n': n
    }
