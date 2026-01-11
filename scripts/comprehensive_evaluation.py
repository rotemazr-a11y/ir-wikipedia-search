"""
Comprehensive Evaluation Script - Main Orchestrator

This script orchestrates the full diagnostic pipeline:
1. Load queries and indices
2. Run diagnostic search on all queries
3. Analyze query characteristics
4. Compute term effectiveness
5. Run ablation analysis (on selected queries)
6. Diagnose failure modes
7. Generate comprehensive report

Usage:
    python comprehensive_evaluation.py \
        --queries data/queries_train.json \
        --indices indices_mini \
        --output diagnostics_report.json

Author: IR Project Team
Date: 2025-12-29
"""

import sys
import json
import pickle
import argparse
from pathlib import Path
from datetime import datetime
from tqdm import tqdm

# Add paths
parent_dir = Path(__file__).parent.parent
backend_path = parent_dir / 'backend'
sys.path.insert(0, str(parent_dir))
sys.path.insert(0, str(backend_path))

# Import modules
from backend.inverted_index_gcp import InvertedIndex
from diagnostic_search import search_with_full_diagnostics
from query_analyzer import (
    analyze_query_characteristics,
    compare_successful_vs_failing_queries,
    analyze_zero_map_queries
)
from term_contribution_analyzer import (
    compute_term_effectiveness_scores,
    ablation_analysis,
    analyze_dominant_terms
)
from root_cause_diagnostics import diagnose_failure_modes


def load_indices(index_dir):
    """Load all indices and metadata."""
    print(f"\n{'='*70}")
    print(f"LOADING INDICES FROM: {index_dir}")
    print(f"{'='*70}")

    # Load inverted indices
    body_index = InvertedIndex.read_index(index_dir, 'body_index')
    print(f"✓ Body index: {len(body_index.df)} terms")

    title_index = InvertedIndex.read_index(index_dir, 'title_index')
    print(f"✓ Title index: {len(title_index.df)} terms")

    anchor_index = InvertedIndex.read_index(index_dir, 'anchor_index')
    print(f"✓ Anchor index: {len(anchor_index.df)} terms")

    # Load metadata
    metadata_path = Path(index_dir) / 'metadata.pkl'
    with open(metadata_path, 'rb') as f:
        metadata = pickle.load(f)
    print(f"✓ Metadata: {metadata['num_docs']} documents")

    # Load PageRank (optional)
    try:
        pagerank_path = Path(index_dir) / 'pagerank.pkl'
        with open(pagerank_path, 'rb') as f:
            pagerank = pickle.load(f)
        print(f"✓ PageRank: {len(pagerank)} scores")
    except FileNotFoundError:
        print("⚠ PageRank not found, using zeros")
        pagerank = {}

    # Load page views (optional)
    try:
        pageviews_path = Path(index_dir) / 'pageviews.pkl'
        with open(pageviews_path, 'rb') as f:
            pageviews = pickle.load(f)
        print(f"✓ Page views: {len(pageviews)} entries")
    except FileNotFoundError:
        print("⚠ Page views not found, using zeros")
        pageviews = {}

    print(f"{'='*70}\n")

    return {
        'body_index': body_index,
        'title_index': title_index,
        'anchor_index': anchor_index,
        'metadata': metadata,
        'pagerank': pagerank,
        'pageviews': pageviews
    }


def load_queries(queries_path):
    """Load queries and relevance judgments."""
    print(f"Loading queries from: {queries_path}")

    with open(queries_path, 'r') as f:
        queries_data = json.load(f)

    print(f"✓ Loaded {len(queries_data)} queries\n")
    return queries_data


def compute_map_at_k(retrieved_docs, relevant_docs, k=10):
    """Compute MAP@K for a single query."""
    if not relevant_docs:
        return 0.0

    top_k = retrieved_docs[:k]
    precisions = []
    num_relevant_seen = 0

    for rank, doc_info in enumerate(top_k, start=1):
        doc_id = doc_info['doc_id']
        if doc_id in relevant_docs:
            num_relevant_seen += 1
            precision_at_rank = num_relevant_seen / rank
            precisions.append(precision_at_rank)

    if not precisions:
        return 0.0

    return sum(precisions) / len(relevant_docs)


def run_comprehensive_evaluation(queries_data, indices, weights, output_path, run_ablation=False):
    """
    Main evaluation loop.

    Parameters:
    -----------
    queries_data : list
        List of query dictionaries with 'query' and 'relevant_docs'
    indices : dict
        Dictionary containing all indices and metadata
    weights : dict
        Field weights for scoring
    output_path : str
        Path to save diagnostic report
    run_ablation : bool
        Whether to run ablation analysis (expensive)

    Returns:
    --------
    dict: Comprehensive diagnostic report
    """
    print(f"\n{'='*70}")
    print(f"RUNNING COMPREHENSIVE EVALUATION")
    print(f"{'='*70}")
    print(f"Total queries: {len(queries_data)}")
    print(f"Run ablation: {run_ablation}")
    print(f"Weights: {weights}")
    print(f"{'='*70}\n")

    report = {
        'metadata': {
            'timestamp': datetime.now().isoformat(),
            'num_queries': len(queries_data),
            'weights': weights,
            'ablation_enabled': run_ablation
        },
        'query_diagnostics': [],
        'summary': {},
        'statistical_analysis': {},
        'recommendations': []
    }

    all_characteristics = []
    all_map_scores = []

    # Process each query
    for i, query_data in enumerate(tqdm(queries_data, desc="Processing queries")):
        query_text = query_data['query']
        relevant_docs = query_data.get('relevant_docs', [])

        print(f"\n[{i+1}/{len(queries_data)}] Processing: {query_text[:60]}...")

        # Run diagnostic search
        diagnostic_data = search_with_full_diagnostics(
            query=query_text,
            body_index=indices['body_index'],
            title_index=indices['title_index'],
            anchor_index=indices['anchor_index'],
            index_dir=str(Path(args.indices).resolve()),
            metadata=indices['metadata'],
            pagerank=indices['pagerank'],
            pageviews=indices['pageviews'],
            **weights,
            remove_stops=True,
            stem=True
        )

        # Compute evaluation metrics
        retrieved_docs = diagnostic_data.get('ranking', {}).get('top_100', [])
        map_score = compute_map_at_k(retrieved_docs, relevant_docs, k=10)
        all_map_scores.append(map_score)

        evaluation_result = {
            'map': map_score,
            'num_relevant': len(relevant_docs),
            'num_retrieved': len(retrieved_docs),
            'num_retrieved_relevant': sum(1 for doc in retrieved_docs[:10] if doc['doc_id'] in relevant_docs)
        }

        print(f"  MAP@10: {map_score:.4f}")

        # Analyze query characteristics
        characteristics = analyze_query_characteristics(
            query_text,
            diagnostic_data,
            evaluation_result
        )
        all_characteristics.append(characteristics)

        # Analyze dominant terms
        dominant_analysis = analyze_dominant_terms(diagnostic_data)

        # Diagnose failure modes
        diagnosis = diagnose_failure_modes(
            query_text,
            diagnostic_data,
            evaluation_result,
            characteristics
        )

        # Term effectiveness (for all terms)
        term_effectiveness = {}
        query_tokens = diagnostic_data.get('query_info', {}).get('tokens', [])
        unique_terms = list(set(query_tokens))

        for term in unique_terms:
            effectiveness = compute_term_effectiveness_scores(
                term,
                diagnostic_data,
                relevant_docs
            )
            term_effectiveness[term] = effectiveness

        # Ablation analysis (only for selected queries if enabled)
        ablation_results = None
        if run_ablation and (map_score == 0.0 or i < 5):  # Run on zero-MAP or first 5
            print(f"  Running ablation analysis...")
            ablation_results = ablation_analysis(
                query_tokens,
                indices['body_index'],
                indices['title_index'],
                indices['anchor_index'],
                str(Path(args.indices).resolve()),
                indices['metadata'],
                indices['pagerank'],
                indices['pageviews'],
                relevant_docs,
                weights
            )

        # Store query diagnostic
        query_diagnostic = {
            'query': query_text,
            'evaluation': evaluation_result,
            'characteristics': characteristics,
            'dominant_analysis': dominant_analysis,
            'term_effectiveness': term_effectiveness,
            'diagnosis': diagnosis,
            'ablation_results': ablation_results,
            'diagnostic_data': diagnostic_data  # Full diagnostic data
        }

        report['query_diagnostics'].append(query_diagnostic)

    # Overall summary
    report['summary'] = {
        'map_at_10': sum(all_map_scores) / len(all_map_scores) if all_map_scores else 0.0,
        'num_queries': len(queries_data),
        'num_zero_map': sum(1 for score in all_map_scores if score == 0.0),
        'best_map': max(all_map_scores) if all_map_scores else 0.0,
        'worst_map': min(all_map_scores) if all_map_scores else 0.0,
        'map_scores': all_map_scores
    }

    print(f"\n{'='*70}")
    print(f"OVERALL SUMMARY")
    print(f"{'='*70}")
    print(f"MAP@10: {report['summary']['map_at_10']:.4f}")
    print(f"Zero-MAP queries: {report['summary']['num_zero_map']}/{len(queries_data)}")
    print(f"Best MAP: {report['summary']['best_map']:.4f}")
    print(f"Worst MAP: {report['summary']['worst_map']:.4f}")
    print(f"{'='*70}\n")

    # Statistical analysis
    print("Running statistical analysis...")
    comparison = compare_successful_vs_failing_queries(all_characteristics, threshold=0.0)
    zero_map_analysis = analyze_zero_map_queries(all_characteristics)

    report['statistical_analysis'] = {
        'successful_vs_failing': comparison,
        'zero_map_analysis': zero_map_analysis
    }

    # Generate recommendations
    print("Generating recommendations...")
    recommendations = generate_recommendations(report)
    report['recommendations'] = recommendations

    # Save report
    print(f"\nSaving diagnostic report to: {output_path}")
    with open(output_path, 'w') as f:
        json.dump(report, f, indent=2, default=str)

    print(f"✓ Report saved successfully!\n")

    return report


def generate_recommendations(report):
    """
    Generate prioritized recommendations from diagnostic report.

    Parameters:
    -----------
    report : dict
        Complete diagnostic report

    Returns:
    --------
    list: Prioritized recommendations
    """
    recommendations = []

    # Extract insights from statistical analysis
    comparison = report['statistical_analysis']['successful_vs_failing']
    insights = comparison.get('insights', [])

    for insight in insights:
        if insight.get('actionable', False):
            recommendations.append({
                'priority': 'HIGH' if insight.get('effect_size', 0) > 0.5 else 'MEDIUM',
                'category': 'Statistical Pattern',
                'issue': insight['finding'],
                'fix': insight.get('recommendation', 'See statistical analysis'),
                'expected_impact': f"Effect size: {insight.get('effect_size', 0):.2f}"
            })

    # Extract common failure modes
    failure_mode_counts = {}
    for qd in report['query_diagnostics']:
        primary_issue = qd['diagnosis'].get('primary_issue')
        if primary_issue:
            failure_mode_counts[primary_issue] = failure_mode_counts.get(primary_issue, 0) + 1

    # Top 3 failure modes
    top_failures = sorted(failure_mode_counts.items(), key=lambda x: x[1], reverse=True)[:3]

    for failure_mode, count in top_failures:
        # Find example fixes
        example_fixes = []
        for qd in report['query_diagnostics']:
            if qd['diagnosis'].get('primary_issue') == failure_mode:
                fixes = qd['diagnosis'].get('recommended_fixes', [])
                example_fixes.extend(fixes)
                if len(example_fixes) >= 2:
                    break

        recommendations.append({
            'priority': 'HIGH',
            'category': 'Common Failure Mode',
            'issue': f"{failure_mode} affects {count} queries",
            'fix': example_fixes[0] if example_fixes else 'See individual query diagnostics',
            'expected_impact': f"May improve {count} queries"
        })

    # Sort by priority
    priority_order = {'HIGH': 0, 'MEDIUM': 1, 'LOW': 2}
    recommendations.sort(key=lambda x: priority_order.get(x['priority'], 3))

    return recommendations


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description='Comprehensive IR System Evaluation')
    parser.add_argument('--queries', type=str, required=True,
                       help='Path to queries JSON file')
    parser.add_argument('--indices', type=str, required=True,
                       help='Path to indices directory')
    parser.add_argument('--output', type=str, default='diagnostics_report.json',
                       help='Output path for diagnostic report')
    parser.add_argument('--ablation', action='store_true',
                       help='Enable ablation analysis (slower)')
    parser.add_argument('--w-body', type=float, default=0.40,
                       help='Body weight')
    parser.add_argument('--w-title', type=float, default=0.25,
                       help='Title weight')
    parser.add_argument('--w-anchor', type=float, default=0.15,
                       help='Anchor weight')
    parser.add_argument('--w-pagerank', type=float, default=0.15,
                       help='PageRank weight')
    parser.add_argument('--w-pageviews', type=float, default=0.05,
                       help='PageViews weight')

    global args
    args = parser.parse_args()

    # Weights
    weights = {
        'w_body': args.w_body,
        'w_title': args.w_title,
        'w_anchor': args.w_anchor,
        'w_pagerank': args.w_pagerank,
        'w_pageviews': args.w_pageviews
    }

    # Load data
    queries_data = load_queries(args.queries)
    indices = load_indices(args.indices)

    # Run evaluation
    report = run_comprehensive_evaluation(
        queries_data,
        indices,
        weights,
        args.output,
        run_ablation=args.ablation
    )

    # Print summary
    print("\n" + "="*70)
    print("EVALUATION COMPLETE")
    print("="*70)
    print(f"Report saved to: {args.output}")
    print(f"Total queries: {report['metadata']['num_queries']}")
    print(f"MAP@10: {report['summary']['map_at_10']:.4f}")
    print(f"Zero-MAP queries: {report['summary']['num_zero_map']}")
    print(f"\nTop recommendations:")
    for i, rec in enumerate(report['recommendations'][:5], 1):
        print(f"  {i}. [{rec['priority']}] {rec['issue']}")
        print(f"     → {rec['fix']}")
    print("="*70)


if __name__ == '__main__':
    main()
