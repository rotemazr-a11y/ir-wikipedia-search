"""
Visualization Generator - Diagnostic Plot Generation

This module generates visualizations from diagnostic reports.
Requires matplotlib and seaborn (optional).

Key Visualizations:
1. Term contribution heatmap
2. IDF distribution comparison
3. Query length vs. MAP scatter
4. Field contribution stacked bars
5. Ablation impact tornado chart

Usage:
    python visualization_generator.py --report diagnostics_report.json --output visualizations/

Author: IR Project Team
Date: 2025-12-29
"""

import json
import argparse
from pathlib import Path
import numpy as np

# Try to import plotting libraries
try:
    import matplotlib.pyplot as plt
    import matplotlib
    matplotlib.use('Agg')  # Non-interactive backend
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False
    print("⚠ matplotlib not available. Visualizations will be skipped.")

try:
    import seaborn as sns
    sns.set_style("whitegrid")
    SEABORN_AVAILABLE = True
except ImportError:
    SEABORN_AVAILABLE = False
    print("⚠ seaborn not available. Using matplotlib defaults.")


def plot_term_contribution_heatmap(report, output_path):
    """
    Generate term contribution heatmap.

    Rows: Queries
    Columns: Terms
    Color: Average contribution to final score
    """
    if not MATPLOTLIB_AVAILABLE:
        print("⚠ Skipping heatmap (matplotlib not available)")
        return

    print("Generating term contribution heatmap...")

    # Extract term contributions from all queries
    query_names = []
    term_contributions = {}

    for qd in report['query_diagnostics'][:20]:  # Limit to first 20 for readability
        query = qd['query'][:30]  # Truncate for display
        query_names.append(query)

        # Get term effectiveness
        for term, effectiveness in qd.get('term_effectiveness', {}).items():
            if term not in term_contributions:
                term_contributions[term] = []
            avg_contrib = effectiveness.get('avg_contribution', 0.0)
            term_contributions[term].append(avg_contrib)

    # Create matrix
    terms = list(term_contributions.keys())[:15]  # Top 15 terms
    matrix = []

    for i, query in enumerate(query_names):
        row = []
        for term in terms:
            values = term_contributions.get(term, [])
            value = values[i] if i < len(values) else 0.0
            row.append(value)
        matrix.append(row)

    matrix = np.array(matrix)

    # Plot
    fig, ax = plt.subplots(figsize=(12, 8))

    if SEABORN_AVAILABLE:
        sns.heatmap(matrix, xticklabels=terms, yticklabels=query_names,
                   cmap='YlOrRd', annot=False, fmt='.2f', ax=ax)
    else:
        im = ax.imshow(matrix, cmap='YlOrRd', aspect='auto')
        ax.set_xticks(np.arange(len(terms)))
        ax.set_yticks(np.arange(len(query_names)))
        ax.set_xticklabels(terms, rotation=45, ha='right')
        ax.set_yticklabels(query_names)
        plt.colorbar(im, ax=ax)

    ax.set_title('Term Contribution to Score Across Queries', fontsize=14, fontweight='bold')
    ax.set_xlabel('Terms', fontsize=12)
    ax.set_ylabel('Queries', fontsize=12)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()

    print(f"✓ Saved: {output_path}")


def plot_idf_distribution_comparison(report, output_path):
    """
    Compare IDF distributions between successful and failing queries.
    """
    if not MATPLOTLIB_AVAILABLE:
        print("⚠ Skipping IDF distribution (matplotlib not available)")
        return

    print("Generating IDF distribution comparison...")

    successful_idfs = []
    failing_idfs = []

    for qd in report['query_diagnostics']:
        map_score = qd['evaluation']['map']
        characteristics = qd['characteristics']
        mean_idf = characteristics['term_specificity'].get('mean_idf', 0.0)

        if map_score > 0.0:
            successful_idfs.append(mean_idf)
        else:
            failing_idfs.append(mean_idf)

    # Plot
    fig, ax = plt.subplots(figsize=(10, 6))

    if successful_idfs and failing_idfs:
        data = [successful_idfs, failing_idfs]
        labels = [f'Successful\n(n={len(successful_idfs)})', f'Failing\n(n={len(failing_idfs)})']

        bp = ax.boxplot(data, labels=labels, patch_artist=True)
        bp['boxes'][0].set_facecolor('lightgreen')
        bp['boxes'][1].set_facecolor('lightcoral')

        ax.set_ylabel('Mean IDF', fontsize=12)
        ax.set_title('IDF Distribution: Successful vs. Failing Queries', fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()

    print(f"✓ Saved: {output_path}")


def plot_query_length_vs_map(report, output_path):
    """
    Scatter plot: Query length vs. MAP@10
    """
    if not MATPLOTLIB_AVAILABLE:
        print("⚠ Skipping scatter plot (matplotlib not available)")
        return

    print("Generating query length vs. MAP scatter plot...")

    query_lengths = []
    map_scores = []
    has_zero_map = []

    for qd in report['query_diagnostics']:
        length = qd['characteristics']['length']['unique_terms']
        map_score = qd['evaluation']['map']

        query_lengths.append(length)
        map_scores.append(map_score)
        has_zero_map.append(map_score == 0.0)

    # Plot
    fig, ax = plt.subplots(figsize=(10, 6))

    colors = ['red' if zero else 'blue' for zero in has_zero_map]
    ax.scatter(query_lengths, map_scores, c=colors, alpha=0.6, s=100)

    ax.set_xlabel('Query Length (Unique Terms)', fontsize=12)
    ax.set_ylabel('MAP@10', fontsize=12)
    ax.set_title('Query Length vs. MAP@10', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)

    # Legend
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor='blue', label='MAP > 0'),
        Patch(facecolor='red', label='MAP = 0')
    ]
    ax.legend(handles=legend_elements)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()

    print(f"✓ Saved: {output_path}")


def plot_field_contribution_analysis(report, output_path):
    """
    Stacked bar chart: Field contributions for successful vs. failing queries
    """
    if not MATPLOTLIB_AVAILABLE:
        print("⚠ Skipping field contribution plot (matplotlib not available)")
        return

    print("Generating field contribution analysis...")

    # Aggregate field contributions
    successful_fields = {'body': [], 'title': [], 'anchor': [], 'pagerank': [], 'pageviews': []}
    failing_fields = {'body': [], 'title': [], 'anchor': [], 'pagerank': [], 'pageviews': []}

    for qd in report['query_diagnostics']:
        map_score = qd['evaluation']['map']

        # Get average field percentages from top-10 docs
        doc_contributions = qd['diagnostic_data']['field_diagnostics'].get('document_contributions', {})
        top_docs = qd['diagnostic_data']['ranking']['top_100'][:10]

        field_pcts = {'body': [], 'title': [], 'anchor': [], 'pagerank': [], 'pageviews': []}
        for doc_info in top_docs:
            doc_id = doc_info['doc_id']
            if doc_id in doc_contributions:
                pcts = doc_contributions[doc_id].get('field_percentages', {})
                for field in field_pcts:
                    field_pcts[field].append(pcts.get(field, 0.0))

        # Average
        avg_pcts = {field: np.mean(values) if values else 0.0 for field, values in field_pcts.items()}

        if map_score > 0.0:
            for field, pct in avg_pcts.items():
                successful_fields[field].append(pct)
        else:
            for field, pct in avg_pcts.items():
                failing_fields[field].append(pct)

    # Compute means
    successful_means = {field: np.mean(values) if values else 0.0 for field, values in successful_fields.items()}
    failing_means = {field: np.mean(values) if values else 0.0 for field, values in failing_fields.items()}

    # Plot
    fig, ax = plt.subplots(figsize=(10, 6))

    fields = ['body', 'title', 'anchor', 'pagerank', 'pageviews']
    x = np.arange(2)
    width = 0.6

    bottom_success = np.zeros(1)
    bottom_fail = np.zeros(1)

    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']

    for i, field in enumerate(fields):
        ax.bar(0, successful_means[field], width, bottom=bottom_success, label=field, color=colors[i])
        ax.bar(1, failing_means[field], width, bottom=bottom_fail, color=colors[i])

        bottom_success += successful_means[field]
        bottom_fail += failing_means[field]

    ax.set_ylabel('Contribution (%)', fontsize=12)
    ax.set_title('Field Contributions: Successful vs. Failing Queries', fontsize=14, fontweight='bold')
    ax.set_xticks([0, 1])
    ax.set_xticklabels(['Successful\n(MAP > 0)', 'Failing\n(MAP = 0)'])
    ax.legend(title='Field', bbox_to_anchor=(1.05, 1), loc='upper left')

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()

    print(f"✓ Saved: {output_path}")


def plot_ablation_impact(query_name, ablation_data, output_path):
    """
    Tornado chart: Impact of removing each term (ablation analysis)
    """
    if not MATPLOTLIB_AVAILABLE:
        print("⚠ Skipping ablation chart (matplotlib not available)")
        return

    print(f"Generating ablation impact chart for: {query_name}...")

    term_ablations = ablation_data.get('term_ablations', {})
    if not term_ablations:
        print("⚠ No ablation data available")
        return

    # Extract term impacts
    terms = []
    map_deltas = []

    for term, data in term_ablations.items():
        if 'error' not in data:
            terms.append(term)
            map_deltas.append(data['map_delta'])

    # Sort by impact (most negative first)
    sorted_data = sorted(zip(terms, map_deltas), key=lambda x: x[1])
    terms, map_deltas = zip(*sorted_data)

    # Plot
    fig, ax = plt.subplots(figsize=(10, max(6, len(terms) * 0.4)))

    colors = ['red' if delta < 0 else 'green' for delta in map_deltas]
    ax.barh(range(len(terms)), map_deltas, color=colors, alpha=0.7)

    ax.set_yticks(range(len(terms)))
    ax.set_yticklabels(terms)
    ax.set_xlabel('MAP Delta (when term removed)', fontsize=12)
    ax.set_title(f'Ablation Impact: {query_name}', fontsize=14, fontweight='bold')
    ax.axvline(x=0, color='black', linestyle='--', linewidth=1)
    ax.grid(True, alpha=0.3, axis='x')

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()

    print(f"✓ Saved: {output_path}")


def generate_all_visualizations(report_path, output_dir):
    """
    Generate all visualizations from diagnostic report.

    Parameters:
    -----------
    report_path : str
        Path to diagnostic report JSON
    output_dir : str
        Directory to save visualizations
    """
    if not MATPLOTLIB_AVAILABLE:
        print("\n⚠ matplotlib not installed. Cannot generate visualizations.")
        print("Install with: pip install matplotlib seaborn")
        return

    print(f"\n{'='*70}")
    print("GENERATING VISUALIZATIONS")
    print(f"{'='*70}\n")

    # Load report
    with open(report_path, 'r') as f:
        report = json.load(f)

    # Create output directory
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Generate visualizations
    try:
        plot_term_contribution_heatmap(
            report,
            output_dir / 'term_contribution_heatmap.png'
        )
    except Exception as e:
        print(f"⚠ Error generating heatmap: {e}")

    try:
        plot_idf_distribution_comparison(
            report,
            output_dir / 'idf_distribution_comparison.png'
        )
    except Exception as e:
        print(f"⚠ Error generating IDF distribution: {e}")

    try:
        plot_query_length_vs_map(
            report,
            output_dir / 'query_length_vs_map.png'
        )
    except Exception as e:
        print(f"⚠ Error generating scatter plot: {e}")

    try:
        plot_field_contribution_analysis(
            report,
            output_dir / 'field_contribution_analysis.png'
        )
    except Exception as e:
        print(f"⚠ Error generating field contribution plot: {e}")

    # Ablation charts for zero-MAP queries
    for i, qd in enumerate(report['query_diagnostics']):
        if qd['evaluation']['map'] == 0.0 and qd.get('ablation_results'):
            try:
                query_name = qd['query'][:30]
                plot_ablation_impact(
                    query_name,
                    qd['ablation_results'],
                    output_dir / f'ablation_impact_query_{i+1}.png'
                )
            except Exception as e:
                print(f"⚠ Error generating ablation chart for query {i+1}: {e}")

    print(f"\n{'='*70}")
    print(f"Visualizations saved to: {output_dir}")
    print(f"{'='*70}\n")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description='Generate diagnostic visualizations')
    parser.add_argument('--report', type=str, required=True,
                       help='Path to diagnostic report JSON')
    parser.add_argument('--output', type=str, default='visualizations',
                       help='Output directory for visualizations')

    args = parser.parse_args()

    generate_all_visualizations(args.report, args.output)


if __name__ == '__main__':
    main()
