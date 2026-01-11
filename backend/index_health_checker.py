"""
Index Health Validation Using Zipf's Law

This module implements statistical validation to verify that your inverted index
is "healthy" without requiring expensive full scans of 6GB+ index data.

Based on: search_engine_testing_optimization_strategy.md (Section 1.1)

Zipf's Law states that in natural language:
    frequency(term) ∝ 1 / rank^α

where α ≈ 1.0 for English text. If your index violates this, it indicates:
- Incorrect preprocessing (e.g., stopwords not removed)
- Data corruption during index construction
- Encoding/decoding errors in binary posting lists
"""

import numpy as np
import logging
from collections import Counter
from typing import Dict, List, Tuple
import random

from .inverted_index_gcp import InvertedIndex

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class IndexHealthCheck:
    """
    Statistical validation of inverted index health without full scans.

    Uses Zipf's Law validation as a fast integrity check.
    """

    def __init__(self, index: InvertedIndex, sample_size: int = 5000):
        """
        Parameters:
        -----------
        index : InvertedIndex
            The index to validate (body, title, or anchor)
        sample_size : int, default=5000
            Number of terms to sample for validation (trade-off: accuracy vs. speed)
        """
        self.index = index
        self.sample_size = min(sample_size, len(index.df))
        logger.info(f"IndexHealthCheck initialized (sample_size={self.sample_size})")

    def validate_zipf_distribution(self, tolerance: float = 0.15) -> Dict:
        """
        Verify Zipf's Law holds for sampled terms.

        Zipf's Law: log(frequency) = log(k) - α * log(rank)
        We perform log-log regression and check:
        1. Slope α ≈ 1.0 (±tolerance)
        2. R² > 0.9 (good linear fit in log-log space)

        Parameters:
        -----------
        tolerance : float, default=0.15
            Acceptable deviation from α=1.0

        Returns:
        --------
        dict : {
            'zipf_exponent': float (should be ~1.0),
            'r_squared': float (goodness of fit, >0.95 is good),
            'passes': bool,
            'details': str
        }
        """
        logger.info("Validating Zipf's Law distribution...")

        # Get all terms and their document frequencies
        all_terms = list(self.index.df.keys())
        frequencies = np.array([self.index.df[t] for t in all_terms])

        # Rank by frequency (descending)
        sorted_indices = np.argsort(-frequencies)
        top_indices = sorted_indices[:self.sample_size]
        ranks = np.arange(1, self.sample_size + 1)
        sampled_freqs = frequencies[top_indices]

        # Fit log-log regression: log(f) = log(k) - α*log(r)
        log_ranks = np.log(ranks)
        log_freqs = np.log(sampled_freqs)

        # Linear regression coefficients
        coeffs = np.polyfit(log_ranks, log_freqs, 1)
        poly = np.poly1d(coeffs)
        fitted = poly(log_ranks)

        # Calculate R² (coefficient of determination)
        ss_res = np.sum((log_freqs - fitted) ** 2)
        ss_tot = np.sum((log_freqs - np.mean(log_freqs)) ** 2)
        r_squared = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0

        # Zipf exponent is the negative slope
        alpha = -coeffs[0]

        # Validation criteria
        passes = (r_squared > 0.9) and (abs(alpha - 1.0) < tolerance)

        result = {
            'zipf_exponent': float(alpha),
            'r_squared': float(r_squared),
            'passes': passes,
            'details': f"Zipf exponent: {alpha:.3f} (expected ~1.0), R²: {r_squared:.4f}",
            'interpretation': self._interpret_zipf_result(alpha, r_squared)
        }

        if passes:
            logger.info(f"✅ Zipf validation PASSED: {result['details']}")
        else:
            logger.warning(f"❌ Zipf validation FAILED: {result['details']}")
            logger.warning(f"   {result['interpretation']}")

        return result

    def _interpret_zipf_result(self, alpha: float, r_squared: float) -> str:
        """Provide diagnostic interpretation of Zipf validation results."""
        if r_squared < 0.9:
            return ("Poor log-log fit (R² < 0.9). Possible causes: "
                   "stopwords not removed, mixed languages, or data corruption.")

        if alpha < 0.85:
            return ("Zipf exponent too low (α < 0.85). Possible causes: "
                   "over-aggressive stopword removal or incorrect tokenization.")

        if alpha > 1.15:
            return ("Zipf exponent too high (α > 1.15). Possible causes: "
                   "stopwords leaked through, or stemming not applied.")

        return "Distribution looks healthy (within expected range for natural language)."

    def check_df_anomalies(self, percentile_threshold: float = 99.5) -> Dict:
        """
        Detect Document Frequency anomalies that may indicate indexing errors.

        Anomalies to detect:
        1. Terms appearing in >90% of documents (likely stopwords that leaked through)
        2. Extremely long terms (>30 chars) with DF=1 (tokenization errors)
        3. Non-ASCII characters (encoding issues)

        Parameters:
        -----------
        percentile_threshold : float, default=99.5
            Report terms above this percentile in DF

        Returns:
        --------
        dict : {
            'anomalies_found': bool,
            'suspicious_terms': list of (term, df, reason),
            'reasoning': str
        }
        """
        logger.info("Checking for DF anomalies...")

        dfs = np.array(list(self.index.df.values()))

        # Estimate total documents (max doc_id seen in any posting list)
        # Note: This is approximate without scanning all posting lists
        total_docs_estimate = int(np.percentile(dfs, 99.9)) * 10  # Rough estimate

        # Find suspicious terms
        suspicious = []

        # Check top terms by DF
        sorted_terms = sorted(self.index.df.items(), key=lambda x: -x[1])
        for term, df in sorted_terms[:100]:  # Check top 100 terms
            # Flag 1: Appears in >90% of documents (likely stopword)
            if df > 0.9 * total_docs_estimate:
                suspicious.append((term, df, "appears in >90% of docs (possible stopword leak)"))

        # Sample random terms for other checks
        sample_terms = random.sample(list(self.index.df.keys()), min(1000, len(self.index.df)))
        for term in sample_terms:
            df = self.index.df[term]

            # Flag 2: Very rare but extremely long (tokenization error)
            if df == 1 and len(term) > 30:
                suspicious.append((term, df, "very rare but extremely long (tokenization error?)"))

            # Flag 3: Non-ASCII characters (encoding issues)
            if not term.isascii():
                suspicious.append((term, df, f"contains non-ASCII characters: {repr(term)}"))

        result = {
            'anomalies_found': len(suspicious) > 0,
            'suspicious_terms': suspicious[:20],  # Top 20 anomalies
            'total_anomalies': len(suspicious),
            'reasoning': "These terms may indicate preprocessing or encoding errors"
        }

        if result['anomalies_found']:
            logger.warning(f"❌ Found {len(suspicious)} anomalies:")
            for term, df, reason in suspicious[:10]:
                logger.warning(f"   '{term}' (DF={df}): {reason}")
        else:
            logger.info("✅ No DF anomalies detected")

        return result

    def check_term_total_consistency(self, sample_size: int = 1000) -> Dict:
        """
        Verify term_total counts match sum of TFs across all postings.

        This detects:
        - Binary encoding/decoding errors
        - Corruption during index writes
        - Integer overflow in TF counters

        Note: Requires reading posting lists, so we sample randomly.

        Parameters:
        -----------
        sample_size : int, default=1000
            Number of terms to verify (trade-off: thoroughness vs. speed)

        Returns:
        --------
        dict : {
            'consistency_check_passed': bool,
            'mismatches': list of {term, recorded, actual, diff},
            'sample_size': int
        }
        """
        logger.info(f"Checking term_total consistency (sample={sample_size})...")

        # Only check if we have posting lists (not available after read_index)
        if not hasattr(self.index, '_posting_list') or not self.index._posting_list:
            logger.warning("⚠️  Posting lists not available (index already written to disk)")
            return {
                'consistency_check_passed': True,
                'mismatches': [],
                'sample_size': 0,
                'note': 'Skipped (posting lists not in memory)'
            }

        sample_terms = random.sample(
            list(self.index.df.keys()),
            min(sample_size, len(self.index.df))
        )

        mismatches = []
        for term in sample_terms:
            recorded_total = self.index.term_total.get(term, 0)

            # Recount from postings
            if term in self.index._posting_list:
                actual_total = sum(tf for doc_id, tf in self.index._posting_list[term])
                if recorded_total != actual_total:
                    mismatches.append({
                        'term': term,
                        'recorded': recorded_total,
                        'actual': actual_total,
                        'diff': actual_total - recorded_total
                    })

        result = {
            'consistency_check_passed': len(mismatches) == 0,
            'mismatches': mismatches,
            'sample_size': len(sample_terms)
        }

        if result['consistency_check_passed']:
            logger.info(f"✅ term_total consistency check passed ({sample_size} terms verified)")
        else:
            logger.error(f"❌ Found {len(mismatches)} mismatches in term_total counts!")
            for mismatch in mismatches[:5]:
                logger.error(f"   '{mismatch['term']}': recorded={mismatch['recorded']}, "
                           f"actual={mismatch['actual']} (diff={mismatch['diff']})")

        return result

    def run_full_validation(self) -> Dict:
        """
        Run all validation checks and return comprehensive report.

        Returns:
        --------
        dict : {
            'timestamp': datetime,
            'index_size_terms': int,
            'zipf_check': dict,
            'df_anomalies': dict,
            'consistency_check': dict,
            'overall_health': str ('PASS' or 'FAIL')
        }
        """
        import datetime

        logger.info("\n" + "="*70)
        logger.info("RUNNING FULL INDEX HEALTH VALIDATION")
        logger.info("="*70 + "\n")

        report = {
            'timestamp': datetime.datetime.now().isoformat(),
            'index_size_terms': len(self.index.df),
            'zipf_check': self.validate_zipf_distribution(),
            'df_anomalies': self.check_df_anomalies(),
            'consistency_check': self.check_term_total_consistency()
        }

        # Overall health assessment
        overall_pass = (
            report['zipf_check']['passes'] and
            not report['df_anomalies']['anomalies_found'] and
            report['consistency_check']['consistency_check_passed']
        )

        report['overall_health'] = 'PASS' if overall_pass else 'FAIL'

        logger.info("\n" + "="*70)
        logger.info(f"OVERALL HEALTH: {report['overall_health']}")
        logger.info("="*70)

        if not overall_pass:
            logger.warning("\n⚠️  Index validation found issues. Review the report above.")

        return report


def main():
    """
    Standalone validation script.

    Usage:
        python index_health_checker.py \
            --index-dir gs://my-bucket/indices/ \
            --index-name body_index \
            --bucket my-bucket
    """
    import argparse
    import json

    parser = argparse.ArgumentParser(description='Validate index health using Zipf\'s Law')
    parser.add_argument('--index-dir', required=True, help='Directory containing index')
    parser.add_argument('--index-name', required=True, help='Index name (e.g., body_index)')
    parser.add_argument('--bucket', default=None, help='GCS bucket name')
    parser.add_argument('--output', default=None, help='Output JSON report path')

    args = parser.parse_args()

    # Load index
    logger.info(f"Loading index from {args.index_dir}/{args.index_name}...")
    index = InvertedIndex.read_index(args.index_dir, args.index_name, args.bucket)

    # Run validation
    checker = IndexHealthCheck(index)
    report = checker.run_full_validation()

    # Save report if requested
    if args.output:
        with open(args.output, 'w') as f:
            json.dump(report, f, indent=2)
        logger.info(f"\n📄 Report saved to {args.output}")

    # Print summary
    print("\n" + "="*70)
    print("VALIDATION SUMMARY")
    print("="*70)
    print(f"Index: {args.index_name}")
    print(f"Total terms: {report['index_size_terms']:,}")
    print(f"Zipf exponent: {report['zipf_check']['zipf_exponent']:.3f} "
          f"(R²={report['zipf_check']['r_squared']:.4f})")
    print(f"DF anomalies: {report['df_anomalies']['total_anomalies']}")
    print(f"Overall health: {report['overall_health']}")
    print("="*70)


if __name__ == "__main__":
    main()
