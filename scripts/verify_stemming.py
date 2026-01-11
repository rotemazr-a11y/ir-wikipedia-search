#!/usr/bin/env python3
"""
Stemming Verification Script

Verifies that the indices were built with stemming enabled by checking for
stemmed terms vs unstemmed terms in the index.
"""

import sys
import os

# Add parent directory to path to import backend modules
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from backend.inverted_index_gcp import InvertedIndex

def verify_stemming(index_dir="indices_mini"):
    """Verify that indices contain stemmed terms."""

    print("="*70)
    print("STEMMING VERIFICATION")
    print("="*70)
    print(f"\nChecking indices in: {index_dir}\n")

    # Load body index
    try:
        body_idx = InvertedIndex.read_index(index_dir, "body_index")
        print(f"✓ Loaded body_index ({len(body_idx.df)} unique terms)\n")
    except Exception as e:
        print(f"✗ Failed to load body index: {e}")
        return False

    # Define test cases: (term, should_exist_if_stemmed)
    test_cases = [
        # Stemmed versions (should exist with stemming)
        ("climb", True, "stemmed from 'climbing'"),
        ("expedit", True, "stemmed from 'expeditions'"),
        ("revolut", True, "stemmed from 'revolution'"),
        ("photograph", True, "stemmed from 'photography'"),
        ("discoveri", True, "stemmed from 'discovery'"),

        # Unstemmed versions (should NOT exist with stemming)
        ("climbing", False, "original form (should be stemmed to 'climb')"),
        ("expeditions", False, "original form (should be stemmed to 'expedit')"),
        ("revolution", False, "original form (should be stemmed to 'revolut')"),
        ("photography", False, "original form (should be stemmed to 'photograph')"),
        ("discovery", False, "original form (should be stemmed to 'discoveri')"),
    ]

    passed = 0
    failed = 0

    print("Stemmed terms (should exist):")
    print("-" * 70)
    for term, should_exist, description in test_cases[:5]:
        exists = term in body_idx.df
        df_count = body_idx.df.get(term, 0) if exists else 0

        if exists == should_exist:
            passed += 1
            status = "✓ PASS"
            print(f"{status}: '{term}' found (DF={df_count}) - {description}")
        else:
            failed += 1
            status = "✗ FAIL"
            print(f"{status}: '{term}' {'found' if exists else 'NOT found'} - {description}")

    print("\nUnstemmed terms (should NOT exist):")
    print("-" * 70)
    for term, should_exist, description in test_cases[5:]:
        exists = term in body_idx.df
        df_count = body_idx.df.get(term, 0) if exists else 0

        if exists == should_exist:
            passed += 1
            status = "✓ PASS"
            print(f"{status}: '{term}' not found - {description}")
        else:
            failed += 1
            status = "✗ FAIL"
            print(f"{status}: '{term}' found (DF={df_count}) - {description}")
            print(f"         WARNING: Index still contains unstemmed terms!")

    print("\n" + "="*70)
    print(f"Results: {passed} passed, {failed} failed")

    if failed == 0:
        print("✓ VERIFICATION PASSED: Stemming is correctly applied!")
    else:
        print("✗ VERIFICATION FAILED: Stemming mismatch detected!")
        print("\nThis means the index was not built with stem=True.")
        print("Please check backend/index_builder.py lines 135, 142, 152")

    print("="*70)

    return failed == 0

if __name__ == "__main__":
    index_dir = sys.argv[1] if len(sys.argv) > 1 else "indices_mini"
    success = verify_stemming(index_dir)
    sys.exit(0 if success else 1)
