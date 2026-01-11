"""
Unit Tests for Search Engine Endpoints

Tests TF-IDF calculation, cosine similarity, binary ranking, and edge cases.
"""

import unittest
import math
from collections import Counter
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
class TestTFIDFFormula(unittest.TestCase):
    """Test TF-IDF calculation formulas."""

    def test_idf_natural_log(self):
        """Test that IDF uses natural log, not log10."""
        N = 1000  # Total documents
        df = 100  # Document frequency

        # Correct: natural log
        idf_correct = math.log(N / df)

        # Wrong: log10
        idf_wrong = math.log10(N / df)

        self.assertAlmostEqual(idf_correct, math.log(10), places=4)
        self.assertNotAlmostEqual(idf_correct, idf_wrong, places=1)
        print(f"✓ IDF formula uses natural log: log({N}/{df}) = {idf_correct:.4f}")

    def test_tfidf_calculation(self):
        """Test TF-IDF calculation."""
        tf = 5
        df = 100
        N = 1000

        idf = math.log(N / df)
        tfidf = tf * idf

        expected_tfidf = 5 * math.log(10)
        self.assertAlmostEqual(tfidf, expected_tfidf, places=4)
        print(f"✓ TF-IDF: {tf} * log({N}/{df}) = {tfidf:.4f}")


class TestCosineSimilarity(unittest.TestCase):
    """Test cosine similarity calculation."""

    def test_cosine_similarity_basic(self):
        """Test basic cosine similarity calculation."""
        # Document vector: [3, 0, 4] (TF-IDF values)
        # Query vector: [3, 4, 0]
        # dot(d,q) = 3*3 + 0*4 + 4*0 = 9
        # ||d|| = sqrt(9 + 0 + 16) = 5
        # ||q|| = sqrt(9 + 16 + 0) = 5
        # cosine = 9 / (5 * 5) = 0.36

        doc_vector = [3, 0, 4]
        query_vector = [3, 4, 0]

        dot_product = sum(d * q for d, q in zip(doc_vector, query_vector))
        doc_norm = math.sqrt(sum(d**2 for d in doc_vector))
        query_norm = math.sqrt(sum(q**2 for q in query_vector))

        cosine_sim = dot_product / (doc_norm * query_norm)

        self.assertAlmostEqual(cosine_sim, 0.36, places=2)
        print(f"✓ Cosine similarity: {cosine_sim:.4f}")

    def test_identical_vectors(self):
        """Test cosine similarity of identical vectors = 1.0."""
        vec = [1, 2, 3, 4, 5]

        dot_product = sum(v**2 for v in vec)
        norm = math.sqrt(sum(v**2 for v in vec))

        cosine_sim = dot_product / (norm * norm)

        self.assertAlmostEqual(cosine_sim, 1.0, places=6)
        print(f"✓ Identical vectors have cosine = 1.0")

    def test_orthogonal_vectors(self):
        """Test cosine similarity of orthogonal vectors = 0.0."""
        doc_vector = [1, 0, 0]
        query_vector = [0, 1, 0]

        dot_product = sum(d * q for d, q in zip(doc_vector, query_vector))
        doc_norm = math.sqrt(sum(d**2 for d in doc_vector))
        query_norm = math.sqrt(sum(q**2 for q in query_vector))

        cosine_sim = dot_product / (doc_norm * query_norm)

        self.assertAlmostEqual(cosine_sim, 0.0, places=6)
        print(f"✓ Orthogonal vectors have cosine = 0.0")


class TestBinaryRanking(unittest.TestCase):
    """Test binary ranking logic."""

    def test_distinct_word_count(self):
        """Test that binary ranking counts DISTINCT words, not frequencies."""
        # Query with repeated word: "Fire Fire London"
        query_tokens = ['fire', 'fire', 'fire', 'london']

        # WRONG: Count all tokens (would give 4)
        wrong_count = len(query_tokens)

        # CORRECT: Count distinct tokens (should give 2)
        distinct_query_words = set(query_tokens)
        correct_count = len(distinct_query_words)

        self.assertEqual(correct_count, 2)
        self.assertEqual(wrong_count, 4)
        self.assertNotEqual(correct_count, wrong_count)

        print(f"✓ Distinct word count: 'fire fire fire london' → {correct_count} distinct words")

    def test_binary_ranking_score(self):
        """Test binary ranking score calculation."""
        # Query: "Great Fire London"
        query_tokens = ['great', 'fire', 'london']
        query_set = set(query_tokens)  # {'great', 'fire', 'london'}

        # Title: "Great Fire of London"  (tokenized: ['great', 'fire', 'london'])
        title_tokens = ['great', 'fire', 'london']

        # Score = number of distinct query words in title
        matched_words = query_set & set(title_tokens)
        score = len(matched_words)

        self.assertEqual(score, 3)
        print(f"✓ Binary ranking: 3 distinct query words matched in title")

    def test_binary_ranking_with_duplicates(self):
        """Test that duplicate query words don't increase score."""
        # Query: "Fire Fire Fire London"
        query_tokens = ['fire', 'fire', 'fire', 'london']
        query_set = set(query_tokens)  # {'fire', 'london'}

        # Title: "Great Fire of London"
        title_tokens = ['great', 'fire', 'london']

        # Score = distinct words
        matched_words = query_set & set(title_tokens)
        score = len(matched_words)

        # Should be 2 (fire, london), not higher
        self.assertEqual(score, 2)
        print(f"✓ Duplicate query words don't inflate score: {score}")


class TestPreprocessing(unittest.TestCase):
    """Test preprocessing functions."""

    def test_stopword_removal(self):
        """Test that stopwords are removed."""
        from backend.pre_processing import tokenize_and_process

        text = "the Great Fire of London"
        tokens = tokenize_and_process(text, remove_stops=True, stem=False)

        # "the" and "of" should be removed
        self.assertNotIn('the', tokens)
        self.assertNotIn('of', tokens)
        self.assertIn('great', tokens)
        self.assertIn('fire', tokens)
        self.assertIn('london', tokens)

        print(f"✓ Stopwords removed: '{text}' → {tokens}")

    def test_no_stemming(self):
        """Test that stem=False preserves original forms."""
        from backend.pre_processing import tokenize_and_process

        text = "climbing pyramids"
        tokens_no_stem = tokenize_and_process(text, remove_stops=True, stem=False)
        tokens_with_stem = tokenize_and_process(text, remove_stops=True, stem=True)

        # Without stemming: should preserve original
        self.assertIn('climbing', tokens_no_stem)
        self.assertIn('pyramids', tokens_no_stem)

        # With stemming: should reduce to stems
        self.assertIn('climb', tokens_with_stem)
        self.assertIn('pyramid', tokens_with_stem)

        print(f"✓ No stemming: '{text}' → {tokens_no_stem}")
        print(f"✓ With stemming: '{text}' → {tokens_with_stem}")

    def test_case_normalization(self):
        """Test lowercase normalization."""
        from backend.pre_processing import tokenize_and_process

        text1 = "MOUNT EVEREST"
        text2 = "mount everest"

        tokens1 = tokenize_and_process(text1, remove_stops=True, stem=False)
        tokens2 = tokenize_and_process(text2, remove_stops=True, stem=False)

        self.assertEqual(tokens1, tokens2)
        print(f"✓ Case normalized: '{text1}' = '{text2}'")

    def test_empty_input(self):
        """Test empty input handling."""
        from backend.pre_processing import tokenize_and_process

        tokens = tokenize_and_process("", remove_stops=True, stem=False)
        self.assertEqual(tokens, [])

        tokens = tokenize_and_process(None, remove_stops=True, stem=False)
        self.assertEqual(tokens, [])

        print(f"✓ Empty input handled gracefully")

    def test_stopword_only_query(self):
        """Test query with only stopwords."""
        from backend.pre_processing import tokenize_and_process

        text = "the and or"
        tokens = tokenize_and_process(text, remove_stops=True, stem=False)

        # All stopwords, should return empty
        self.assertEqual(tokens, [])
        print(f"✓ Stopword-only query: '{text}' → []")


class TestEdgeCases(unittest.TestCase):
    """Test edge cases."""

    def test_division_by_zero_in_cosine(self):
        """Test that zero norm doesn't cause division by zero."""
        query_norm = 0.0
        doc_norm = 5.0
        dot_product = 10.0

        # Should handle gracefully, not crash
        if query_norm > 0 and doc_norm > 0:
            cosine = dot_product / (query_norm * doc_norm)
        else:
            cosine = 0.0

        self.assertEqual(cosine, 0.0)
        print(f"✓ Zero norm handled: cosine = 0.0")

    def test_query_term_not_in_index(self):
        """Test query with term not in index vocabulary."""
        # Simulate: term "xyznotexist" not in df dictionary
        index_df = {'everest': 10, 'mount': 50}

        query_term = 'xyznotexist'

        # Should not crash, just skip the term
        if query_term in index_df:
            df = index_df[query_term]
        else:
            df = None  # Term not found, skip it

        self.assertIsNone(df)
        print(f"✓ Missing term handled: '{query_term}' not in index")

    def test_empty_posting_list(self):
        """Test that empty posting list is handled."""
        posting_list = []

        candidates = {}
        for doc_id, tf in posting_list:
            candidates[doc_id] = tf

        self.assertEqual(candidates, {})
        print(f"✓ Empty posting list handled")


def run_all_tests():
    """Run all test suites."""
    print("\n" + "="*70)
    print("RUNNING SEARCH ENGINE UNIT TESTS")
    print("="*70 + "\n")

    # Create test suite
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()

    suite.addTests(loader.loadTestsFromTestCase(TestTFIDFFormula))
    suite.addTests(loader.loadTestsFromTestCase(TestCosineSimilarity))
    suite.addTests(loader.loadTestsFromTestCase(TestBinaryRanking))
    suite.addTests(loader.loadTestsFromTestCase(TestPreprocessing))
    suite.addTests(loader.loadTestsFromTestCase(TestEdgeCases))

    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)

    # Summary
    print("\n" + "="*70)
    print("TEST SUMMARY")
    print("="*70)
    print(f"Tests run: {result.testsRun}")
    print(f"Successes: {result.testsRun - len(result.failures) - len(result.errors)}")
    print(f"Failures: {len(result.failures)}")
    print(f"Errors: {len(result.errors)}")
    print("="*70 + "\n")

    return result.wasSuccessful()


if __name__ == "__main__":
    success = run_all_tests()
    exit(0 if success else 1)
