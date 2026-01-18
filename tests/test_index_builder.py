# test_index_builder.py
"""
Unit tests for the inverted index builder components.
Run locally before deploying to GCP Dataproc.

Usage:
    python test_index_builder.py
"""

import os
import sys
import tempfile
import unittest
from collections import Counter

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from pre_processing import (
    tokenize_and_process, 
    tokenize_no_stem, 
    tokenize, 
    remove_stopwords,
    stem_tokens,
    ALL_STOPWORDS
)
from spimi_block_builder import SPIMIBlockBuilder, create_body_builder


class TestPreProcessing(unittest.TestCase):
    """Tests for pre_processing.py"""
    
    def test_tokenize_basic(self):
        """Test basic tokenization."""
        text = "Hello World! This is a test."
        tokens = tokenize(text)
        self.assertIn("hello", tokens)
        self.assertIn("world", tokens)
        self.assertIn("this", tokens)
        self.assertIn("test", tokens)
    
    def test_tokenize_short_words_filtered(self):
        """Words with <3 chars should be filtered by regex."""
        text = "I am a big fan of AI and ML"
        tokens = tokenize(text)
        # 'I', 'am', 'a', 'AI', 'ML' should not be in tokens (too short)
        self.assertNotIn("i", tokens)
        self.assertNotIn("am", tokens)
        self.assertNotIn("a", tokens)
        self.assertIn("big", tokens)
        self.assertIn("fan", tokens)
    
    def test_stopword_removal(self):
        """Test stopword removal."""
        tokens = ["the", "quick", "brown", "fox", "is", "running"]
        filtered = remove_stopwords(tokens)
        self.assertNotIn("the", filtered)
        self.assertNotIn("is", filtered)
        self.assertIn("quick", filtered)
        self.assertIn("brown", filtered)
        self.assertIn("fox", filtered)
        self.assertIn("running", filtered)
    
    def test_stemming(self):
        """Test Porter stemming."""
        tokens = ["running", "jumps", "happily", "foxes"]
        stemmed = stem_tokens(tokens)
        self.assertIn("run", stemmed)
        self.assertIn("jump", stemmed)
        self.assertIn("happili", stemmed)  # Porter stems 'happily' to 'happili'
        self.assertIn("fox", stemmed)
    
    def test_full_pipeline_with_stemming(self):
        """Test full pipeline with stemming enabled."""
        text = "The quick brown foxes are jumping over the lazy dogs."
        tokens = tokenize_and_process(text, use_stemming=True)
        
        # Should not have stopwords
        self.assertNotIn("the", tokens)
        self.assertNotIn("are", tokens)
        self.assertNotIn("over", tokens)
        
        # Should be stemmed
        self.assertIn("fox", tokens)  # foxes -> fox
        self.assertIn("jump", tokens)  # jumping -> jump
        self.assertIn("dog", tokens)   # dogs -> dog
    
    def test_full_pipeline_without_stemming(self):
        """Test full pipeline without stemming."""
        text = "The quick brown foxes are jumping"
        tokens = tokenize_no_stem(text)
        
        # Should not have stopwords
        self.assertNotIn("the", tokens)
        self.assertNotIn("are", tokens)
        
        # Should NOT be stemmed
        self.assertIn("foxes", tokens)
        self.assertIn("jumping", tokens)
    
    def test_empty_input(self):
        """Test empty input handling."""
        self.assertEqual(tokenize_and_process(""), [])
        self.assertEqual(tokenize_and_process(None), [])
    
    def test_corpus_stopwords(self):
        """Test corpus-specific stopwords."""
        text = "See the category for more references and external links"
        tokens = tokenize_and_process(text, use_stemming=False)
        
        self.assertNotIn("see", tokens)
        self.assertNotIn("category", tokens)
        self.assertNotIn("references", tokens)
        self.assertNotIn("external", tokens)
        self.assertNotIn("links", tokens)


class TestSPIMIBlockBuilder(unittest.TestCase):
    """Tests for spimi_block_builder.py"""
    
    def test_basic_add_and_finalize(self):
        """Test basic posting addition and finalization."""
        builder = SPIMIBlockBuilder(memory_threshold_mb=1, index_name="test")
        
        # Add some postings
        builder.add_posting("hello", 1, 2)
        builder.add_posting("world", 1, 1)
        builder.add_posting("hello", 2, 3)
        builder.add_posting("test", 3, 1)
        
        # Finalize and collect results
        results = dict(builder.finalize())
        
        self.assertIn("hello", results)
        self.assertIn("world", results)
        self.assertIn("test", results)
        
        # Check postings
        self.assertEqual(len(results["hello"]), 2)  # 2 docs with "hello"
        self.assertEqual(len(results["world"]), 1)
        self.assertEqual(len(results["test"]), 1)
    
    def test_memory_threshold_triggers_flush(self):
        """Test that memory threshold triggers block flush."""
        # Very small threshold to force flushing
        builder = SPIMIBlockBuilder(memory_threshold_mb=0.001, index_name="test")
        
        # Add many postings to trigger flush
        for i in range(1000):
            builder.add_posting(f"term_{i}", i, 1)
        
        # Should have flushed at least once
        self.assertGreater(builder.flushes, 0)
        
        # Finalize should still work correctly
        results = dict(builder.finalize())
        self.assertEqual(len(results), 1000)
    
    def test_postings_sorted_by_doc_id(self):
        """Test that postings are sorted by doc_id after finalization."""
        builder = SPIMIBlockBuilder(memory_threshold_mb=1, index_name="test")
        
        # Add postings in non-sorted order
        builder.add_posting("term", 5, 1)
        builder.add_posting("term", 1, 2)
        builder.add_posting("term", 10, 1)
        builder.add_posting("term", 3, 3)
        
        results = dict(builder.finalize())
        postings = results["term"]
        
        # Check doc_ids are sorted
        doc_ids = [doc_id for doc_id, tf in postings]
        self.assertEqual(doc_ids, sorted(doc_ids))
    
    def test_multiple_blocks_merge_correctly(self):
        """Test k-way merge of multiple blocks."""
        builder = SPIMIBlockBuilder(memory_threshold_mb=0.0001, index_name="test")
        
        # Add postings that will definitely span multiple blocks
        terms = ["alpha", "beta", "gamma", "delta"]
        for i in range(100):
            for term in terms:
                builder.add_posting(term, i, 1)
        
        # Finalize
        results = dict(builder.finalize())
        
        # Each term should have 100 postings
        for term in terms:
            self.assertEqual(len(results[term]), 100)
    
    def test_stats_tracking(self):
        """Test statistics tracking."""
        builder = SPIMIBlockBuilder(memory_threshold_mb=1, index_name="test")
        
        builder.add_posting("a", 1, 1)
        builder.add_posting("b", 1, 2)
        builder.add_posting("a", 2, 1)
        
        stats = builder.get_stats()
        
        self.assertEqual(stats["total_postings"], 3)
        self.assertEqual(stats["unique_terms"], 2)
        self.assertEqual(stats["index_name"], "test")


class TestIntegration(unittest.TestCase):
    """Integration tests simulating the full pipeline."""
    
    def test_document_processing_simulation(self):
        """Simulate processing a document through the pipeline."""
        # Sample document
        doc_id = 12345
        body = "Python is a programming language. Python is popular."
        title = "Python Programming Language"
        anchor = "Learn Python here"
        
        # Process body
        body_tokens = tokenize_and_process(body, use_stemming=True)
        body_builder = SPIMIBlockBuilder(memory_threshold_mb=1, index_name="body")
        for term, tf in Counter(body_tokens).items():
            body_builder.add_posting(term, doc_id, tf)
        
        # Process title
        title_tokens = tokenize_no_stem(title)
        title_builder = SPIMIBlockBuilder(memory_threshold_mb=1, index_name="title")
        for term, tf in Counter(title_tokens).items():
            title_builder.add_posting(term, doc_id, tf)
        
        # Finalize
        body_index = dict(body_builder.finalize())
        title_index = dict(title_builder.finalize())
        
        # Body should have stemmed terms
        self.assertIn("python", body_index)  # 'python' stemmed is still 'python'
        self.assertIn("program", body_index)  # 'programming' stems to 'program'
        self.assertIn("languag", body_index)  # 'language' stems to 'languag'
        
        # Title should NOT be stemmed
        self.assertIn("python", title_index)
        self.assertIn("programming", title_index)
        self.assertIn("language", title_index)
    
    def test_tf_calculation(self):
        """Test term frequency is calculated correctly."""
        doc_id = 1
        text = "cat cat cat dog dog bird"
        
        tokens = tokenize_and_process(text, use_stemming=False)
        builder = SPIMIBlockBuilder(memory_threshold_mb=1, index_name="test")
        
        for term, tf in Counter(tokens).items():
            builder.add_posting(term, doc_id, tf)
        
        results = dict(builder.finalize())
        
        # Check TFs
        cat_posting = results["cat"][0]
        dog_posting = results["dog"][0]
        bird_posting = results["bird"][0]
        
        self.assertEqual(cat_posting[1], 3)   # cat appears 3 times
        self.assertEqual(dog_posting[1], 2)   # dog appears 2 times
        self.assertEqual(bird_posting[1], 1)  # bird appears 1 time


def run_tests():
    """Run all tests and print summary."""
    print("=" * 60)
    print("Running Inverted Index Builder Tests")
    print("=" * 60)
    
    # Create test suite
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()
    
    suite.addTests(loader.loadTestsFromTestCase(TestPreProcessing))
    suite.addTests(loader.loadTestsFromTestCase(TestSPIMIBlockBuilder))
    suite.addTests(loader.loadTestsFromTestCase(TestIntegration))
    
    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    # Print summary
    print("\n" + "=" * 60)
    if result.wasSuccessful():
        print("✅ All tests passed!")
    else:
        print(f"❌ Tests failed: {len(result.failures)} failures, {len(result.errors)} errors")
    print("=" * 60)
    
    return result.wasSuccessful()


if __name__ == '__main__':
    success = run_tests()
    sys.exit(0 if success else 1)
