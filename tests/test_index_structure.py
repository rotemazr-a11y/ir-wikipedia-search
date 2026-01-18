# test_index_structure.py
"""
Tests to validate the expected structure from the index builder.
Run these tests BEFORE deploying to GCP to catch issues early.
Also run AFTER indexing to validate output structure.

Usage:
    # Before GCP (unit tests with mock data):
    python -m pytest tests/test_index_structure.py -v
    
    # After GCP (validate actual output):
    python tests/test_index_structure.py --validate-gcs --bucket bucket_207916263
"""

import os
import sys
import struct
import pickle
import tempfile
import unittest
from pathlib import Path
from collections import defaultdict
from typing import Dict, List, Tuple, Any

# Add parent directories to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'indexing'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'frontend'))


class TestInvertedIndexStructure(unittest.TestCase):
    """Test the InvertedIndex class structure and serialization."""
    
    @classmethod
    def setUpClass(cls):
        """Import InvertedIndex and create test fixtures."""
        try:
            from inverted_index_gcp import InvertedIndex, MultiFileWriter, MultiFileReader
            cls.InvertedIndex = InvertedIndex
            cls.MultiFileWriter = MultiFileWriter
            cls.MultiFileReader = MultiFileReader
        except ImportError as e:
            raise unittest.SkipTest(f"Could not import InvertedIndex: {e}")
        
        cls.temp_dir = tempfile.mkdtemp()
    
    def test_inverted_index_has_required_attributes(self):
        """InvertedIndex must have df and posting_locs dictionaries."""
        idx = self.InvertedIndex()
        
        self.assertTrue(hasattr(idx, 'df'), "InvertedIndex missing 'df' attribute")
        self.assertTrue(hasattr(idx, 'posting_locs'), "InvertedIndex missing 'posting_locs' attribute")
        self.assertIsInstance(idx.df, dict, "'df' should be a dict")
        self.assertIsInstance(idx.posting_locs, dict, "'posting_locs' should be a dict")
    
    def test_multifile_writer_creates_bin_files(self):
        """MultiFileWriter should create .bin files for posting lists."""
        base_dir = os.path.join(self.temp_dir, 'test_writer')
        os.makedirs(base_dir, exist_ok=True)
        
        idx = self.InvertedIndex()
        
        with self.MultiFileWriter(base_dir, 'test_index') as writer:
            # Write some postings
            term = 'hello'
            postings = [(1, 3), (5, 2), (10, 1)]  # (doc_id, tf)
            
            # Pack postings as binary
            posting_bytes = b''.join(struct.pack('I', doc_id) + struct.pack('H', tf) 
                                      for doc_id, tf in postings)
            
            writer.write(posting_bytes, term)
            idx.posting_locs[term] = writer.posting_locs[term]
        
        # Verify .bin file was created
        bin_files = list(Path(base_dir).glob('*.bin'))
        self.assertGreater(len(bin_files), 0, "No .bin files created")
    
    def test_posting_format_6_bytes(self):
        """Each posting should be 6 bytes: 4 (doc_id) + 2 (tf)."""
        doc_id = 12345678
        tf = 42
        
        packed = struct.pack('I', doc_id) + struct.pack('H', tf)
        self.assertEqual(len(packed), 6, "Posting should be exactly 6 bytes")
        
        # Verify we can unpack
        unpacked_doc_id = struct.unpack('I', packed[:4])[0]
        unpacked_tf = struct.unpack('H', packed[4:6])[0]
        
        self.assertEqual(unpacked_doc_id, doc_id)
        self.assertEqual(unpacked_tf, tf)
    
    def test_posting_locs_format(self):
        """posting_locs should map term -> list of (file, offset, length)."""
        idx = self.InvertedIndex()
        
        # Simulate what the index builder produces
        idx.posting_locs['test_term'] = [('test_index_0.bin', 0, 18)]  # 3 postings * 6 bytes
        
        locs = idx.posting_locs['test_term']
        self.assertIsInstance(locs, list)
        self.assertEqual(len(locs[0]), 3, "Each loc should be (file, offset, length)")
        
        file_name, offset, length = locs[0]
        self.assertIsInstance(file_name, str)
        self.assertIsInstance(offset, int)
        self.assertIsInstance(length, int)
    
    def test_df_contains_document_frequencies(self):
        """df dict should map term -> count of documents containing term."""
        idx = self.InvertedIndex()
        idx.df['python'] = 1500
        idx.df['java'] = 2000
        
        self.assertEqual(idx.df['python'], 1500)
        self.assertIsInstance(idx.df['python'], int)
    
    def test_pickle_serialization(self):
        """InvertedIndex should be serializable with pickle."""
        idx = self.InvertedIndex()
        idx.df = {'term1': 100, 'term2': 200}
        idx.posting_locs = {'term1': [('file.bin', 0, 6)]}
        
        # Serialize
        pkl_path = os.path.join(self.temp_dir, 'test_index.pkl')
        with open(pkl_path, 'wb') as f:
            pickle.dump(idx, f)
        
        # Deserialize
        with open(pkl_path, 'rb') as f:
            loaded_idx = pickle.load(f)
        
        self.assertEqual(loaded_idx.df, idx.df)
        self.assertEqual(loaded_idx.posting_locs, idx.posting_locs)


class TestSPIMIBlockBuilder(unittest.TestCase):
    """Test the SPIMI block builder structure."""
    
    @classmethod
    def setUpClass(cls):
        try:
            from spimi_block_builder import SPIMIBlockBuilder, create_body_builder
            cls.SPIMIBlockBuilder = SPIMIBlockBuilder
            cls.create_body_builder = staticmethod(create_body_builder)
        except ImportError as e:
            raise unittest.SkipTest(f"Could not import SPIMIBlockBuilder: {e}")
        
        cls.temp_dir = tempfile.mkdtemp()
    
    def test_builder_has_required_methods(self):
        """SPIMIBlockBuilder must have add_posting, finalize methods."""
        builder = self.create_body_builder()
        
        self.assertTrue(hasattr(builder, 'add_posting'), "Missing 'add_posting' method")
        self.assertTrue(hasattr(builder, 'finalize'), "Missing 'finalize' method")
        self.assertTrue(callable(builder.add_posting))
        self.assertTrue(callable(builder.finalize))
    
    def test_add_posting_updates_dictionary(self):
        """add_posting should add term->(doc_id, tf) mappings."""
        builder = self.create_body_builder()
        
        builder.add_posting('hello', 1, 5)
        builder.add_posting('world', 1, 3)
        builder.add_posting('hello', 2, 2)
        
        # Check internal dictionary (stored in current_block)
        self.assertIn('hello', builder.current_block)
        self.assertEqual(len(builder.current_block['hello']), 2)
    
    def test_memory_tracking(self):
        """Builder should track memory usage."""
        builder = self.create_body_builder()
        
        initial_memory = builder.current_memory
        
        # Add many postings
        for i in range(1000):
            builder.add_posting(f'term_{i}', i, 1)
        
        self.assertGreater(builder.current_memory, initial_memory)
    
    def test_finalize_returns_inverted_index(self):
        """finalize() should return an InvertedIndex object."""
        try:
            from inverted_index_gcp import InvertedIndex
        except ImportError:
            self.skipTest("inverted_index_gcp requires google-cloud-storage")
        
        builder = self.create_body_builder()
        
        builder.add_posting('test', 1, 1)
        builder.add_posting('test', 2, 2)
        
        result = builder.finalize()
        
        self.assertIsInstance(result, InvertedIndex)
        self.assertIn('test', result.df)
        self.assertEqual(result.df['test'], 2)  # 2 documents


class TestPreProcessing(unittest.TestCase):
    """Test the tokenization and preprocessing functions."""
    
    @classmethod
    def setUpClass(cls):
        try:
            from pre_processing import tokenize_no_stem, tokenize_and_process
            cls.tokenize_no_stem = tokenize_no_stem
            cls.tokenize_and_process = tokenize_and_process
        except ImportError as e:
            raise unittest.SkipTest(f"Could not import preprocessing: {e}")
    
    def test_tokenize_removes_stopwords(self):
        """Tokenizer should remove common stopwords."""
        text = "The quick brown fox jumps over the lazy dog"
        tokens = self.tokenize_no_stem(text)
        
        # 'the' and 'over' should be removed
        self.assertNotIn('the', tokens)
        self.assertNotIn('over', tokens)
        
        # Content words should remain
        self.assertIn('quick', tokens)
        self.assertIn('brown', tokens)
        self.assertIn('fox', tokens)
    
    def test_tokenize_lowercases(self):
        """All tokens should be lowercase."""
        text = "Python Java JavaScript"
        tokens = self.tokenize_no_stem(text)
        
        for token in tokens:
            self.assertEqual(token, token.lower())
    
    def test_tokenize_handles_empty_string(self):
        """Empty string should return empty list."""
        tokens = self.tokenize_no_stem("")
        self.assertEqual(tokens, [])
    
    def test_tokenize_handles_none(self):
        """None input should return empty list."""
        tokens = self.tokenize_no_stem(None)
        self.assertEqual(tokens, [])
    
    def test_tokenize_minimum_length(self):
        """Tokens shorter than 2 characters should be removed."""
        text = "I am a test of short words x y z ab"
        tokens = self.tokenize_no_stem(text)
        
        for token in tokens:
            self.assertGreaterEqual(len(token), 2)


class TestSearchFrontendStructure(unittest.TestCase):
    """Test the search frontend structure before deployment."""
    
    @classmethod
    def setUpClass(cls):
        try:
            from search_frontend import (
                tokenize_no_stem, compute_tfidf_cosine, binary_ranking,
                combined_search, engine, app
            )
            cls.tokenize_no_stem = tokenize_no_stem
            cls.compute_tfidf_cosine = compute_tfidf_cosine
            cls.binary_ranking = binary_ranking
            cls.combined_search = combined_search
            cls.engine = engine
            cls.app = app
        except ImportError as e:
            raise unittest.SkipTest(f"Could not import search_frontend: {e}")
    
    def test_app_has_all_routes(self):
        """Flask app should have all 6 required routes."""
        rules = [rule.rule for rule in self.app.url_map.iter_rules()]
        
        required_routes = ['/search', '/search_body', '/search_title', 
                          '/search_anchor', '/get_pagerank', '/get_pageview']
        
        for route in required_routes:
            self.assertIn(route, rules, f"Missing route: {route}")
    
    def test_engine_has_required_attributes(self):
        """SearchEngine should have all required data stores."""
        required_attrs = ['body_index', 'title_index', 'anchor_index',
                         'pagerank', 'pageviews', 'doc_titles', 'doc_lengths']
        
        for attr in required_attrs:
            self.assertTrue(hasattr(self.engine, attr), f"Engine missing: {attr}")
    
    def test_tokenize_consistency(self):
        """Frontend tokenizer should match preprocessing tokenizer."""
        text = "Machine Learning in Python"
        
        frontend_tokens = self.tokenize_no_stem(text)
        
        # Should contain relevant tokens
        self.assertIn('machine', frontend_tokens)
        self.assertIn('learning', frontend_tokens)
        self.assertIn('python', frontend_tokens)
    
    def test_empty_index_handling(self):
        """Search functions should handle None indices gracefully."""
        tokens = ['test', 'query']
        
        # These should not raise, just return empty
        result = self.compute_tfidf_cosine(tokens, None, '', 10)
        self.assertEqual(result, [])
        
        result = self.binary_ranking(tokens, None, '')
        self.assertEqual(result, [])


class TestDocumentMetadataStructure(unittest.TestCase):
    """Test expected structure of doc_titles.pkl and doc_lengths.pkl."""
    
    def test_doc_titles_format(self):
        """doc_titles.pkl should be Dict[int, str]."""
        # Create mock data
        doc_titles = {
            1: "Main Page",
            12: "Python (programming language)",
            156: "Machine learning"
        }
        
        # Verify structure
        for doc_id, title in doc_titles.items():
            self.assertIsInstance(doc_id, int)
            self.assertIsInstance(title, str)
    
    def test_doc_lengths_format(self):
        """doc_lengths.pkl should be Dict[int, int]."""
        doc_lengths = {
            1: 150,
            12: 5000,
            156: 8500
        }
        
        for doc_id, length in doc_lengths.items():
            self.assertIsInstance(doc_id, int)
            self.assertIsInstance(length, int)
            self.assertGreaterEqual(length, 0)
    
    def test_pagerank_format(self):
        """pagerank.pkl should be Dict[int, float] with values in [0, 1]."""
        pagerank = {
            1: 0.15,
            12: 0.00023,
            156: 0.00045
        }
        
        for doc_id, score in pagerank.items():
            self.assertIsInstance(doc_id, int)
            self.assertIsInstance(score, float)
            self.assertGreaterEqual(score, 0.0)
            self.assertLessEqual(score, 1.0)
    
    def test_pageviews_format(self):
        """pageviews.pkl should be Dict[int, int] with non-negative values."""
        pageviews = {
            1: 1000000,
            12: 50000,
            156: 25000
        }
        
        for doc_id, views in pageviews.items():
            self.assertIsInstance(doc_id, int)
            self.assertIsInstance(views, int)
            self.assertGreaterEqual(views, 0)


class TestBinaryPostingIO(unittest.TestCase):
    """Test binary posting list reading/writing."""
    
    def test_write_and_read_postings(self):
        """Write postings to binary and read them back."""
        postings = [(100, 5), (200, 3), (300, 1), (400, 10)]
        
        with tempfile.NamedTemporaryFile(suffix='.bin', delete=False) as f:
            # Write
            for doc_id, tf in postings:
                f.write(struct.pack('I', doc_id))
                f.write(struct.pack('H', tf))
            temp_path = f.name
        
        # Read back
        read_postings = []
        with open(temp_path, 'rb') as f:
            while True:
                data = f.read(6)
                if not data:
                    break
                doc_id = struct.unpack('I', data[:4])[0]
                tf = struct.unpack('H', data[4:6])[0]
                read_postings.append((doc_id, tf))
        
        self.assertEqual(postings, read_postings)
        
        # Cleanup
        os.unlink(temp_path)
    
    def test_large_doc_id_handling(self):
        """Test that large doc_ids (up to 4B) are handled correctly."""
        large_doc_id = 2**31 - 1  # Max signed 32-bit
        tf = 100
        
        packed = struct.pack('I', large_doc_id) + struct.pack('H', tf)
        unpacked_doc_id = struct.unpack('I', packed[:4])[0]
        
        self.assertEqual(unpacked_doc_id, large_doc_id)
    
    def test_max_tf_handling(self):
        """Test that tf up to 65535 is handled correctly."""
        doc_id = 1000
        max_tf = 65535  # Max uint16
        
        packed = struct.pack('I', doc_id) + struct.pack('H', max_tf)
        unpacked_tf = struct.unpack('H', packed[4:6])[0]
        
        self.assertEqual(unpacked_tf, max_tf)


# ============================================================================
# GCS Validation Mode
# ============================================================================

def validate_gcs_output(bucket_name: str):
    """Validate the structure of index output in GCS."""
    from google.cloud import storage
    
    client = storage.Client()
    bucket = client.bucket(bucket_name)
    
    print(f"\n{'='*60}")
    print(f"Validating GCS output in bucket: {bucket_name}")
    print('='*60)
    
    errors = []
    warnings = []
    
    # Check for index directories
    index_types = ['body_index', 'title_index', 'anchor_index']
    
    for idx_type in index_types:
        prefix = f'indices/{idx_type}/'
        blobs = list(bucket.list_blobs(prefix=prefix, max_results=10))
        
        if not blobs:
            errors.append(f"Missing index: {idx_type}")
            continue
        
        print(f"\n✓ Found {idx_type}:")
        
        # Check for .pkl file
        pkl_files = [b for b in blobs if b.name.endswith('.pkl')]
        bin_files = [b for b in blobs if b.name.endswith('.bin')]
        
        if not pkl_files:
            errors.append(f"  Missing .pkl metadata for {idx_type}")
        else:
            print(f"  - Metadata: {pkl_files[0].name} ({pkl_files[0].size} bytes)")
            
            # Try to load and validate
            try:
                import io
                content = pkl_files[0].download_as_bytes()
                idx = pickle.loads(content)
                print(f"    - Terms in df: {len(idx.df)}")
                print(f"    - Terms in posting_locs: {len(idx.posting_locs)}")
            except Exception as e:
                errors.append(f"  Failed to load {idx_type} metadata: {e}")
        
        if not bin_files:
            errors.append(f"  Missing .bin posting files for {idx_type}")
        else:
            print(f"  - Posting files: {len(bin_files)}")
            total_size = sum(b.size for b in bin_files)
            print(f"  - Total size: {total_size / 1024 / 1024:.2f} MB")
    
    # Check for doc metadata
    print("\nChecking document metadata:")
    for meta_file in ['doc_titles.pkl', 'doc_lengths.pkl']:
        blob = bucket.blob(meta_file)
        if blob.exists():
            print(f"  ✓ {meta_file} ({blob.size} bytes)")
        else:
            warnings.append(f"Missing {meta_file}")
            print(f"  ⚠ {meta_file} not found")
    
    # Summary
    print(f"\n{'='*60}")
    print("VALIDATION SUMMARY")
    print('='*60)
    
    if errors:
        print(f"\n❌ ERRORS ({len(errors)}):")
        for e in errors:
            print(f"   - {e}")
    
    if warnings:
        print(f"\n⚠️  WARNINGS ({len(warnings)}):")
        for w in warnings:
            print(f"   - {w}")
    
    if not errors and not warnings:
        print("\n✅ All validations passed!")
    
    return len(errors) == 0


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='Test index structure')
    parser.add_argument('--validate-gcs', action='store_true', 
                       help='Validate GCS output instead of running unit tests')
    parser.add_argument('--bucket', type=str, default='bucket_207916263',
                       help='GCS bucket name for validation')
    
    args = parser.parse_args()
    
    if args.validate_gcs:
        success = validate_gcs_output(args.bucket)
        sys.exit(0 if success else 1)
    else:
        # Run unit tests
        unittest.main(argv=[''], exit=True, verbosity=2)
