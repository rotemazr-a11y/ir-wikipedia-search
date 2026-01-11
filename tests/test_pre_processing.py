import unittest
# No 'backend.' prefix here - assuming flat directory structure
from backend.pre_processing import tokenize_and_process, get_term_counts

class TestPreProcessing(unittest.TestCase):

    def test_assignment1_regex(self):
        """
        CRITICAL: Matches Assignment 1 requirement.
        Ensures internal punctuation is kept but external is stripped.
        """
        text = "The user's state-of-the-art engine."
        tokens = tokenize_and_process(text, remove_stops=False)
        
        # 'the' should be lowercased to 'the'
        self.assertIn("the", tokens)
        # 'user's' should be one token
        self.assertIn("user's", tokens)
        # 'state-of-the-art' should be one token
        self.assertIn("state-of-the-art", tokens)
        # The period at the end should be gone
        self.assertNotIn("engine.", tokens)
        self.assertIn("engine", tokens)
        print("✅ Regex: Complex tokens preserved correctly.")

    def test_stopword_logic(self):
        """Ensures the NLTK stopword list is actually filtering."""
        text = "This is a test of the search engine"
        # 'this', 'is', 'a', 'of', 'the' are standard stopwords
        tokens = tokenize_and_process(text, remove_stops=True)
        
        expected_keywords = ['test', 'search', 'engine']
        self.assertEqual(tokens, expected_keywords)
        print("✅ Stopwords: Noise words filtered correctly.")

    def test_unicode_and_special(self):
        """Assignment 1 requires handling hashtags and different characters."""
        text = "Search #python 2025! שלום"
        tokens = tokenize_and_process(text, remove_stops=False)
        
        self.assertIn("#python", tokens)
        self.assertIn("2025", tokens)
        self.assertIn("שלום", tokens)
        print("✅ Unicode: Special characters and Hebrew handled.")

    def test_empty_inputs(self):
        """Robustness check: Ensure code doesn't crash on empty data."""
        self.assertEqual(tokenize_and_process(""), [])
        self.assertEqual(tokenize_and_process(None), [])
        print("✅ Robustness: Empty strings handled.")

if __name__ == "__main__":
    unittest.main()