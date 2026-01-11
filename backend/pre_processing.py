"""
Pre-processing utilities for text tokenization and normalization.
Compatible with both local and Spark environments.
"""

# CRITICAL FIX: Must be FIRST, before any other imports
import sys
try:
    import regex
    if not hasattr(regex, 'Pattern'):
        regex.Pattern = type(regex.compile(''))
except ImportError:
    pass

import re
import nltk
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer

# Initialize stemmer and stopwords
porter_stemmer = PorterStemmer()

# Handle stopwords loading gracefully
try:
    english_stopwords = frozenset(stopwords.words('english'))
except LookupError:
    # If stopwords not downloaded, use a minimal set
    english_stopwords = frozenset([
        'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for',
        'of', 'with', 'as', 'by', 'is', 'was', 'are', 'were', 'been', 'be',
        'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would', 'could',
        'should', 'may', 'might', 'must', 'can', 'this', 'that', 'these', 'those'
    ])

# Regex patterns for tokenization
RE_WORD = re.compile(r"""[\w']+""", re.UNICODE)
NUM_REGEX = re.compile(r'^\d+$')


def tokenize_and_process(text, remove_stops=True, stem=False):
    """
    Tokenize and optionally remove stopwords and apply stemming.
    
    Args:
        text (str): Input text to process
        remove_stops (bool): Remove stopwords if True
        stem (bool): Apply Porter stemming if True
        
    Returns:
        list: List of processed tokens
    """
    if not text:
        return []
    
    # Convert to lowercase and tokenize
    tokens = [token.lower() for token in RE_WORD.findall(text)]
    
    # Remove pure numeric tokens
    tokens = [token for token in tokens if not NUM_REGEX.match(token)]
    
    # Remove stopwords if requested
    if remove_stops:
        tokens = [token for token in tokens if token not in english_stopwords]
    
    # Apply stemming if requested
    if stem:
        tokens = [porter_stemmer.stem(token) for token in tokens]
    
    return tokens