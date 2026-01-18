# pre_processing.py
"""
Text preprocessing module for Wikipedia search engine.
Provides tokenization, stopword removal, and optional stemming.

NOTE: Uses hardcoded stopwords to avoid NLTK download issues on GCP Dataproc.
"""

import re
from typing import List

# Regex pattern for tokenization
# Matches words starting with #, @, or alphanumeric, followed by 2-24 chars
# that may include apostrophes or hyphens
RE_WORD = re.compile(r"""[\#\@\w](['\-]?\w){2,24}""", re.UNICODE)

# Hardcoded English stopwords (from NLTK) - avoids download issues on GCP
ENGLISH_STOPWORDS = frozenset([
    'i', 'me', 'my', 'myself', 'we', 'our', 'ours', 'ourselves', 'you', "you're",
    "you've", "you'll", "you'd", 'your', 'yours', 'yourself', 'yourselves', 'he',
    'him', 'his', 'himself', 'she', "she's", 'her', 'hers', 'herself', 'it', "it's",
    'its', 'itself', 'they', 'them', 'their', 'theirs', 'themselves', 'what', 'which',
    'who', 'whom', 'this', 'that', "that'll", 'these', 'those', 'am', 'is', 'are',
    'was', 'were', 'be', 'been', 'being', 'have', 'has', 'had', 'having', 'do',
    'does', 'did', 'doing', 'a', 'an', 'the', 'and', 'but', 'if', 'or', 'because',
    'as', 'until', 'while', 'of', 'at', 'by', 'for', 'with', 'about', 'against',
    'between', 'into', 'through', 'during', 'before', 'after', 'above', 'below',
    'to', 'from', 'up', 'down', 'in', 'out', 'on', 'off', 'over', 'under', 'again',
    'further', 'then', 'once', 'here', 'there', 'when', 'where', 'why', 'how', 'all',
    'each', 'few', 'more', 'most', 'other', 'some', 'such', 'no', 'nor', 'not',
    'only', 'own', 'same', 'so', 'than', 'too', 'very', 's', 't', 'can', 'will',
    'just', 'don', "don't", 'should', "should've", 'now', 'd', 'll', 'm', 'o', 're',
    've', 'y', 'ain', 'aren', "aren't", 'couldn', "couldn't", 'didn', "didn't",
    'doesn', "doesn't", 'hadn', "hadn't", 'hasn', "hasn't", 'haven', "haven't",
    'isn', "isn't", 'ma', 'mightn', "mightn't", 'mustn', "mustn't", 'needn',
    "needn't", 'shan', "shan't", 'shouldn', "shouldn't", 'wasn', "wasn't", 'weren',
    "weren't", 'won', "won't", 'wouldn', "wouldn't"
])

# Corpus-specific stopwords for Wikipedia
CORPUS_STOPWORDS = frozenset([
    "category", "references", "also", "links", "external", 
    "see", "thumb", "wikipedia", "article", "page"
])

ALL_STOPWORDS = ENGLISH_STOPWORDS.union(CORPUS_STOPWORDS)

# Lazy-loaded stemmer
_stemmer = None

def get_stemmer():
    """Lazy initialization of Porter Stemmer."""
    global _stemmer
    if _stemmer is None:
        from nltk.stem.porter import PorterStemmer
        _stemmer = PorterStemmer()
    return _stemmer


def tokenize(text: str) -> List[str]:
    """
    Tokenize text using regex pattern.
    
    Args:
        text: Input string to tokenize.
        
    Returns:
        List of lowercase tokens.
    """
    if not text:
        return []
    return [token.group().lower() for token in RE_WORD.finditer(text.lower())]


def remove_stopwords(tokens: List[str]) -> List[str]:
    """
    Remove stopwords from token list.
    
    Args:
        tokens: List of tokens.
        
    Returns:
        Filtered list without stopwords.
    """
    return [token for token in tokens if token not in ALL_STOPWORDS]


def stem_tokens(tokens: List[str]) -> List[str]:
    """
    Apply Porter Stemming to tokens.
    
    Args:
        tokens: List of tokens.
        
    Returns:
        List of stemmed tokens.
    """
    stemmer = get_stemmer()
    return [stemmer.stem(token) for token in tokens]


def tokenize_and_process(text: str, use_stemming: bool = True) -> List[str]:
    """
    Full text processing pipeline: tokenize, remove stopwords, optionally stem.
    
    Args:
        text: The input string to process.
        use_stemming: Flag to enable/disable stemming (default: True).
        
    Returns:
        A list of processed tokens.
        
    Example:
        >>> tokenize_and_process("The quick brown foxes are jumping!")
        ['quick', 'brown', 'fox', 'jump']
    """
    if not text:
        return []
    
    # Step 1: Tokenize
    tokens = tokenize(text)
    
    # Step 2: Remove stopwords
    tokens = remove_stopwords(tokens)
    
    # Step 3: Apply stemming if enabled
    if use_stemming:
        tokens = stem_tokens(tokens)
    
    return tokens


def tokenize_no_stem(text: str) -> List[str]:
    """
    Tokenize and remove stopwords WITHOUT stemming.
    Required for title and anchor search endpoints.
    
    Args:
        text: The input string to process.
        
    Returns:
        A list of processed tokens (not stemmed).
    """
    return tokenize_and_process(text, use_stemming=False)


# For backwards compatibility and direct import
__all__ = [
    'tokenize_and_process',
    'tokenize_no_stem', 
    'tokenize',
    'remove_stopwords',
    'stem_tokens',
    'ALL_STOPWORDS',
    'RE_WORD'
]
