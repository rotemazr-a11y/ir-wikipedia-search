"""
Text Processing Module for Information Retrieval
Includes tokenization, normalization, and stop word removal
Based on standard IR practices and course materials
"""

import re
from collections import Counter

# Standard English stop words list
# Based on common IR stop word lists (NLTK, Lucene, etc.)
ENGLISH_STOP_WORDS = frozenset([
    'a', 'about', 'above', 'after', 'again', 'against', 'all', 'am', 'an', 'and', 'any', 'are',
    'aren', 'arent', 'as', 'at', 'be', 'because', 'been', 'before', 'being', 'below', 'between',
    'both', 'but', 'by', 'can', 'cannot', 'could', 'couldn', 'couldnt', 'did', 'didn', 'didnt',
    'do', 'does', 'doesn', 'doesnt', 'doing', 'don', 'dont', 'down', 'during', 'each', 'few',
    'for', 'from', 'further', 'had', 'hadn', 'hadnt', 'has', 'hasn', 'hasnt', 'have', 'haven',
    'havent', 'having', 'he', 'her', 'here', 'hers', 'herself', 'him', 'himself', 'his', 'how',
    'i', 'if', 'in', 'into', 'is', 'isn', 'isnt', 'it', 'its', 'itself', 'let', 'll', 'm', 'me',
    'more', 'most', 'mustn', 'mustnt', 'my', 'myself', 'no', 'nor', 'not', 'of', 'off', 'on',
    'once', 'only', 'or', 'other', 'ought', 'our', 'ours', 'ourselves', 'out', 'over', 'own',
    're', 's', 'same', 'shan', 'shant', 'she', 'should', 'shouldn', 'shouldnt', 'so', 'some',
    'such', 't', 'than', 'that', 'the', 'their', 'theirs', 'them', 'themselves', 'then', 'there',
    'these', 'they', 'this', 'those', 'through', 'to', 'too', 'under', 'until', 'up', 've', 'very',
    'was', 'wasn', 'wasnt', 'we', 'were', 'weren', 'werent', 'what', 'when', 'where', 'which',
    'while', 'who', 'whom', 'why', 'will', 'with', 'won', 'wont', 'would', 'wouldn', 'wouldnt',
    'you', 'your', 'yours', 'yourself', 'yourselves'
])

# Regex pattern for tokenization
# Matches sequences of alphanumeric characters (including underscores)
# This handles words, numbers, and common patterns like "state-of-the-art"
TOKEN_PATTERN = re.compile(r'\b\w+\b', re.UNICODE)


def tokenize(text):
    """
    Tokenize text into individual tokens (words).

    Uses regex to split text on word boundaries, keeping alphanumeric sequences.
    Handles:
    - Standard word boundaries
    - Numbers
    - Hyphenated words (broken into separate tokens)
    - Apostrophes and contractions

    Parameters:
    -----------
    text : str
        The input text to tokenize

    Returns:
    --------
    list of str
        List of tokens extracted from the text

    Example:
    --------
    >>> tokenize("Hello, world! This is a test.")
    ['Hello', 'world', 'This', 'is', 'a', 'test']
    """
    if not text:
        return []

    # Find all word tokens using the regex pattern
    tokens = TOKEN_PATTERN.findall(text)

    return tokens


def normalize(text):
    """
    Normalize text by converting to lowercase.

    Normalization is a crucial step in IR to ensure that:
    - "Bank" and "bank" are treated as the same term
    - Queries match documents regardless of case

    Parameters:
    -----------
    text : str
        The input text to normalize

    Returns:
    --------
    str
        Normalized (lowercase) text

    Example:
    --------
    >>> normalize("Hello WORLD")
    'hello world'
    """
    if not text:
        return ""

    # Convert to lowercase
    return text.lower()


def remove_stopwords(tokens, stopwords=None):
    """
    Remove stop words from a list of tokens.

    Stop words are common words (like "the", "is", "at", etc.) that:
    - Appear very frequently (high in Zipf distribution)
    - Have low discriminative power for retrieval
    - Can be removed to save space and improve efficiency

    Parameters:
    -----------
    tokens : list of str
        List of tokens to filter
    stopwords : set or frozenset, optional
        Set of stop words to remove. If None, uses ENGLISH_STOP_WORDS

    Returns:
    --------
    list of str
        Filtered list of tokens with stop words removed

    Example:
    --------
    >>> remove_stopwords(['this', 'is', 'a', 'test', 'document'])
    ['test', 'document']
    """
    if stopwords is None:
        stopwords = ENGLISH_STOP_WORDS

    # Filter out tokens that are in the stop words set
    # Using lowercase comparison to ensure case-insensitive matching
    return [token for token in tokens if token.lower() not in stopwords]


def tokenize_and_process(text, remove_stops=True, custom_stopwords=None):
    """
    Complete text processing pipeline: normalize, tokenize, and optionally remove stop words.

    This is the main function to use for processing both documents and queries.
    It combines all text processing steps:
    1. Normalization (lowercasing)
    2. Tokenization (splitting into words)
    3. Stop word removal (optional)

    Parameters:
    -----------
    text : str
        The input text to process
    remove_stops : bool, default=True
        Whether to remove stop words
    custom_stopwords : set or frozenset, optional
        Custom stop words set. If None, uses ENGLISH_STOP_WORDS

    Returns:
    --------
    list of str
        Processed list of tokens

    Example:
    --------
    >>> tokenize_and_process("The quick brown fox jumps over the lazy dog")
    ['quick', 'brown', 'fox', 'jumps', 'lazy', 'dog']

    >>> tokenize_and_process("Hello World!", remove_stops=False)
    ['hello', 'world']
    """
    # Step 1: Normalize (convert to lowercase)
    normalized_text = normalize(text)

    # Step 2: Tokenize (split into words)
    tokens = tokenize(normalized_text)

    # Step 3: Remove stop words (if requested)
    if remove_stops:
        tokens = remove_stopwords(tokens, custom_stopwords)

    return tokens


def get_term_counts(tokens):
    """
    Count the frequency of each term in a list of tokens.

    This is useful for:
    - Building inverted indices with term frequencies (TF)
    - Calculating TF-IDF weights
    - Understanding document content

    Parameters:
    -----------
    tokens : list of str
        List of tokens to count

    Returns:
    --------
    Counter
        Dictionary-like object mapping terms to their frequencies

    Example:
    --------
    >>> tokens = ['apple', 'banana', 'apple', 'cherry', 'banana', 'apple']
    >>> get_term_counts(tokens)
    Counter({'apple': 3, 'banana': 2, 'cherry': 1})
    """
    return Counter(tokens)


# Example usage and testing
if __name__ == "__main__":
    # Test text
    test_text = "The quick brown fox jumps over the lazy dog. It's a beautiful day!"

    print("Original text:")
    print(test_text)
    print()

    # Test normalization
    print("Normalized text:")
    print(normalize(test_text))
    print()

    # Test tokenization
    print("Tokens:")
    tokens = tokenize(normalize(test_text))
    print(tokens)
    print()

    # Test tokenization with stop word removal
    print("Tokens (without stop words):")
    filtered_tokens = tokenize_and_process(test_text, remove_stops=True)
    print(filtered_tokens)
    print()

    # Test term counting
    print("Term frequencies:")
    term_counts = get_term_counts(filtered_tokens)
    print(term_counts)
    print()

    # Test with query
    query = "How does the quick brown fox jump?"
    print(f"Query: {query}")
    print("Processed query tokens:")
    query_tokens = tokenize_and_process(query)
    print(query_tokens)
