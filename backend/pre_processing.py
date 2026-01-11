
import re
import nltk
from collections import Counter
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer

# 1. THE ASSIGNMENT 1 REGEX
RE_WORD = re.compile(r"[\#\w](['\w-]*\w)?", re.UNICODE | re.VERBOSE)

def _get_stop_words():
    """Returns a hardcoded list of NLTK stopwords to avoid network/permission issues in Spark Workers."""
    return frozenset(['i', 'me', 'my', 'myself', 'we', 'our', 'ours', 'ourselves', 'you', "you're", "you've", "you'll", "you'd", 'your', 'yours', 'yourself', 'yourselves', 'he', 'him', 'his', 'himself', 'she', "she's", 'her', 'hers', 'herself', 'it', "it's", 'its', 'itself', 'they', 'them', 'their', 'theirs', 'themselves', 'what', 'which', 'who', 'whom', 'this', 'that', "that'll", 'these', 'those', 'am', 'is', 'are', 'was', 'were', 'be', 'been', 'being', 'have', 'has', 'had', 'having', 'do', 'does', 'did', 'doing', 'a', 'an', 'the', 'and', 'but', 'if', 'or', 'because', 'as', 'until', 'while', 'of', 'at', 'by', 'for', 'with', 'about', 'against', 'between', 'into', 'through', 'during', 'before', 'after', 'above', 'below', 'to', 'from', 'up', 'down', 'in', 'out', 'on', 'off', 'over', 'under', 'again', 'further', 'then', 'once', 'here', 'there', 'when', 'where', 'why', 'how', 'all', 'any', 'both', 'each', 'few', 'more', 'most', 'other', 'some', 'such', 'no', 'nor', 'not', 'only', 'own', 'same', 'so', 'than', 'too', 'very', 's', 't', 'can', 'will', 'just', 'don', "don't", 'should', "should've", 'now', 'd', 'll', 'm', 'o', 're', 've', 'y', 'ain', 'aren', "aren't", 'couldn', "couldn't", 'didn', "didn't", 'doesn', "doesn't", 'hadn', "hadn't", 'hasn', "hasn't", 'haven', "haven't", 'isn', "isn't", 'ma', 'mightn', "mightn't", 'mustn', "mustn't", 'needn', "needn't", 'shan', "shan't", 'shouldn', "shouldn't", 'wasn', "wasn't", 'weren', "weren't", 'won', "won't", 'wouldn', "wouldn't"])
ENGLISH_STOP_WORDS = _get_stop_words()
_stemmer = PorterStemmer()

def _get_stemmer():
    global _stemmer
    if _stemmer is None:
        _stemmer = PorterStemmer()
    return _stemmer

def tokenize_and_process(text, remove_stops=True, stem=True):
    if not text:
        return []

    # 1. Tokenize (Assignment 1 Regex)
    tokens = [token.group() for token in RE_WORD.finditer(text.lower())]

    # 2. Stopword Removal
    if remove_stops:
        tokens = [t for t in tokens if t not in ENGLISH_STOP_WORDS]

    # 3. Stemming & Cleanup
    stemmer = _get_stemmer()
    processed_tokens = []
    
    for t in tokens:
        # Step 3a: Stem if requested
        if stem:
            t = stemmer.stem(t)
        
        # Step 3b: Strip trailing apostrophes (Fixes "user's" -> "user'" -> "user")
        t = t.rstrip("'")
        
        if t: # Ensure we don't add empty strings
            processed_tokens.append(t)

    return processed_tokens

def get_term_counts(tokens):
    return Counter(tokens)