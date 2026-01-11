"""
Wikipedia Search Engine - Backend Module

This package contains the core indexing and retrieval components:
- Text preprocessing with Porter stemming
- Inverted index construction
- Distributed PySpark indexing
- Champion lists and adaptive retrieval
- PageRank computation
- Index health validation
"""

__version__ = "1.0.0"
__author__ = "IR Project Team"

# Import key classes for convenient access
from .pre_processing import tokenize_and_process, get_term_counts
from .inverted_index_gcp import InvertedIndex
from .index_builder import IndexBuilder
from .index_health_checker import IndexHealthCheck

__all__ = [
    'tokenize_and_process',
    'get_term_counts',
    'InvertedIndex',
    'IndexBuilder',
    'IndexHealthCheck',
]
