"""
Index Builder for Wikipedia Search Engine

This module provides utilities for building inverted indices from Wikipedia documents.
It integrates with the text_processing module for tokenization and the inverted_index_gcp
module for index storage.

Three types of indices are built:
1. Body Index: For TF-IDF cosine similarity search on article body text
2. Title Index: For binary ranking on article titles
3. Anchor Index: For binary ranking on anchor text (link text pointing to articles)

The implementation uses an inverted index approach where:
- Each term maps to a posting list of (doc_id, term_frequency) pairs
- Document frequency (DF) is maintained for each term
- Total term frequency is tracked across all documents
- Posting lists are written to disk to handle large-scale data

Usage:
    from backend.index_builder import IndexBuilder

    builder = IndexBuilder()

    # Add documents
    builder.add_document(
        doc_id=1,
        title="Python",
        body="Python is a programming language",
        anchors=["programming", "python language"]
    )

    # Build and write indices
    builder.build_indices(output_dir="indices/")
"""

import logging
from collections import defaultdict, Counter
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Set
import pickle
import json
# FIXED: Use relative imports for backend package
from .pre_processing import tokenize_and_process
from .inverted_index_gcp import InvertedIndex

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class IndexBuilder:
    """
    Builder class for creating inverted indices for Wikipedia search engine.

    Creates three separate indices:
    1. Body Index: For TF-IDF cosine similarity search on article body text
    2. Title Index: For binary ranking on article titles
    3. Anchor Index: For binary ranking on anchor text (link text pointing to articles)

    The inverted index approach used:
    - Term-based inverted index: Maps each term to a posting list
    - Posting list: List of (doc_id, term_frequency) tuples
    - Supports efficient query processing for ranked retrieval
    - Stores document frequency (DF) and term totals for TF-IDF calculation

    Attributes:
        body_docs (dict): Mapping of doc_id to body tokens
        title_docs (dict): Mapping of doc_id to title tokens
        anchor_docs (dict): Mapping of doc_id to anchor tokens
        doc_titles (dict): Mapping of doc_id to original title string
        doc_lengths (dict): Mapping of doc_id to document length (for normalization)
        num_docs (int): Total number of documents processed
    """

    def __init__(self):
        """Initialize empty index builder."""
        # Document collections for each index type
        # These store the tokenized versions of documents
        self.body_docs = {}      # doc_id -> [body_tokens]
        self.title_docs = {}     # doc_id -> [title_tokens]
        self.anchor_docs = {}    # doc_id -> [anchor_tokens]

        # Metadata storage
        self.doc_titles = {}     # doc_id -> original title string
        self.doc_lengths = {}    # doc_id -> body document length (for normalization)

        # Statistics
        self.num_docs = 0

        logger.info("IndexBuilder initialized")

    def add_document(self,
                     doc_id: int,
                     title: str = "",
                     body: str = "",
                     anchors: Optional[List[str]] = None):
        """
        Add a single document to the index builder.

        The document is processed through the text_processing pipeline:
        1. Normalization (lowercase)
        2. Tokenization (regex-based word extraction)
        3. Stop word removal

        Parameters:
        -----------
        doc_id : int
            Unique document identifier (Wikipedia article ID)
        title : str, optional
            Article title
        body : str, optional
            Article body text
        anchors : list of str, optional
            List of anchor texts (link texts) pointing to this article

        Example:
        --------
        >>> builder = IndexBuilder()
        >>> builder.add_document(
        ...     doc_id=12345,
        ...     title="Python (programming language)",
        ...     body="Python is a high-level programming language...",
        ...     anchors=["Python", "python programming", "Python language"]
        ... )
        """
        # Store original title for retrieval results
        self.doc_titles[doc_id] = title

        # Process title
        # CHANGED: Use stem=True to match query-time processing (fixes stemming mismatch bug)
        if title:
            title_tokens = tokenize_and_process(title, remove_stops=True, stem=True)
            if title_tokens:  # Only add if there are tokens after processing
                self.title_docs[doc_id] = title_tokens

        # Process body
        # CHANGED: Use stem=True to match query-time processing (fixes stemming mismatch bug)
        if body:
            body_tokens = tokenize_and_process(body, remove_stops=True, stem=True)
            if body_tokens:  # Only add if there are tokens after processing
                self.body_docs[doc_id] = body_tokens
                self.doc_lengths[doc_id] = len(body_tokens)

        # Process anchors
        # CHANGED: Use stem=True to match query-time processing (fixes stemming mismatch bug)
        if anchors:
            # Combine all anchor texts for this document
            all_anchor_text = " ".join(anchors)
            anchor_tokens = tokenize_and_process(all_anchor_text, remove_stops=True, stem=True)
            if anchor_tokens:  # Only add if there are tokens after processing
                self.anchor_docs[doc_id] = anchor_tokens

        self.num_docs += 1

        # Log progress every 10,000 documents
        if self.num_docs % 10000 == 0:
            logger.info(f"Processed {self.num_docs:,} documents")

    def add_documents_batch(self, documents: List[Dict]):
        """
        Add multiple documents in batch.

        Parameters:
        -----------
        documents : list of dict
            List of document dictionaries, each containing:
            - 'doc_id': int (required)
            - 'title': str (optional)
            - 'body': str (optional)
            - 'anchors': list of str (optional)

        Example:
        --------
        >>> docs = [
        ...     {'doc_id': 1, 'title': 'Python', 'body': 'Python is...'},
        ...     {'doc_id': 2, 'title': 'Java', 'body': 'Java is...'}
        ... ]
        >>> builder.add_documents_batch(docs)
        """
        logger.info(f"Adding batch of {len(documents):,} documents")

        for doc in documents:
            self.add_document(
                doc_id=doc['doc_id'],
                title=doc.get('title', ''),
                body=doc.get('body', ''),
                anchors=doc.get('anchors', None)
            )

        logger.info(f"Batch complete. Total documents: {self.num_docs:,}")

    def build_indices(self,
                      output_dir: str = "indices",
                      bucket_name: Optional[str] = None) -> Tuple[InvertedIndex, InvertedIndex, InvertedIndex]:
        """
        Build all three inverted indices and write them to disk/GCP.

        Creates:
        1. Body Index: Used for TF-IDF cosine similarity ranking
        2. Title Index: Used for binary ranking based on query terms in titles
        3. Anchor Index: Used for binary ranking based on query terms in anchor text

        Each index is an InvertedIndex object that maintains:
        - df: Document frequency for each term
        - term_total: Total frequency of each term across all documents
        - posting_locs: File locations of posting lists on disk

        Parameters:
        -----------
        output_dir : str, default="indices"
            Directory to write index files (local or GCP path)
        bucket_name : str, optional
            GCP bucket name for cloud storage. If None, writes locally.

        Returns:
        --------
        tuple of InvertedIndex
            (body_index, title_index, anchor_index)

        Example:
        --------
        >>> builder = IndexBuilder()
        >>> # ... add documents ...
        >>> body_idx, title_idx, anchor_idx = builder.build_indices("indices/")
        >>> print(f"Body index has {len(body_idx.df)} unique terms")
        """
        logger.info("="*70)
        logger.info("Building inverted indices...")
        logger.info("="*70)

        # Create output directory if it doesn't exist (for local storage)
        if bucket_name is None:
            output_path = Path(output_dir)
            output_path.mkdir(parents=True, exist_ok=True)

        # Build body index
        logger.info(f"\n[1/3] Building BODY index with {len(self.body_docs):,} documents")
        body_index = InvertedIndex(docs=self.body_docs)
        body_index.write_index(output_dir, 'body_index', bucket_name)
        logger.info(f"✓ Body index complete: {len(body_index.df):,} unique terms, "
                   f"{sum(body_index.df.values()):,} postings")

        # Build title index
        logger.info(f"\n[2/3] Building TITLE index with {len(self.title_docs):,} documents")
        title_index = InvertedIndex(docs=self.title_docs)
        title_index.write_index(output_dir, 'title_index', bucket_name)
        logger.info(f"✓ Title index complete: {len(title_index.df):,} unique terms, "
                   f"{sum(title_index.df.values()):,} postings")

        # Build anchor index
        logger.info(f"\n[3/3] Building ANCHOR index with {len(self.anchor_docs):,} documents")
        anchor_index = InvertedIndex(docs=self.anchor_docs)
        anchor_index.write_index(output_dir, 'anchor_index', bucket_name)
        logger.info(f"✓ Anchor index complete: {len(anchor_index.df):,} unique terms, "
                   f"{sum(anchor_index.df.values()):,} postings")

        # Save metadata
        self._save_metadata(output_dir, bucket_name)

        logger.info("\n" + "="*70)
        logger.info("All indices built successfully!")
        logger.info("="*70)

        return body_index, title_index, anchor_index

    def _save_metadata(self, output_dir: str, bucket_name: Optional[str] = None):
        """
        Save document metadata (titles, lengths, etc.) to disk.

        Metadata includes:
        - doc_titles: Mapping of doc_id to title string
        - doc_lengths: Mapping of doc_id to body length
        - num_docs: Total number of documents
        """
        metadata = {
            'doc_titles': self.doc_titles,
            'doc_lengths': self.doc_lengths,
            'num_docs': self.num_docs
        }

        metadata_path = Path(output_dir) / 'metadata.pkl'

        if bucket_name is None:
            # Local storage
            with open(metadata_path, 'wb') as f:
                pickle.dump(metadata, f)
        else:
            # GCP storage
            from backend.inverted_index_gcp import get_bucket, _open
            bucket = get_bucket(bucket_name)
            with _open(str(metadata_path), 'wb', bucket) as f:
                pickle.dump(metadata, f)

        logger.info(f"✓ Metadata saved: {self.num_docs:,} documents")

    @staticmethod
    def load_metadata(output_dir: str, bucket_name: Optional[str] = None) -> Dict:
        """
        Load document metadata from disk.

        Parameters:
        -----------
        output_dir : str
            Directory containing metadata file
        bucket_name : str, optional
            GCP bucket name. If None, loads from local storage.

        Returns:
        --------
        dict
            Metadata dictionary with doc_titles, doc_lengths, num_docs
        """
        metadata_path = Path(output_dir) / 'metadata.pkl'

        if bucket_name is None:
            # Local storage
            with open(metadata_path, 'rb') as f:
                return pickle.load(f)
        else:
            # GCP storage
            from backend.inverted_index_gcp import get_bucket, _open
            bucket = get_bucket(bucket_name)
            with _open(str(metadata_path), 'rb', bucket) as f:
                return pickle.load(f)

    def get_stats(self) -> Dict:
        """
        Get statistics about the documents processed.

        Returns:
        --------
        dict
            Statistics including document counts, vocabulary sizes, avg lengths
        """
        # Calculate unique terms (vocabulary) for each index type
        body_vocab = set()
        for tokens in self.body_docs.values():
            body_vocab.update(tokens)

        title_vocab = set()
        for tokens in self.title_docs.values():
            title_vocab.update(tokens)

        anchor_vocab = set()
        for tokens in self.anchor_docs.values():
            anchor_vocab.update(tokens)

        # Calculate average document lengths
        avg_body_length = (sum(self.doc_lengths.values()) / len(self.doc_lengths)
                          if self.doc_lengths else 0)

        avg_title_length = (sum(len(t) for t in self.title_docs.values()) / len(self.title_docs)
                           if self.title_docs else 0)

        avg_anchor_length = (sum(len(t) for t in self.anchor_docs.values()) / len(self.anchor_docs)
                            if self.anchor_docs else 0)

        stats = {
            'total_documents': self.num_docs,
            'documents_with_body': len(self.body_docs),
            'documents_with_title': len(self.title_docs),
            'documents_with_anchors': len(self.anchor_docs),
            'body_vocabulary_size': len(body_vocab),
            'title_vocabulary_size': len(title_vocab),
            'anchor_vocabulary_size': len(anchor_vocab),
            'avg_body_length': avg_body_length,
            'avg_title_length': avg_title_length,
            'avg_anchor_length': avg_anchor_length,
        }

        return stats

    def print_stats(self):
        """Print formatted statistics about the documents processed."""
        stats = self.get_stats()

        print("\n" + "="*70)
        print("INDEX BUILDER STATISTICS")
        print("="*70)
        print(f"Total Documents:           {stats['total_documents']:>12,}")
        print(f"Documents with Body:       {stats['documents_with_body']:>12,}")
        print(f"Documents with Title:      {stats['documents_with_title']:>12,}")
        print(f"Documents with Anchors:    {stats['documents_with_anchors']:>12,}")
        print()
        print(f"Body Vocabulary Size:      {stats['body_vocabulary_size']:>12,} unique terms")
        print(f"Title Vocabulary Size:     {stats['title_vocabulary_size']:>12,} unique terms")
        print(f"Anchor Vocabulary Size:    {stats['anchor_vocabulary_size']:>12,} unique terms")
        print()
        print(f"Avg Body Length:           {stats['avg_body_length']:>12.1f} tokens")
        print(f"Avg Title Length:          {stats['avg_title_length']:>12.1f} tokens")
        print(f"Avg Anchor Length:         {stats['avg_anchor_length']:>12.1f} tokens")
        print("="*70 + "\n")


# ==============================================================================
# Utility Functions for Wikipedia Processing
# ==============================================================================

def process_wikipedia_dump(dump_path: str, output_dir: str = "indices") -> IndexBuilder:
    """
    Process a Wikipedia dump and build indices.

    This is a placeholder function. In practice, you would use libraries like:
    - gensim.corpora.WikiCorpus
    - mwxml
    - or custom XML parsing

    Parameters:
    -----------
    dump_path : str
        Path to Wikipedia XML dump file
    output_dir : str
        Directory to write index files

    Returns:
    --------
    IndexBuilder
        Populated index builder (not yet built)
    """
    logger.warning("process_wikipedia_dump is a placeholder. Implement with WikiCorpus or mwxml.")
    return IndexBuilder()


def load_training_queries(queries_path: str = "data/queries_train.json") -> List[Dict]:
    """
    Load training queries from JSON file.

    Parameters:
    -----------
    queries_path : str
        Path to queries JSON file

    Returns:
    --------
    list of dict
        List of queries with query text and relevant doc IDs
    """
    with open(queries_path, 'r', encoding='utf-8') as f:
        queries = json.load(f)
    return queries


# ==============================================================================
# Example Usage and Testing
# ==============================================================================

if __name__ == "__main__":
    print("\n" + "="*70)
    print("INDEX BUILDER MODULE - EXAMPLE USAGE")
    print("="*70 + "\n")

    # Create index builder
    builder = IndexBuilder()

    # Sample documents (simulating Wikipedia articles)
    sample_docs = [
        {
            'doc_id': 1,
            'title': 'Python (programming language)',
            'body': '''Python is a high-level, interpreted programming language with dynamic
                      semantics. It was created by Guido van Rossum and first released in 1991.
                      Python's design philosophy emphasizes code readability with its notable use
                      of significant indentation. Python is dynamically typed and garbage-collected.
                      It supports multiple programming paradigms including structured, object-oriented
                      and functional programming.''',
            'anchors': ['Python', 'Python programming', 'Python language', 'Python software']
        },
        {
            'doc_id': 2,
            'title': 'Information retrieval',
            'body': '''Information retrieval is the process of obtaining information system
                      resources that are relevant to an information need from a collection of
                      those resources. Searches can be based on full-text or other content-based
                      indexing. Information retrieval is the science of searching for information
                      in a document, searching for documents themselves, and also searching for
                      metadata that describe data, and for databases of texts, images or sounds.''',
            'anchors': ['IR', 'information retrieval', 'search', 'document retrieval']
        },
        {
            'doc_id': 3,
            'title': 'Machine learning',
            'body': '''Machine learning is a field of inquiry devoted to understanding and
                      building methods that learn from data and improve their performance on
                      some set of tasks. It is seen as a part of artificial intelligence.
                      Machine learning algorithms build a model based on sample data, known as
                      training data, in order to make predictions or decisions without being
                      explicitly programmed to do so.''',
            'anchors': ['ML', 'machine learning', 'learning algorithms', 'AI learning']
        },
        {
            'doc_id': 4,
            'title': 'Natural language processing',
            'body': '''Natural language processing is an interdisciplinary subfield of linguistics,
                      computer science, and artificial intelligence concerned with the interactions
                      between computers and human language, in particular how to program computers
                      to process and analyze large amounts of natural language data. The goal is
                      a computer capable of understanding the contents of documents.''',
            'anchors': ['NLP', 'natural language processing', 'language processing', 'NLP systems']
        },
        {
            'doc_id': 5,
            'title': 'Search engine',
            'body': '''A search engine is a software system that is designed to carry out web
                      searches, which means to search the World Wide Web in a systematic way for
                      particular information specified in a textual web search query. The search
                      results are generally presented in a line of results, often referred to as
                      search engine results pages.''',
            'anchors': ['search engine', 'web search', 'search', 'search system']
        },
        {
            'doc_id': 6,
            'title': 'TF-IDF',
            'body': '''TF-IDF (term frequency-inverse document frequency) is a numerical statistic
                      that is intended to reflect how important a word is to a document in a collection
                      or corpus. It is often used as a weighting factor in searches of information
                      retrieval, text mining, and user modeling. The tf-idf value increases proportionally
                      to the number of times a word appears in the document.''',
            'anchors': ['TF-IDF', 'term frequency', 'inverse document frequency', 'tf idf']
        }
    ]

    # Add documents
    print("Adding sample documents...")
    builder.add_documents_batch(sample_docs)

    # Print statistics
    builder.print_stats()

    # Build indices
    print("Building indices...\n")
    body_idx, title_idx, anchor_idx = builder.build_indices(output_dir="test_indices")

    print("\n" + "="*70)
    print("INDEX BUILDING COMPLETE!")
    print("="*70)
    print(f"\nOutput directory: test_indices/")
    print("\nGenerated files:")
    print("  - body_index.pkl        (body text index)")
    print("  - title_index.pkl       (title index)")
    print("  - anchor_index.pkl      (anchor text index)")
    print("  - metadata.pkl          (document metadata)")
    print("  - *_posting_locs.pickle (posting list locations)")

    # Test reading the indices back
    print("\n" + "="*70)
    print("TESTING INDEX READING")
    print("="*70)

    loaded_body_idx = InvertedIndex.read_index("test_indices", "body_index")
    loaded_title_idx = InvertedIndex.read_index("test_indices", "title_index")
    loaded_anchor_idx = InvertedIndex.read_index("test_indices", "anchor_index")

    print(f"\nBody Index:   {len(loaded_body_idx.df):,} unique terms")
    print(f"Title Index:  {len(loaded_title_idx.df):,} unique terms")
    print(f"Anchor Index: {len(loaded_anchor_idx.df):,} unique terms")

    # Show sample terms
    print("\nSample terms from body index:")
    sample_terms = list(loaded_body_idx.df.keys())[:10]
    for term in sample_terms:
        df = loaded_body_idx.df[term]
        total_tf = loaded_body_idx.term_total[term]
        print(f"  '{term}': DF={df}, Total TF={total_tf}")

    # Load metadata
    metadata = IndexBuilder.load_metadata("test_indices")
    print(f"\nMetadata loaded: {metadata['num_docs']} documents")
    print(f"Sample document titles:")
    for doc_id, title in list(metadata['doc_titles'].items())[:3]:
        print(f"  Doc {doc_id}: {title}")

    print("\n" + "="*70)
    print("EXAMPLE COMPLETED SUCCESSFULLY!")
    print("="*70 + "\n")
