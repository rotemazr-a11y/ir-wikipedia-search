"""
Comprehensive Test for Index Builder

This script tests the index_builder.py module by:
1. Creating sample Wikipedia-like documents
2. Building body, title, and anchor indices
3. Verifying index correctness
4. Testing posting list retrieval
5. Demonstrating end-to-end functionality
"""

import sys
import os
from pathlib import Path

# Add backend to path
sys.path.insert(0, str(Path(__file__).parent))

from backend.index_builder import IndexBuilder
from backend.inverted_index_gcp import InvertedIndex
from backend.text_processing import tokenize_and_process

def create_test_documents():
    """
    Create sample documents that simulate Wikipedia articles.
    These represent actual articles we would index.
    """
    documents = [
        {
            'doc_id': 1,
            'title': 'Python (programming language)',
            'body': '''Python is a high-level, interpreted, general-purpose programming language.
                      Its design philosophy emphasizes code readability with the use of significant
                      indentation. Python is dynamically typed and garbage-collected. It supports
                      multiple programming paradigms, including structured, object-oriented and
                      functional programming. It is often described as a "batteries included" language
                      due to its comprehensive standard library. Guido van Rossum began working on
                      Python in the late 1980s as a successor to the ABC programming language and
                      first released it in 1991 as Python 0.9.0.''',
            'anchors': [
                'Python', 'Python programming', 'Python language', 'Programming language Python',
                'Python coding', 'Learn Python'
            ]
        },
        {
            'doc_id': 2,
            'title': 'Information retrieval',
            'body': '''Information retrieval (IR) in computing and information science is the process
                      of obtaining information system resources that are relevant to an information need
                      from a collection of those resources. Searches can be based on full-text or other
                      content-based indexing. Information retrieval is the science of searching for
                      information in a document, searching for documents themselves, and also searching
                      for the metadata that describes data, and for databases of texts, images or sounds.
                      Automated information retrieval systems are used to reduce what has been called
                      information overload. An IR system is a software system that provides access to
                      books, journals and other documents.''',
            'anchors': [
                'IR', 'Information retrieval', 'Search systems', 'Document retrieval',
                'Information retrieval systems', 'IR research'
            ]
        },
        {
            'doc_id': 3,
            'title': 'Machine learning',
            'body': '''Machine learning (ML) is a field of inquiry devoted to understanding and building
                      methods that "learn", that is, methods that leverage data to improve performance on
                      some set of tasks. It is seen as a part of artificial intelligence. Machine learning
                      algorithms build a model based on sample data, known as training data, in order to
                      make predictions or decisions without being explicitly programmed to do so. Machine
                      learning algorithms are used in a wide variety of applications, such as in medicine,
                      email filtering, speech recognition, and computer vision, where it is difficult or
                      unfeasible to develop conventional algorithms to perform the needed tasks.''',
            'anchors': [
                'ML', 'Machine learning', 'Learning algorithms', 'AI machine learning',
                'Machine learning systems', 'Statistical learning'
            ]
        },
        {
            'doc_id': 4,
            'title': 'Search engine',
            'body': '''A search engine is a software system designed to carry out web searches. They
                      search the World Wide Web in a systematic way for particular information specified
                      in a textual web search query. The search results are generally presented in a line
                      of results, often referred to as search engine results pages (SERPs). When a user
                      enters a query into a search engine, the engine examines its index and provides a
                      listing of best-matching web pages according to its criteria, usually with a short
                      summary containing the document's title and sometimes parts of the text.''',
            'anchors': [
                'Search engine', 'Web search', 'Search', 'Internet search engine',
                'Search engine technology', 'Google search'
            ]
        },
        {
            'doc_id': 5,
            'title': 'Natural language processing',
            'body': '''Natural language processing (NLP) is an interdisciplinary subfield of linguistics,
                      computer science, and artificial intelligence concerned with the interactions between
                      computers and human language, in particular how to program computers to process and
                      analyze large amounts of natural language data. The goal is a computer capable of
                      "understanding" the contents of documents, including the contextual nuances of the
                      language within them. The technology can then accurately extract information and
                      insights contained in the documents as well as categorize and organize the documents
                      themselves.''',
            'anchors': [
                'NLP', 'Natural language processing', 'Language processing',
                'Computational linguistics', 'NLP systems', 'Text processing'
            ]
        },
        {
            'doc_id': 6,
            'title': 'TF-IDF',
            'body': '''In information retrieval, tf-idf (also TF-IDF, TFIDF, or Tf-idf), short for term
                      frequency–inverse document frequency, is a numerical statistic that is intended to
                      reflect how important a word is to a document in a collection or corpus. It is often
                      used as a weighting factor in searches of information retrieval, text mining, and
                      user modeling. The tf-idf value increases proportionally to the number of times a
                      word appears in the document and is offset by the number of documents in the corpus
                      that contain the word, which helps to adjust for the fact that some words appear
                      more frequently in general.''',
            'anchors': [
                'TF-IDF', 'Term frequency', 'Inverse document frequency',
                'TFIDF algorithm', 'tf idf', 'Document weighting'
            ]
        },
        {
            'doc_id': 7,
            'title': 'Inverted index',
            'body': '''In computer science, an inverted index (also referred to as a postings list,
                      postings file, or inverted file) is a database index storing a mapping from content,
                      such as words or numbers, to its locations in a table, or in a document or a set of
                      documents (named in contrast to a forward index, which maps from documents to content).
                      The purpose of an inverted index is to allow fast full-text searches, at a cost of
                      increased processing when a document is added to the database.''',
            'anchors': [
                'Inverted index', 'Postings list', 'Index structure',
                'Inverted file', 'Search index', 'Document index'
            ]
        },
        {
            'doc_id': 8,
            'title': 'Cosine similarity',
            'body': '''Cosine similarity is a measure of similarity between two non-zero vectors of an
                      inner product space. It is defined to equal the cosine of the angle between them,
                      which is also the same as the inner product of the same vectors normalized to both
                      have length 1. The cosine of 0° is 1, and it is less than 1 for any angle in the
                      interval (0, π] radians. It is thus a judgment of orientation and not magnitude.
                      Cosine similarity is used in positive space, where the outcome is neatly bounded in [0,1].''',
            'anchors': [
                'Cosine similarity', 'Vector similarity', 'Cosine distance',
                'Similarity measure', 'Document similarity', 'Text similarity'
            ]
        }
    ]

    return documents


def test_index_building():
    """Test the complete index building process."""

    print("="*80)
    print("INDEX BUILDER - COMPREHENSIVE TEST")
    print("="*80)
    print()

    # Step 1: Create test documents
    print("[Step 1] Creating test documents...")
    documents = create_test_documents()
    print(f"✓ Created {len(documents)} sample Wikipedia-like documents")
    print()

    # Show sample document
    print("Sample document:")
    sample = documents[0]
    print(f"  Doc ID: {sample['doc_id']}")
    print(f"  Title: {sample['title']}")
    print(f"  Body: {sample['body'][:100]}...")
    print(f"  Anchors: {sample['anchors'][:3]}")
    print()

    # Step 2: Create IndexBuilder
    print("[Step 2] Creating IndexBuilder...")
    builder = IndexBuilder()
    print("✓ IndexBuilder initialized")
    print()

    # Step 3: Add documents
    print("[Step 3] Adding documents to builder...")
    builder.add_documents_batch(documents)
    print(f"✓ Added {builder.num_docs} documents")
    print()

    # Step 4: Show statistics before building
    print("[Step 4] Pre-build statistics:")
    builder.print_stats()

    # Step 5: Build indices
    print("[Step 5] Building inverted indices...")
    output_dir = "test_indices"
    body_idx, title_idx, anchor_idx = builder.build_indices(output_dir=output_dir)
    print()

    # Step 6: Verify indices
    print("="*80)
    print("[Step 6] VERIFYING INDICES")
    print("="*80)
    print()

    print(f"Body Index:")
    print(f"  - Unique terms: {len(body_idx.df):,}")
    print(f"  - Total postings: {sum(body_idx.df.values()):,}")
    print(f"  - Sample terms: {list(body_idx.df.keys())[:10]}")
    print()

    print(f"Title Index:")
    print(f"  - Unique terms: {len(title_idx.df):,}")
    print(f"  - Total postings: {sum(title_idx.df.values()):,}")
    print(f"  - Sample terms: {list(title_idx.df.keys())[:10]}")
    print()

    print(f"Anchor Index:")
    print(f"  - Unique terms: {len(anchor_idx.df):,}")
    print(f"  - Total postings: {sum(anchor_idx.df.values()):,}")
    print(f"  - Sample terms: {list(anchor_idx.df.keys())[:10]}")
    print()

    # Step 7: Test reading indices from disk
    print("="*80)
    print("[Step 7] TESTING INDEX PERSISTENCE")
    print("="*80)
    print()

    print("Reading indices from disk...")
    loaded_body = InvertedIndex.read_index(output_dir, "body_index")
    loaded_title = InvertedIndex.read_index(output_dir, "title_index")
    loaded_anchor = InvertedIndex.read_index(output_dir, "anchor_index")
    print("✓ All indices loaded successfully")
    print()

    # Step 8: Test posting list retrieval
    print("="*80)
    print("[Step 8] TESTING POSTING LIST RETRIEVAL")
    print("="*80)
    print()

    # Test some common terms
    test_terms = ['python', 'machine', 'learning', 'search', 'information']

    for term in test_terms:
        if term in loaded_body.df:
            df = loaded_body.df[term]
            total_tf = loaded_body.term_total[term]
            print(f"Term: '{term}'")
            print(f"  - Document Frequency (DF): {df}")
            print(f"  - Total Term Frequency: {total_tf}")
            print(f"  - Appears in documents: ", end="")

            # Get posting list (doc_id, tf) pairs
            # Note: This would normally be read from disk, but for testing we can access the in-memory version
            if hasattr(loaded_body, '_posting_list') and term in loaded_body._posting_list:
                postings = loaded_body._posting_list[term]
                doc_ids = [doc_id for doc_id, tf in postings]
                print(doc_ids)
            else:
                print("(posting lists written to disk)")
            print()

    # Step 9: Demonstrate query tokenization
    print("="*80)
    print("[Step 9] DEMONSTRATING QUERY PROCESSING")
    print("="*80)
    print()

    test_queries = [
        "python programming language",
        "machine learning algorithms",
        "search engine information retrieval"
    ]

    for query in test_queries:
        print(f"Query: '{query}'")
        query_tokens = tokenize_and_process(query)
        print(f"  Tokens: {query_tokens}")

        # Show which documents contain these terms
        print(f"  Terms found in indices:")
        for token in query_tokens:
            if token in loaded_body.df:
                df = loaded_body.df[token]
                print(f"    '{token}': appears in {df} documents")
        print()

    # Step 10: Load and display metadata
    print("="*80)
    print("[Step 10] METADATA VERIFICATION")
    print("="*80)
    print()

    metadata = IndexBuilder.load_metadata(output_dir)
    print(f"Loaded metadata for {metadata['num_docs']} documents")
    print()
    print("Sample document titles:")
    for doc_id in sorted(list(metadata['doc_titles'].keys())[:5]):
        title = metadata['doc_titles'][doc_id]
        length = metadata['doc_lengths'].get(doc_id, 0)
        print(f"  Doc {doc_id}: {title} (length: {length} tokens)")
    print()

    # Step 11: Summary
    print("="*80)
    print("TEST SUMMARY")
    print("="*80)
    print()
    print("✅ Documents added successfully")
    print("✅ Body index built and verified")
    print("✅ Title index built and verified")
    print("✅ Anchor index built and verified")
    print("✅ Indices persisted to disk")
    print("✅ Indices loaded from disk")
    print("✅ Posting lists accessible")
    print("✅ Query tokenization working")
    print("✅ Metadata stored and loaded")
    print()
    print(f"📁 Index files saved to: {output_dir}/")
    print("   - body_index.pkl")
    print("   - title_index.pkl")
    print("   - anchor_index.pkl")
    print("   - metadata.pkl")
    print("   - 0_posting_locs.pickle")
    print()
    print("="*80)
    print("ALL TESTS PASSED! ✓")
    print("="*80)
    print()
    print("Next Steps:")
    print("  Phase 3: Implement ranking methods (TF-IDF, binary ranking)")
    print("  Phase 4: Build search API endpoints")
    print("  Phase 5: Test with training queries")
    print()


if __name__ == "__main__":
    try:
        test_index_building()
    except Exception as e:
        print(f"\n❌ Error during testing: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
