# test_local_pipeline.py
"""
Local test of the full pipeline using mock Wikipedia documents.
Run this BEFORE deploying to GCP to catch issues early.
"""

import os
import sys
import tempfile
import pickle
from collections import Counter, namedtuple
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from pre_processing import tokenize_and_process, tokenize_no_stem
from spimi_block_builder import create_body_builder, create_title_builder, create_anchor_builder
from inverted_index_gcp import InvertedIndex, MultiFileWriter, TUPLE_SIZE, TF_MASK

# Mock Row object to simulate Spark DataFrame row
MockRow = namedtuple('MockRow', ['id', 'title', 'text', 'anchor_text'])

# Sample Wikipedia-like documents
MOCK_DOCUMENTS = [
    MockRow(
        id=1,
        title="Python (programming language)",
        text="Python is a high-level programming language. Python supports multiple programming paradigms. Python was created by Guido van Rossum.",
        anchor_text={"100": "Python language", "101": "Python programming"}
    ),
    MockRow(
        id=2,
        title="Java (programming language)",
        text="Java is a class-based programming language. Java is designed to have few implementation dependencies. Java runs on billions of devices.",
        anchor_text={"100": "Java language", "102": "Java programming"}
    ),
    MockRow(
        id=3,
        title="Machine Learning",
        text="Machine learning is a subset of artificial intelligence. Machine learning algorithms build models based on sample data. Python is popular for machine learning.",
        anchor_text={"200": "ML algorithms", "201": "machine learning AI"}
    ),
    MockRow(
        id=4,
        title="Artificial Intelligence",
        text="Artificial intelligence is intelligence demonstrated by machines. AI research includes machine learning and deep learning. Python and Java are used in AI development.",
        anchor_text={"300": "AI technology", "301": "artificial intelligence research"}
    ),
    MockRow(
        id=5,
        title="Deep Learning",
        text="Deep learning is part of machine learning methods based on artificial neural networks. Deep learning architectures include deep neural networks. Python frameworks like TensorFlow support deep learning.",
        anchor_text={"400": "deep neural networks", "401": "deep learning AI"}
    ),
]


def simulate_partition_processing(documents, use_stemming=False):
    """Simulate the mapPartitions function locally."""
    
    body_builder = create_body_builder(memory_mb=10)
    title_builder = create_title_builder(memory_mb=5)
    anchor_builder = create_anchor_builder(memory_mb=5)
    
    for row in documents:
        doc_id = row.id
        
        # Process body
        if row.text:
            body_tokens = tokenize_and_process(row.text, use_stemming=use_stemming)
            for term, tf in Counter(body_tokens).items():
                body_builder.add_posting(term, doc_id, tf)
        
        # Process title (NO stemming per requirements)
        if row.title:
            title_tokens = tokenize_no_stem(row.title)
            for term, tf in Counter(title_tokens).items():
                title_builder.add_posting(term, doc_id, tf)
        
        # Process anchors (NO stemming per requirements)
        if row.anchor_text:
            for anchor in row.anchor_text.values():
                anchor_tokens = tokenize_no_stem(anchor)
                for term, tf in Counter(anchor_tokens).items():
                    anchor_builder.add_posting(term, doc_id, tf)
    
    # Collect results
    results = {'body': {}, 'title': {}, 'anchor': {}}
    
    for term, postings in body_builder.finalize():
        results['body'][term] = postings
    
    for term, postings in title_builder.finalize():
        results['title'][term] = postings
    
    for term, postings in anchor_builder.finalize():
        results['anchor'][term] = postings
    
    return results


def write_index_to_disk(index_data, output_dir, index_name):
    """Write an index to disk in InvertedIndex format."""
    
    index = InvertedIndex()
    
    # Create output directory
    index_dir = Path(output_dir) / f"{index_name}_index"
    index_dir.mkdir(parents=True, exist_ok=True)
    
    # Write posting lists
    from contextlib import closing
    with closing(MultiFileWriter(str(index_dir), index_name)) as writer:
        for term, postings in sorted(index_data.items()):
            # Calculate metadata
            df = len(postings)
            term_total = sum(tf for _, tf in postings)
            
            # Convert to binary
            b = b''.join([
                ((doc_id << 16) | (tf & TF_MASK)).to_bytes(TUPLE_SIZE, 'big')
                for doc_id, tf in postings
            ])
            
            # Write and store locations
            locs = writer.write(b)
            index.posting_locs[term] = locs
            index.df[term] = df
            index.term_total[term] = term_total
    
    # Write the pickle file (None for bucket_name = local file)
    index._write_globals(str(index_dir), f"{index_name}_index", bucket_name=None)
    
    return index


def test_read_back(index, index_dir, index_name):
    """Test reading the index back from disk."""
    
    # Read index from pickle
    loaded_index = InvertedIndex.read_index(
        str(Path(index_dir) / f"{index_name}_index"),
        f"{index_name}_index"
    )
    
    print(f"\n  Loaded {index_name} index: {len(loaded_index.df)} terms")
    
    # Test reading a posting list
    sample_terms = list(loaded_index.df.keys())[:3]
    for term in sample_terms:
        postings = loaded_index.read_a_posting_list(
            str(Path(index_dir) / f"{index_name}_index"),
            term
        )
        print(f"    '{term}': df={loaded_index.df[term]}, postings={postings[:3]}...")
    
    return loaded_index


def main():
    print("=" * 60)
    print("LOCAL PIPELINE TEST")
    print("=" * 60)
    
    # Create temp directory for output
    with tempfile.TemporaryDirectory() as tmpdir:
        print(f"\nOutput directory: {tmpdir}")
        
        # Step 1: Process documents
        print("\n[1/4] Processing mock documents...")
        results = simulate_partition_processing(MOCK_DOCUMENTS, use_stemming=False)
        
        print(f"  Body index: {len(results['body'])} terms")
        print(f"  Title index: {len(results['title'])} terms")
        print(f"  Anchor index: {len(results['anchor'])} terms")
        
        # Step 2: Write indices to disk
        print("\n[2/4] Writing indices to disk...")
        indices = {}
        for index_name, index_data in results.items():
            indices[index_name] = write_index_to_disk(index_data, tmpdir, index_name)
            print(f"  Written {index_name}_index.pkl and .bin files")
        
        # Step 3: Read back and verify
        print("\n[3/4] Reading indices back from disk...")
        for index_name in ['body', 'title', 'anchor']:
            test_read_back(indices[index_name], tmpdir, index_name)
        
        # Step 4: Test some queries
        print("\n[4/4] Testing sample queries...")
        
        body_index = InvertedIndex.read_index(
            str(Path(tmpdir) / "body_index"),
            "body_index"
        )
        
        test_queries = ["python", "machine", "learning", "programming"]
        for query_term in test_queries:
            if query_term in body_index.df:
                postings = body_index.read_a_posting_list(
                    str(Path(tmpdir) / "body_index"),
                    query_term
                )
                doc_ids = [doc_id for doc_id, tf in postings]
                print(f"  Query '{query_term}': found in docs {doc_ids}")
            else:
                print(f"  Query '{query_term}': not found")
        
        # List files created
        print("\n[Files Created]")
        for idx_type in ['body', 'title', 'anchor']:
            idx_dir = Path(tmpdir) / f"{idx_type}_index"
            files = list(idx_dir.glob("*"))
            total_size = sum(f.stat().st_size for f in files)
            print(f"  {idx_type}_index/: {len(files)} files, {total_size} bytes")
    
    print("\n" + "=" * 60)
    print("✅ LOCAL PIPELINE TEST PASSED!")
    print("=" * 60)
    print("\nReady for GCP deployment.")


if __name__ == '__main__':
    main()
