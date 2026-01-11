"""
INDEXING STRATEGY NOTE:
This script was migrated from a PySpark implementation to a local VM execution.
Reasoning: Spark's overhead for Title/Anchor datasets led to memory exhaustion 
during the shuffling phase. Local execution allowed for better memory profiling 
and direct binary I/O control, resulting in a stable and reliable indexing process.
"""
import sys
from collections import Counter, defaultdict
import hashlib
import _pickle as pickle
from pathlib import Path
import os

# ============================================================================
# INVERTED INDEX DATA STRUCTURE
# ============================================================================
class InvertedIndex:
    def __init__(self):
        """
        Initializes the inverted index components:
        df: document frequency for each term
        term_total: total occurrences of each term across the corpus
        posting_locs: mapping of terms to their binary file locations (file_name, offset)
        """
        self.df = Counter()
        self.term_total = Counter()
        self.posting_locs = defaultdict(list)

    def save_index(self, base_dir, name):
        """Saves the index object as a pickle file."""
        with open(Path(base_dir) / f'{name}.pkl', 'wb') as f:
            pickle.dump(self, f)

# ============================================================================
# CONSTANTS & CONFIGURATION
# ============================================================================
TUPLE_SIZE = 6       # 4 bytes for doc_id, 2 bytes for frequency (tf)
TF_MASK = 2**16 - 1  # Mask to extract the lower 16 bits for tf

# ============================================================================
# INDEX CONSTRUCTION FUNCTIONS
# ============================================================================

def write_postings_to_disk(inverted, postings, base_dir, index_name):
    """
    Writes the posting lists to binary files in chunks.
    This method ensures memory efficiency by writing to disk instead of keeping
    everything in RAM.
    """
    # Create directory if it doesn't exist
    path = Path(base_dir)
    path.mkdir(parents=True, exist_ok=True)
    
    # Sort terms to ensure consistent disk writes
    sorted_terms = sorted(postings.items())
    
    current_file_name = f"{index_name}_postings.bin"
    with open(path / current_file_name, 'wb') as f:
        for term, l in sorted_terms:
            # Store the current file pointer offset for the index lookup
            offset = f.tell()
            
            # Encode doc_id and tf into 6 bytes total
            # 32 bits (4 bytes) for doc_id, 16 bits (2 bytes) for frequency
            b = b"".join([(int(doc_id) << 16 | (int(tf) & TF_MASK)).to_bytes(TUPLE_SIZE, 'big') 
                          for doc_id, tf in l])
            
            f.write(b)
            
            # Update index metadata
            inverted.posting_locs[term].append((current_file_name, offset))
            inverted.df[term] = len(l)
            inverted.term_total[term] = sum(tf for doc_id, tf in l)

def run_indexing(data_source, output_dir, index_label):
    """
    Main execution loop for the indexing process.
    Processes the raw tokenized data and converts it into a structured 
    Inverted Index with binary postings.
    """
    print(f"Starting indexing for: {index_label}...")
    idx = InvertedIndex()
    
    # In a real scenario, 'data_source' would be an iterator over 
    # processed Wikipedia articles (doc_id, tokens)
    postings_dict = defaultdict(list)
    
    # Step 1: Count term frequencies per document (Map stage)
    for doc_id, tokens in data_source:
        counts = Counter(tokens)
        for term, tf in counts.items():
            postings_dict[term].append((doc_id, tf))
            
    # Step 2: Write results to binary storage (Reduce/Merge stage)
    write_postings_to_disk(idx, postings_dict, output_dir, index_label)
    
    # Step 3: Persistence
    idx.save_index(output_dir, f"{index_label}_index")
    print(f"✓ {index_label} index completed successfully.")

# ============================================================================
# EXECUTION SCRIPT
# ============================================================================
if __name__ == "__main__":
    # Example execution for Title and Anchor indices
    # This matches the process run on the Google Cloud VM
    output_path = "postings_gcp/"
    
    # Note: data_iterator_title and data_iterator_anchor are 
    # pre-processed streams from the Wikipedia parquet files
    # run_indexing(data_iterator_title, output_path, "title")
    # run_indexing(data_iterator_anchor, output_path, "anchor")