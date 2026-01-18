### **Objective: Build a Memory-Efficient, Multi-Index Spark Pipeline for Wikipedia**

Your primary task is to create a robust and scalable PySpark application to construct three inverted indexes (for body text, title, and anchor text) from a large Wikipedia corpus (6.3 million documents).

This project has a critical constraint: **you must design the system to avoid Out-Of-Memory (OOM) errors that occurred in previous attempts.** The root cause of these failures was a naive `flatMap` approach that generated over 3 billion Python objects, leading to a shuffle size exceeding 250 GB.

Your solution **must** be based on the **SPIMI (Single-Pass In-Memory Indexing)** algorithm implemented within Spark's `mapPartitions` transformation. This will allow for memory-bounded, distributed processing and dramatically reduce shuffle data size.

---

### **Core Architectural Requirements**

1.  **No Naive `flatMap`:** You are explicitly forbidden from using an RDD transformation that emits individual `(term, doc_id)` tuples for every word in the corpus. Your design must pre-aggregate postings within each Spark partition.

2.  **SPIMI-on-Spark with `mapPartitions`:** The core of your application will be a `mapPartitions` call that processes documents and builds partial inverted indexes locally on each worker.

3.  **Single-Pass Multi-Index Construction:** Your `mapPartitions` function must process the document corpus **only once**. In this single pass, it should instantiate three separate index builders (for body, title, and anchor) and build them concurrently to minimize I/O.

4.  **Memory-Bounded Processing:** Each index builder must be memory-bounded. When an in-memory dictionary reaches a predefined size limit (e.g., 500 MB), its contents must be sorted and written to a temporary block file on the worker's local disk. The in-memory dictionary is then cleared, preventing OOM errors.

5.  **Efficient Merging:**
    *   **Intra-Partition Merge:** After processing all documents in a partition, the temporary on-disk blocks must be merged using a memory-efficient **k-way merge** (e.g., using a min-heap).
    *   **Inter-Partition Merge:** The `mapPartitions` function will yield aggregated `(index_type, term, full_postings_list)` tuples. The final index will be constructed from this RDD using a `reduceByKey` transformation to merge postings for the same term from different partitions.

---

### **Required Code Structure & Modules**

You are to generate three separate, well-documented, and production-ready Python files.

**1. `pre_processing.py`**
A utility module for text processing.

```python
# pre_processing.py

import nltk
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer

# Ensure necessary NLTK data is downloaded
nltk.download('stopwords')

english_stopwords = frozenset(stopwords.words('english'))
corpus_stopwords = ["category", "references", "also", "links", "external", "see", "thumb"]
all_stopwords = english_stopwords.union(corpus_stopwords)
RE_WORD = re.compile(r"""[\#\@\w](['\-]?\w){2,24}""", re.UNICODE)
stemmer = PorterStemmer()

def tokenize_and_process(text: str, use_stemming: bool = True) -> list[str]:
    """
    Tokenizes, removes stopwords, and optionally stems the text.
    - text: The input string.
    - use_stemming: Flag to enable/disable stemming.
    - returns: A list of processed tokens.
    """
    # Your implementation here:
    # 1. Tokenize using RE_WORD.
    # 2. Filter out stopwords.
    # 3. Apply stemming if use_stemming is True.
    tokens = [token.group() for token in RE_WORD.finditer(text.lower())]
    tokens = [token for token in tokens if token not in all_stopwords]
    if use_stemming:
        tokens = [stemmer.stem(token) for token in tokens]
    return tokens

```

**2. `spimi_block_builder.py`**
A dedicated module containing a class that implements the core memory-bounded SPIMI logic for a single partition.

```python
# spimi_block_builder.py

import pickle
import heapq
import tempfile
from pathlib import Path
from collections import defaultdict, Counter

class SPIMIBlockBuilder:
    """
    Implements the SPIMI algorithm for a single Spark partition.
    - Builds a memory-bounded dictionary for postings.
    - Flushes blocks to local disk when memory threshold is reached.
    - Merges blocks using a streaming k-way merge.
    """

    def __init__(self, memory_threshold_mb: int = 500):
        """
        - memory_threshold_mb: The memory limit in megabytes for the in-memory dictionary.
        """
        # Your implementation here for the constructor
        # Initialize attributes: memory_threshold, current_block, current_memory,
        # temp_dir, list of block paths, etc.

    def add_posting(self, term: str, doc_id: int, tf: int):
        """
        Adds a posting to the current in-memory block.
        If the memory threshold is exceeded, triggers a flush to disk.
        """
        # Your implementation here:
        # 1. Add the posting to self.current_block.
        # 2. Estimate the memory increase (be mindful of new vs. existing terms).
        # 3. If self.current_memory >= self.memory_threshold, call _write_block().

    def _write_block(self):
        """
        Writes the current in-memory block to a temporary file on local disk.
        Uses a streaming pickle format (one pickle.dump per term).
        """
        # Your implementation here:
        # 1. Create a unique block file path in self.temp_dir.
        # 2. Sort the terms in the current block.
        # 3. Open the file and iterate through the sorted terms.
        # 4. For each term, write `pickle.dump((term, postings_list), f)`.
        # 5. Add the block path to the list of written blocks.
        # 6. Clear the current_block and reset the memory counter.

    def finalize(self) -> iter:
        """
        Called after all documents in the partition have been processed.
        - Writes any remaining data in the current block to disk.
        - Performs a streaming k-way merge of all written blocks.
        - Yields the final, merged (term, postings_list) tuples.
        """
        # Your implementation here:
        # 1. Call _write_block() to flush any remaining postings.
        # 2. Implement the k-way merge using a min-heap (heapq).
        #    - Open all block files for reading.
        #    - Use a helper generator to read one pickled term at a time from each file.
        #    - Push the first term from each block onto the heap.
        #    - In a loop, pop the smallest term, merge postings for that term from
        #      other blocks in the heap, and yield the result.
        # 3. Clean up (delete) the temporary block files.


```

**3. `pyspark_index_builder.py`**
The main Spark driver application. This script will orchestrate the entire index-building pipeline.

```python
# pyspark_index_builder.py

import sys
from pyspark.sql import SparkSession
from google.cloud import storage

from spimi_block_builder import SPIMIBlockBuilder
from pre_processing import tokenize_and_process
# You will also need to import the existing InvertedIndex class.
# Assume it is available in a file named inverted_index_gcp.py in the zipped dependencies.

def build_all_indices_per_partition(doc_partition: iter) -> iter:
    """
    The main worker function for `mapPartitions`. It builds all three indices
    (body, title, anchor) in a single pass over the documents in a partition.
    """
    # 1. Instantiate three separate SPIMIBlockBuilder objects:
    #    - body_builder, title_builder, anchor_builder

    # 2. Iterate through `doc_partition` which contains (doc_id, wikipedia_document) tuples.
    #    For each document:
    #    a. Extract body, title, and anchor text.
    #    b. Process each text field using tokenize_and_process().
    #    c. Calculate term frequencies (e.g., using collections.Counter).
    #    d. Add postings to the corresponding builder (e.g., body_builder.add_posting(...)).

    # 3. After the loop, call finalize() on each builder.
    #    - for term, postings in body_builder.finalize(): yield ('body', term, postings)
    #    - for term, postings in title_builder.finalize(): yield ('title', term, postings)
    #    - for term, postings in anchor_builder.finalize(): yield ('anchor', term, postings)


def main(input_path, output_path, bucket_name):
    """
    - input_path: GCS path to the Parquet files of the Wikipedia corpus.
    - output_path: GCS path to write the final index files.
    - bucket_name: The name of the GCS bucket.
    """
    # 1. Initialize SparkSession.

    # 2. Read the corpus Parquet files into an RDD.
    #    docs_rdd = spark.read.parquet(input_path).rdd.map(lambda row: (row.id, row.text))

    # 3. Run the main SPIMI pipeline.
    #    all_indices_rdd = docs_rdd.mapPartitions(build_all_indices_per_partition)

    # 4. Cache `all_indices_rdd` if memory allows, as it will be used three times.
    #    all_indices_rdd.persist()

    # 5. For each index type ('body', 'title', 'anchor'):
    #    a. Filter `all_indices_rdd` for the current type.
    #    b. Use `reduceByKey` to merge postings lists for the same term from different partitions.
    #    c. The output of this stage is the final RDD for one complete inverted index.
    #       (term, final_postings_list)

    #    d. **Write the Index to GCS (Crucial Step):**
    #       - This is where you must produce output compatible with the provided `InvertedIndex` class.
    #       - You need to write the posting lists to `.bin` files in a distributed manner.
    #         One way is to use `foreachPartition` on your final RDD. Inside, use the logic
    #         from `InvertedIndex.write_a_posting_list` to write blocks of postings.
    #       - After writing the postings, you need to collect the metadata:
    #         i.   Document Frequency (df) for each term.
    #         ii.  Total term frequency (term_total) for each term.
    #         iii. The locations of the written posting lists (`posting_locs`).
    #       - Finally, on the driver, create an `InvertedIndex` object, populate its
    #         metadata fields (`df`, `term_total`, `posting_locs`), and use
    #         `_write_globals` (or `write_index`) to save the final `_index.pkl` file.

    # 6. Unpersist the cached RDD.

if __name__ == '__main__':
    # Parse command line arguments for input_path, output_path, bucket_name
    # and call main().
    ...
```

---

### **Contract: Final Index Format**

Your Spark application's output **must** be readable by the following `InvertedIndex` class, which will be provided in a file named `inverted_index_gcp.py` and included in the job's dependencies. The Spark job is responsible for creating the `.pkl` and `.bin` files that this class manages.

```python
# inverted_index_gcp.py (This is the target format for your output)

import pickle
from google.cloud import storage
from collections import defaultdict, Counter
from contextlib import closing
from pathlib import Path
import itertools

# --- [Content of the provided inverted_index_gcp.py file] ---
# (Include the full class definition here, including MultiFileWriter,
# MultiFileReader, and the InvertedIndex class itself).
# It's provided in a previous turn's output.
# For brevity, I am not repeating the whole class here in my thought process.
# But the prompt to the LLM must contain the full class.
# ... (Full content of inverted_index_gcp.py) ...
```
You can find the full content of `inverted_index_gcp.py` in the previous turns.

### **Final Deliverables**

Please provide the complete, production-ready code for the following three files:
1.  `pre_processing.py`
2.  `spimi_block_builder.py`
3.  `pyspark_index_builder.py`

The code should be well-commented, include docstrings and type hints, and have robust logging to facilitate debugging during execution on the Dataproc cluster.
