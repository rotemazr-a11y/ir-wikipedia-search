# spimi_block_builder.py
"""
SPIMI (Single-Pass In-Memory Indexing) Block Builder.
Memory-bounded inverted index construction for Spark partitions.

This module implements the core SPIMI algorithm:
1. Build in-memory dictionary of postings
2. Flush to disk when memory threshold exceeded
3. K-way merge all blocks on finalization
"""

import os
import pickle
import heapq
import tempfile
import logging
from pathlib import Path
from collections import defaultdict
from typing import Iterator, Tuple, List, Dict, Any, Generator

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Memory estimation constants
BYTES_PER_POSTING = 12      # (doc_id: 4 bytes, tf: 4 bytes) + tuple overhead
NEW_TERM_OVERHEAD = 60      # Average term string + dict key overhead
BYTES_PER_MB = 1024 * 1024


class SPIMIBlockBuilder:
    """
    Implements the SPIMI algorithm for a single Spark partition.
    
    Features:
    - Memory-bounded dictionary for postings
    - Automatic flushing to local disk when threshold reached
    - Streaming k-way merge using min-heap for finalization
    
    Attributes:
        memory_threshold_bytes: Maximum memory before flushing (in bytes).
        current_block: In-memory dictionary of term -> [(doc_id, tf), ...].
        current_memory: Estimated current memory usage.
        block_paths: List of paths to flushed block files.
        temp_dir: Temporary directory for block files.
    """
    
    def __init__(self, memory_threshold_mb: int = 250, index_name: str = "index"):
        """
        Initialize the SPIMI block builder.
        
        Args:
            memory_threshold_mb: Memory limit in megabytes before flushing.
            index_name: Name prefix for temp files (for debugging).
        """
        self.memory_threshold_bytes = memory_threshold_mb * BYTES_PER_MB
        self.index_name = index_name
        
        # In-memory block: term -> list of (doc_id, tf)
        self.current_block: Dict[str, List[Tuple[int, int]]] = defaultdict(list)
        self.current_memory: int = 0
        
        # Temp directory and block file tracking
        self.temp_dir = tempfile.mkdtemp(prefix=f"spimi_{index_name}_")
        self.block_paths: List[str] = []
        self.block_counter: int = 0
        
        # Statistics
        self.total_postings: int = 0
        self.total_terms_seen: int = 0
        self.flushes: int = 0
        
        logger.debug(f"SPIMIBlockBuilder initialized: {index_name}, "
                     f"threshold={memory_threshold_mb}MB, temp_dir={self.temp_dir}")
    
    def add_posting(self, term: str, doc_id: int, tf: int) -> None:
        """
        Add a posting to the current in-memory block.
        
        Args:
            term: The term/token.
            doc_id: Document ID.
            tf: Term frequency in the document.
        """
        # Check if this is a new term (for memory estimation)
        is_new_term = term not in self.current_block
        
        # Add the posting
        self.current_block[term].append((doc_id, tf))
        self.total_postings += 1
        
        # Update memory estimate
        if is_new_term:
            self.current_memory += NEW_TERM_OVERHEAD + len(term)
            self.total_terms_seen += 1
        self.current_memory += BYTES_PER_POSTING
        
        # Check if we need to flush
        if self.current_memory >= self.memory_threshold_bytes:
            self._write_block()
    
    def _write_block(self) -> None:
        """
        Write the current in-memory block to a temporary file.
        Uses streaming pickle format (one pickle.dump per term) for memory efficiency.
        """
        if not self.current_block:
            return
        
        # Create unique block file path
        block_path = os.path.join(
            self.temp_dir, 
            f"block_{self.block_counter:04d}.pkl"
        )
        self.block_counter += 1
        
        # Sort terms and write
        sorted_terms = sorted(self.current_block.keys())
        
        with open(block_path, 'wb') as f:
            for term in sorted_terms:
                postings = self.current_block[term]
                # Sort postings by doc_id for efficient merging later
                postings.sort(key=lambda x: x[0])
                pickle.dump((term, postings), f)
        
        # Track the block file
        self.block_paths.append(block_path)
        self.flushes += 1
        
        logger.debug(f"Flushed block {self.block_counter - 1}: "
                     f"{len(sorted_terms)} terms, {self.current_memory / BYTES_PER_MB:.2f}MB")
        
        # Clear the in-memory block
        self.current_block.clear()
        self.current_memory = 0
    
    def _read_block_iterator(self, block_path: str) -> Generator[Tuple[str, List], None, None]:
        """
        Generator that reads terms one at a time from a block file.
        
        Args:
            block_path: Path to the block file.
            
        Yields:
            (term, postings_list) tuples.
        """
        with open(block_path, 'rb') as f:
            while True:
                try:
                    term, postings = pickle.load(f)
                    yield term, postings
                except EOFError:
                    break
    
    def _kway_merge(self) -> Generator[Tuple[str, List[Tuple[int, int]]], None, None]:
        """
        Perform a k-way merge of all block files using a min-heap.
        
        Yields:
            (term, merged_postings_list) tuples in sorted order.
        """
        if not self.block_paths:
            return
        
        # Initialize iterators for each block
        iterators = [self._read_block_iterator(path) for path in self.block_paths]
        
        # Min-heap: (term, postings, iterator_index, iterator)
        # We use iterator_index to break ties consistently
        heap: List[Tuple[str, List, int, Any]] = []
        
        # Prime the heap with first term from each block
        for idx, it in enumerate(iterators):
            try:
                term, postings = next(it)
                heapq.heappush(heap, (term, postings, idx, it))
            except StopIteration:
                pass  # Empty block
        
        current_term = None
        current_postings: List[Tuple[int, int]] = []
        
        while heap:
            term, postings, idx, iterator = heapq.heappop(heap)
            
            # If new term, yield the previous one
            if current_term is not None and term != current_term:
                # Sort merged postings by doc_id
                current_postings.sort(key=lambda x: x[0])
                yield current_term, current_postings
                current_postings = []
            
            current_term = term
            current_postings.extend(postings)
            
            # Advance the iterator we just popped from
            try:
                next_term, next_postings = next(iterator)
                heapq.heappush(heap, (next_term, next_postings, idx, iterator))
            except StopIteration:
                pass  # This block is exhausted
        
        # Don't forget the last term
        if current_term is not None and current_postings:
            current_postings.sort(key=lambda x: x[0])
            yield current_term, current_postings
    
    def finalize(self) -> Generator[Tuple[str, List[Tuple[int, int]]], None, None]:
        """
        Finalize the index building process.
        
        1. Flush any remaining in-memory data
        2. Perform k-way merge of all blocks
        3. Clean up temporary files
        
        Yields:
            (term, postings_list) tuples for the complete merged index.
        """
        logger.info(f"Finalizing {self.index_name}: {self.total_postings} postings, "
                    f"{self.total_terms_seen} unique terms, {self.flushes} flushes")
        
        # Flush any remaining data
        if self.current_block:
            self._write_block()
        
        # If no blocks were written, nothing to yield
        if not self.block_paths:
            return
        
        # Perform k-way merge
        try:
            for term, postings in self._kway_merge():
                yield term, postings
        finally:
            # Clean up temp files
            self._cleanup()
    
    def _cleanup(self) -> None:
        """Remove all temporary block files and directory."""
        for path in self.block_paths:
            try:
                os.remove(path)
            except OSError as e:
                logger.warning(f"Failed to remove temp file {path}: {e}")
        
        try:
            os.rmdir(self.temp_dir)
        except OSError as e:
            logger.warning(f"Failed to remove temp dir {self.temp_dir}: {e}")
        
        logger.debug(f"Cleanup complete for {self.index_name}")
    
    def get_stats(self) -> Dict[str, Any]:
        """Return statistics about the index building process."""
        return {
            'index_name': self.index_name,
            'total_postings': self.total_postings,
            'unique_terms': self.total_terms_seen,
            'flushes': self.flushes,
            'memory_threshold_mb': self.memory_threshold_bytes / BYTES_PER_MB,
            'blocks_written': len(self.block_paths)
        }


# Convenience factory functions for the three index types
def create_body_builder(memory_mb: int = 250) -> SPIMIBlockBuilder:
    """Create a SPIMI builder for body text (largest allocation)."""
    return SPIMIBlockBuilder(memory_threshold_mb=memory_mb, index_name="body")


def create_title_builder(memory_mb: int = 100) -> SPIMIBlockBuilder:
    """Create a SPIMI builder for title text (smallest allocation)."""
    return SPIMIBlockBuilder(memory_threshold_mb=memory_mb, index_name="title")


def create_anchor_builder(memory_mb: int = 150) -> SPIMIBlockBuilder:
    """Create a SPIMI builder for anchor text (medium allocation)."""
    return SPIMIBlockBuilder(memory_threshold_mb=memory_mb, index_name="anchor")


__all__ = [
    'SPIMIBlockBuilder',
    'create_body_builder',
    'create_title_builder', 
    'create_anchor_builder'
]
