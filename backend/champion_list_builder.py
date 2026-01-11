"""
Champion List Builder for Fast Query Processing

This module implements the two-tier indexing strategy to meet the 35-second SLA:
- Tier 1: Champion lists (top r=500 docs per term, selected by TF × PageRank)
- Tier 2: Full posting lists (used only as fallback)

Architecture Decision (from Phase 2):
- Use adaptive retrieval: Start with Tier 1, expand to Tier 2 if <10 results found
- Champion list size: r=500 (balances speed vs. recall)

Reference: search_engine_testing_optimization_strategy.md lines 554-637
"""

import logging
from pathlib import Path
from collections import defaultdict
from typing import List, Tuple, Dict, Optional
import pickle
import heapq
import math

from .inverted_index_gcp import InvertedIndex, get_bucket, _open, TUPLE_SIZE, TF_MASK, MultiFileWriter
from contextlib import closing

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ChampionListBuilder:
    """
    Builds champion lists (Tier 1) from existing full index (Tier 2).

    The champion list for a term contains the top-r documents ranked by:
        score(d, t) = TF(t, d) × (1 + PageRank(d))

    This prioritizes documents where:
    - The term appears frequently (high TF)
    - The document is authoritative (high PageRank)
    """

    def __init__(self,
                 body_index: InvertedIndex,
                 pagerank_dict: Dict[int, float],
                 champion_list_size: int = 500):
        """
        Initialize champion list builder.

        Parameters:
        -----------
        body_index : InvertedIndex
            Full body index (Tier 2)
        pagerank_dict : dict
            Mapping doc_id -> PageRank score
        champion_list_size : int, default=500
            Number of top docs to keep per term (r value)
        """
        self.body_index = body_index
        self.pagerank = pagerank_dict
        self.r = champion_list_size
        logger.info(f"ChampionListBuilder initialized with r={self.r}")

    def build_champion_lists(self,
                            base_dir: str,
                            output_dir: str,
                            bucket_name: Optional[str] = None,
                            df_threshold: int = 100) -> Dict[str, List[Tuple[int, int]]]:
        """
        Build champion lists for all terms with DF > df_threshold.

        For rare terms (DF <= df_threshold), we'll use the full posting list
        directly (no need for champion list).

        Parameters:
        -----------
        base_dir : str
            Directory containing the full index posting lists
        output_dir : str
            Output directory for champion lists
        bucket_name : str, optional
            GCS bucket name
        df_threshold : int, default=100
            Only create champion lists for terms with DF > threshold

        Returns:
        --------
        dict : term -> [(doc_id, tf), ...] champion lists
        """
        logger.info("="*70)
        logger.info(f"Building champion lists (r={self.r})")
        logger.info("="*70)

        champion_lists = {}
        terms_processed = 0
        terms_with_champions = 0

        # Process each term
        for term in self.body_index.df.keys():
            df = self.body_index.df[term]

            # Only build champion list for high-frequency terms
            if df <= df_threshold:
                continue

            # Read full posting list from Tier 2
            posting_list = self.body_index.read_a_posting_list(base_dir, term, bucket_name)

            # Score each document by TF × (1 + PageRank)
            scored_docs = []
            for doc_id, tf in posting_list:
                pagerank_score = self.pagerank.get(doc_id, 0.0)
                score = tf * (1 + pagerank_score)
                scored_docs.append((score, doc_id, tf))

            # Select top-r documents
            top_r = heapq.nlargest(self.r, scored_docs, key=lambda x: x[0])
            champion_list = [(doc_id, tf) for score, doc_id, tf in top_r]

            champion_lists[term] = champion_list
            terms_with_champions += 1

            terms_processed += 1
            if terms_processed % 10000 == 0:
                logger.info(f"Processed {terms_processed:,} terms, {terms_with_champions:,} have champions")

        logger.info(f"\n✓ Champion lists built for {terms_with_champions:,} terms")
        logger.info(f"  (skipped {len(self.body_index.df) - terms_with_champions:,} rare terms with DF <= {df_threshold})")

        # Write champion lists to disk
        self._write_champion_lists(champion_lists, output_dir, bucket_name)

        return champion_lists

    def _write_champion_lists(self,
                             champion_lists: Dict[str, List[Tuple[int, int]]],
                             output_dir: str,
                             bucket_name: Optional[str] = None):
        """
        Write champion lists to binary files (same format as posting lists).

        This allows us to reuse the existing MultiFileReader infrastructure.
        """
        logger.info(f"Writing champion lists to {output_dir}...")

        champion_index = InvertedIndex()

        # Populate df and term_total for champion lists
        for term, champion_list in champion_lists.items():
            champion_index.df[term] = len(champion_list)
            champion_index.term_total[term] = sum(tf for doc_id, tf in champion_list)

        # Write binary posting lists
        with closing(MultiFileWriter(output_dir, 'champion_lists', bucket_name)) as writer:
            for term, champion_list in champion_lists.items():
                # Encode as binary (same format as full index)
                b = b''.join([
                    (doc_id << 16 | (tf & TF_MASK)).to_bytes(TUPLE_SIZE, 'big')
                    for doc_id, tf in champion_list
                ])

                locs = writer.write(b)
                champion_index.posting_locs[term].extend(locs)

        # Save metadata
        champion_index.write_index(output_dir, 'champion_lists', bucket_name)

        # Save posting locations
        posting_locs_path = Path(output_dir) / 'champion_lists_posting_locs.pickle'
        bucket = None if bucket_name is None else get_bucket(bucket_name)
        with _open(str(posting_locs_path), 'wb', bucket) as f:
            pickle.dump(champion_index.posting_locs, f)

        logger.info(f"✓ Champion lists written to {output_dir}/champion_lists.pkl")


class AdaptiveTierRetriever:
    """
    Adaptive two-tier retrieval strategy.

    Strategy (from Phase 2, Question 2: Option C):
    1. Start with Tier 1 (champion lists) for all query terms
    2. If <10 results found, expand to Tier 2 (full posting lists)
    3. Combine scores and re-rank
    """

    def __init__(self,
                 body_index: InvertedIndex,
                 champion_index: InvertedIndex,
                 base_dir: str,
                 champion_dir: str,
                 pagerank_dict: Dict[int, float],
                 bucket_name: Optional[str] = None,
                 min_results_threshold: int = 10):
        """
        Initialize adaptive retriever.

        Parameters:
        -----------
        body_index : InvertedIndex
            Full index (Tier 2)
        champion_index : InvertedIndex
            Champion list index (Tier 1)
        base_dir : str
            Directory for Tier 2 posting lists
        champion_dir : str
            Directory for Tier 1 posting lists
        pagerank_dict : dict
            PageRank scores
        bucket_name : str, optional
            GCS bucket
        min_results_threshold : int, default=10
            Expand to Tier 2 if <this many results in Tier 1
        """
        self.body_index = body_index
        self.champion_index = champion_index
        self.base_dir = base_dir
        self.champion_dir = champion_dir
        self.pagerank = pagerank_dict
        self.bucket_name = bucket_name
        self.min_results = min_results_threshold

        logger.info(f"AdaptiveTierRetriever initialized (min_results={min_results_threshold})")

    def search(self,
              query_tokens: List[str],
              k: int = 100,
              w_body: float = 1.0,
              w_pagerank: float = 0.1) -> List[Tuple[int, float]]:
        """
        Adaptive two-tier search.

        Parameters:
        -----------
        query_tokens : list of str
            Tokenized and stemmed query terms
        k : int, default=100
            Number of results to return
        w_body : float, default=1.0
            Weight for TF-IDF score
        w_pagerank : float, default=0.1
            Weight for PageRank score

        Returns:
        --------
        list : [(doc_id, score), ...] ranked results
        """
        # PHASE 1: Try Tier 1 (champion lists)
        candidates_tier1 = self._search_tier1(query_tokens)

        # Check if we have enough results
        if len(candidates_tier1) >= self.min_results:
            logger.debug(f"Tier 1 sufficient: {len(candidates_tier1)} candidates")
            candidates = candidates_tier1
        else:
            # PHASE 2: Expand to Tier 2 (full index)
            logger.debug(f"Tier 1 insufficient ({len(candidates_tier1)} candidates), expanding to Tier 2")
            candidates_tier2 = self._search_tier2(query_tokens)

            # Merge candidates (union of both tiers)
            candidates = self._merge_candidates(candidates_tier1, candidates_tier2)

        # Compute final scores: w_body * TF-IDF + w_pagerank * PageRank
        final_scores = {}
        for doc_id, tfidf_score in candidates.items():
            pagerank_score = self.pagerank.get(doc_id, 0.0)
            final_scores[doc_id] = w_body * tfidf_score + w_pagerank * pagerank_score

        # Rank and return top-k
        ranked = sorted(final_scores.items(), key=lambda x: -x[1])[:k]
        return ranked

    def _search_tier1(self, query_tokens: List[str]) -> Dict[int, float]:
        """Search using champion lists only."""
        candidates = defaultdict(float)
        N = len(self.body_index.df)  # Total number of documents

        for term in query_tokens:
            if term not in self.champion_index.df:
                continue

            # Read champion list
            champion_list = self.champion_index.read_a_posting_list(
                self.champion_dir, term, self.bucket_name
            )

            df = self.body_index.df.get(term, 1)
            idf = math.log(N / df)

            for doc_id, tf in champion_list:
                tfidf = tf * idf
                candidates[doc_id] += tfidf

        return dict(candidates)

    def _search_tier2(self, query_tokens: List[str]) -> Dict[int, float]:
        """Search using full posting lists."""
        candidates = defaultdict(float)
        N = len(self.body_index.df)

        for term in query_tokens:
            if term not in self.body_index.df:
                continue

            # Read full posting list
            posting_list = self.body_index.read_a_posting_list(
                self.base_dir, term, self.bucket_name
            )

            df = self.body_index.df[term]
            idf = math.log(N / df)

            for doc_id, tf in posting_list:
                tfidf = tf * idf
                candidates[doc_id] += tfidf

        return dict(candidates)

    def _merge_candidates(self,
                         tier1: Dict[int, float],
                         tier2: Dict[int, float]) -> Dict[int, float]:
        """
        Merge candidates from both tiers (take maximum score for each doc).
        """
        merged = tier1.copy()
        for doc_id, score in tier2.items():
            if doc_id in merged:
                merged[doc_id] = max(merged[doc_id], score)
            else:
                merged[doc_id] = score
        return merged


def main():
    """
    Example usage: Build champion lists from existing index.

    Usage:
        python champion_list_builder.py \
            --index-dir gs://my-bucket/indices/ \
            --output-dir gs://my-bucket/champion_lists/ \
            --pagerank gs://my-bucket/pagerank.pkl
    """
    import argparse

    parser = argparse.ArgumentParser(description='Build champion lists from full index')
    parser.add_argument('--index-dir', required=True, help='Directory containing full index')
    parser.add_argument('--output-dir', required=True, help='Output directory for champion lists')
    parser.add_argument('--pagerank', required=True, help='Path to PageRank pickle file')
    parser.add_argument('--bucket', default=None, help='GCS bucket name')
    parser.add_argument('--r', type=int, default=500, help='Champion list size')

    args = parser.parse_args()

    # Load full index
    logger.info(f"Loading full index from {args.index_dir}...")
    body_index = InvertedIndex.read_index(args.index_dir, 'body_index', args.bucket)

    # Load PageRank
    logger.info(f"Loading PageRank from {args.pagerank}...")
    bucket = None if args.bucket is None else get_bucket(args.bucket)
    with _open(args.pagerank, 'rb', bucket) as f:
        pagerank_dict = pickle.load(f)

    # Build champion lists
    builder = ChampionListBuilder(body_index, pagerank_dict, champion_list_size=args.r)
    champion_lists = builder.build_champion_lists(
        base_dir=args.index_dir,
        output_dir=args.output_dir,
        bucket_name=args.bucket
    )

    logger.info("\n" + "="*70)
    logger.info("CHAMPION LISTS BUILD COMPLETE!")
    logger.info("="*70)


if __name__ == "__main__":
    main()
