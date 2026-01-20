#set document(title: "Information Retrieval Project: Wikipedia Search Engine")
#set page(paper: "a4", margin: (x: 2.5cm, y: 2cm))
#set text(font: "Arial", size: 11pt)
#set par(justify: true, leading: 0.8em)

#text(size: 14pt, weight: "bold")[Information Retrieval Project: Wikipedia Search Engine]

#v(0.4cm)

#text(weight: "bold")[Students:]
- Tomer Filo
  - ID: 206969750
  - Mail: filot\@post.bgu.ac.il
- Rotem Azriel
  - ID: 207916263
  - Mail: rotemazr\@post.bgu.ac.il

#v(0.3cm)

#text(weight: "bold")[Project Links:]
- *GitHub Repository:* #link("https://github.com/RotemAzriel/wikipedia-search-engine")
- *Google Cloud Storage Bucket:* bucket_207916263

#v(0.3cm)

#text(weight: "bold")[Bucket Structure]

#table(
  columns: (auto, auto, auto),
  stroke: 0.5pt,
  [*Category*], [*File*], [*Size*],
  [postings_gcp], [body/bin], [~1.5 GB],
  [], [body/pkl], [~8 MB],
  [], [title/bin], [~50 MB],
  [], [title/pkl], [~67 MB],
  [], [anchor/bin], [~800 MB],
  [], [anchor/pkl], [~824 MB],
  [Python Code], [inverted_index_gcp.py], [],
  [], [pre_processing.py], [],
  [], [pyspark_index_builder.py], [],
  [], [search_frontend.py], [],
  [Metadata], [doc_titles.pkl], [~200 MB],
  [], [doc_lengths.pkl], [~100 MB],
  [], [pagerank_dict.pkl], [~85 MB],
  [], [pageview_dict.pkl], [~74 MB],
)

#pagebreak()

#text(weight: "bold", size: 12pt)[System Architecture & Implementation]
#v(0.2cm)

Our search engine is built on a multi-index architecture designed for high precision and low latency. The engine integrates three distinct inverted indices:

1. *Body Index (BM25):* Optimized for broad content matching. It uses a 4-byte DocID and 2-byte TF structure (6-byte tuples). BM25 provides better term frequency saturation than standard TF-IDF, with parameters k1=1.5 and b=0.75 for length normalization.

2. *Title Index (In-Memory):* The primary relevance signal. Instead of fetching from GCS, we load the entire doc_titles dictionary (6.3M entries) into RAM at startup. Query terms are matched directly against titles, eliminating network latency entirely.

3. *Anchor Index (Binary TF-IDF):* Captures how the Wikipedia community describes a page through link text. Uses 6-byte tuples and binary ranking based on distinct query term matches.

#v(0.3cm)
#text(weight: "bold", size: 12pt)[Key Experiments & Takeaways]
#v(0.2cm)

#text(weight: "bold")[Experiment 1: The "Slow Response" Problem]

*Context:* Initial version had query times of ~30 seconds, approaching the 35-second limit. We discovered that fetching posting lists sequentially from GCS was the bottleneck. Each index fetch took ~3 seconds due to network latency.

*Solution:* We implemented parallel fetching using ThreadPoolExecutor, querying body and anchor indices simultaneously. We also moved title matching entirely to RAM, eliminating one GCS round-trip completely.

#v(0.2cm)
#text(weight: "bold")[Experiment 2: BM25 vs TF-IDF Comparison]

We tested both scoring methods on the body index. Standard TF-IDF performed adequately but struggled with longer documents where term frequency saturation became an issue. BM25's logarithmic dampening of term frequency proved more effective, particularly for queries with common terms appearing many times in Wikipedia articles.

#v(0.2cm)
#text(weight: "bold")[Experiment 3: Title Weight Optimization]

We experimented with different weight distributions. Initial equal weighting (0.33 each) yielded P\@10 of 0.35. Increasing title weight to 0.35 while reducing body to 0.45 improved P\@10 to 0.54. This makes sense for Wikipedia where users often search for exact article names (navigational queries).

#v(0.2cm)
#text(weight: "bold")[Experiment 4: PageRank and PageView Integration]

Adding PageRank (weight 0.07) helped surface authoritative articles over obscure stubs. PageView data (weight 0.03) provided a popularity signal that correlated well with user expectations. Together, these global signals prevented low-quality pages from ranking high even when they matched query terms.

#pagebreak()

#text(weight: "bold", size: 12pt)[Final Version - Combined Scoring Model]
#v(0.2cm)

The search engine utilizes BM25 scoring to evaluate the relevance of the document body, normalized to ensure consistent scaling across different query lengths.

This is augmented by an in-memory title matching signal that grants higher priority to documents where query terms appear in the title. Unlike traditional index-based approaches, our title search operates entirely in RAM against the 6.3M doc_titles dictionary.

To further refine results, the engine integrates global importance metrics: PageRank for structural authority and PageViews (log-scaled) for historical popularity.

Finally, Anchor Text frequencies are incorporated to leverage how other Wikipedia articles reference the document. These components are weighted and aggregated to produce a final relevance score, optimized for both retrieval precision and computational efficiency using parallelized execution.

#v(0.3cm)
#align(center)[
  #box(stroke: 1pt, inset: 10pt)[
    $"Score" = 0.45 dot "BM25"_"norm" + 0.35 dot "Title"_"match" + 0.10 dot "Anchor" + 0.07 dot "PageRank" + 0.03 dot "PageView"$
  ]
]

#v(0.4cm)

#text(weight: "bold")[Performance Evaluation]

#figure(
  table(
    columns: 6,
    stroke: 0.5pt,
    [*Version*], [*Configuration*], [*P\@5*], [*P\@10*], [*F1\@30*], [*Time (s)*],
    [V1], [Body BM25 only], [0.36], [0.28], [0.11], [30],
    [V2], [+ In-Memory Title], [0.45], [0.38], [0.18], [25],
    [V3], [+ Anchor], [0.52], [0.45], [0.22], [20],
    [V4], [+ PageRank], [0.58], [0.52], [0.27], [15],
    [*Final*], [*+ PageView*], [*0.67*], [*0.54*], [*0.30*], [*12*],
  )
)

#v(0.4cm)

#text(weight: "bold")[Technical Implementation Details]

*Index Construction (SPIMI):* We used Single-Pass In-Memory Indexing on Google Cloud Dataproc with a 3-worker cluster. The algorithm processes documents in parallel, builds in-memory dictionaries until memory thresholds are reached (250MB for body, 150MB for anchor, 100MB for title), then flushes sorted blocks to disk. A final k-way merge using min-heap produces the complete index.

*Preprocessing Pipeline:* Tokenization uses regex pattern `[\\#\\@\\w](['-]?\\w){2,24}` to extract words 3-24 characters. We remove 137 stopwords (NLTK + corpus-specific like "wikipedia", "category", "references"). No stemming is applied to preserve exact matching for navigational queries.

*Query Processing:* Each query triggers parallel GCS fetches for body and anchor posting lists while title matching runs in-memory. Results are merged using a candidate set approach where documents appearing in any index are scored across all signals. Final ranking returns top 100 documents sorted by combined score.

#v(0.3cm)
#align(center)[
  #text(weight: "bold", size: 12pt)[Average Precision\@10 = 0.543]
]

#pagebreak()

#text(weight: "bold", size: 12pt)[Qualitative Evaluation]
#v(0.2cm)

#figure(
  table(
    columns: (auto, 1fr, auto),
    stroke: 0.5pt,
    align: (center, left, center),
    [*\#*], [*Query*], [*P\@10*],
    [1], [Mount Everest climbing expeditions], [0.60],
    [2], [Great Fire of London 1666], [1.00],
    [3], [Nanotechnology materials science], [0.50],
    [6], [Printing press invention Gutenberg], [0.90],
    [7], [Ancient Egypt pyramids pharaohs], [0.90],
    [8], [Gothic literature Mary Shelley], [0.20],
    [11], [Wright brothers first flight], [0.80],
    [17], [Renaissance architecture Florence Italy], [0.70],
    [23], [Green Revolution agriculture yield], [0.80],
    [28], [Stonehenge prehistoric monument], [0.90],
  ),
  caption: [Selected queries showing range of performance (best, average, worst)]
)

#v(0.3cm)

*What worked well:* The primary driver for success was *Title-Term Congruence*. For queries involving specific entities or historical events (e.g., "Mount Everest" or "Great Fire of London"), the high weight on title matching (0.35) allowed the engine to surface exact matches immediately. The in-memory approach also ensured zero latency for title lookups.

*Exact Match Precision:* Since we use exact string matching for titles, the engine successfully matched specific nomenclature, ensuring that canonical Wikipedia articles appeared at the top. Queries like "Stonehenge prehistoric monument" achieved P\@10 of 0.90 because the Wikipedia article title directly matches.

#v(0.2cm)

We analyzed failures in queries like "Gothic literature Mary Shelley" (P\@10 = 0.20):

*What didn't work well (Term Dispersion):* The query contained multiple distinct entities. The engine suffered from topic drift because "Gothic" and "literature" matched different documents than "Mary Shelley". No single document contained all terms with high frequency.

*The "Zero Precision" Cases:* Some semantically related results failed to match Ground Truth IDs. This suggests the engine finds "relevant" content but not always the "authoritative" aggregate pages expected by evaluators.

*Dominant Factor - Lack of Phrase Matching:* The engine's bag-of-words approach treats all query terms as independent signals. Without a Proximity Score, it cannot prioritize documents where terms appear as a single concept.

*Proposed Fix:* Implementing Bigram Indexing or Positional Postings would solve this by rewarding documents where query terms appear in close proximity, effectively filtering out term dispersion noise. Additionally, query expansion using word embeddings could help connect related concepts.
