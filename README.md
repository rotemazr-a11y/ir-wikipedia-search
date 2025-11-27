# ir-wikipedia-search
Final project – information retrieval search engine over English Wikipedia (Ben-Gurion University IR course)

# Wikipedia Search Engine – IR Course Final Project

Final project for the Information Retrieval course – building a search engine over the **English Wikipedia**.

The goal is to implement an end-to-end IR system that:

- Indexes the full English Wikipedia dump (body, titles, anchors, links, pageviews).
- Exposes several HTTP endpoints (via Flask) to serve queries.
- Implements and combines multiple retrieval signals (tf-idf, titles, anchors, PageRank, pageviews).
- Is deployable on Google Cloud Platform (GCP).
- Is evaluated using a provided set of training queries (`queries_train.json`).

---

## 1. Project Overview

This project consists of two main parts:

1. **Backend (IR engine)**
   - Processes the Wikipedia dump offline.
   - Builds inverted indexes for:
     - Article body
     - Titles
     - Anchor texts
   - Builds a link graph and computes **PageRank**.
   - Aggregates **pageviews** (August 2021).
   - Implements the ranking logic: scoring, combining signals, returning ranked lists of documents.

2. **Frontend (Flask HTTP API)**
   - Exposes a set of endpoints that the course staff's evaluation scripts will call.
   - For a given query or list of document IDs, it calls the backend logic and returns JSON.

The system must be able to handle **queries over the entire English Wikipedia**, return results within the required time constraints, and achieve a reasonable retrieval quality on a hidden test set.

---

## 2. Repository Structure

```text
.
├── backend/
│   ├── __init__.py
│   ├── text_processing.py      # ✅ Text tokenization, normalization, stop words
│   ├── inverted_index_gcp.py   # Provided helper for managing inverted indexes on GCP
│   ├── index_builder.py        # Index construction utilities
│   └── search_logic.py         # (TODO) main search / ranking logic
│
├── frontend/
│   ├── __init__.py
│   └── search_frontend.py      # Provided Flask app with required endpoints
│
├── data/
│   ├── queries_train.json      # Provided train queries + relevance labels
│   └── ...                     # (You) small metadata files, local configs, etc.
│
├── scripts/
│   ├── startup_script_gcp.sh   # Provided startup script for GCP VM
│   └── run_frontend_in_gcp.sh  # Provided helper to run frontend on GCP
│
├── notebooks/
│   └── run_frontend_in_colab.ipynb   # Provided Colab notebook for local/remote testing
│
├── README.md
├── requirements.txt
├── .gitignore
└── LICENSE
```

---

## 3. Implementation Progress

### ✅ Phase 1: Text Processing (COMPLETED)

**Module**: `backend/text_processing.py`

Implemented complete text processing pipeline following IR best practices:

#### Components

1. **Tokenization**
   - Uses regex pattern for word boundary detection
   - Handles: words, numbers, hyphenated terms, contractions
   - Function: `tokenize(text)` returns list of tokens

2. **Normalization**
   - Converts text to lowercase for case-insensitive matching
   - Ensures "Bank" and "bank" are treated identically
   - Function: `normalize(text)` returns lowercase string

3. **Stop Words Removal**
   - 150+ common English stop words (based on NLTK/Lucene)
   - Implements Zipf's law: removes high-frequency, low-discriminative words
   - Improves index size and retrieval speed
   - Function: `remove_stopwords(tokens)` returns filtered token list

4. **Complete Pipeline**
   - Single function for end-to-end processing
   - Used for both documents and queries
   - Function: `tokenize_and_process(text, remove_stops=True)` returns processed tokens

5. **Term Frequency Counting**
   - Returns Counter object with term frequencies
   - Essential for TF-IDF calculation
   - Function: `get_term_counts(tokens)` returns Counter object

#### Usage Examples

```python
from backend.text_processing import tokenize_and_process, get_term_counts

# Process a document
doc_text = "Information Retrieval is the science of searching for information."
tokens = tokenize_and_process(doc_text)
# Output: ['information', 'retrieval', 'science', 'searching', 'information']

# Get term frequencies
term_freq = get_term_counts(tokens)
# Output: Counter({'information': 2, 'retrieval': 1, 'science': 1, 'searching': 1})

# Process a query
query = "How to search for information?"
query_tokens = tokenize_and_process(query)
# Output: ['search', 'information']
```

#### Testing

Run the module directly to see demonstrations:
```bash
python backend/text_processing.py
```

#### Design Decisions

- **No Stemming**: Following project requirements (not used in search_body, search_title, search_anchor)
- **Regex-based**: Efficient for large-scale processing
- **Modular**: Each function serves single purpose for easy testing
- **Configurable**: Supports custom stop word lists

---

### 🔨 Phase 2: Index Building (IN PROGRESS)

**Goal**: Build inverted indices for body, title, and anchor text

**Tasks**:
- [ ] Process Wikipedia dump and build indices
- [ ] Calculate document frequencies (DF)
- [ ] Calculate term frequencies (TF)
- [ ] Store indices in GCP bucket
- [ ] Implement posting list compression

---

### 📊 Phase 3: Ranking Methods (PENDING)

**Required endpoints** (from `frontend/search_frontend.py`):

1. **`/search`** - Best combined ranking (your design)
2. **`/search_body`** - TF-IDF cosine similarity on body
3. **`/search_title`** - Binary ranking on titles
4. **`/search_anchor`** - Binary ranking on anchor text
5. **`/get_pagerank`** - Return PageRank scores
6. **`/get_pageview`** - Return page view counts

---

### 🚀 Phase 4: Deployment (PENDING)

- [ ] Test locally with Colab notebook
- [ ] Deploy to GCP
- [ ] Optimize for <35 second query time
- [ ] Test with training queries

---

### 📈 Phase 5: Evaluation (PENDING)

- [ ] Measure Precision@5 and F1@30
- [ ] Measure average retrieval time
- [ ] Experiment with improvements
- [ ] Document findings in report

---

## 4. Requirements

### Minimum Requirements

- ✅ Tokenization, normalization, stop word removal
- ⬜ Functional search engine over entire Wikipedia
- ⬜ Testable via URL (testing period: Jan 28-30, 2025)
- ⬜ Query time <35 seconds
- ⬜ Average Precision@10 >0.1
- ⬜ No external API calls
- ⬜ Clean code repository
- ⬜ Report (4 pages max)

### Full Requirements

- ⬜ Support 5 ranking methods (10 points)
- ⬜ Efficiency: <1 second preferred (7 points)
- ⬜ Results quality: Precision@5 & F1@30 (18 points)
- ⬜ Experimentation & evaluation (15 points)
- ⬜ Reporting & presentation (4 points)

---

## 5. Getting Started

### Installation

```bash
# Clone the repository
git clone https://github.com/rotemazr-a11y/ir-wikipedia-search.git
cd ir-wikipedia-search

# Install dependencies
pip install -r requirements.txt
```

### Testing Text Processing

```bash
# Test tokenization module
python backend/text_processing.py

# Example output:
# Original: "The quick brown fox jumps over the lazy dog"
# Tokens: ['quick', 'brown', 'fox', 'jumps', 'lazy', 'dog']
```

### Running Frontend Locally

```bash
# Start Flask server
python frontend/search_frontend.py

# Server will run on http://localhost:8080
```

---

## 6. Technical Stack

- **Language**: Python 3.x
- **Web Framework**: Flask
- **Cloud Platform**: Google Cloud Platform (GCP)
- **Storage**: Google Cloud Storage buckets
- **Key Libraries**:
  - `re` - Regular expressions for tokenization
  - `collections` - Counter for term frequencies
  - `pickle` - Index serialization
  - `google-cloud-storage` - GCP integration
  - `flask` - HTTP API

---

## 7. Course Information

**Course**: 372-1-4406 Information Retrieval
**Semester**: Winter 2024-2025
**Institution**: Ben-Gurion University
**Project Weight**: 25% of final grade (optional, can only improve grade)

---

## 8. License

MIT License - Academic project for Information Retrieval course

---

## 9. Contributors

See commit history for contributors and contributions.
