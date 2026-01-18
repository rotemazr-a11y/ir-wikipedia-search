# Wikipedia Search Engine

A search engine for English Wikipedia built as part of the Information Retrieval course (2025-2026).

## Project Overview

This project implements a full-text search engine over the entire English Wikipedia corpus (~6.3M documents). The engine supports multiple ranking methods including TF-IDF cosine similarity, binary title/anchor matching, PageRank, and PageView scoring.

## Project Structure

```
PROJECT/
├── README.md                    # This file
├── queries_train.json           # Training queries for evaluation
│
├── indexing/                    # Index building pipeline
│   ├── pyspark_index_builder.py # Main Spark driver for SPIMI indexing
│   ├── spimi_block_builder.py   # SPIMI algorithm implementation
│   ├── inverted_index_gcp.py    # Inverted index class with GCS support
│   ├── pre_processing.py        # Tokenization and text preprocessing
│   ├── compute_doc_metadata.py  # Generate doc_titles.pkl and doc_lengths.pkl
│   ├── compute_pagerank.py      # PageRank calculation using link graph
│   └── compute_pageviews.py     # Process Wikipedia pageview data
│
├── frontend/                    # Search engine API
│   ├── search_frontend.py       # Flask app with 6 search endpoints
│   └── run_frontend_in_colab.ipynb  # Notebook for Colab testing
│
├── scripts/                     # Deployment scripts
│   ├── run_frontend_in_gcp.sh   # GCP Compute Engine deployment
│   ├── startup_script_gcp.sh    # VM startup configuration
│   └── deploy_to_gcp.sh         # Deployment automation
│
├── tests/                       # Testing and evaluation
│   └── evaluate_search.py       # Evaluation script (P@5, P@10, F1@30)
│
└── docs/                        # Documentation
    └── ...
```

## Components

### 1. Indexing Pipeline (`indexing/`)

#### `pyspark_index_builder.py`
Main PySpark driver that builds inverted indices for the entire Wikipedia corpus using the SPIMI (Single-Pass In-Memory Indexing) algorithm.

**Features:**
- Processes corpus in parallel across Spark workers
- Builds 3 separate indices: body, title, anchor text
- Memory-bounded block building with automatic flushing
- K-way merge for combining sorted blocks

**Usage:**
```bash
gcloud dataproc jobs submit pyspark \
    --cluster=YOUR_CLUSTER \
    --region=us-central1 \
    --py-files=gs://YOUR_BUCKET/deps.zip \
    gs://YOUR_BUCKET/pyspark_index_builder.py \
    -- \
    --input gs://YOUR_BUCKET/corpus \
    --output gs://YOUR_BUCKET/indices \
    --bucket YOUR_BUCKET
```

#### `spimi_block_builder.py`
Implements the SPIMI algorithm for efficient index construction:
- In-memory dictionary building
- Automatic block flushing when memory threshold reached
- Final k-way merge using min-heap

#### `inverted_index_gcp.py`
Inverted index class supporting:
- Binary posting list storage
- GCS read/write operations
- Efficient posting list retrieval

#### `pre_processing.py`
Text preprocessing pipeline:
- Tokenization using regex pattern (matches words 3-24 chars)
- Stopword removal (English + corpus-specific)
- No stemming (per project requirements)

#### `compute_pagerank.py`
Computes PageRank scores from Wikipedia link graph:
- Extracts links from anchor text
- Iterative PageRank calculation (10 iterations)
- Outputs `pagerank.pkl`

#### `compute_pageviews.py`
Processes Wikipedia pageview data:
- Parses August 2021 pageview dump
- Maps page titles to document IDs
- Outputs `pageviews.pkl`

#### `compute_doc_metadata.py`
Extracts document metadata:
- `doc_titles.pkl`: Mapping of doc_id → title
- `doc_lengths.pkl`: Mapping of doc_id → word count

### 2. Search Frontend (`frontend/`)

#### `search_frontend.py`
Flask-based REST API implementing 6 endpoints:

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/search` | GET | Combined search (TF-IDF + PageRank + PageView) |
| `/search_body` | GET | TF-IDF cosine similarity on body text |
| `/search_title` | GET | Binary ranking on title matches |
| `/search_anchor` | GET | Binary ranking on anchor text |
| `/get_pagerank` | POST | Returns PageRank scores for doc IDs |
| `/get_pageview` | POST | Returns pageview counts for doc IDs |

**Ranking Methods:**
1. **TF-IDF Cosine Similarity**: Log-weighted TF with IDF, normalized by document length
2. **Binary Ranking**: Counts distinct query terms matching in title/anchor
3. **Combined Search**: Weighted combination of TF-IDF, PageRank, and PageView

### 3. Evaluation (`tests/`)

#### `evaluate_search.py`
Comprehensive evaluation script:
- Computes P@5, P@10, F1@30, AP@10
- Measures query response time
- Calculates grading metric (harmonic mean of P@5 and F1@30)

**Usage:**
```bash
# Test against Flask server
python tests/evaluate_search.py --url http://localhost:8080

# Test specific endpoint
python tests/evaluate_search.py --endpoint search_body
```

## Deployment

### Google Cloud Storage
All index files are stored in: `gs://bucket_207916263/`

**Index Structure:**
```
gs://bucket_207916263/
├── indices/
│   ├── body_index/          # Body text inverted index
│   ├── title_index/         # Title inverted index
│   └── anchor_index/        # Anchor text inverted index
├── pagerank/
│   └── pagerank.pkl         # PageRank scores
├── pageviews/
│   └── pageviews.pkl        # Pageview counts
├── doc_titles.pkl           # Document ID → Title mapping
└── doc_lengths.pkl          # Document ID → Length mapping
```

### Running on GCP

1. **Start Compute Engine VM:**
```bash
bash scripts/run_frontend_in_gcp.sh
```

2. **Access the search engine:**
```
http://YOUR_EXTERNAL_IP:8080/search?query=hello+world
```

### Running in Colab
Open `frontend/run_frontend_in_colab.ipynb` and follow the instructions.

## Requirements

- Python 3.8+
- PySpark 3.x (for indexing)
- Flask (for frontend)
- google-cloud-storage (for GCS access)

## Authors

- Student ID: 207916263

## License

This project is for educational purposes as part of the Information Retrieval course.
