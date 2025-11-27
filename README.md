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
   - Exposes a set of endpoints that the course staff’s evaluation scripts will call.
   - For a given query or list of document IDs, it calls the backend logic and returns JSON.

The system must be able to handle **queries over the entire English Wikipedia**, return results within the required time constraints, and achieve a reasonable retrieval quality on a hidden test set.

---

## 2. Repository Structure

```text
.
├── backend/
│   ├── __init__.py
│   ├── inverted_index_gcp.py   # Provided helper for managing inverted indexes on GCP
│   └── search_logic.py         # (You) main search / ranking logic (tf-idf, BM25, PageRank, etc.)
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
└── LICENSE (optional, e.g. MIT)
