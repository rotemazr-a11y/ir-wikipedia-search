# Wikipedia Search Engine - Information Retrieval Project

Course: 372-1-4406 Information Retrieval
Semester: Winter 2024-2025

## Project Overview

This project implements a search engine for English Wikipedia as part of the IR course final project. The search engine will support multiple ranking methods including TF-IDF, binary ranking, PageRank, and page view ranking.

## Project Structure

```
.
├── text_processing.py      # Text processing module (tokenization, normalization, stop words)
├── inverted_index_gcp.py   # Inverted index implementation (provided by staff)
├── search_frontend.py      # Flask frontend for search engine (to be implemented)
├── run_frontend_in_colab.ipynb  # Colab notebook for testing
├── run_frontend_in_gcp.sh  # GCP deployment script
├── startup_script_gcp.sh   # GCP instance setup script
└── queries_train.json      # Training queries for evaluation
```

## Implementation Progress

### Phase 1: Text Processing ✅ COMPLETED

The text processing module (`text_processing.py`) implements the following components:

#### 1. **Tokenization**
- Converts input text into individual tokens (words)
- Uses regex pattern matching (`\b\w+\b`) to extract alphanumeric sequences
- Handles:
  - Standard words
  - Numbers
  - Hyphenated words (split into separate tokens)
  - Contractions and apostrophes

**Function**: `tokenize(text)`

**Example**:
```python
>>> tokenize("Hello, world! This is a test.")
['Hello', 'world', 'This', 'is', 'a', 'test']
```

#### 2. **Normalization**
- Converts all text to lowercase for case-insensitive matching
- Ensures "Bank" and "bank" are treated identically
- Critical for query-document matching

**Function**: `normalize(text)`

**Example**:
```python
>>> normalize("Hello WORLD")
'hello world'
```

#### 3. **Stop Word Removal**
- Removes common English stop words (e.g., "the", "is", "at", "of")
- Based on Zipf's law: high-frequency words with low discriminative power
- Uses a standard stop word list (similar to NLTK/Lucene)
- Improves:
  - Index efficiency (smaller index size)
  - Query processing speed
  - Retrieval precision

**Function**: `remove_stopwords(tokens, stopwords=None)`

**Stop words included**: 150+ common English words including articles, prepositions, pronouns, and common verbs

**Example**:
```python
>>> remove_stopwords(['this', 'is', 'a', 'test', 'document'])
['test', 'document']
```

#### 4. **Complete Processing Pipeline**
- Combines all processing steps in correct order:
  1. Normalization (lowercase)
  2. Tokenization (split into words)
  3. Stop word removal (optional)

**Function**: `tokenize_and_process(text, remove_stops=True, custom_stopwords=None)`

**Example**:
```python
>>> tokenize_and_process("The quick brown fox jumps over the lazy dog")
['quick', 'brown', 'fox', 'jumps', 'lazy', 'dog']
```

#### 5. **Term Frequency Counting**
- Counts occurrences of each term in a token list
- Essential for TF-IDF calculation
- Returns a Counter object for efficient frequency queries

**Function**: `get_term_counts(tokens)`

**Example**:
```python
>>> tokens = ['apple', 'banana', 'apple', 'cherry']
>>> get_term_counts(tokens)
Counter({'apple': 2, 'banana': 1, 'cherry': 1})
```

### Key Design Decisions

1. **No Stemming**: Following project requirements, stemming is NOT implemented for the basic search methods (search_body, search_title, search_anchor)

2. **Regex-based Tokenization**: Uses `\b\w+\b` pattern which:
   - Captures word boundaries accurately
   - Handles Unicode characters
   - Efficient for large-scale processing

3. **Configurable Stop Words**: Allows custom stop word lists while providing a comprehensive default set

4. **Modular Design**: Each function serves a single purpose, making it easy to:
   - Test individual components
   - Modify behavior as needed
   - Reuse in different contexts

## Usage Examples

### Processing a Document

```python
from text_processing import tokenize_and_process, get_term_counts

# Example document
doc_text = "Information Retrieval is the science of searching for information in documents."

# Process the document
tokens = tokenize_and_process(doc_text)
print("Tokens:", tokens)
# Output: ['information', 'retrieval', 'science', 'searching', 'information', 'documents']

# Get term frequencies
term_freq = get_term_counts(tokens)
print("Term frequencies:", term_freq)
# Output: Counter({'information': 2, 'retrieval': 1, 'science': 1, 'searching': 1, 'documents': 1})
```

### Processing a Query

```python
from text_processing import tokenize_and_process

# Example query
query = "How to search for information?"

# Process the query
query_tokens = tokenize_and_process(query)
print("Query tokens:", query_tokens)
# Output: ['search', 'information']
```

### Using with Inverted Index

```python
from inverted_index_gcp import InvertedIndex
from text_processing import tokenize_and_process

# Create inverted index
docs = {
    1: tokenize_and_process("The quick brown fox"),
    2: tokenize_and_process("The lazy dog sleeps"),
    3: tokenize_and_process("Quick brown dogs and foxes")
}

index = InvertedIndex(docs)
```

## Testing

Run the text processing module directly to see example outputs:

```bash
python text_processing.py
```

This will demonstrate:
- Normalization
- Tokenization
- Stop word removal
- Term frequency counting
- Query processing

## Next Steps

### Phase 2: Index Building (Pending)
- [ ] Build inverted indices for Wikipedia corpus:
  - Body index (for TF-IDF search)
  - Title index (for binary ranking)
  - Anchor text index (for link-based ranking)
- [ ] Calculate and store document frequencies (DF)
- [ ] Calculate and store term frequencies (TF)
- [ ] Implement posting list compression

### Phase 3: Ranking Implementation (Pending)
- [ ] Implement TF-IDF with cosine similarity (search_body)
- [ ] Implement binary ranking for titles (search_title)
- [ ] Implement binary ranking for anchors (search_anchor)
- [ ] Implement PageRank ranking (get_pagerank)
- [ ] Implement page view ranking (get_pageview)

### Phase 4: Search Engine Integration (Pending)
- [ ] Implement the main search method (combining multiple signals)
- [ ] Optimize for speed (<35 seconds per query)
- [ ] Test on training queries
- [ ] Deploy to GCP

### Phase 5: Evaluation & Optimization (Pending)
- [ ] Measure Precision@5 and F1@30
- [ ] Measure average retrieval time
- [ ] Optimize query processing
- [ ] Experiment with query expansion, re-ranking, etc.

## Requirements Met

✅ Text tokenization
✅ Normalization (lowercasing)
✅ Stop word removal
✅ Modular, clean code structure
✅ Comprehensive documentation

## Technical Details

**Language**: Python 3.x
**Key Libraries**:
- `re` - Regular expressions for tokenization
- `collections.Counter` - Efficient term frequency counting
- `pickle` - Serialization (from inverted_index_gcp.py)
- `flask` - Web framework (from search_frontend.py)

## Notes

- The tokenizer follows standard IR practices as taught in course lectures
- Stop word list is based on common IR libraries (NLTK, Lucene)
- No external API calls or services used (as per project requirements)
- All processing is done locally for maximum control and efficiency

## License

Academic project for Information Retrieval course (372-1-4406)
