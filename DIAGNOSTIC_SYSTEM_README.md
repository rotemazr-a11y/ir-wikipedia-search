# Query Term Contribution Analysis & Diagnostic System

## Executive Summary

A comprehensive diagnostic system for analyzing IR query performance, identifying failure modes, and providing actionable recommendations for improvement.

**Current Performance:**
- **MAP@10:** 0.0245 (on mini corpus with 42 documents)
- **Zero-MAP Queries:** 10/30 (33% failure rate)
- **Best Query:** 0.0732 (DNA double helix discovery)
- **Worst Queries:** 0.0000 (10 queries with zero results)

---

## System Overview

This diagnostic system provides deep insights into query-level performance through:

1. **Instrumented Search Backend** - Term-level scoring transparency
2. **Statistical Analysis** - Compare successful vs. failing queries
3. **Ablation Testing** - Measure individual term contributions
4. **Automated Failure Detection** - Identify 8 distinct failure modes
5. **Visualization** - Generate diagnostic plots (requires matplotlib)

---

## Modules

### 1. `scripts/diagnostic_search.py` (350 lines)

**Purpose:** Instrumented search backend exposing term-level analytics

**Key Functions:**
- `compute_tfidf_with_diagnostics()` - Enhanced TF-IDF with tracking
- `multi_field_fusion_with_diagnostics()` - Field contribution tracking
- `search_with_full_diagnostics()` - Complete diagnostic pipeline

**Diagnostic Data Provided:**
- Query-level: tokens, TF-IDF values, norms, OOV terms
- Term-level: DF, IDF, matched docs, effectiveness
- Document-level: matched terms, score contributions per term
- Field-level: body/title/anchor/PageRank/pageviews contributions

### 2. `scripts/query_analyzer.py` (460 lines)

**Purpose:** Statistical analysis of query characteristics

**Key Functions:**
- `analyze_query_characteristics()` - Comprehensive query stats
- `compare_successful_vs_failing_queries()` - Statistical comparison
- `analyze_zero_map_queries()` - Zero-MAP pattern analysis

**Metrics Analyzed:**
- Query length, term specificity (IDF), vocabulary coverage
- Coherence (IDF variance), field matching patterns
- Statistical tests (t-test, effect sizes, confidence intervals)

### 3. `scripts/term_contribution_analyzer.py` (420 lines)

**Purpose:** Term effectiveness and ablation testing

**Key Functions:**
- `compute_term_effectiveness_scores()` - Per-term retrieval metrics
- `ablation_analysis()` - Systematically remove terms to measure impact
- `analyze_dominant_terms()` - Identify term dominance issues

**Metrics:**
- Recall/precision per term, average contribution
- MAP delta when term removed, rank changes
- Dominance ratios, field distribution

### 4. `scripts/root_cause_diagnostics.py` (580 lines)

**Purpose:** Automated failure mode detection

**Failure Modes Detected:**
1. **OOV_TERMS** - Query terms not in index
2. **SEMANTIC_DILUTION** - Multiple unrelated concepts
3. **DOMINANT_TERM** - One common term overwhelming others
4. **FIELD_MISMATCH** - Terms in wrong field (title vs. body)
5. **PAGERANK_BIAS** - Popular docs displacing relevant ones
6. **COMPOUND_PHRASE** - Multi-word entities (e.g., "Mount Everest")
7. **LOW_TERM_SPECIFICITY** - All terms too common (low IDF)
8. **INSUFFICIENT_MATCHES** - Too few candidates retrieved

**Output:**
- Primary failure mode with severity score
- Supporting evidence and contributing factors
- Actionable recommendations per failure mode

### 5. `scripts/comprehensive_evaluation.py` (400 lines)

**Purpose:** Main orchestration script

**Usage:**
```bash
python scripts/comprehensive_evaluation.py \
    --queries data/queries_train_formatted.json \
    --indices indices_mini \
    --output diagnostics_report.json \
    [--ablation]  # Optional: enable expensive ablation testing
```

**Process:**
1. Load queries and indices
2. Run diagnostic search on all queries
3. Analyze query characteristics
4. Compute term effectiveness
5. Run ablation (if enabled)
6. Diagnose failure modes
7. Generate statistical comparisons
8. Produce comprehensive JSON report

### 6. `scripts/visualization_generator.py` (350 lines)

**Purpose:** Generate diagnostic visualizations

**Visualizations:**
1. **Term Contribution Heatmap** - Queries × Terms
2. **IDF Distribution** - Successful vs. failing queries
3. **Query Length vs. MAP** - Scatter plot
4. **Field Contribution** - Stacked bar chart
5. **Ablation Impact** - Tornado charts

**Usage:**
```bash
python scripts/visualization_generator.py \
    --report diagnostics_report.json \
    --output visualizations/
```

**Note:** Requires `matplotlib` and `seaborn` (optional)

---

## Key Findings

### Top Recommendations (from diagnostics_report.json)

1. **[HIGH] Title Matches Correlate with Success**
   - Statistical finding: Successful queries have higher title_matches
   - **Recommendation:** Increase title weight from 0.25 to 0.40
   - **Expected Impact:** May improve retrieval for queries with title-matching docs

2. **[HIGH] Semantic Dilution (affects 15/30 queries)**
   - Issue: Multiple unrelated concepts diluting scores
   - **Recommendation:**
     - Consider splitting multi-concept queries
     - Boost weights of rarer, more specific terms
     - Use query segmentation or concept extraction

3. **[HIGH] Compound Phrases (affects 7/30 queries)**
   - Issue: Named entities treated as separate tokens
   - Examples: "Mount Everest", "Wright brothers", "Great Fire"
   - **Recommendation:**
     - Treat capitalized sequences as bigrams
     - Implement named entity recognition
     - Use proximity scoring for adjacent terms

4. **[MEDIUM] Low Term Specificity**
   - Issue: Some queries have all common terms (low mean IDF)
   - **Recommendation:**
     - Add more specific terms to queries
     - Consider query expansion with synonyms
     - Increase minimum IDF threshold

### Zero-MAP Queries Analysis

**Queries with Zero Results (10/30):**
1. Mount Everest climbing expeditions (46 relevant docs)
2. Great Fire of London 1666 (39 relevant docs)
3. Robotics automation industry (43 relevant docs)
4. Wright brothers first flight (44 relevant docs)
5. Renaissance architecture Florence Italy (46 relevant docs)
6. Silk Road trade cultural exchange (47 relevant docs)
7. Green Revolution agriculture yield (49 relevant docs)
8. Roman aqueducts engineering innovation (48 relevant docs)
9. Coffee history Ethiopia trade (43 relevant docs)
10. Ballet origins France Russia (45 relevant docs)

**Common Patterns:**
- Most have 40+ relevant docs available (content exists!)
- Many contain compound phrases or named entities
- Multi-concept queries (e.g., "Silk Road" + "trade" + "cultural exchange")
- Historical events with specific terms

**Primary Issues:**
- **SEMANTIC_DILUTION:** 50% (terms too diverse)
- **COMPOUND_PHRASE:** 30% (named entities split)
- **FIELD_MISMATCH:** 20% (terms in title/anchor but not body)

---

## Statistical Insights

### Successful vs. Failing Queries Comparison

**Significant Differences (p < 0.05):**

| Metric | Successful (MAP>0) | Failing (MAP=0) | Difference | Interpretation |
|--------|-------------------|-----------------|------------|----------------|
| **Title Matches** | 2.8 | 0.9 | +1.9 | Successful queries match more titles |
| **Mean IDF** | 1.65 | 1.42 | +0.23 | Successful queries have rarer terms |
| **IDF Variance** | 0.45 | 0.68 | -0.23 | Failing queries less coherent |
| **Body Matches** | 28.5 | 18.2 | +10.3 | Successful queries match more docs |

**Non-Significant:**
- Query length (both ~4-5 unique terms)
- OOV percentage (both ~5-10%)
- Anchor matches (both similar)

---

## Usage Guide

### Quick Start

1. **Run Full Diagnostic Evaluation:**
```bash
cd "/path/to/ir_proj_20251213 2"

python scripts/comprehensive_evaluation.py \
    --queries data/queries_train_formatted.json \
    --indices indices_mini \
    --output diagnostics_report.json
```

2. **Generate Visualizations (optional):**
```bash
pip install matplotlib seaborn  # If not installed

python scripts/visualization_generator.py \
    --report diagnostics_report.json \
    --output visualizations/
```

3. **Examine Results:**
```bash
# View summary statistics
jq '.summary' diagnostics_report.json

# View recommendations
jq '.recommendations' diagnostics_report.json

# View specific query diagnosis
jq '.query_diagnostics[0].diagnosis' diagnostics_report.json
```

### Analyzing Individual Queries

The diagnostic report contains detailed analysis for each query. Example structure:

```json
{
  "query": "Mount Everest climbing expeditions",
  "evaluation": {
    "map": 0.0,
    "num_relevant": 46,
    "num_retrieved": 10
  },
  "diagnosis": {
    "primary_issue": "COMPOUND_PHRASE",
    "evidence": {
      "compound_phrases": ["Mount Everest"],
      "severity": 9.0
    },
    "recommended_fixes": [
      "Query contains compound phrases: Mount Everest",
      "Consider treating these as bigrams or phrases",
      "Use proximity scoring to boost documents with adjacent terms"
    ]
  },
  "characteristics": {
    "term_specificity": {
      "mean_idf": 1.808,
      "idf_variance": 0.523
    },
    "field_matching": {
      "body_matches": 12,
      "title_matches": 2,
      "anchor_matches": 5
    }
  }
}
```

### Customizing Field Weights

Test different weight configurations:

```bash
python scripts/comprehensive_evaluation.py \
    --queries data/queries_train_formatted.json \
    --indices indices_mini \
    --output diagnostics_title_heavy.json \
    --w-body 0.35 \
    --w-title 0.40 \
    --w-anchor 0.15 \
    --w-pagerank 0.05 \
    --w-pageviews 0.05
```

### Enable Ablation Testing

**Warning:** Computationally expensive (runs N+1 searches per query where N = unique terms)

```bash
python scripts/comprehensive_evaluation.py \
    --queries data/queries_train_formatted.json \
    --indices indices_mini \
    --output diagnostics_with_ablation.json \
    --ablation
```

Ablation results show MAP delta when each term is removed, helping identify:
- **Critical terms** (large negative delta)
- **Redundant terms** (near-zero delta)
- **Harmful terms** (positive delta - query improves without it!)

---

## Interpreting Results

### Failure Mode Severity

- **0-3:** Low severity, minor issue
- **4-6:** Medium severity, noticeable impact
- **7-10:** High severity, critical issue

### Effect Size (Cohen's d)

- **< 0.2:** Negligible difference
- **0.2-0.5:** Small effect
- **0.5-0.8:** Medium effect
- **> 0.8:** Large effect

### IDF Values

- **< 0.5:** Very common term (high DF)
- **0.5-1.5:** Moderate specificity
- **1.5-2.5:** Specific term (good discriminator)
- **> 2.5:** Rare term (very specific)

---

## Actionable Next Steps

### Immediate Improvements (Based on Diagnostic Findings)

1. **Increase Title Weight**
   - Current: 0.25
   - Recommended: 0.40
   - Rationale: Strong correlation with success

2. **Implement Bigram Matching**
   - Detect capitalized sequences ("Mount Everest", "Great Fire")
   - Treat as single units in tokenization
   - Or boost proximity scores for adjacent terms

3. **Query-Specific Fixes**
   - For multi-concept queries: segment or boost rare terms
   - For named entity queries: add NER preprocessing
   - For common-term queries: expand with synonyms

4. **Term Weighting Adjustments**
   - Apply sublinear TF scaling to reduce common term impact
   - Boost rare terms (IDF > 2.0) more aggressively

### Long-term Enhancements

1. **Query Expansion**
   - Add synonyms for low-IDF terms
   - Use word embeddings for semantic expansion

2. **Named Entity Recognition**
   - Preprocess queries to identify entities
   - Treat entities as single units

3. **Query Classification**
   - Detect query type (entity, concept, multi-topic)
   - Apply different retrieval strategies per type

4. **Learning to Rank**
   - Use diagnostic features as input
   - Train on successful vs. failing patterns

---

## File Structure

```
scripts/
├── diagnostic_search.py              # Instrumented search backend
├── query_analyzer.py                 # Statistical analysis
├── term_contribution_analyzer.py     # Ablation testing
├── root_cause_diagnostics.py         # Failure mode detection
├── comprehensive_evaluation.py       # Main orchestrator
├── visualization_generator.py        # Plot generation
└── test_diagnostic_search.py         # Unit test example

Output:
├── diagnostics_report.json           # Full diagnostic report (2MB)
├── visualizations/                   # PNG plots (if generated)
│   ├── term_contribution_heatmap.png
│   ├── idf_distribution_comparison.png
│   ├── query_length_vs_map.png
│   ├── field_contribution_analysis.png
│   └── ablation_impact_query_*.png
```

---

## Dependencies

**Required:**
- Python 3.7+
- numpy
- tqdm

**Optional (for full functionality):**
- scipy (statistical tests)
- matplotlib (visualizations)
- seaborn (enhanced visualizations)

**Install all:**
```bash
pip install numpy tqdm scipy matplotlib seaborn
```

---

## Troubleshooting

### Issue: "ModuleNotFoundError: No module named 'scipy'"
**Solution:** The system works without scipy using simplified statistical tests. Install scipy for full functionality:
```bash
pip install scipy
```

### Issue: "matplotlib not available"
**Solution:** Visualizations are optional. Install if needed:
```bash
pip install matplotlib seaborn
```

### Issue: Evaluation is slow
**Solution:**
- Disable ablation testing (remove `--ablation` flag)
- Run on subset of queries
- Use smaller index (indices_mini vs. full index)

### Issue: Report file is very large
**Solution:** The full diagnostic data is included. To reduce size:
- Remove `diagnostic_data` field from query_diagnostics
- Run on fewer queries
- Use compression: `gzip diagnostics_report.json`

---

## Performance Notes

**Evaluation Time (30 queries on indices_mini):**
- Without ablation: ~5-10 seconds
- With ablation: ~2-5 minutes (N searches per query)

**Memory Usage:**
- Peak: ~200MB (indices loaded in memory)
- Report file: ~2MB (full diagnostics)

**Scalability:**
- Tested on: 42 documents, 30 queries
- Should scale to: 1000s of documents, 100s of queries
- For larger corpora: consider sampling or parallel processing

---

## Citation

If using this diagnostic system in research or publications:

```
IR Project Diagnostic System (2025)
Query Term Contribution Analysis & Failure Mode Detection
https://github.com/yourusername/ir_project
```

---

## Contact & Support

For questions, issues, or contributions:
- See implementation in `scripts/` directory
- Check `diagnostics_report.json` for example output
- Modify thresholds and weights as needed for your use case

---

**Generated:** 2025-12-29
**Version:** 1.0
**Authors:** IR Project Team
