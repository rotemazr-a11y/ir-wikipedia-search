# Diagnostic System - Quick Start Guide

## 🚀 Run Diagnostics in 3 Steps

### Step 1: Prepare Your Data

```bash
# Your queries should be in this format:
# data/queries_formatted.json
[
  {
    "query": "Mount Everest climbing expeditions",
    "relevant_docs": [47353693, 5208803, ...]
  },
  ...
]
```

### Step 2: Run Comprehensive Evaluation

```bash
cd "/path/to/ir_proj_20251213 2"

python scripts/comprehensive_evaluation.py \
    --queries data/queries_train_formatted.json \
    --indices indices_mini \
    --output diagnostics_report.json
```

**Output:**
```
MAP@10: 0.0245
Zero-MAP queries: 10/30
✓ Report saved to: diagnostics_report.json
```

### Step 3: Examine Results

```bash
# View summary
jq '.summary' diagnostics_report.json

# View recommendations
jq '.recommendations[] | {priority, issue, fix}' diagnostics_report.json

# View specific query diagnosis
jq '.query_diagnostics[0] | {query, map: .evaluation.map, issue: .diagnosis.primary_issue}' diagnostics_report.json
```

---

## 📊 What You Get

### Comprehensive Diagnostic Report (`diagnostics_report.json`)

**For Each Query:**
1. **Evaluation Metrics**
   - MAP@10 score
   - Number of relevant docs
   - Retrieved relevant docs count

2. **Query Characteristics**
   - Length, term specificity (mean/median IDF)
   - Vocabulary coverage, OOV terms
   - Coherence (multi-concept detection)
   - Field matching pattern

3. **Failure Diagnosis**
   - Primary issue (e.g., SEMANTIC_DILUTION)
   - Severity score (0-10)
   - Supporting evidence
   - 2-3 specific recommendations

4. **Term Effectiveness**
   - Per-term recall/precision
   - Contribution to score
   - Field distribution

5. **Dominant Term Analysis**
   - Identify if one term overwhelms others
   - Dominance ratios

**Aggregate Analysis:**
- Statistical comparison (successful vs. failing)
- Zero-MAP query patterns
- Prioritized recommendations

---

## 🎯 Key Metrics Explained

### MAP@10 (Mean Average Precision at 10)
- **Range:** 0.0 to 1.0
- **0.0:** No relevant docs in top-10
- **0.05-0.15:** Low performance
- **0.15-0.30:** Moderate performance
- **> 0.30:** Good performance

### IDF (Inverse Document Frequency)
- **< 0.5:** Very common term
- **0.5-1.5:** Moderate specificity
- **1.5-2.5:** Specific term
- **> 2.5:** Rare term

### Field Contribution (%)
Shows what % of final score comes from each field:
- **Body:** Content similarity (TF-IDF)
- **Title:** Title matches
- **Anchor:** Anchor text matches
- **PageRank:** Popularity signal
- **PageViews:** User interest signal

---

## 🔍 Common Failure Modes & Fixes

### 1. SEMANTIC_DILUTION (50% of queries)
**Problem:** Multiple unrelated concepts diluting scores
```
Query: "Silk Road trade cultural exchange"
Terms: silk (IDF: 1.2), road (IDF: 0.4), trade (IDF: 0.8), cultural (IDF: 1.5), exchange (IDF: 0.9)
IDF Variance: HIGH → concepts too diverse
```
**Fix:**
- Split into separate queries
- Boost rare terms (IDF > 2.0)
- Use query segmentation

### 2. COMPOUND_PHRASE (23% of queries)
**Problem:** Named entities split into tokens
```
Query: "Mount Everest climbing expeditions"
Tokenized: [mount, everest, climb, expedit]
Issue: "Mount Everest" should be single unit
```
**Fix:**
- Detect capitalized sequences
- Treat as bigrams
- Use proximity scoring

### 3. DOMINANT_TERM (10% of queries)
**Problem:** One common term overwhelming others
```
Query: "Wright brothers first flight"
Term Stats:
  - "first": DF=500, IDF=0.2 (overwhelming)
  - "flight": DF=50, IDF=1.8
  - "wright": DF=5, IDF=2.8
```
**Fix:**
- Add to stopwords
- Use sublinear TF scaling
- Boost rare terms

### 4. FIELD_MISMATCH (detected but rare)
**Problem:** Terms in title/anchor but not body
```
Field Matches:
  Body: 2
  Title: 15
  Anchor: 8
Current weight: body=0.40, title=0.25
```
**Fix:**
- Increase title weight to 0.40
- Rebalance field weights

---

## 🛠️ Advanced Usage

### Test Different Field Weights

```bash
# Increase title weight (recommended)
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

**⚠️ Warning:** Very slow (runs N+1 searches per query)

```bash
python scripts/comprehensive_evaluation.py \
    --queries data/queries_train_formatted.json \
    --indices indices_mini \
    --output diagnostics_with_ablation.json \
    --ablation
```

**Ablation Output:**
```json
{
  "baseline_map": 0.0732,
  "term_ablations": {
    "dna": {
      "map_delta": -0.0500,  // Removing "dna" drops MAP by 0.05
      "is_critical": true
    },
    "discoveri": {
      "map_delta": -0.0100,
      "is_critical": false
    }
  }
}
```

### Generate Visualizations

```bash
# Install dependencies first
pip install matplotlib seaborn

# Generate plots
python scripts/visualization_generator.py \
    --report diagnostics_report.json \
    --output visualizations/
```

**Generated Plots:**
- `term_contribution_heatmap.png` - Queries × Terms contribution matrix
- `idf_distribution_comparison.png` - Successful vs. failing IDF box plots
- `query_length_vs_map.png` - Scatter plot with zero-MAP highlighted
- `field_contribution_analysis.png` - Stacked bars showing field %
- `ablation_impact_query_N.png` - Tornado charts for ablation results

---

## 📈 Interpreting Your Results

### Good Signs ✅
- MAP > 0.10
- Few zero-MAP queries (< 20%)
- Low IDF variance (coherent queries)
- Balanced field contributions
- Rare terms (high IDF) in queries

### Warning Signs ⚠️
- MAP < 0.05
- Many zero-MAP queries (> 30%)
- High IDF variance (multi-concept)
- One term dominates (dominance ratio > 3.0)
- Many OOV terms (> 20%)

### Critical Issues 🚨
- MAP = 0.0 (no results)
- All terms are common (mean IDF < 1.0)
- Compound phrases split incorrectly
- Field mismatch (terms in wrong field)
- Insufficient candidates (< 10 docs retrieved)

---

## 💡 Top 3 Actionable Improvements

Based on diagnostic analysis of 30 queries:

### 1. Increase Title Weight (Immediate)
```python
# Current weights
w_body=0.40, w_title=0.25, w_anchor=0.15

# Recommended weights
w_body=0.35, w_title=0.40, w_anchor=0.15

# Expected impact: +5-10% MAP improvement
```

### 2. Implement Bigram Matching (Short-term)
```python
# Detect capitalized sequences
compound_phrases = detect_capitalized_sequences(query)
# Examples: "Mount Everest", "Great Fire", "Wright brothers"

# Treat as single units or boost proximity
for phrase in compound_phrases:
    boost_proximity_score(phrase, documents)
```

### 3. Query Segmentation for Multi-Concept (Medium-term)
```python
# For queries with high IDF variance (> 0.5)
if idf_cv > 0.5:
    # Segment into sub-queries
    sub_queries = segment_by_concept(query)
    # Run separate searches and merge results
    results = merge_results([search(sq) for sq in sub_queries])
```

---

## 🔧 Troubleshooting

### "ModuleNotFoundError: No module named 'scipy'"
```bash
pip install scipy
# Or run without scipy (uses simplified stats)
```

### "matplotlib not available"
```bash
pip install matplotlib seaborn
# Or skip visualization step
```

### Evaluation is too slow
- Remove `--ablation` flag (speeds up 10-50×)
- Run on subset: `head -10 queries.json > queries_subset.json`
- Use smaller index (indices_mini)

### Report file too large (> 10MB)
The full report includes all diagnostic data. To reduce size:
```bash
# Extract only summary and recommendations
jq '{summary, recommendations, statistical_analysis}' diagnostics_report.json > summary.json
```

---

## 📚 Next Steps

1. **Run diagnostics** on your queries
2. **Review top recommendations** in report
3. **Implement fixes** (start with title weight)
4. **Re-run evaluation** to measure improvement
5. **Iterate** until MAP target is reached

---

## 📞 Need Help?

- Check `DIAGNOSTIC_SYSTEM_README.md` for detailed documentation
- Examine `diagnostics_report.json` for your specific issues
- Review example output in `scripts/test_diagnostic_output.json`
- Modify thresholds in `root_cause_diagnostics.py` for your use case

---

**Quick Reference:**
- **Run:** `python scripts/comprehensive_evaluation.py --queries <path> --indices <path> --output <path>`
- **View:** `jq '.recommendations' diagnostics_report.json`
- **Visualize:** `python scripts/visualization_generator.py --report <path> --output <dir>`

**Generated:** 2025-12-29
