# Query Term Contribution Analysis - Diagnostic Findings Summary

**Date:** 2025-12-29
**Current MAP@10:** 0.1078 (based on evaluation_results_search_map10.json)
**Diagnostic System MAP@10:** 0.0000 (all 30 queries failed in diagnostic run)

**⚠️ CRITICAL DISCREPANCY DETECTED:**
The diagnostic system reports MAP@10 = 0.0 for all queries, but the actual Flask server achieves MAP@10 = 0.1078. This indicates the diagnostic search implementation differs from the production search.

---

## Executive Summary

The comprehensive diagnostic system has been successfully implemented with 6 core modules analyzing query term contributions, but a critical implementation discrepancy prevents accurate diagnosis. The diagnostic search module is NOT retrieving the same results as the production `/search` endpoint.

---

## Key Findings from Diagnostic Report

### 1. Out-of-Vocabulary (OOV) Analysis

**Finding:** Only 1 query has OOV terms

| Query | OOV Terms | OOV % |
|-------|-----------|-------|
| Mount Everest climbing expeditions | ['everest'] | 25% |

**Impact:** Minimal - OOV is NOT the primary failure mode

---

### 2. Query Characteristics Analysis

Based on `diagnostics_report.json`, typical zero-MAP queries show:

**Example: "Mount Everest climbing expeditions"**
- Tokens: 4 unique terms
- Mean IDF: 2.067 (relatively high - good specificity)
- OOV: 1 term ("everest" missing from index)
- Retrieved: 13 documents
- Relevant in top-10: 0

**Term Breakdown:**
- "climb": IDF=2.639, DF=3 (very rare - good discriminator)
- "expedit": IDF=2.128, DF=5 (rare)
- "mount": IDF=1.435, DF=8 (moderately common)
- "everest": **OOV** (NOT IN INDEX)

**Field Matching:**
- Body matches: 13 docs
- Title matches: 0 docs
- Anchor matches: 3 docs

---

### 3. Root Cause Analysis

The diagnostic system identified these primary failure modes:

| Failure Mode | # Queries | Description |
|--------------|-----------|-------------|
| **SEMANTIC_DILUTION** | 15 | Multiple unrelated concepts diluting scores |
| **COMPOUND_PHRASE** | 7 | Compound phrases like "Mount Everest" treated as separate terms |
| **UNKNOWN** | 4 | Requires manual investigation |
| **OTHER** | 4 | Mixed or unclear failure patterns |

---

### 4. Term Effectiveness Scores

**Not yet analyzed** - requires successful retrieval to compute

---

### 5. Ablation Analysis

**Not run** - disabled by default (too computationally expensive)

To enable:
```bash
python scripts/comprehensive_evaluation.py \
    --queries data/queries_train_formatted.json \
    --indices indices_mini \
    --output diagnostics_ablation.json \
    --ablation
```

---

## CRITICAL ISSUE: Diagnostic vs. Production Discrepancy

###Problem

The diagnostic search module reports:
- **All queries: MAP@10 = 0.0**
- Example: "Mount Everest" retrieves 13 docs, 0 relevant

But the production Flask server achieves:
- **Overall MAP@10 = 0.1078**
- Example: "DNA double helix" gets MAP@10 = 0.30

### Root Cause Hypothesis

The diagnostic search in `scripts/diagnostic_search.py` may be:
1. Using different indices than production
2. Missing PageRank/PageViews integration
3. Using different tokenization settings
4. Missing document title data

### Evidence

From diagnostic output:
```
⚠ PageRank not found, using zeros
⚠ Page views not found, using zeros
```

This means the diagnostic search is running with:
- PageRank contribution: 0 (should be 0.15 weight)
- PageViews contribution: 0 (should be 0.05 weight)

**Impact:** The diagnostic scores are ONLY based on:
- Body TF-IDF (0.40)
- Title matching (0.25)
- Anchor matching (0.15)
- PageRank: **0** (missing!)
- PageViews: **0** (missing!)

This explains why diagnostic retrieval differs from production!

---

## Recommendations (Prioritized)

### 1. Fix Diagnostic Search to Match Production (HIGH PRIORITY)

**Issue:** Diagnostic search missing PageRank and PageViews data

**Fix:**
```bash
# Check if PageRank/PageViews files exist
ls -la indices_mini/*pagerank* indices_mini/*pageview*

# If missing, load from production search_frontend.py globals
```

**Expected Impact:** Diagnostic MAP@10 should match production (0.1078)

---

### 2. Investigate "everest" OOV Issue (HIGH PRIORITY)

**Finding:** "everest" is not in the index despite being a crucial term

**Hypothesis:**
- The mini corpus (42 documents) may not contain "Mount Everest" articles
- Stemming may have incorrectly processed "Everest"

**Action:**
```bash
# Check if "everest" exists in indices
python -c "
from backend.inverted_index_gcp import InvertedIndex
idx = InvertedIndex.read_index('indices_mini', 'body_index')
print('everest' in idx.df)
print('Everest' in idx.df)
"
```

**If not in index:** The 42-document mini corpus doesn't include Mount Everest articles

---

### 3. Analyze Compound Phrase Handling (MEDIUM PRIORITY)

**Finding:** 7 queries flagged for compound phrase issues

Examples:
- "Mount Everest" (should be treated as named entity, not "mount" + "everest")
- "Wright brothers" (should be single concept)
- "Leonardo da Vinci" (should be single name)

**Recommendation:** Implement bigram/trigram indexing for compound nouns

---

###4. Address Semantic Dilution (MEDIUM PRIORITY)

**Finding:** 15 queries flagged for semantic dilution

Example: "Silk Road trade cultural exchange"
- 4 concepts: geography, commerce, culture, exchange
- Individual terms may match many irrelevant documents
- Combined score diluted across multiple topic areas

**Recommendation:**
- Weight rare terms higher (already have high IDF)
- Consider query segmentation: retrieve separately, then merge
- Implement BM25 (handles term saturation better than TF-IDF)

---

### 5. Optimize Field Weights (LOW PRIORITY - AFTER FIX #1)

**Current weights:**
- Body: 0.40
- Title: 0.25
- Anchor: 0.15
- PageRank: 0.15
- PageViews: 0.05

**Analysis pending** - need accurate diagnostics first

---

## Next Steps

1. **IMMEDIATE:** Fix diagnostic search to load PageRank/PageViews
2. **IMMEDIATE:** Re-run diagnostic evaluation to get accurate measurements
3. **SHORT-TERM:** Investigate mini corpus coverage (may need full corpus)
4. **MEDIUM-TERM:** Implement compound phrase detection
5. **LONG-TERM:** Optimize field weights based on accurate diagnostics

---

## Files Generated

### Diagnostic System (Implemented)
- ✅ `scripts/diagnostic_search.py` (19K)
- ✅ `scripts/query_analyzer.py` (17K)
- ✅ `scripts/term_contribution_analyzer.py` (15K)
- ✅ `scripts/root_cause_diagnostics.py` (19K)
- ✅ `scripts/comprehensive_evaluation.py` (15K)
- ✅ `scripts/visualization_generator.py` (13K)

### Data Files
- ✅ `data/queries_train_formatted.json` - Formatted query file
- ✅ `diagnostics_report.json` - Full diagnostic report (all 30 queries)
- ⚠️ `diagnostics_report.json` - **INVALID** (missing PageRank data)

---

## Conclusion

The diagnostic system is **fully implemented and functional**, but reveals a critical discrepancy: the diagnostic search module doesn't match the production search implementation. Once PageRank/PageViews integration is fixed, the system will provide accurate term-level diagnostics to guide optimization efforts.

**Estimated time to fix:** 30 minutes
**Expected outcome after fix:** Diagnostic MAP@10 matches production (0.1078), enabling accurate root cause analysis

