# Migration Guide - Import Path Updates

After organizing files into folders, you need to update import statements in your code.

## ⚠️ Required Changes

### 1. Test Files

**File:** `tests/test_full_system_integrity.py`
**File:** `tests/test_pre_processing.py`

**OLD Imports:**
```python
from pre_processing import tokenize_and_process, get_term_counts
from inverted_index_gcp import InvertedIndex, TUPLE_SIZE, TF_MASK
from index_health_checker import IndexHealthCheck
```

**NEW Imports:**
```python
from backend.pre_processing import tokenize_and_process, get_term_counts
from backend.inverted_index_gcp import InvertedIndex, TUPLE_SIZE, TF_MASK
from backend.index_health_checker import IndexHealthCheck
```

---

### 2. Backend Internal Imports

**File:** `backend/index_builder.py` (Line 43-44)

**OLD:**
```python
from backend.text_processing import tokenize_and_process, get_term_counts
from backend.inverted_index_gcp import InvertedIndex
```

**NEW:**
```python
from .pre_processing import tokenize_and_process, get_term_counts
from .inverted_index_gcp import InvertedIndex
```
*Note: Use relative imports within the same package*

---

### 3. PySpark Index Builder

**File:** `backend/pyspark_index_builder.py` (Lines 29-32)

**OLD:**
```python
from pre_processing import tokenize_and_process
from inverted_index_gcp import InvertedIndex, MultiFileWriter, TUPLE_SIZE, TF_MASK
```

**NEW:**
```python
from .pre_processing import tokenize_and_process
from .inverted_index_gcp import InvertedIndex, MultiFileWriter, TUPLE_SIZE, TF_MASK
```

---

### 4. Champion List Builder

**File:** `backend/champion_list_builder.py` (Line 22)

**OLD:**
```python
from inverted_index_gcp import InvertedIndex, get_bucket, _open, TUPLE_SIZE, TF_MASK, MultiFileWriter
```

**NEW:**
```python
from .inverted_index_gcp import InvertedIndex, get_bucket, _open, TUPLE_SIZE, TF_MASK, MultiFileWriter
```

---

### 5. PageRank Computer

**File:** `backend/pagerank_computer.py` (Line 49)

**OLD:**
```python
from inverted_index_gcp import get_bucket, _open
```

**NEW:**
```python
from .inverted_index_gcp import get_bucket, _open
```

---

### 6. Index Health Checker

**File:** `backend/index_health_checker.py` (Line 22)

**OLD:**
```python
from inverted_index_gcp import InvertedIndex
```

**NEW:**
```python
from .inverted_index_gcp import InvertedIndex
```

---

## 🔧 Quick Fix Script

Run this to automatically update all import statements:

```bash
# Update test files
sed -i '' 's/^from pre_processing/from backend.pre_processing/g' tests/*.py
sed -i '' 's/^from inverted_index_gcp/from backend.inverted_index_gcp/g' tests/*.py
sed -i '' 's/^from index_health_checker/from backend.index_health_checker/g' tests/*.py

# Update backend internal imports (to relative imports)
cd backend
sed -i '' 's/^from pre_processing/from .pre_processing/g' *.py
sed -i '' 's/^from inverted_index_gcp/from .inverted_index_gcp/g' *.py
sed -i '' 's/^from index_health_checker/from .index_health_checker/g' *.py
cd ..
```

---

## ✅ Verification

After making changes, verify everything works:

```bash
# Test imports
python -c "from backend import tokenize_and_process, InvertedIndex"

# Run tests
python tests/test_full_system_integrity.py

# Run individual module
python backend/index_health_checker.py --help
```

---

## 📝 Notes

### Running Scripts

**Before (old structure):**
```bash
python pyspark_index_builder.py --input data.parquet --output indices/
```

**After (new structure):**
```bash
python backend/pyspark_index_builder.py --input data.parquet --output indices/
# or use -m flag
python -m backend.pyspark_index_builder --input data.parquet --output indices/
```

### Jupyter Notebooks

In notebooks, use absolute imports:
```python
import sys
sys.path.insert(0, '..')  # Add parent directory to path

from backend.pre_processing import tokenize_and_process
from backend.inverted_index_gcp import InvertedIndex
```

### GCP Dataproc Submission

When submitting PySpark jobs, the path changes:

**Before:**
```bash
gcloud dataproc jobs submit pyspark pyspark_index_builder.py ...
```

**After:**
```bash
gcloud dataproc jobs submit pyspark backend/pyspark_index_builder.py ...
```

Or upload the entire `backend/` folder to GCS and reference it:
```bash
gsutil -m cp -r backend/ gs://your-bucket/code/
gcloud dataproc jobs submit pyspark gs://your-bucket/code/backend/pyspark_index_builder.py ...
```

---

## 🚨 Common Errors & Solutions

### Error: ModuleNotFoundError: No module named 'pre_processing'

**Cause:** Using old import path
**Fix:** Change to `from backend.pre_processing import ...`

### Error: ImportError: attempted relative import beyond top-level package

**Cause:** Running a backend file directly with relative imports
**Fix:** Run as module: `python -m backend.pyspark_index_builder`

### Error: ModuleNotFoundError: No module named 'backend'

**Cause:** Running from wrong directory
**Fix:** Run from project root: `cd ir_proj_20251213/`

---

**Status:** Ready for migration
**Estimated Time:** 5-10 minutes to update all imports
