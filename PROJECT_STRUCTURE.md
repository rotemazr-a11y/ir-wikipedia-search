# Wikipedia Search Engine - Project Structure

**Last Updated:** December 22, 2025
**Organization:** Complete & Production-Ready

---

## 📂 Directory Structure

```
ir_proj_20251213/
│
├── 📁 backend/                          # Core Backend Components (8 files)
│   ├── __init__.py                     # Package initialization
│   ├── pre_processing.py               # ⭐ Text tokenization + Porter stemming
│   ├── inverted_index_gcp.py           # Base inverted index (GCS-compatible)
│   ├── index_builder.py                # Legacy single-machine builder
│   ├── pyspark_index_builder.py        # ⭐ Distributed indexing (559 lines)
│   ├── champion_list_builder.py        # ⭐ Two-tier retrieval (371 lines)
│   ├── pagerank_computer.py            # ⭐ PageRank computation (339 lines)
│   ├── index_health_checker.py         # ⭐ Zipf validation (384 lines)
│   └── search_frontend.py              # Flask query service
│
├── 📁 scripts/                          # Deployment Scripts (2 files)
│   ├── run_frontend_in_gcp.sh          # Deploy Flask to GCE
│   └── startup_script_gcp.sh           # GCE startup configuration
│
├── 📁 tests/                            # Test Suite (3 files)
│   ├── __init__.py                     # Package initialization
│   ├── test_full_system_integrity.py   # ⭐ Comprehensive tests (419 lines)
│   └── test_pre_processing.py          # Unit tests for preprocessing
│
├── 📁 docs/                             # Documentation (6 files)
│   ├── IMPLEMENTATION_SUMMARY.md       # ⭐ Complete technical guide
│   ├── README_SYSTEM_TESTS.md          # Testing guide & CI/CD
│   ├── search_engine_testing_optimization_strategy.md  # Testing strategy
│   ├── 03 Text op and indexing.pdf     # Assignment reference
│   ├── 07system.pdf                    # System architecture
│   └── IR final project 2025-2026 (1).pdf  # Project description
│
├── 📁 notebooks/                        # Jupyter Notebooks (3 files)
│   ├── assignment1.ipynb               # Assignment 1 implementation
│   ├── assignment_2-2.ipynb            # Assignment 2 BSBI merge
│   └── run_frontend_in_colab.ipynb     # Colab deployment notebook
│
├── 📁 data/                             # Data Files (1 file)
│   └── queries_train.json              # 30 training queries with ground truth
│
├── 📁 deployment/                       # Deployment Configs (2 files)
│   ├── dataproc_config.yaml            # GCP Dataproc cluster config
│   └── install_dependencies.sh         # Cluster initialization script
│
├── 📄 README.md                         # ⭐ Main project documentation
├── 📄 requirements.txt                  # Python dependencies
├── 📄 .gitignore                        # Git ignore rules
└── 📄 PROJECT_STRUCTURE.md              # This file

⭐ = New/Enhanced for production
```

---

## 🎯 File Categories

### Core Backend (Production-Ready)

| File | Purpose | Lines | Status |
|------|---------|-------|--------|
| `backend/pre_processing.py` | Text tokenization + Porter stemming | 95 | ✅ Enhanced |
| `backend/pyspark_index_builder.py` | Distributed MapReduce indexing | 559 | ✅ NEW |
| `backend/champion_list_builder.py` | Two-tier adaptive retrieval | 371 | ✅ NEW |
| `backend/pagerank_computer.py` | PageRank via DataFrames | 339 | ✅ NEW |
| `backend/index_health_checker.py` | Zipf validation & health checks | 384 | ✅ NEW |
| `backend/inverted_index_gcp.py` | Base inverted index | 202 | ✅ Original |
| `backend/index_builder.py` | Single-machine builder (legacy) | 575 | ⚠️ Legacy |
| `backend/search_frontend.py` | Flask query service | ~200 | ✅ Original |

### Tests

| File | Purpose | Coverage |
|------|---------|----------|
| `tests/test_full_system_integrity.py` | End-to-end integration tests | 13 test methods |
| `tests/test_pre_processing.py` | Unit tests for preprocessing | 5 test methods |

### Documentation

| File | Purpose | Pages |
|------|---------|-------|
| `docs/IMPLEMENTATION_SUMMARY.md` | Complete technical documentation | ~15 |
| `docs/README_SYSTEM_TESTS.md` | Testing guide & CI/CD | ~8 |
| `docs/search_engine_testing_optimization_strategy.md` | Testing strategy (provided) | ~30 |

### Scripts

| File | Purpose |
|------|---------|
| `scripts/run_frontend_in_gcp.sh` | Deploy Flask app to GCE |
| `scripts/startup_script_gcp.sh` | GCE instance startup |
| `deployment/install_dependencies.sh` | Dataproc cluster initialization |

### Data

| File | Purpose | Size |
|------|---------|------|
| `data/queries_train.json` | 30 training queries | ~15KB |

---

## 🔄 Workflow Overview

### Development Workflow

```
1. Edit code in backend/
2. Run tests: python tests/test_full_system_integrity.py
3. Validate: python backend/index_health_checker.py
4. Document changes in docs/
```

### Production Deployment

```
1. Configure: deployment/dataproc_config.yaml
2. Index Wikipedia: backend/pyspark_index_builder.py
3. Compute PageRank: backend/pagerank_computer.py
4. Build Champions: backend/champion_list_builder.py
5. Validate: backend/index_health_checker.py
6. Deploy: scripts/run_frontend_in_gcp.sh
```

---

## 📊 Statistics

| Metric | Count |
|--------|-------|
| **Total Files** | 30 files |
| **Backend Files** | 8 files |
| **Test Files** | 2 files |
| **Documentation Files** | 6 files |
| **Script Files** | 4 files |
| **Total Lines of Code** | ~2,500 lines |
| **New Production Code** | 1,848 lines |
| **Test Code** | 419 lines |

---

## 🎨 Color Legend

- 📁 **Folder** - Directory
- 📄 **File** - Regular file
- ⭐ **New/Enhanced** - Production implementation
- ✅ **Ready** - Production-ready
- ⚠️ **Legacy** - Old implementation (keep for reference)

---

## 🚀 Quick Navigation

### For Development
- **Start here:** [`README.md`](README.md)
- **Run tests:** [`tests/test_full_system_integrity.py`](tests/test_full_system_integrity.py)
- **Main logic:** [`backend/pyspark_index_builder.py`](backend/pyspark_index_builder.py)

### For Deployment
- **GCP config:** [`deployment/dataproc_config.yaml`](deployment/dataproc_config.yaml)
- **Deploy script:** [`scripts/run_frontend_in_gcp.sh`](scripts/run_frontend_in_gcp.sh)
- **Dependencies:** [`requirements.txt`](requirements.txt)

### For Understanding
- **Technical docs:** [`docs/IMPLEMENTATION_SUMMARY.md`](docs/IMPLEMENTATION_SUMMARY.md)
- **Testing guide:** [`docs/README_SYSTEM_TESTS.md`](docs/README_SYSTEM_TESTS.md)
- **Strategy:** [`docs/search_engine_testing_optimization_strategy.md`](docs/search_engine_testing_optimization_strategy.md)

---

## 📝 Notes

### Import Paths (After Organization)

**Before:**
```python
from pre_processing import tokenize_and_process
from inverted_index_gcp import InvertedIndex
```

**After:**
```python
from backend.pre_processing import tokenize_and_process
from backend.inverted_index_gcp import InvertedIndex
```

Or use package imports:
```python
from backend import tokenize_and_process, InvertedIndex
```

### Running Tests

**Before:**
```bash
python test_full_system_integrity.py
```

**After:**
```bash
python tests/test_full_system_integrity.py
# or
python -m pytest tests/
```

---

## ✅ Organization Checklist

- [x] Backend code organized in `backend/`
- [x] Tests organized in `tests/`
- [x] Documentation in `docs/`
- [x] Scripts in `scripts/`
- [x] Deployment configs in `deployment/`
- [x] Data files in `data/`
- [x] Notebooks in `notebooks/`
- [x] Root-level README created
- [x] requirements.txt created
- [x] .gitignore created
- [x] `__init__.py` files added

---

**Organization Status:** ✅ **COMPLETE**
**Project Status:** ✅ **PRODUCTION-READY**
