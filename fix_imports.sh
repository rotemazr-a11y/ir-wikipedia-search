#!/bin/bash
# Automatic Import Path Fixer
# Updates all import statements to work with new folder structure

set -e

echo "🔧 Fixing import paths for organized project structure..."

# Fix tests/ imports (use absolute imports to backend)
echo "📁 Updating tests/..."
find tests/ -name "*.py" -type f -exec sed -i '' \
    -e 's/^from pre_processing import/from backend.pre_processing import/g' \
    -e 's/^from inverted_index_gcp import/from backend.inverted_index_gcp import/g' \
    -e 's/^from index_health_checker import/from backend.index_health_checker import/g' \
    -e 's/^from index_builder import/from backend.index_builder import/g' \
    -e 's/^from pyspark_index_builder import/from backend.pyspark_index_builder import/g' \
    -e 's/^from champion_list_builder import/from backend.champion_list_builder import/g' \
    -e 's/^from pagerank_computer import/from backend.pagerank_computer import/g' \
    {} \;

# Fix backend/ internal imports (use relative imports within package)
echo "📁 Updating backend/..."
find backend/ -name "*.py" -type f ! -name "__init__.py" -exec sed -i '' \
    -e 's/^from pre_processing import/from .pre_processing import/g' \
    -e 's/^from inverted_index_gcp import/from .inverted_index_gcp import/g' \
    -e 's/^from index_health_checker import/from .index_health_checker import/g' \
    -e 's/^from index_builder import/from .index_builder import/g' \
    -e 's/^from pyspark_index_builder import/from .pyspark_index_builder import/g' \
    -e 's/^from champion_list_builder import/from .champion_list_builder import/g' \
    -e 's/^from pagerank_computer import/from .pagerank_computer import/g' \
    -e 's/^from backend\./from ./g' \
    {} \;

echo "✅ Import paths updated!"
echo ""
echo "🧪 Verifying changes..."

# Test if imports work
python3 -c "from backend import tokenize_and_process, InvertedIndex" 2>&1 && \
    echo "✅ Backend package imports work" || \
    echo "❌ Backend package imports failed"

python3 -c "from backend.pre_processing import tokenize_and_process" 2>&1 && \
    echo "✅ Pre-processing imports work" || \
    echo "❌ Pre-processing imports failed"

echo ""
echo "📋 Next steps:"
echo "   1. Run tests: python tests/test_full_system_integrity.py"
echo "   2. Verify modules: python -m backend.index_health_checker --help"
echo "   3. Check notebooks and update any import statements there"
