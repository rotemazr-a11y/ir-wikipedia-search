#!/bin/bash
# Deploy the fixed search_runtime.py

echo "Deploying fixed search_runtime.py..."

# Backup current version
if [ -f search_runtime.py ]; then
    cp search_runtime.py search_runtime.py.backup
    echo "✓ Backed up current version to search_runtime.py.backup"
fi

# Replace with fixed version
cp search_runtime_fixed.py search_runtime.py
echo "✓ Deployed fixed version"

echo ""
echo "Changes made:"
echo "  1. Added robust metadata fallbacks (default doc_length=500)"
echo "  2. Added comprehensive DEBUG logging for every search"
echo "  3. Fixed read_a_posting_list usage with proper error handling"
echo "  4. Added preprocessing alignment (lowercase + stem)"
echo "  5. Logs show: tokens, posting lists fetched, top scores"
echo ""
echo "Test with:"
echo "  python3 -c \"from search_runtime import initialize_engine, get_engine; engine = initialize_engine(); results = engine.search('israel'); print(f'Results: {len(results)}')\""
echo ""
echo "Or start Flask:"
echo "  python3 search_frontend.py"
