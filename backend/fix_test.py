#!/usr/bin/env python3
"""
Quick fix to update test_search.py on the VM.
Run this to fix the .df -> .posting_locs issue.
"""
import sys

# Read the file
with open('test_search.py', 'r') as f:
    content = f.read()

# Replace all .df references with .posting_locs
content = content.replace('engine.body_index.df', 'engine.body_index.posting_locs')
content = content.replace('engine.title_index.df', 'engine.title_index.posting_locs')
content = content.replace('engine.anchor_index.df', 'engine.anchor_index.posting_locs')

# Write back
with open('test_search.py', 'w') as f:
    f.write(content)

print("✓ Fixed test_search.py")
print("  Changed .df to .posting_locs")
print("\nNow run: python3 test_search.py")
