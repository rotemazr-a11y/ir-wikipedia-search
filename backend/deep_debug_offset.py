#!/usr/bin/env python3
"""
Check if document titles match the documents in the index
"""
from google.cloud import storage
import pickle

BUCKET_NAME = '206969750_bucket'

client = storage.Client()
bucket = client.bucket(BUCKET_NAME)

# Load title index
print("Loading title index...")
blob = bucket.blob('postings_gcp/title/pkl/title_index.pkl')
title_idx = pickle.loads(blob.download_as_bytes())

if isinstance(title_idx, dict):
    posting_locs = title_idx.get('posting_locs', {})
    df = title_idx.get('df', {})
else:
    posting_locs = getattr(title_idx, 'posting_locs', {})
    df = getattr(title_idx, 'df', {})

# Load metadata
print("Loading metadata...")
blob = bucket.blob('postings_gcp/metadata.pkl')
metadata = pickle.loads(blob.download_as_bytes())
doc_titles = metadata.get('doc_titles') or metadata.get('titles') or {}

print(f"\nMetadata stats:")
print(f"  Total documents with titles: {len(doc_titles)}")

# Now check: does the doc_id from 'stonehenge' actually have 'Stonehenge' in its title?
print("\n" + "="*60)
print("CHECKING: Does doc 4509951 contain 'Stonehenge'?")
print("="*60)

doc_id = 4509951
actual_title = doc_titles.get(doc_id, "NOT FOUND")
print(f"  Doc ID: {doc_id}")
print(f"  Actual title: {actual_title}")
print(f"  Contains 'stonehenge'? {'✓ YES' if 'stonehenge' in actual_title.lower() else '✗ NO'}")

# Check what the term 'stonehenge' thinks it should return
print("\n" + "="*60)
print("ALL documents containing 'stonehenge' in title:")
print("="*60)

import os
fname_base = os.path.basename(str(posting_locs['stonehenge'][0][0]))
parts = fname_base.replace('.bin', '').split('_')
if len(parts) >= 3:
    field_name = parts[0]
    file_num = parts[-1]
    corrected_fname = f"{field_name}_{file_num}.bin"
else:
    corrected_fname = fname_base

corrected_path = f"postings_gcp/title/bin/{corrected_fname}"
blob = bucket.blob(corrected_path)

file_name, offset = posting_locs['stonehenge'][0]
TUPLE_SIZE = 6
read_size = df['stonehenge'] * TUPLE_SIZE

content = blob.download_as_bytes(start=offset, end=offset + read_size - 1)

print(f"\nTerm: 'stonehenge' (df={df['stonehenge']})")
for i in range(df['stonehenge']):
    chunk = content[i*TUPLE_SIZE : (i+1)*TUPLE_SIZE]
    if len(chunk) == TUPLE_SIZE:
        doc_id_found = int.from_bytes(chunk[:4], 'big')
        tf_val = int.from_bytes(chunk[4:], 'big')
        title = doc_titles.get(doc_id_found, "NOT FOUND")
        print(f"  Doc {doc_id_found}: {title}")
        
# Let's also search for documents that SHOULD contain stonehenge
print("\n" + "="*60)
print("REVERSE CHECK: Finding docs with 'stonehenge' in title")
print("="*60)

stonehenge_docs = []
for doc_id, title in list(doc_titles.items())[:100000]:  # Check first 100k
    if 'stonehenge' in title.lower():
        stonehenge_docs.append((doc_id, title))

print(f"\nFound {len(stonehenge_docs)} documents with 'stonehenge' in title:")
for doc_id, title in stonehenge_docs[:10]:
    print(f"  Doc {doc_id}: {title}")
    
    # Check if this doc is in the title index
    if doc_id == 4509951:
        print(f"    ^ This IS in the title index!")
    else:
        print(f"    ^ This is NOT in the title index (or we haven't checked all)")

print("\n" + "="*60)
print("DIAGNOSIS")
print("="*60)

if stonehenge_docs:
    expected_doc = stonehenge_docs[0][0]
    indexed_doc = 4509951
    
    if expected_doc == indexed_doc:
        print("✓ Index is correct!")
    else:
        print(f"✗ MISMATCH!")
        print(f"  Title index says doc {indexed_doc}")
        print(f"  But actual title is at doc {expected_doc}")
        print(f"\n  This suggests a problem during indexing!")