import sys
from collections import Counter, OrderedDict
import itertools
from itertools import islice, count, groupby
import pandas as pd
import os
import re
import io
from operator import itemgetter
from time import time
from pathlib import Path
import pickle
from google.cloud import storage
from collections import defaultdict
from contextlib import closing

PROJECT_ID = 'durable-tracer-479509-q2'
def get_bucket(bucket_name):
    return storage.Client().bucket(bucket_name)


class GCSBinaryWriter:
    """Wrapper to write binary data to GCS using upload_from_string (compatible with older gcs library)."""
    def __init__(self, bucket, blob_path):
        self._bucket = bucket
        self._blob_path = blob_path
        self._buffer = io.BytesIO()
        self._blob = bucket.blob(blob_path)
        self.name = blob_path
        
    def write(self, data):
        return self._buffer.write(data)
    
    def tell(self):
        return self._buffer.tell()
    
    def seek(self, pos):
        return self._buffer.seek(pos)
    
    def close(self):
        # Upload the buffer contents to GCS
        self._buffer.seek(0)
        self._blob.upload_from_file(self._buffer)
        self._buffer.close()
    
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
        return False


class GCSBinaryReader:
    """Wrapper to read binary data from GCS (compatible with older gcs library)."""
    def __init__(self, bucket, blob_path):
        self._bucket = bucket
        self._blob_path = blob_path
        self._blob = bucket.blob(blob_path)
        # Download full blob content - use download_as_string for max compatibility
        # (returns bytes despite the name, available since earliest versions)
        try:
            self._data = self._blob.download_as_bytes()
        except AttributeError:
            # Fallback for very old versions
            self._data = self._blob.download_as_string()
        self._pos = 0
        self.name = blob_path
        
    def read(self, n=-1):
        if n < 0:
            result = self._data[self._pos:]
            self._pos = len(self._data)
        else:
            result = self._data[self._pos:self._pos + n]
            self._pos += len(result)
        return result
    
    def readline(self, limit=-1):
        """Read until newline or end of data. Required for pickle.load()."""
        start = self._pos
        idx = self._data.find(b'\n', start)
        if idx == -1:
            # No newline found, read to end
            result = self._data[start:]
            self._pos = len(self._data)
        else:
            # Include the newline
            result = self._data[start:idx + 1]
            self._pos = idx + 1
        if limit > 0 and len(result) > limit:
            result = result[:limit]
            self._pos = start + limit
        return result
    
    def seek(self, pos):
        self._pos = pos
        return self._pos
    
    def tell(self):
        return self._pos
    
    def close(self):
        pass  # Nothing to close
    
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
        return False


def _open(path, mode, bucket=None):
    if bucket is None:
        return open(path, mode)
    
    # If path is a full GCS URI (gs://bucket_name/...), extract just the blob path
    if path.startswith('gs://'):
        # Extract blob path: gs://bucket_name/path/to/file -> path/to/file
        parts = path[5:].split('/', 1)  # Remove 'gs://' and split on first /
        if len(parts) > 1:
            path = parts[1]  # Get everything after bucket name
        else:
            path = ''
    
    # Use wrapper classes for GCS compatibility
    if 'w' in mode:
        return GCSBinaryWriter(bucket, path)
    else:
        return GCSBinaryReader(bucket, path)

# Let's start with a small block size of 30 bytes just to test things out. 
BLOCK_SIZE = 1999998


def _make_path(base_dir, filename, is_gcs=False):
    """
    Construct a path safely for both local and GCS.
    Avoids using Path() which corrupts GCS URIs.
    """
    if is_gcs:
        # For GCS: use string concatenation (Path() strips the double slash)
        base = base_dir.rstrip('/')
        return f"{base}/{filename}"
    else:
        # For local: use Path for proper OS handling
        return str(Path(base_dir) / filename)


class MultiFileWriter:
    """ Sequential binary writer to multiple files of up to BLOCK_SIZE each. """
    def __init__(self, base_dir, name, bucket_name=None):
        self._base_dir = base_dir  # Keep as string, don't use Path()!
        self._name = name
        self._bucket = None if bucket_name is None else get_bucket(bucket_name)
        self._is_gcs = bucket_name is not None
        
        def make_file_path(i):
            filename = f'{name}_{i:03}.bin'
            return _make_path(self._base_dir, filename, self._is_gcs)
        
        self._file_gen = (_open(make_file_path(i), 'wb', self._bucket) 
                          for i in itertools.count())
        self._f = next(self._file_gen)
           
    def write(self, b):
        locs = []
        while len(b) > 0:
            pos = self._f.tell()
            remaining = BLOCK_SIZE - pos
            # if the current file is full, close and open a new one.
            if remaining == 0:  
                self._f.close()
                self._f = next(self._file_gen)
                pos, remaining = 0, BLOCK_SIZE
            self._f.write(b[:remaining])
            name = self._f.name if hasattr(self._f, 'name') else self._f._blob.name
            locs.append((name, pos))
            b = b[remaining:]
        return locs

    def close(self):
        self._f.close()

class MultiFileReader:
    """ Sequential binary reader of multiple files of up to BLOCK_SIZE each. """
    def __init__(self, base_dir, bucket_name=None):
        self._base_dir = base_dir  # Keep as string, don't use Path()!
        self._bucket = None if bucket_name is None else get_bucket(bucket_name)
        self._is_gcs = bucket_name is not None
        self._open_files = {}

    def read(self, locs, n_bytes):
        b = []
        for f_name, offset in locs:
            # f_name might be a full GCS path (gs://...) or relative path
            if f_name.startswith('gs://'):
                # Already a full GCS path - use as-is
                full_path = f_name
            elif self._is_gcs:
                # Relative path, need to construct GCS path
                full_path = _make_path(self._base_dir, f_name, self._is_gcs)
            else:
                # Local path
                if f_name.startswith(self._base_dir):
                    full_path = f_name
                else:
                    full_path = _make_path(self._base_dir, f_name, self._is_gcs)
            
            if full_path not in self._open_files:
                self._open_files[full_path] = _open(full_path, 'rb', self._bucket)
            f = self._open_files[full_path]
            f.seek(offset)
            n_read = min(n_bytes, BLOCK_SIZE - offset)
            b.append(f.read(n_read))
            n_bytes -= n_read
        return b''.join(b)
  
    def close(self):
        for f in self._open_files.values():
            f.close()

    def __exit__(self, exc_type, exc_value, traceback):
        self.close()
        return False 

TUPLE_SIZE = 6       # We're going to pack the doc_id and tf values in this 
                     # many bytes.
TF_MASK = 2 ** 16 - 1 # Masking the 16 low bits of an integer


class InvertedIndex:  
    def __init__(self, docs={}):
        """ Initializes the inverted index and add documents to it (if provided).
        Parameters:
        -----------
          docs: dict mapping doc_id to list of tokens
        """
        # stores document frequency per term
        self.df = Counter()
        # stores total frequency per term
        self.term_total = Counter()
        # stores posting list per term while building the index (internally), 
        # otherwise too big to store in memory.
        self._posting_list = defaultdict(list)
        # mapping a term to posting file locations, which is a list of 
        # (file_name, offset) pairs. Since posting lists are big we are going to
        # write them to disk and just save their location in this list. We are 
        # using the MultiFileWriter helper class to write fixed-size files and store
        # for each term/posting list its list of locations. The offset represents 
        # the number of bytes from the beginning of the file where the posting list
        # starts. 
        self.posting_locs = defaultdict(list)

        for doc_id, tokens in docs.items():
            self.add_doc(doc_id, tokens)

    def add_doc(self, doc_id, tokens):
        """ Adds a document to the index with a given `doc_id` and tokens. It counts
            the tf of tokens, then update the index (in memory, no storage 
            side-effects).
        """
        w2cnt = Counter(tokens)
        self.term_total.update(w2cnt)
        for w, cnt in w2cnt.items():
            self.df[w] = self.df.get(w, 0) + 1
            self._posting_list[w].append((doc_id, cnt))

    def write_index(self, base_dir, name, bucket_name=None):
        """ Write the in-memory index to disk. Results in the file: 
            (1) `name`.pkl containing the global term stats (e.g. df).
        """
        #### GLOBAL DICTIONARIES ####
        self._write_globals(base_dir, name, bucket_name)

    def _write_globals(self, base_dir, name, bucket_name):
        is_gcs = bucket_name is not None
        path = _make_path(base_dir, f'{name}.pkl', is_gcs)
        bucket = None if bucket_name is None else get_bucket(bucket_name)
        with _open(path, 'wb', bucket) as f:
            pickle.dump(self, f)

    def __getstate__(self):
        """ Modify how the object is pickled by removing the internal posting lists
            from the object's state dictionary. 
        """
        state = self.__dict__.copy()
        del state['_posting_list']
        return state

    def posting_lists_iter(self, base_dir, bucket_name=None):
        """ A generator that reads one posting list from disk and yields 
            a (word:str, [(doc_id:int, tf:int), ...]) tuple.
        """
        with closing(MultiFileReader(base_dir, bucket_name)) as reader:
            for w, locs in self.posting_locs.items():
                b = reader.read(locs, self.df[w] * TUPLE_SIZE)
                posting_list = []
                for i in range(self.df[w]):
                    doc_id = int.from_bytes(b[i*TUPLE_SIZE:i*TUPLE_SIZE+4], 'big')
                    tf = int.from_bytes(b[i*TUPLE_SIZE+4:(i+1)*TUPLE_SIZE], 'big')
                    posting_list.append((doc_id, tf))
                yield w, posting_list

    def read_a_posting_list(self, base_dir, w, bucket_name=None):
        posting_list = []
        if not w in self.posting_locs:
            return posting_list
        with closing(MultiFileReader(base_dir, bucket_name)) as reader:
            locs = self.posting_locs[w]
            b = reader.read(locs, self.df[w] * TUPLE_SIZE)
            for i in range(self.df[w]):
                doc_id = int.from_bytes(b[i*TUPLE_SIZE:i*TUPLE_SIZE+4], 'big')
                tf = int.from_bytes(b[i*TUPLE_SIZE+4:(i+1)*TUPLE_SIZE], 'big')
                posting_list.append((doc_id, tf))
        return posting_list

    @staticmethod
    def write_a_posting_list(b_w_pl, base_dir, bucket_name=None):
        posting_locs = defaultdict(list)
        bucket_id, list_w_pl = b_w_pl
        is_gcs = bucket_name is not None
        
        with closing(MultiFileWriter(base_dir, bucket_id, bucket_name)) as writer:
            for w, pl in list_w_pl: 
                # convert to bytes
                b = b''.join([(doc_id << 16 | (tf & TF_MASK)).to_bytes(TUPLE_SIZE, 'big')
                              for doc_id, tf in pl])
                # write to file(s)
                locs = writer.write(b)
                # save file locations to index
                posting_locs[w].extend(locs)
            path = _make_path(base_dir, f'{bucket_id}_posting_locs.pickle', is_gcs)
            bucket = None if bucket_name is None else get_bucket(bucket_name)
            with _open(path, 'wb', bucket) as f:
                pickle.dump(posting_locs, f)
        return bucket_id


    @staticmethod
    def read_index(base_dir, name, bucket_name=None):
        is_gcs = bucket_name is not None
        path = _make_path(base_dir, f'{name}.pkl', is_gcs)
        bucket = None if bucket_name is None else get_bucket(bucket_name)
        with _open(path, 'rb', bucket) as f:
            return pickle.load(f)