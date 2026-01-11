import sys
from collections import Counter, OrderedDict, defaultdict
import itertools
from itertools import islice, count, groupby
import pandas as pd
import os
import re
from operator import itemgetter
from time import time
from pathlib import Path
import pickle
from google.cloud import storage
from contextlib import closing
import io

PROJECT_ID = 'assignment3-479509'
BLOCK_SIZE = 1999998
TUPLE_SIZE = 6
TF_MASK = 2 ** 16 - 1

def get_bucket(bucket_name):
    return storage.Client(PROJECT_ID).bucket(bucket_name)

def _open(path, mode, bucket=None):
    if bucket is None:
        return open(path, mode)
    
    blob = bucket.blob(str(path))
    if 'w' in mode:
        return GCSCompatibleWriter(blob, str(path))
    else:
        return io.BytesIO(blob.download_as_bytes())

class GCSCompatibleWriter:
    def __init__(self, blob, name):
        self.blob = blob
        self.name = name  
        self.buffer = io.BytesIO()
        self.closed = False
        
    def write(self, b, *args): 
        # ה-*args כאן כדי למנוע את ה-TypeError שקיבלת מה-Spark Worker
        return self.buffer.write(b)
    
    def tell(self):
        return self.buffer.tell()
    
    def close(self):
        if not self.closed:
            self.buffer.seek(0)
            self.blob.upload_from_file(self.buffer)
            self.buffer.close()
            self.closed = True
            
    def __enter__(self): return self
    def __exit__(self, exc_type, exc_val, exc_tb): self.close()

class MultiFileWriter:
    def __init__(self, base_dir, name, bucket_name=None):
        self._base_dir = Path(base_dir)
        self._name = name
        self._bucket = None if bucket_name is None else get_bucket(bucket_name)
        self._file_gen = (_open(str(self._base_dir / f'{name}_{i:03}.bin'), 
                                'wb', self._bucket) 
                          for i in itertools.count())
        self._f = next(self._file_gen)
           
    def write(self, b):
        locs = []
        while len(b) > 0:
            pos = self._f.tell()
            remaining = BLOCK_SIZE - pos
            if remaining == 0:  
                self._f.close()
                self._f = next(self._file_gen)
                pos, remaining = 0, BLOCK_SIZE
            
            self._f.write(b[:remaining])
            # חשוב: שומרים רק את שם הקובץ ללא נתיב מלא ל-GCS
            name = Path(self._f.name).name
            locs.append((name, pos))
            b = b[remaining:]
        return locs

    def close(self):
        self._f.close()

class MultiFileReader:
    def __init__(self, base_dir, bucket_name=None):
        self._base_dir = Path(base_dir)
        self._bucket = None if bucket_name is None else get_bucket(bucket_name)
        self._open_files = {}

    def read(self, locs, n_bytes):
        b = []
        for f_name, offset in locs:
            full_path = str(self._base_dir / f_name)
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

class InvertedIndex:  
    def __init__(self, docs={}):
        self.df = Counter()
        self.term_total = Counter()
        self._posting_list = defaultdict(list)
        self.posting_locs = defaultdict(list)

        for doc_id, tokens in docs.items():
            self.add_doc(doc_id, tokens)

    def add_doc(self, doc_id, tokens):
        w2cnt = Counter(tokens)
        self.term_total.update(w2cnt)
        for w, cnt in w2cnt.items():
            self.df[w] = self.df.get(w, 0) + 1
            self._posting_list[w].append((doc_id, cnt))

    def write_index(self, base_dir, name, bucket_name=None):
        self._write_posting_lists(base_dir, name, bucket_name)
        self._write_globals(base_dir, name, bucket_name)

    def _write_posting_lists(self, base_dir, name, bucket_name):
        with closing(MultiFileWriter(base_dir, name, bucket_name)) as writer:
            for term in sorted(self._posting_list.keys()):
                posting_list = self._posting_list[term]
                b = b''.join([(doc_id << 16 | (tf & TF_MASK)).to_bytes(TUPLE_SIZE, 'big')
                              for doc_id, tf in posting_list])
                locs = writer.write(b)
                self.posting_locs[term] = locs

    def _write_globals(self, base_dir, name, bucket_name):
        path = str(Path(base_dir) / f'{name}.pkl')
        bucket = None if bucket_name is None else get_bucket(bucket_name)
        with _open(path, 'wb', bucket) as f:
            pickle.dump(self, f)

    def __getstate__(self):
        state = self.__dict__.copy()
        if '_posting_list' in state:
            del state['_posting_list']
        return state

    def posting_lists_iter(self, base_dir, bucket_name=None):
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
        
        with closing(MultiFileWriter(base_dir, bucket_id, bucket_name)) as writer:
            for w, pl in list_w_pl: 
                b = b''.join([(doc_id << 16 | (tf & TF_MASK)).to_bytes(TUPLE_SIZE, 'big')
                              for doc_id, tf in pl])
                locs = writer.write(b)
                posting_locs[w].extend(locs)
            
            path = str(Path(base_dir) / f'{bucket_id}_posting_locs.pickle')
            bucket = None if bucket_name is None else get_bucket(bucket_name)
            with _open(path, 'wb', bucket) as f:
                pickle.dump(posting_locs, f)
        return bucket_id

    @staticmethod
    def read_index(base_dir, name, bucket_name=None):
        path = str(Path(base_dir) / f'{name}.pkl')
        bucket = None if bucket_name is None else get_bucket(bucket_name)
        with _open(path, 'rb', bucket) as f:
            return pickle.load(f)