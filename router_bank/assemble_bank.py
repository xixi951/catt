"""Assemble a complete router reference bank from the released vectors and rebuilt GPQA vectors.

The released bank (`embeddings_without_gpqa.npy`, 590 x 1024) holds every reference row except the 171 GPQA
(non-Diamond) rows, in the order of `rows.csv` (`vector_released` = 1). Rebuild the GPQA rows as described in
README.md, save them as a (171, 1024) float32 array in `rows.csv` order (`vector_released` = 0), and run:

  python assemble_bank.py rows.csv embeddings_without_gpqa.npy gpqa_rebuilt.npy OUT/embeddings.npy

Copy OUT/embeddings.npy next to orchestration/v3/v2/exp003/snapshot/data/router/battles.csv and manifest.json.
The script prints the sha256 of the result; it equals the original (manifest.json, embeddings_npy_sha256) only if
the rebuilt vectors are bit-identical, which depends on hardware and library versions.
"""
import csv
import hashlib
import sys

import numpy as np

rows_csv, released, rebuilt, out = sys.argv[1:5]
rows = list(csv.DictReader(open(rows_csv)))
a, g = np.load(released).astype(np.float32), np.load(rebuilt).astype(np.float32)
keep = [int(r['row']) for r in rows if r['vector_released'] == '1']
miss = [int(r['row']) for r in rows if r['vector_released'] == '0']
assert a.shape == (len(keep), 1024) and g.shape == (len(miss), 1024), (a.shape, g.shape)
full = np.zeros((len(rows), 1024), dtype=np.float32)
full[keep] = a
full[miss] = g
np.save(out, full)
print(out, hashlib.sha256(open(out, 'rb').read()).hexdigest())
