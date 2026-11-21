"""Shared paths, deterministic provenance and atomic result writes."""
import hashlib
import json
import os
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / 'config/trained_gnn_v2_20261013.json'
RUN = ROOT / 'results/trained_gnn_v2_20261013'
CACHE = ROOT / 'data/spectral_v2'
EVIDENCE = ROOT / 'evidence/local_only/inspection_v2'
SEED = 20261012

def sha(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for block in iter(lambda: f.read(1048576), b''):
            h.update(block)
    return h.hexdigest()

def seed(*parts):
    return int.from_bytes(hashlib.sha256('|'.join(map(str, (SEED,) + parts)).encode()).digest()[:4], 'little')

def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))

def write(path, obj):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(obj, indent=2, ensure_ascii=False, allow_nan=False,
                             default=lambda x: x.item() if isinstance(x, np.generic) else str(x)), encoding='utf-8')
    os.replace(tmp, path)

def people():
    return sorted(read(EVIDENCE / 'prediction_eligibility_manifest.json')['include'], key=lambda x: x['subject_id'])

def contexts(person):
    return ['first190', 'native230', 'last190'] if person['T'] == 230 else ['first190']

def cache_path(person, context):
    return CACHE / (person['subject_id'] + '_' + context + '.npz')
