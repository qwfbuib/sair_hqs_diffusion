"""Local entrypoints use an immutable snapshot, never import the E: project."""
import os
import sys
from pathlib import Path
import hashlib
import json

ROOT = Path(__file__).resolve().parent
for key in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[key] = '1'
sys.dont_write_bytecode = True
sys.path.insert(0, str(ROOT / 'vendor'))
if os.environ.get('SAIR_USE_LOCAL_CPU_TORCH') == '1' and (ROOT / 'runtime').is_dir():
    sys.path.insert(0, str(ROOT / 'runtime'))

def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()

def write_json(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding='utf-8')

def new_output(path, protected):
    out = Path(path).resolve()
    for item in protected:
        p = Path(item).resolve()
        if out == p or out.is_relative_to(p) or p.is_relative_to(out):
            raise ValueError(f'Output overlaps a protected input: {p}')
    out.mkdir(parents=True, exist_ok=False)
    return out
