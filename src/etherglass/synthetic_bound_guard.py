"""Separate trust anchor for the generated-only synthetic bound example."""
from __future__ import annotations
import hashlib
import json
from pathlib import Path
import numpy as np
from .uncertainty import model_directory, source_fingerprints

_BINDING_SHA256 = '35dfd5aa65e8834cbdef94e171b2354f3d33e811b2f510e60102c891ae8ae433'

_RECORD_SHA256 = '9676136fdad9615c52741da051faba0f8c8c784106d072fd397b37a257a1a46d'


def _sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _record_path():
    return Path(__file__).resolve().parent / 'resources/models/synthetic-bound-v1.json'


def record_sha256():
    return _sha(_record_path())


def _binding_path():
    return _record_path().with_name('synthetic-bound-implementation-v3.json')


def verify_binding():
    if record_sha256() != _RECORD_SHA256:
        raise ValueError('Synthetic bound unavailable: calibration record identity changed.')
    payload = _record_path().read_bytes()
    if hashlib.sha256(payload).hexdigest() != _RECORD_SHA256:
        raise ValueError('Synthetic bound unavailable: calibration record identity changed.')
    record = json.loads(payload)
    payload = _binding_path().read_bytes()
    if hashlib.sha256(payload).hexdigest() != _BINDING_SHA256:
        raise ValueError('Synthetic bound unavailable: implementation binding identity changed.')
    binding = json.loads(payload)
    evidence = _record_path().with_name('synthetic-bound-equivalence-v3.json')
    if (binding['version'] != '3' or binding['parent_record_sha256'] != _RECORD_SHA256
            or _sha(evidence) != binding['equivalence_evidence_sha256']):
        raise ValueError('Synthetic bound unavailable: implementation evidence identity changed.')
    previous_binding = _record_path().with_name('synthetic-bound-implementation-v2.json')
    previous_evidence = _record_path().with_name('synthetic-bound-equivalence-v2.json')
    if (_sha(previous_binding) != binding['previous_implementation_binding_sha256']
            or _sha(previous_evidence) != binding['previous_equivalence_evidence_sha256']):
        raise ValueError('Synthetic bound unavailable: implementation predecessor identity changed.')
    sources = source_fingerprints()
    sources['etherglass/calibrated_example.py'] = _sha(Path(__file__).resolve().with_name('calibrated_example.py'))
    folder = model_directory()
    checkpoints = {str(i): _sha(folder / f'seed{i}' / 'hybrid.pt') for i in range(3)}
    if sources != binding['source_sha256'] or checkpoints != record['checkpoint_sha256']:
        raise ValueError('Synthetic bound unavailable: source or checkpoint identity changed.')
    if _sha(folder / 'hybrid.pt') != checkpoints['0']:
        raise ValueError('Synthetic bound unavailable: canonical checkpoint differs from seed 0.')
    try:
        import scipy, torch
    except ImportError as error:
        raise ValueError('Synthetic bound unavailable: install the replayed neural runtime.') from error
    versions = {'numpy': np.__version__, 'scipy': scipy.__version__, 'torch': torch.__version__}
    if versions != record['runtime']:
        raise ValueError('Synthetic bound unavailable: numerical runtime differs from the replayed runtime.')
    return {**record, 'calibration_record_sha256': _RECORD_SHA256,
            'implementation_binding_sha256': _BINDING_SHA256,
            'implementation_binding_version': binding['version']}
