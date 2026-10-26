from pathlib import Path
import hashlib
import json
import numpy as np
from etherglass.io import load_capture

ROOT=Path(__file__).resolve().parents[1]


def test_bundled_external_provenance_and_lengths():
    path=ROOT/'examples'/'real'/'radiopile-its-40.npz'
    meta=json.loads(path.with_name('radiopile-its-40.manifest.json').read_text())
    assert hashlib.sha256(path.read_bytes()).hexdigest()==meta['subset_sha256']
    iq=np.load(path,allow_pickle=False)['iq']
    assert iq.shape==(40,2,500)
    assert np.isfinite(iq).all()
    assert meta['license']=='CC-BY-4.0'


def test_continuous_real_sigmf_checksum_and_schema():
    path=ROOT/'examples'/'real'/'camras-danuri-65536.sigmf-data'
    if not path.exists():
        import pytest
        pytest.skip('Continuous external excerpt not installed')
    from sigmf import SigMFFile
    cap=load_capture(path)
    assert len(cap.samples)==65536
    assert cap.sample_rate==500000
    SigMFFile(metadata=cap.metadata).validate()
