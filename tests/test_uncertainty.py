import hashlib
import json
import numpy as np
from etherglass import uncertainty


def test_error_bound_invalidated_when_checkpoint_bytes_change(tmp_path,monkeypatch):
    from etherglass import separation
    paths=[]
    for seed in [0,1,2]:
        path=tmp_path/f'seed{seed}'/'hybrid.pt'
        path.parent.mkdir();path.write_bytes(f'fixture-weights-{seed}'.encode());paths.append(path)
    monkeypatch.setattr(uncertainty,'model_directory',lambda:tmp_path)
    monkeypatch.setattr(separation,'separate',lambda x,*args,**kwargs:{'components':[x*.75,x*.25]})
    cal={'nominal_coverage':.9,'quantile':2,'calibration_examples':96,'scope':'synthetic fixture',
         'inference_source_sha256':uncertainty.source_fingerprints(),
         'checkpoint_sha256':{str(s):hashlib.sha256(p.read_bytes()).hexdigest() for s,p in enumerate(paths)}}
    (tmp_path/'uncertainty.json').write_text(json.dumps(cal))
    x=np.ones(512,dtype='complex64')
    valid=uncertainty.ensemble_indicator(x,1e6,0,125000)
    assert valid['status']=='synthetic-calibrated'
    assert valid['waveform_error_rms_upper']>0
    paths[1].write_bytes(b'retrained-different-checkpoint')
    invalid=uncertainty.ensemble_indicator(x,1e6,0,125000)
    assert invalid['status']=='indicator'
    assert 'waveform_error_rms_upper' not in invalid
    assert 'changed after calibration' in invalid['reason']


def test_supplied_estimate_must_use_canonical_ensemble_seed(tmp_path,monkeypatch):
    for seed in [0,1,2]:
        path=tmp_path/f'seed{seed}'/'hybrid.pt';path.parent.mkdir();path.write_bytes(b'ensemble')
    (tmp_path/'hybrid.pt').write_bytes(b'different-model')
    monkeypatch.setattr(uncertainty,'model_directory',lambda:tmp_path)
    x=np.ones(128,dtype='complex64')
    u=uncertainty.ensemble_indicator(x,1e6,0,125000,estimate=x)
    assert u['status']=='unavailable'
    assert 'differs' in u['reason']
