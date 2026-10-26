import hashlib
import json
from pathlib import Path
import numpy as np

ROOT=Path(__file__).resolve().parents[1]


def test_classifier_provenance_and_report_denominators():
    model=json.loads((ROOT/'models'/'classifier.json').read_text())
    metrics=json.loads((ROOT/'models'/'classifier_metrics.json').read_text())
    assert hashlib.sha256((ROOT/'models'/'classifier.json').read_bytes()).hexdigest()==metrics['model_sha256']
    for path,sha in model['training']['source_sha256'].items():
        fit_path=model['training'].get('source_snapshots',{}).get(path,path)
        assert hashlib.sha256((ROOT/fit_path).read_bytes()).hexdigest()==sha,path
    for path,sha in model.get('runtime_provenance',{}).get('source_sha256',{}).items():
        assert hashlib.sha256((ROOT/path).read_bytes()).hexdigest()==sha,path
    features=ROOT/'models'/'classifier_features.npz'
    assert hashlib.sha256(features.read_bytes()).hexdigest()==model['training']['features_sha256']
    data=np.load(features,allow_pickle=False)
    assert len(data['test_known'])==metrics['known_test_n']==1000
    assert len(data['test_unknown'])==metrics['unknown_test_n']==1000
    assert sum(x['n'] for x in metrics['unknown_by_family'].values())==1000
    # Fresh final tests cannot overlap train/calibration or exploratory ranges.
    assert model['training']['test_known_seed_start']==1100000
    assert model['training']['test_unknown_seed_start']==1200000


def test_three_seed_uncertainty_is_finite_and_bound_is_explicitly_scoped():
    from etherglass.synthetic import generate_example
    from etherglass.uncertainty import ensemble_indicator
    from etherglass.separation import DEFAULT_CHECKPOINT
    from etherglass.classification import model_path
    assert DEFAULT_CHECKPOINT.parent==ROOT/'models'
    assert model_path()==ROOT/'models'/'classifier.json'
    folder=DEFAULT_CHECKPOINT.parent
    # Release CI requires complete uncertainty artifacts, rather than silently
    # reporting the availability of a single seed as an uncertainty check.
    assert all((folder/f'seed{s}'/'hybrid.pt').is_file() for s in [0,1,2])
    assert (folder/'uncertainty.json').is_file()
    x=generate_example(990711,n=4096,split='test')['mixture']
    u=ensemble_indicator(x,1e6,0,125000)
    assert u['status']=='synthetic-calibrated'
    assert np.isfinite(u['relative_disagreement_rms'])
    assert np.isfinite(u['waveform_error_rms_upper']) and u['waveform_error_rms_upper']>=0
    assert 'synthetic' in u['scope'].lower()
    assert u['nominal_coverage']==.9
