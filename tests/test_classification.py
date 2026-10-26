"""Classification guards and physical features, not accuracy-on-training assertions."""
import numpy as np
import pytest

from etherglass.classification import FEATURE_NAMES, classify, extract_features
from etherglass.synthetic import generate_psk


def test_abstains_without_prior():
    r=classify(np.ones(2048,dtype=complex),1e6)
    assert r["status"] == "unknown"
    assert "symbol-rate" in r["reason"]


@pytest.mark.parametrize("samples,fs,sr",[
    (np.ones(8,dtype=complex),1e6,125000),
    (np.full(1024,np.nan+1j),1e6,125000),
    (np.ones(1024,dtype=complex),0,125000),
    (np.ones(1024,dtype=complex),1e6,1e6),
    (np.ones(2048,dtype=complex),1e6,62500),
])
def test_invalid_inputs_abstain(samples,fs,sr):
    assert classify(samples,fs,sr)["status"] == "unknown"


def test_moment_features_invariant_to_complex_gain():
    rng=np.random.default_rng(918)
    x,_,_=generate_psk(rng,2048,family="QPSK",timing=.3,cfo=.0001)
    f=extract_features(x,1e6,125000)
    g=extract_features(x*3.2*np.exp(.7j),1e6,125000)
    assert f.shape == (len(FEATURE_NAMES),)
    np.testing.assert_allclose(f,g,rtol=2e-4,atol=2e-4)


def test_missing_artifact_abstains(tmp_path):
    x,_,_=generate_psk(np.random.default_rng(18),2048)
    r=classify(x,1e6,125000,model=tmp_path/"missing.json")
    assert r["status"] == "unknown"
    assert "unavailable" in r["reason"]


def test_zero_capture_abstains():
    assert classify(np.zeros(2048,dtype=complex),1e6,125000)["status"] == "unknown"


def test_inference_uses_exact_calibrated_window(monkeypatch):
    import etherglass.classification as module
    x,_,_=generate_psk(np.random.default_rng(209),2048,family="BPSK")
    original=module.extract_features
    observed=[]
    def spy(candidate,fs,sr):
        observed.append(len(candidate))
        return original(candidate,fs,sr)
    monkeypatch.setattr(module,"extract_features",spy)
    first=module.classify(x,1e6,125000)
    extended=module.classify(np.r_[x,np.full(8192,1e5+3e5j)],1e6,125000)
    assert observed == [2048,2048]
    assert first == extended


def test_replaced_classifier_artifact_invalidates_cache(tmp_path):
    import json
    from etherglass.classification import _load_model
    p=tmp_path/'classifier.json'
    p.write_text(json.dumps({'feature_names':FEATURE_NAMES,'fixture':1}))
    assert _load_model(str(p))['fixture']==1
    p.write_text(json.dumps({'feature_names':FEATURE_NAMES,'fixture':22}))
    assert _load_model(str(p))['fixture']==22
