"""The conditional route admits only its bound generated population."""
import subprocess
import sys
import numpy as np
import pytest
from etherglass import calibrated_example as ce
from etherglass import synthetic_bound_guard as guard


def test_runtime_contract():
    torch = pytest.importorskip('torch')
    scipy = pytest.importorskip('scipy')
    if (np.__version__, scipy.__version__, torch.__version__) != ('2.5.4', '1.18.1', '2.6.0'):
        with pytest.raises(ValueError, match='numerical runtime'):
            ce.generated_example()
    else:
        result = ce.generated_example()
        assert result['status'] == 'synthetic-conditional-bound'
        assert result['sample_count'] == 4096 and result['waveform_error_rms_upper'] > 0
        assert 'exchangeability' in result['limitations'] and 'Not physical' in result['limitations']


def test_unsupported_runtime_refused(monkeypatch):
    torch = pytest.importorskip('torch')
    monkeypatch.setattr(torch, '__version__', '2.9.1+cpu')
    with pytest.raises(ValueError, match='numerical runtime'):
        ce.generated_example()


def test_numeric_source_drift_refused(monkeypatch):
    monkeypatch.setattr(guard, 'source_fingerprints', lambda: {'etherglass/spectral.py': '0'*64})
    with pytest.raises(ValueError, match='identity changed'):
        ce.generated_example()


def test_wrapper_source_drift_refused(monkeypatch):
    actual = guard._sha
    monkeypatch.setattr(guard, '_sha', lambda p: '0'*64 if p.name == 'calibrated_example.py' else actual(p))
    with pytest.raises(ValueError, match='identity changed'):
        ce.generated_example()


def test_checkpoint_drift_refused(monkeypatch):
    actual = guard._sha
    monkeypatch.setattr(guard, '_sha', lambda p: '0'*64 if p.parent.name == 'seed1' else actual(p))
    with pytest.raises(ValueError, match='identity changed'):
        ce.generated_example()


def test_wrong_generated_length_refused(monkeypatch):
    monkeypatch.setattr(ce, 'verify_binding', lambda: {})
    monkeypatch.setattr(ce, 'generate_example', lambda *a, **k: {'mixture': np.zeros(4095, dtype=np.complex64)})
    with pytest.raises(ValueError, match='frame length'):
        ce.generated_example()


def test_cli_rejects_import_and_selection(tmp_path):
    for extra in [('--path', 'claimed.sigmf-data'), ('--start', '1'), ('--stop', '100'), ('--seed', '123')]:
        cmd = [sys.executable, '-m', 'etherglass.cli', 'synthetic-bound-example', '--output', str(tmp_path/'no'), *extra]
        result = subprocess.run(cmd, text=True, capture_output=True)
        assert result.returncode != 0 and 'unrecognized arguments' in result.stderr
        assert not (tmp_path/'no').exists()


def test_calibration_record_drift_refused(monkeypatch):
    monkeypatch.setattr(guard, 'record_sha256', lambda: '0'*64)
    with pytest.raises(ValueError, match='calibration record identity changed'):
        ce.generated_example()


def test_canonical_checkpoint_drift_refused(monkeypatch):
    actual = guard._sha
    monkeypatch.setattr(guard, '_sha', lambda p: '0'*64 if p.name == 'hybrid.pt' and p.parent.name == 'models' else actual(p))
    with pytest.raises(ValueError, match='canonical checkpoint'):
        ce.generated_example()


def test_missing_torch_is_friendly(monkeypatch):
    monkeypatch.setitem(sys.modules, 'torch', None)
    with pytest.raises(ValueError, match='install the replayed neural runtime'):
        ce.generated_example()
