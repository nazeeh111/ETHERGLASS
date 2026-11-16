"""The scientific core is useful without neural or web runtimes."""
import json
import os
from pathlib import Path
import subprocess
import sys
import textwrap
import tomllib

import pytest

ROOT = Path(__file__).resolve().parents[1]
BLOCK_OPTIONAL = """
import importlib.abc
import sys
class NoOptionalRuntime(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split('.')[0] in {'torch', 'fastapi', 'uvicorn'}:
            raise ModuleNotFoundError('Optional import forbidden: ' + fullname, name=fullname)
sys.meta_path.insert(0, NoOptionalRuntime())
"""


def run_core(code, *args):
    env = dict(os.environ, PYTHONPATH=str(ROOT / "src"), PYTHONDONTWRITEBYTECODE="1")
    return subprocess.run(
        [sys.executable, "-B", "-c", BLOCK_OPTIONAL + textwrap.dedent(code), *map(str, args)],
        capture_output=True, text=True, timeout=30, env=env,
    )


@pytest.mark.parametrize("method", ["bandpass", "matched", "wiener", "notch"])
def test_classical_analysis_and_export_without_optional_runtimes(method):
    result = run_core("""
        import io, json, zipfile
        import numpy as np
        from etherglass.analysis import analyze, public_report
        from etherglass.io import Capture, export_zip, load_bytes
        from etherglass.synthetic import generate_example
        example = generate_example(719, n=4096, split='nominal')
        original = example['mixture'].copy()
        capture = Capture(original.copy(), 1e6, 0)
        result = analyze(capture, symbol_rate=125000, method=sys.argv[1])
        target, residual = result['_arrays']
        np.testing.assert_allclose(target + residual, original, atol=1e-6, rtol=1e-6)
        np.testing.assert_array_equal(capture.samples, original)
        assert np.isfinite(target).all() and np.isfinite(residual).all()
        assert result['separation']['model']['trained'] is False
        report = public_report(result)
        json.dumps(report, allow_nan=False)
        with zipfile.ZipFile(io.BytesIO(export_zip(target, 1e6, 0, report))) as archive:
            restored = load_bytes(archive.read('capture.sigmf-data'),
                                  metadata=json.loads(archive.read('capture.sigmf-meta')))
            np.testing.assert_array_equal(restored.samples, target)
        assert not any(name in sys.modules for name in ('torch', 'fastapi', 'uvicorn'))
    """, method)
    assert result.returncode == 0, result.stderr


def test_missing_neural_runtime_is_actionable_without_substitution():
    result = run_core("""
        import numpy as np
        from etherglass.separation import separate
        try:
            separate(np.ones(2048, dtype=np.complex64), 1e6,
                     symbol_rate=125000, method='hybrid')
        except ValueError as error:
            assert 'etherglass[neural]' in str(error), str(error)
        else:
            raise AssertionError('Neural method was silently replaced')
    """)
    assert result.returncode == 0, result.stderr


def test_default_cli_demo_exports_a_complete_classical_record(tmp_path):
    destination = tmp_path / "demo"
    result = run_core("""
        import runpy
        sys.argv = ['etherglass', 'demo', '--output', sys.argv[1]]
        runpy.run_module('etherglass.cli', run_name='__main__')
    """, destination)
    assert result.returncode == 0, result.stderr
    report = json.loads((destination / "analysis.json").read_text())
    assert report["separation"]["method"] == "bandpass"
    assert report["num_samples"] == 16384
    assert report["provenance"]["license"]
    assert {p.name for p in destination.iterdir()} == {
        "analysis.json", "mixture.sigmf-data", "mixture.sigmf-meta", "target.zip", "residual.zip",
    }
    assert report["separation"]["diagnostics"]["mixture_consistency_max_abs"] < 1e-6


def test_cli_serve_explains_missing_web_capability():
    result = run_core("""
        import runpy
        sys.argv = ['etherglass', 'serve']
        runpy.run_module('etherglass.cli', run_name='__main__')
    """)
    assert result.returncode == 2
    assert "etherglass[web]" in result.stderr
    assert "Traceback" not in result.stderr


def test_core_dependency_metadata_does_not_require_neural_or_web_runtimes():
    project = tomllib.loads((ROOT / "pyproject.toml").read_text())["project"]
    requirements = project["dependencies"]
    for name in ("torch", "fastapi", "uvicorn", "python-multipart"):
        assert not any(r.startswith(name) for r in requirements)
    assert {"neural", "web", "all", "test-core"} <= project["optional-dependencies"].keys()


def test_uncertainty_identity_includes_the_shared_numerical_helpers():
    from etherglass.uncertainty import source_fingerprints

    fingerprints = source_fingerprints()
    assert "etherglass/spectral.py" in fingerprints
    assert all(len(digest) == 64 for digest in fingerprints.values())
