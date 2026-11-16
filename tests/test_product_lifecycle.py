"""Product receipts and refusal paths preserve the analyzed result and user files."""
import hashlib
import io
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

import numpy as np
import pytest
from fastapi.testclient import TestClient
from etherglass.app import app
from etherglass.io import encode_sigmf, load_bytes

ROOT = Path(__file__).resolve().parents[1]


def test_export_receipt_rejects_replaced_result_and_preserves_selected_bytes():
    x = (np.arange(4096)/4096 + 1j*np.arange(4096)[::-1]/4096).astype('complex64')
    raw, meta = encode_sigmf(x, 1e6, 0)
    with TestClient(app) as client:
        imported = client.post('/api/import', files={'data_file': ('owned', raw),
                            'meta_file': ('owned.json', json.dumps(meta).encode())})
        assert imported.status_code == 200
        capture_id = imported.json()['capture_id']
        first = client.post('/api/analyze', json={'capture_id': capture_id,
                'start_sample': 0, 'stop_sample': 1024, 'inspect_only': True}).json()
        first_id = first['analysis_id']
        exported = client.get(f'/api/export/{capture_id}', params={'component': 'mixture', 'expected_analysis_id': first_id})
        assert exported.status_code == 200
        with zipfile_from(exported.content) as z:
            report = json.loads(z.read('analysis.json'))
            capture = load_bytes(z.read('capture.sigmf-data'), metadata=json.loads(z.read('capture.sigmf-meta')))
        assert report['analysis_id'] == first_id and report['selection'] == first['selection']
        np.testing.assert_array_equal(capture.samples, x[:1024])
        second = client.post('/api/analyze', json={'capture_id': capture_id,
                'start_sample': 1024, 'stop_sample': 2048, 'inspect_only': True}).json()
        assert second['analysis_id'] != first_id
        assert client.get(f'/api/export/{capture_id}', params={'expected_analysis_id': first_id}).status_code == 409
        assert client.get(f'/api/export/{capture_id}', params={'expected_analysis_id': second['analysis_id']}).status_code == 200
        assert client.get(f'/api/export/{capture_id}').status_code == 200  # Existing caller compatibility.
        assert client.get('/api/export/expired', params={'expected_analysis_id': first_id}).status_code == 404


def zipfile_from(payload):
    import zipfile
    return zipfile.ZipFile(io.BytesIO(payload))


def cli(*args):
    return subprocess.run([sys.executable, '-B', '-m', 'etherglass.cli', *map(str,args)],
                          env=dict(os.environ, PYTHONPATH=str(ROOT/'src'), PYTHONDONTWRITEBYTECODE='1'),
                          capture_output=True, text=True, timeout=30)


def test_refused_input_does_not_create_destination(tmp_path):
    malformed = tmp_path/'malformed.cf32'; malformed.write_bytes(b'123')
    destination = tmp_path/'must-not-exist'
    result = cli('analyze', malformed, '--datatype', 'cf32_le', '--sample-rate', '1000000',
                 '--center-frequency', '0', '--output', destination)
    assert result.returncode == 2 and result.stderr
    assert not destination.exists()


def test_existing_output_bytes_are_preserved(tmp_path):
    destination = tmp_path/'existing'; destination.mkdir()
    sentinel = destination/'owned.txt'; sentinel.write_bytes(b'preserve actual bytes')
    result = cli('demo', '--output', destination)
    assert result.returncode == 2
    assert sorted(p.name for p in destination.iterdir()) == ['owned.txt']
    assert sentinel.read_bytes() == b'preserve actual bytes'


def test_current_manifest_rejects_unlisted_eligible_file(tmp_path):
    scripts = tmp_path/'scripts'; scripts.mkdir()
    shutil.copy2(ROOT/'scripts/verify_artifacts.py', scripts/'verify_artifacts.py')
    (tmp_path/'ARTIFACTS.json').write_text(json.dumps({'version':'fixture','algorithm':'SHA256','files':{}}))
    (tmp_path/'unlisted.txt').write_bytes(b'new current artifact')
    result = subprocess.run([sys.executable, '-B', str(scripts/'verify_artifacts.py')], capture_output=True, text=True)
    assert result.returncode != 0
    assert 'unlisted' in result.stderr


def test_export_snapshot_stays_coherent_during_reanalysis(monkeypatch):
    import importlib
    import threading
    from concurrent.futures import ThreadPoolExecutor
    module = importlib.import_module('etherglass.app')
    x = (np.arange(4096)/4096 + 1j*np.arange(4096)[::-1]/4096).astype('complex64')
    raw, meta = encode_sigmf(x, 1e6, 0)
    entered_export, release_export, writer_attempt, entered_analysis = [threading.Event() for _ in range(4)]
    original_export, original_analyze = module.export_zip, module.analyze

    def hold_snapshot(*args, **kwargs):
        entered_export.set()
        assert release_export.wait(5)
        return original_export(*args, **kwargs)

    def watch_analysis(*args, **kwargs):
        if kwargs.get('start_sample') == 1024:
            entered_analysis.set()
        return original_analyze(*args, **kwargs)

    with TestClient(app) as client:
        imported = client.post('/api/import', files={'data_file': ('owned', raw),
                            'meta_file': ('owned.json', json.dumps(meta).encode())}).json()
        ident = imported['capture_id']
        first = client.post('/api/analyze', json={'capture_id': ident,
                            'start_sample': 0, 'stop_sample': 1024, 'inspect_only': True}).json()
        monkeypatch.setattr(module, 'export_zip', hold_snapshot)
        monkeypatch.setattr(module, 'analyze', watch_analysis)
        with ThreadPoolExecutor(max_workers=2) as pool:
            download = pool.submit(client.get, f'/api/export/{ident}',
                                   params={'expected_analysis_id': first['analysis_id']})
            assert entered_export.wait(5)

            def replace():
                writer_attempt.set()
                return client.post('/api/analyze', json={'capture_id': ident,
                            'start_sample': 1024, 'stop_sample': 2048, 'inspect_only': True})

            writer = pool.submit(replace)
            try:
                assert writer_attempt.wait(5)
                assert not entered_analysis.wait(.1)
            finally:
                release_export.set()
            exported, replaced = download.result(timeout=5), writer.result(timeout=5)
        assert exported.status_code == replaced.status_code == 200
        assert entered_analysis.is_set() and replaced.json()['analysis_id'] != first['analysis_id']
        with zipfile_from(exported.content) as z:
            report = json.loads(z.read('analysis.json'))
            capture = load_bytes(z.read('capture.sigmf-data'), metadata=json.loads(z.read('capture.sigmf-meta')))
        assert report['analysis_id'] == first['analysis_id'] and report['selection'] == first['selection']
        np.testing.assert_array_equal(capture.samples, x[:1024])


def test_failed_imports_preserve_four_successful_capture_exports():
    import importlib
    module = importlib.import_module('etherglass.app')
    x = np.linspace(0, 1, 1024, dtype=np.float32).astype('complex64')
    raw, meta = encode_sigmf(x, 1e6, 0)
    bad = {**meta, 'captures': [{'core:sample_start': 0, 'core:frequency': 0},
                              {'core:sample_start': 32, 'core:frequency': 600}]}
    with TestClient(app) as client:
        saved = []
        for i in range(4):
            response = client.post('/api/import', files={'data_file': (f'owned-{i}', raw),
                                  'meta_file': ('owned.json', json.dumps(meta).encode())})
            assert response.status_code == 200
            result = response.json()
            response = client.get(f"/api/export/{result['capture_id']}", params={'expected_analysis_id': result['analysis_id']})
            with zipfile_from(response.content) as archive:
                saved.append((result, {n: archive.read(n) for n in archive.namelist()}))
        with module.lock:
            initial_keys = list(module.store)
        for _ in range(4):
            refused = client.post('/api/import', files={'data_file': ('owned-invalid', raw),
                                  'meta_file': ('owned.json', json.dumps(bad).encode())})
            assert refused.status_code == 422
        with module.lock:
            assert list(module.store) == initial_keys
        for result, members in saved:
            response = client.get(f"/api/export/{result['capture_id']}", params={'expected_analysis_id': result['analysis_id']})
            assert response.status_code == 200
            with zipfile_from(response.content) as archive:
                assert {n: archive.read(n) for n in archive.namelist()} == members


def test_concurrent_successful_imports_commit_complete_results_with_capacity_four():
    import importlib
    import threading
    from concurrent.futures import ThreadPoolExecutor
    module = importlib.import_module('etherglass.app')
    raw, meta = encode_sigmf(np.ones(1024, dtype='complex64'), 48000, 0)
    start = threading.Barrier(8)
    with TestClient(app) as client:
        def import_one(i):
            start.wait(timeout=5)
            response = client.post('/api/import', files={'data_file': (f'concurrent-owned-{i}', raw),
                                   'meta_file': ('owned.json', json.dumps(meta).encode())})
            assert response.status_code == 200, response.text
            return response.json()
        with ThreadPoolExecutor(max_workers=8) as pool:
            responses = list(pool.map(import_one, range(8)))
        by_id = {r['capture_id']: r for r in responses}
        assert len(by_id) == 8
        with module.lock:
            residents = {key: item['result']['analysis_id'] for key,item in module.store.items()}
        assert len(residents) == 4 and set(residents) <= set(by_id)
        for key, result_id in residents.items():
            assert result_id == by_id[key]['analysis_id']
            assert client.get(f'/api/export/{key}', params={'expected_analysis_id': result_id}).status_code == 200


def test_null_prior_inspection_crop_reset_and_reconstruction_refusal():
    raw, meta = encode_sigmf(np.ones(2048, dtype='complex64'), 48000, 0)
    with TestClient(app) as client:
        response = client.post('/api/import', files={'data_file': ('low-rate', raw),
                               'meta_file': ('owned.json', json.dumps(meta).encode())})
        assert response.status_code == 200
        initial = response.json();assert initial['selection']['symbol_rate'] is None
        body = {'capture_id': initial['capture_id'], 'start_sample': 48,
                'stop_sample': 960, 'target_center': 0, 'symbol_rate': None, 'inspect_only': True}
        cropped = client.post('/api/analyze', json=body)
        assert cropped.status_code == 200
        current = cropped.json();assert current['num_samples'] == 912 and current['selection']['symbol_rate'] is None
        refused = client.post('/api/analyze', json={**body, 'inspect_only': False, 'method': 'matched'})
        assert refused.status_code == 422
        assert client.get(f"/api/export/{initial['capture_id']}", params={'expected_analysis_id': current['analysis_id']}).status_code == 200
        reset = client.post('/api/analyze', json={**body, 'start_sample': 0, 'stop_sample': 2048})
        assert reset.status_code == 200 and reset.json()['selection']['symbol_rate'] is None
        supplied = client.post('/api/analyze', json={**body, 'symbol_rate': 6000, 'method': 'matched', 'inspect_only': False})
        assert supplied.status_code == 200 and supplied.json()['selection']['symbol_rate'] == 6000


@pytest.fixture
def generated_sdist_layout(tmp_path):
    scripts = tmp_path/'scripts'; scripts.mkdir()
    shutil.copy2(ROOT/'scripts/verify_artifacts.py', scripts/'verify_artifacts.py')
    package = tmp_path/'src/etherglass'; package.mkdir(parents=True)
    (package/'__init__.py').write_text('__version__ = "0.1.1.dev0"\n')
    (package/'synthetic.py').write_text('# owned source inventory fixture\n')
    (tmp_path/'README.md').write_text('Owned distribution fixture.\n')
    (tmp_path/'pyproject.toml').write_text('[project]\nname = "etherglass"\nversion = "0.1.1.dev0"\n')
    records = {}
    for path in tmp_path.rglob('*'):
        if path.is_file():
            records[path.relative_to(tmp_path).as_posix()] = {'sha256': hashlib.sha256(path.read_bytes()).hexdigest(), 'bytes': path.stat().st_size}
    (tmp_path/'ARTIFACTS.json').write_text(json.dumps({'version':'0.1.1.dev0','algorithm':'SHA256','files':records}))
    metadata = b'Metadata-Version: 2.4\nName: etherglass\nVersion: 0.1.1.dev0\n\n'+(tmp_path/'README.md').read_bytes()
    (tmp_path/'PKG-INFO').write_bytes(metadata)
    (tmp_path/'setup.cfg').write_bytes(b'[egg_info]\ntag_build = \ntag_date = 0\n\n')
    egg = tmp_path/'src/etherglass.egg-info'; egg.mkdir()
    (egg/'PKG-INFO').write_bytes(metadata)
    members = [*records, 'ARTIFACTS.json', 'src/etherglass.egg-info/PKG-INFO', 'src/etherglass.egg-info/SOURCES.txt']
    (egg/'SOURCES.txt').write_text('\n'.join(members)+'\n')
    return tmp_path


def check_artifacts(root):
    return subprocess.run([sys.executable, '-B', str(root/'scripts/verify_artifacts.py')], capture_output=True, text=True)


def test_consistent_generated_sdist_metadata_is_not_an_unlisted_source(generated_sdist_layout):
    result = check_artifacts(generated_sdist_layout)
    assert result.returncode == 0, result.stderr


@pytest.mark.parametrize('mutation', ['real_configuration', 'unlisted_file', 'wrong_version', 'wrong_name', 'ordinary_checkout', 'missing_source_entry'])
def test_sdist_recognition_retains_strict_source_coverage(generated_sdist_layout, mutation):
    root = generated_sdist_layout
    if mutation == 'real_configuration':
        with (root/'setup.cfg').open('a') as output:
            output.write('[build_ext]\ninplace = 1\n')
    elif mutation == 'unlisted_file':
        (root/'unlisted.txt').write_bytes(b'new unsealed source')
    elif mutation == 'missing_source_entry':
        inventory = root/'src/etherglass.egg-info/SOURCES.txt'
        inventory.write_text(inventory.read_text().replace('src/etherglass/synthetic.py\n', ''))
    elif mutation in ('wrong_version', 'wrong_name'):
        before, after = (b'Version: 0.1.1.dev0', b'Version: 0.1.2.dev0') if mutation == 'wrong_version' else (b'Name: etherglass', b'Name: other-package')
        for path in (root/'PKG-INFO', root/'src/etherglass.egg-info/PKG-INFO'):
            path.write_bytes(path.read_bytes().replace(before, after))
    else:
        shutil.rmtree(root/'src/etherglass.egg-info')
    result = check_artifacts(root)
    assert result.returncode != 0
    assert 'unlisted:' in result.stderr
