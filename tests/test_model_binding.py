"""Checkpoint identity must describe the weights consumed by every frame."""
import hashlib
import os
from pathlib import Path
import shutil

import numpy as np
import pytest

from etherglass import separation as sep
from etherglass.synthetic import generate_example


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


@pytest.fixture
def checkpoints(tmp_path):
    pytest.importorskip('torch')
    folder = sep.DEFAULT_CHECKPOINT.parent
    canonical = tmp_path / 'hybrid.pt'
    seed0, seed1 = tmp_path/'seed0.pt', tmp_path/'seed1.pt'
    shutil.copy2(folder/'seed0/hybrid.pt', seed0)
    shutil.copy2(folder/'seed1/hybrid.pt', seed1)
    # Git checkouts do not preserve equal mtimes; construct the collision here.
    stamp = 1_700_000_000_000_000_000
    for checkpoint in (seed0, seed1):
        os.utime(checkpoint, ns=(stamp, stamp))
    shutil.copy2(seed0, canonical)
    sep._load_model_cached.cache_clear()
    yield canonical, seed0, seed1
    sep._load_model_cached.cache_clear()


def test_copy2_warm_restore_uses_restored_content(checkpoints):
    # Regression: equal-size, equal-time restoration retained different weights.
    canonical, seed0, seed1 = checkpoints
    assert seed0.stat().st_size == seed1.stat().st_size
    assert seed0.stat().st_mtime_ns == seed1.stat().st_mtime_ns
    shutil.copy2(seed1, canonical)
    _, warm = sep.load_model(str(canonical))
    assert warm['checkpoint_sha256'] == sha(seed1)
    shutil.copy2(seed0, canonical)
    _, restored = sep.load_model(str(canonical))
    assert restored['checkpoint_sha256'] == sha(seed0)


def test_one_model_for_frames_and_metadata(checkpoints, monkeypatch):
    canonical, seed0, seed1 = checkpoints
    x = generate_example(4411, n=7600, split='test')['mixture']
    control = sep.separate(x, 1e6, 0, 125000, checkpoint=canonical)
    sep._load_model_cached.cache_clear()
    original = sep.frame_estimate
    frames = []

    def replace_between_frames(*args, **kwargs):
        result = original(*args, **kwargs)
        frames.append(1)
        # The next frame and final metadata must not reopen this replaced path.
        shutil.copy2(seed1, canonical)
        stamp = seed1.stat().st_mtime_ns + len(frames)
        os.utime(canonical, ns=(stamp, stamp))
        return result

    monkeypatch.setattr(sep, 'frame_estimate', replace_between_frames)
    actual = sep.separate(x, 1e6, 0, 125000, checkpoint=canonical)
    assert len(frames) > 1
    assert actual['model']['checkpoint_sha256'] == sha(seed0)
    np.testing.assert_array_equal(actual['components'][0], control['components'][0])


def test_expected_digest_rejected_before_inference(checkpoints, monkeypatch):
    canonical, seed0, seed1 = checkpoints
    shutil.copy2(seed1, canonical)
    monkeypatch.setattr(sep, 'frame_estimate', lambda *a, **k: pytest.fail('inference started'))
    with pytest.raises(ValueError, match='consumed for inference'):
        sep.separate(np.ones(128, dtype=np.complex64), 1e6, 0, 125000,
                     checkpoint=canonical, expected_sha256=sha(seed0))


def test_deserializes_exact_captured_bytes(checkpoints, monkeypatch):
    canonical, seed0, seed1 = checkpoints
    expected = sha(seed0)
    original = Path.read_bytes

    def replace_after_read(path):
        payload = original(path)
        if path == canonical:
            shutil.copy2(seed1, canonical)
        return payload

    monkeypatch.setattr(Path, 'read_bytes', replace_after_read)
    _, info = sep.load_model(str(canonical), expected_sha256=expected)
    assert info['checkpoint_sha256'] == expected
    assert sha(canonical) == sha(seed1)


@pytest.fixture
def bound_folder(checkpoints, monkeypatch, tmp_path):
    canonical, seed0, seed1 = checkpoints
    folder = tmp_path/'bound-models'
    shutil.copytree(sep.DEFAULT_CHECKPOINT.parent, folder)
    monkeypatch.setattr(sep, 'DEFAULT_CHECKPOINT', folder/'hybrid.pt')
    return folder


def test_bound_copy2_warm_restore_matches_clean_control(bound_folder, supported_bound):
    from etherglass import calibrated_example as ce
    from etherglass.uncertainty import ensemble_indicator
    from etherglass.synthetic_bound_guard import verify_binding
    canonical = bound_folder/'hybrid.pt'
    seed0, seed1 = bound_folder/'seed0/hybrid.pt', bound_folder/'seed1/hybrid.pt'
    shutil.copy2(seed1, canonical)
    sep.load_model(str(canonical))
    shutil.copy2(seed0, canonical)
    record = verify_binding()
    result = ce.generated_example()
    assert result['status'] == 'synthetic-conditional-bound'
    example = generate_example(result['seed'], n=4096, split='test')
    assert hashlib.sha256(example['mixture'].tobytes()).hexdigest() == result['mixture_sha256']
    sep._load_model_cached.cache_clear()
    clean = sep.separate(example['mixture'], 1e6, 0, 125000)['components'][0]
    indicator = ensemble_indicator(example['mixture'], 1e6, 0, 125000, estimate=clean)
    assert result['waveform_error_rms_upper'] == float(np.sqrt(record['quantile']*indicator['error_scale_power']))
    assert result['observed_synthetic_error_rms'] == float(np.sqrt(np.mean(np.abs(clean-example['target'])**2)))


def test_bound_guard_to_load_replacement_refused(bound_folder, supported_bound, monkeypatch):
    from etherglass import calibrated_example as ce
    original = ce.separate

    def replace_before_load(*args, **kwargs):
        shutil.copy2(bound_folder/'seed1/hybrid.pt', bound_folder/'hybrid.pt')
        return original(*args, **kwargs)

    monkeypatch.setattr(ce, 'separate', replace_before_load)
    with pytest.raises(ValueError, match='consumed for inference'):
        ce.generated_example()


def test_bound_ensemble_replacement_refused(bound_folder, supported_bound, monkeypatch):
    from etherglass import calibrated_example as ce
    original = ce.ensemble_indicator

    def replace_before_ensemble(*args, **kwargs):
        shutil.copy2(bound_folder/'seed2/hybrid.pt', bound_folder/'seed1/hybrid.pt')
        return original(*args, **kwargs)

    monkeypatch.setattr(ce, 'ensemble_indicator', replace_before_ensemble)
    with pytest.raises(ValueError, match='consumed for inference'):
        ce.generated_example()


def test_ensemble_hash_to_load_replacement_refused(bound_folder, monkeypatch):
    from etherglass.uncertainty import ensemble_indicator
    expected = {str(i): sha(bound_folder/f'seed{i}/hybrid.pt') for i in range(3)}
    original = sep.separate

    def replace_before_load(*args, **kwargs):
        path = kwargs.get('checkpoint')
        if path == bound_folder/'seed1/hybrid.pt':
            shutil.copy2(bound_folder/'seed2/hybrid.pt', path)
        return original(*args, **kwargs)

    monkeypatch.setattr(sep, 'separate', replace_before_load)
    with pytest.raises(ValueError, match='consumed for inference'):
        ensemble_indicator(np.ones(128, dtype=np.complex64), 1e6, 0, 125000,
                           expected_digests=expected)


@pytest.fixture
def supported_bound():
    import scipy
    torch = pytest.importorskip('torch')
    if (np.__version__, scipy.__version__, torch.__version__) != ('2.5.4', '1.18.1', '2.6.0'):
        pytest.skip('Positive bound controls require the admitted numerical runtime; refusal is tested separately.')


def test_successor_binding_drift_refused(tmp_path, monkeypatch):
    from etherglass import synthetic_bound_guard as guard
    altered = tmp_path/'binding.json'
    altered.write_bytes(guard._binding_path().read_bytes()+b' ')
    monkeypatch.setattr(guard, '_binding_path', lambda: altered)
    with pytest.raises(ValueError, match='implementation binding identity'):
        guard.verify_binding()


def test_successor_equivalence_receipt_drift_refused(monkeypatch):
    from etherglass import synthetic_bound_guard as guard
    original = guard._sha
    monkeypatch.setattr(guard, '_sha', lambda p: '0'*64 if p.name == 'synthetic-bound-equivalence-v3.json' else original(p))
    with pytest.raises(ValueError, match='implementation evidence identity'):
        guard.verify_binding()


def test_public_models_and_metadata_are_caller_owned(checkpoints, tmp_path):
    canonical, seed0, seed1 = checkpoints
    user = tmp_path/'user-copy.pt'
    shutil.copyfile(canonical, user)
    first, first_info = sep.load_model(str(user))
    second, second_info = sep.load_model(str(canonical))
    repeated, repeated_info = sep.load_model(str(user))
    assert first is not second and first is not repeated
    assert not first.training and not second.training and not repeated.training
    assert first_info['stages'] is not second_info['stages']
    assert first_info['stages'] is not repeated_info['stages']
    for name, parameter in first.named_parameters():
        assert parameter.data_ptr() != dict(second.named_parameters())[name].data_ptr()
        assert parameter.data_ptr() != dict(repeated.named_parameters())[name].data_ptr()
    for name, tensor in first_info['state_dict'].items():
        assert tensor.data_ptr() != second_info['state_dict'][name].data_ptr()
        assert tensor.data_ptr() != repeated_info['state_dict'][name].data_ptr()


@pytest.mark.parametrize('mutation', ['load_state_dict', 'parameter', 'state_dict', 'nested_metadata'])
def test_consumer_mutation_does_not_change_canonical_inference(checkpoints, tmp_path, mutation):
    import torch
    canonical, seed0, seed1 = checkpoints
    user = tmp_path/'user-experiment.pt'
    shutil.copyfile(canonical, user)
    x = generate_example(4411, n=4096, split='test')['mixture']
    control = sep.separate(x, 1e6, 0, 125000, checkpoint=canonical)
    model, info = sep.load_model(str(user))
    expected_state = {k: v.clone() for k,v in info['state_dict'].items()}
    expected_stages = list(info['stages'])
    if mutation == 'load_state_dict':
        alternative, _ = sep.load_model(str(seed1))
        model.load_state_dict(alternative.state_dict())
    elif mutation == 'parameter':
        with torch.no_grad():
            next(model.parameters()).add_(1)
    elif mutation == 'state_dict':
        next(iter(info['state_dict'].values())).add_(1)
    else:
        info['stages'].append('consumer-owned experiment')
    assert sha(canonical) == sha(seed0) == sha(user)
    fresh, fresh_info = sep.load_model(str(canonical))
    assert fresh_info['stages'] == expected_stages
    for name, tensor in fresh_info['state_dict'].items():
        assert torch.equal(tensor, expected_state[name])
    actual = sep.separate(x, 1e6, 0, 125000, checkpoint=canonical)
    assert actual['model'] == control['model']
    np.testing.assert_array_equal(actual['components'][0], control['components'][0])


@pytest.mark.parametrize('user_source', ['copy', 'canonical'])
def test_consumer_model_mutation_cannot_change_generated_bound(bound_folder, supported_bound, tmp_path, user_source):
    from etherglass import calibrated_example as ce
    from etherglass.uncertainty import ensemble_indicator
    from etherglass.synthetic_bound_guard import verify_binding
    canonical = bound_folder/'hybrid.pt'
    before = {str(p): sha(p) for p in [canonical, *(bound_folder/f'seed{i}/hybrid.pt' for i in range(3))]}
    user = tmp_path/'user-independent.pt' if user_source == 'copy' else canonical
    if user_source == 'copy':
        shutil.copyfile(canonical, user)
    model, info = sep.load_model(str(user))
    alternative, _ = sep.load_model(str(bound_folder/'seed1/hybrid.pt'))
    model.load_state_dict(alternative.state_dict())
    record = verify_binding()
    issued = ce.generated_example()
    example = generate_example(issued['seed'], n=4096, split='test')
    assert hashlib.sha256(example['mixture'].tobytes()).hexdigest() == issued['mixture_sha256']
    sep._load_model_cached.cache_clear()
    clean = sep.separate(example['mixture'], 1e6, 0, 125000)['components'][0]
    indicator = ensemble_indicator(example['mixture'], 1e6, 0, 125000, estimate=clean)
    assert issued['waveform_error_rms_upper'] == float(np.sqrt(record['quantile']*indicator['error_scale_power']))
    assert issued['observed_synthetic_error_rms'] == float(np.sqrt(np.mean(np.abs(clean-example['target'])**2)))
    assert all(sha(Path(p)) == h for p,h in before.items())


@pytest.mark.parametrize('name', ['synthetic-bound-implementation-v2.json', 'synthetic-bound-equivalence-v2.json'])
def test_previous_implementation_records_remain_bound(monkeypatch, name):
    from etherglass import synthetic_bound_guard as guard
    original = guard._sha
    monkeypatch.setattr(guard, '_sha', lambda p: '0'*64 if p.name == name else original(p))
    with pytest.raises(ValueError, match='implementation predecessor identity'):
        guard.verify_binding()
