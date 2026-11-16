"""Three-training-seed disagreement and split-calibrated waveform error bounds.

Calibration is conditional on the stated synthetic shift distribution. It is
not a guarantee for arbitrary imported radio recordings or unknown modulations.
"""
from __future__ import annotations
import json
import hashlib
from pathlib import Path
import numpy as np


def model_directory() -> Path:
    from .separation import DEFAULT_CHECKPOINT
    return DEFAULT_CHECKPOINT.parent


def source_fingerprints() -> dict:
    folder=Path(__file__).resolve().parent
    return {f'etherglass/{name}':hashlib.sha256((folder/name).read_bytes()).hexdigest()
            for name in ['models.py','spectral.py','separation.py','uncertainty.py','synthetic.py']}


def ensemble_indicator(x, sample_rate, target_center, symbol_rate, estimate=None, *, expected_digests=None):
    from .separation import separate
    folder = model_directory()
    paths = [folder / f"seed{s}" / "hybrid.pt" for s in [0, 1, 2]]
    if not all(p.exists() for p in paths):
        return {"status": "unavailable", "reason": "Three trained seed checkpoints are required for disagreement estimation."}
    digests = {str(s): hashlib.sha256(path.read_bytes()).hexdigest() for s,path in enumerate(paths)}
    if expected_digests is not None and digests != expected_digests:
        raise ValueError("Checkpoint consumed for inference differs from the calibration record.")
    if estimate is not None:
        canonical = folder / 'hybrid.pt'
        if not canonical.is_file() or hashlib.sha256(canonical.read_bytes()).hexdigest() != digests['0']:
            return {"status": "unavailable", "reason": "Canonical target model differs from ensemble seed0. Rebuild the ensemble and recalibrate its error bound."}
    outputs = []
    for i, path in enumerate(paths):
        if i == 0 and estimate is not None:
            outputs.append(np.asarray(estimate))
        else:
            outputs.append(separate(x, sample_rate, target_center, symbol_rate,
                                    method="hybrid", checkpoint=path,
                                    expected_sha256=expected_digests[str(i)] if expected_digests is not None else None)["components"][0])
    outputs = np.asarray(outputs, dtype=np.complex128)
    mean = outputs.mean(axis=0)
    disagreement = float(np.mean(np.abs(outputs - mean) ** 2))
    power = float(np.mean(np.abs(np.asarray(x, dtype=np.complex128)) ** 2))
    scale = disagreement + 0.01 * power
    if not np.isfinite(scale):
        raise ValueError("Input exceeds the supported numerical range for uncertainty estimation.")
    result = {"status": "indicator", "training_seeds": [0, 1, 2],
              "disagreement_rms": float(np.sqrt(disagreement)),
              "relative_disagreement_rms": float(np.sqrt(disagreement / max(power, 1e-20))),
              "error_scale_power": scale,
              "reason": "Training-seed disagreement can be small when every model is wrong."}
    # Arbitrary captures and selections never inherit synthetic coverage.
    return result
