"""Conservative, trained moment-feature PSK recognition with a separate support gate.

A confidence is conditional on the supported synthetic BPSK/QPSK domain; the
support gate is measured against finite probes, not a guarantee for arbitrary RF.
No payload demodulation is performed.
"""
from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path
from typing import Any

import numpy as np
from scipy.signal import fftconvolve, welch

FEATURE_NAMES = [
    "symbol_c20", "symbol_c40", "symbol_c42", "symbol_radial_cv",
    "symbol_envelope_fourth", "sample_c20", "sample_c40",
    "bandwidth_to_symbol_rate", "spectral_flatness", "peak_bin_fraction",
    "lag_symbol_correlation", "symbol_lag_correlation",
    "raw_envelope_cv", "cfo_to_symbol_rate",
]


def _rrc(sps: float, beta: float = 0.35, span: int = 10) -> np.ndarray:
    """Unit-energy root-raised-cosine receive filter, sampled in symbols."""
    count = int(np.ceil(span * sps))
    count += count % 2
    t = np.arange(-count // 2, count // 2 + 1) / sps
    h = np.empty_like(t)
    for k, u in enumerate(t):
        if abs(u) < 1e-10:
            h[k] = 1 + beta * (4 / np.pi - 1)
        elif abs(abs(u) - 1 / (4 * beta)) < 1e-8:
            h[k] = beta / np.sqrt(2) * (
                (1 + 2 / np.pi) * np.sin(np.pi / (4 * beta))
                + (1 - 2 / np.pi) * np.cos(np.pi / (4 * beta)))
        else:
            h[k] = (np.sin(np.pi * u * (1-beta))
                    + 4 * beta * u * np.cos(np.pi * u * (1+beta))) / (
                        np.pi * u * (1-(4*beta*u)**2))
    return h / np.linalg.norm(h)


def _moments(z: np.ndarray) -> tuple[float, float, float, float, float]:
    z = z - np.mean(z)
    z = z / np.sqrt(np.mean(abs(z)**2) + 1e-15)
    m20 = np.mean(z*z)
    m40 = np.mean(z**4)
    m42 = np.mean(abs(z)**4)
    return (float(abs(m20)), float(abs(m40 - 3*m20*m20)),
            float(m42 - abs(m20)**2 - 2),
            float(np.std(abs(z)) / (np.mean(abs(z)) + 1e-15)), float(m42))


def extract_features(x: np.ndarray, sample_rate: float, symbol_rate: float) -> np.ndarray:
    """Extract source-independent physical features with an explicit rate prior.

    A fourth-power carrier estimate removes small residual offsets. The receive
    filter is beta=.35 RRC, and blind timing chooses the lowest envelope scatter
    among integer sample phases. These assumptions are part of the model domain.
    """
    z = np.asarray(x, dtype=np.complex128).reshape(-1)
    if len(z) < 512 or not np.all(np.isfinite(z)):
        raise ValueError("At least 512 finite complex samples are required")
    if not (np.isfinite(sample_rate) and np.isfinite(symbol_rate)
            and sample_rate > 0 and symbol_rate > 0):
        raise ValueError("Sample rate and symbol rate must be finite and positive")
    sps = sample_rate / symbol_rate
    if not any(abs(sps-v) <= .02 for v in (6, 8, 10, 12)) or len(z) / sps < 64:
        raise ValueError("Calibrated domain: 6, 8, 10 or 12 samples per symbol, and at least 64 symbols")
    scale = max(float(np.max(abs(z.real))), float(np.max(abs(z.imag))))
    if scale == 0:
        raise ValueError("Capture has no measurable signal energy.")
    z = z / scale
    z = z / np.sqrt(np.mean(abs(z)**2) + 1e-15)
    n = len(z)
    p4 = np.fft.fft(z**4 * np.hanning(n), n=2*n)
    f4 = np.fft.fftfreq(2*n)
    # Restrict estimate to +/- 0.25 symbol rates; wider offsets must be downmixed.
    keep = abs(f4) <= 1.0 / sps
    peak = np.flatnonzero(keep)[np.argmax(abs(p4[keep]))]
    cfo = float(f4[peak] / 4)
    q = z * np.exp(-2j*np.pi*cfo*np.arange(n))
    filtered = fftconvolve(q, _rrc(sps), mode="same")
    margin = int(6 * sps)
    timing_candidates = []
    # Fractional sample rates are interpolated onto symbol-spaced coordinates.
    for offset in range(max(1, int(np.ceil(sps)))):
        locations = np.arange(margin + offset, n-margin, sps)
        symbols = np.interp(locations, np.arange(n), filtered.real) + 1j*np.interp(
            locations, np.arange(n), filtered.imag)
        m = _moments(symbols)
        timing_candidates.append((m[3], symbols, m))
    _, symbols, symbol_m = min(timing_candidates, key=lambda t: t[0])
    sample_m = _moments(q)
    f, psd = welch(z, fs=1.0, nperseg=min(n, 512), return_onesided=False)
    order = np.argsort(f)
    f, psd = f[order], psd[order]
    psd = np.maximum(psd, 1e-20)
    weights = psd / psd.sum()
    cdf = np.cumsum(weights)
    low = f[min(np.searchsorted(cdf, .025), len(f)-1)]
    high = f[min(np.searchsorted(cdf, .975), len(f)-1)]
    lag = int(round(sps))
    sample_correlation = abs(np.vdot(q[:-lag], q[lag:])) / (
        np.linalg.norm(q[:-lag]) * np.linalg.norm(q[lag:]) + 1e-15)
    symbol_correlation = abs(np.vdot(symbols[:-1], symbols[1:])) / (
        np.linalg.norm(symbols[:-1]) * np.linalg.norm(symbols[1:]) + 1e-15)
    return np.asarray([
        *symbol_m, sample_m[0], sample_m[1], (high-low)*sps,
        np.exp(np.mean(np.log(psd))) / np.mean(psd), weights.max(),
        sample_correlation, symbol_correlation,
        np.std(abs(q)) / (np.mean(abs(q)) + 1e-15), abs(cfo)*sps,
    ], dtype=np.float64)


def _sigmoid(a: float | np.ndarray) -> Any:
    return 1 / (1 + np.exp(-np.clip(a, -40, 40)))


@lru_cache(maxsize=4)
def _load_model(path: str) -> dict:
    with open(path, encoding="utf-8") as handle:
        model = json.load(handle)
    if model.get("feature_names") != FEATURE_NAMES:
        raise ValueError("Classifier feature schema does not match this version")
    return model


def model_path() -> Path:
    # Source checkout and wheel both carry the same immutable JSON artifact.
    packaged = Path(__file__).parent / "resources" / "models" / "classifier.json"
    if packaged.exists():
        return packaged
    return Path(__file__).resolve().parents[2] / "models" / "classifier.json"


def classify(x: np.ndarray, sample_rate: float, symbol_rate: float | None = None,
             model: str | Path | None = None) -> dict:
    """Classify a centered, isolated candidate or return an explicit abstention.

    `confidence` is the conditional class probability when accepted, zero on
    abstention. `conditional_confidence` retains the BPSK/QPSK probability. `support_score` is separately calibrated known-vs-probe probability.
    Neither is a posterior probability that arbitrary real-world RF is known.
    """
    unknown = {"label": "unknown", "confidence": 0.0, "status": "unknown"}
    if symbol_rate is None:
        return {**unknown, "reason": "Supply a symbol-rate prior for supported BPSK/QPSK recognition."}
    try:
        candidate = np.asarray(x).reshape(-1)
        if len(candidate) < 2048:
            return {**unknown, "reason": "At least 2,048 samples are required for the calibrated classifier window."}
        f = extract_features(candidate[:2048], float(sample_rate), float(symbol_rate))
    except (ValueError, TypeError, OverflowError) as exc:
        return {**unknown, "reason": str(exc)}
    try:
        artifact = _load_model(str(model or model_path()))
    except (OSError, ValueError, json.JSONDecodeError):
        return {**unknown, "reason": "Trained classifier artifact is unavailable; no label was forced."}
    standardized = (f - np.asarray(artifact["mean"])) / np.asarray(artifact["scale"])
    class_logit = float(np.dot(artifact["class_coef"], standardized) + artifact["class_intercept"])
    class_probability = float(_sigmoid(class_logit / artifact["class_temperature"]))
    support_logit = float(np.dot(artifact["support_coef"], standardized) + artifact["support_intercept"])
    support_probability = float(_sigmoid(artifact["support_calibration_slope"] * support_logit
                                         + artifact["support_calibration_intercept"]))
    confidence = max(class_probability, 1-class_probability)
    supported = support_probability >= artifact["support_threshold"] and confidence >= artifact["confidence_threshold"]
    result = {
        "label": ("QPSK" if class_probability >= .5 else "BPSK") if supported else "unknown",
        "confidence": confidence if supported else 0.0,
        "conditional_confidence": confidence,
        "status": "supported" if supported else "unknown",
        "reason": "Within calibrated synthetic PSK support; verify the rate and isolated-signal assumptions."
                  if supported else "Outside calibrated PSK support or insufficient class certainty; abstaining.",
        "support_score": support_probability,
        "support_threshold": artifact["support_threshold"],
        "conditional_probabilities": {"BPSK": 1-class_probability, "QPSK": class_probability},
        "scope": "First 2,048 samples of centered RRC BPSK/QPSK, beta≈0.35, 6/8/10/12 samples/symbol. Finite synthetic unknown probes only.",
    }
    return result
