"""Measured visual summaries and conservative occupancy estimates."""
from __future__ import annotations

import numpy as np
from scipy import signal
from scipy.ndimage import median_filter, binary_closing, label


def power_db(x):
    return 10 * np.log10(np.maximum(np.asarray(x), 1e-20))


def spectrum(x: np.ndarray, fs: float, bins: int = 512) -> dict:
    x = np.asarray(x, dtype=np.complex128)
    n = min(len(x), bins)
    f, p = signal.welch(x, fs=fs, nperseg=n, noverlap=n // 2, return_onesided=False,
                        detrend=False, scaling="density")
    return {"frequency_hz": np.fft.fftshift(f).tolist(), "power_db": power_db(np.fft.fftshift(p)).tolist(),
            "units": "dB (input unit²/Hz)", "fft_size": n}


def waterfall(x: np.ndarray, fs: float, bins: int = 256, max_frames: int = 128) -> dict:
    x = np.asarray(x, dtype=np.complex128)
    n = min(bins, len(x))
    hop = max(n // 2, int(np.ceil(max(0, len(x) - n) / max(1, max_frames - 1))))
    starts = np.arange(0, len(x) - n + 1, hop, dtype=int)
    window = signal.windows.hann(n, sym=False)
    frames = np.stack([x[t:t + n] * window for t in starts])
    p = np.abs(np.fft.fftshift(np.fft.fft(frames, axis=-1), axes=-1)) ** 2 / (fs * np.sum(window ** 2))
    return {"time_s": ((starts + n / 2) / fs).tolist(),
            "frequency_hz": np.fft.fftshift(np.fft.fftfreq(n, 1 / fs)).tolist(),
            "power_db": power_db(p).tolist(), "hop_samples": hop, "fft_size": n,
            "units": "dB (input unit²/Hz)"}


def occupied_regions(spec: dict) -> list[dict]:
    """Heuristic energy detector, not an emitter count or calibrated CFAR."""
    p = np.asarray(spec["power_db"])
    f = np.asarray(spec["frequency_hz"])
    smooth = median_filter(p, size=5, mode="wrap")
    floor = float(np.percentile(smooth, 20))
    threshold = floor + 9.0
    mask = binary_closing(smooth > threshold, structure=np.ones(3))
    labels, count = label(mask)
    result = []
    step = float(f[1] - f[0])
    for i in range(1, count + 1):
        idx = np.flatnonzero(labels == i)
        if len(idx) < 2:
            continue
        result.append({"frequency_low_hz": float(f[idx[0]] - step / 2),
                       "frequency_high_hz": float(f[idx[-1]] + step / 2),
                       "power_db": float(np.max(p[idx])), "threshold_db": threshold,
                       "method": "median PSD > 20th-percentile floor + 9 dB"})
    return result


def waveform(x: np.ndarray, fs: float, points: int = 1024) -> dict:
    # Show contiguous time samples: downsampling without anti-alias filtering
    # would invent low-frequency features. Full-capture navigation uses selection.
    n = min(len(x), points)
    return {"time_s": (np.arange(n) / fs).tolist(), "i": x[:n].real.tolist(), "q": x[:n].imag.tolist(),
            "displayed_samples": n, "total_samples": len(x)}


def constellation(x: np.ndarray, fs: float, symbol_rate: float | None = None, points: int = 1200) -> dict:
    # This is raw I/Q density; no unsignalled clock/phase recovery or symbol-EVM claims.
    n = min(len(x), points)
    idx = np.linspace(0, len(x) - 1, n, dtype=int)
    return {"i": x[idx].real.tolist(), "q": x[idx].imag.tolist(),
            "kind": "raw sample density (no symbol synchronization)"}


def visual_summary(x: np.ndarray, fs: float, symbol_rate: float | None = None) -> dict:
    return {"spectrum": spectrum(x, fs), "waveform": waveform(x, fs),
            "constellation": constellation(x, fs, symbol_rate)}


def diagnostics(x: np.ndarray) -> dict:
    x = np.asarray(x, dtype=np.complex128)
    power = float(np.mean(np.abs(x) ** 2))
    peak = float(np.max(np.abs(x)))
    return {"rms": float(np.sqrt(power)), "peak_magnitude": peak,
            "papr_db": float(10 * np.log10(max(peak * peak, 1e-20) / max(power, 1e-20))),
            "dc_i": float(x.real.mean()), "dc_q": float(x.imag.mean()),
            "iq_power_imbalance_db": float(10 * np.log10(max(float(np.mean(x.real ** 2)), 1e-20) /
                                                       max(float(np.mean(x.imag ** 2)), 1e-20)))}
