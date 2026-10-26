"""Reproducible, labelled complex-baseband PSK mixtures (generated data: CC0).

All frequencies in this module are cycles/sample. This is a signal-level
laboratory simulation, not a protocol encoder or a message decoder.
"""
from __future__ import annotations

import numpy as np
from scipy.signal import fftconvolve


def rrc_taps(sps: float = 8, rolloff: float = .35, span: int = 12) -> np.ndarray:
    """Unit-energy root-raised-cosine pulse with removable limits evaluated."""
    if not np.isfinite(sps) or sps < 2 or not 0 < rolloff <= 1:
        raise ValueError("sps must be >=2 and rolloff must be in (0,1]")
    count = int(np.ceil(span * sps)) | 1
    t = (np.arange(count) - count // 2) / sps
    b = rolloff
    h = np.empty_like(t)
    zero = np.isclose(t, 0)
    edge = np.isclose(np.abs(t), 1 / (4 * b))
    normal = ~(zero | edge)
    h[zero] = 1 + b * (4 / np.pi - 1)
    h[edge] = b / np.sqrt(2) * ((1 + 2 / np.pi) * np.sin(np.pi / (4 * b))
                                 + (1 - 2 / np.pi) * np.cos(np.pi / (4 * b)))
    u = t[normal]
    h[normal] = (np.sin(np.pi*u*(1-b)) + 4*b*u*np.cos(np.pi*u*(1+b))) / (np.pi*u*(1-(4*b*u)**2))
    return h / np.linalg.norm(h)


def generate_psk(rng: np.random.Generator, n: int, sps: int = 8,
                 family: str = "QPSK", rolloff: float = .35,
                 timing: float = 0., cfo: float = 0.,
                 multipath: bool = False) -> tuple[np.ndarray, np.ndarray, dict]:
    """Generate pulse-shaped PSK; return received IQ, transmitted symbols, provenance.

    ``timing`` is fractional samples, ``cfo`` is cycles/sample. Guard symbols
    precede/follow the returned crop, avoiding artificial zero edge bursts.
    """
    family = family.upper()
    if family not in {"BPSK", "QPSK"} or n < 64 or int(sps) != sps:
        raise ValueError("supported synthesis: BPSK/QPSK, n>=64, integer sps")
    h = rrc_taps(sps, rolloff)
    guard = len(h) + 2*sps
    count = int(np.ceil((n+2*guard)/sps)) + 4
    order = 2 if family == "BPSK" else 4
    labels = rng.integers(0, order, count)
    symbols = np.exp(2j*np.pi*labels/order + (1j*np.pi/4 if order == 4 else 0))
    impulses = np.zeros(count*sps, dtype=np.complex128)
    impulses[::sps] = symbols
    shaped = fftconvolve(impulses, h, mode="same")
    t = np.arange(n) + guard + timing
    z = np.interp(t, np.arange(len(shaped)), shaped.real) + 1j*np.interp(t, np.arange(len(shaped)), shaped.imag)
    taps = np.array([1.+0j])
    if multipath:
        taps = np.zeros(6, dtype=complex)
        taps[0] = 1
        taps[2] = rng.uniform(.1,.35)*np.exp(1j*rng.uniform(-np.pi,np.pi))
        taps[5] = rng.uniform(.03,.15)*np.exp(1j*rng.uniform(-np.pi,np.pi))
        # Retain pre-crop guard samples so channel edge transients are physical.
        full = np.interp(np.arange(n+guard)+timing, np.arange(len(shaped)), shaped.real) + 1j*np.interp(np.arange(n+guard)+timing, np.arange(len(shaped)), shaped.imag)
        z = fftconvolve(full, taps, mode="full")[guard:guard+n]
    phase = float(rng.uniform(-np.pi,np.pi))
    z *= np.exp(1j*(2*np.pi*cfo*np.arange(n)+phase))
    z /= np.sqrt(np.mean(np.abs(z)**2)) + 1e-12
    return z.astype(np.complex64), symbols.astype(np.complex64), {
        "family":family,"samples_per_symbol":sps,"rolloff":rolloff,
        "timing_samples":float(timing),"carrier_offset_cycles_per_sample":float(cfo),
        "phase_rad":phase,"channel_taps":[[float(v.real),float(v.imag)] for v in taps],
        "symbol_crop_offset_samples":guard,"symbol_labels":labels.tolist(),
    }


def generate_example(seed: int, n: int = 2048, split: str = "train",
                     snr_db: float | None = None, sir_db: float | None = None,
                     family: str | None = None, impairment: str | None = None) -> dict:
    """Generate clean components and their exact mixture.

    Train: stationary single/dual tones, mild timing/CFO. Adapt/test shift:
    chirps, multipath, larger timing/CFO. Split names select distributions;
    callers own disjoint seed ranges (the experiment runner records them).
    SNR is target/noise power; SIR is target/interference power, at the crop.
    """
    if split not in {"train","adapt","test","nominal"}:
        raise ValueError("unknown synthetic split")
    if n < 128:
        raise ValueError("n must be >=128")
    rng = np.random.default_rng(seed)
    shifted = split in {"adapt","test"}
    snr_db = float(rng.uniform(10,30) if snr_db is None else snr_db)
    sir_db = float(rng.uniform(-12,8) if sir_db is None else sir_db)
    family = family or str(rng.choice(["BPSK","QPSK"]))
    timing = float(rng.uniform(0,1 if shifted else .3))
    cfo = float(rng.uniform(-.003,.003) if shifted else rng.uniform(-.0005,.0005))
    multipath = shifted and impairment != "flat"
    if impairment == "multipath": multipath = True
    if impairment == "timing": timing = .85
    if impairment == "carrier": cfo = .004
    target,symbols,meta = generate_psk(rng,n,family=family,timing=timing,cfo=cfo,multipath=multipath)
    t = np.arange(n)
    if shifted:
        kind = "chirp"
        start = rng.uniform(-.075,.075)
        width = rng.choice([-1,1])*rng.uniform(.006,.035)
        phase = 2*np.pi*(start*t + .5*width*t*t/n) + rng.uniform(-np.pi,np.pi)
        interference = np.exp(1j*phase)
        intmeta = {"kind":kind,"start_cycles_per_sample":float(start),"sweep_cycles_per_sample":float(width)}
    else:
        count = int(rng.integers(1,4))
        frequencies = rng.uniform(-.09,.09,count)
        phases = rng.uniform(-np.pi,np.pi,count)
        interference = np.sum(np.exp(1j*(2*np.pi*frequencies[:,None]*t + phases[:,None])),axis=0)
        intmeta = {"kind":"tones","frequencies_cycles_per_sample":frequencies.tolist()}
    interference /= np.sqrt(np.mean(np.abs(interference)**2)) + 1e-12
    interference *= 10**(-sir_db/20)
    noise = (rng.standard_normal(n)+1j*rng.standard_normal(n))/np.sqrt(2)
    noise *= 10**(-snr_db/20)/(np.sqrt(np.mean(np.abs(noise)**2))+1e-12)
    interference = interference.astype(np.complex64)
    noise = noise.astype(np.complex64)
    mixture = (target+interference+noise).astype(np.complex64)
    meta.update({"seed":seed,"distribution":split,"snr_db":snr_db,"sir_db":sir_db,
                 "sample_rate":1e6,"symbol_rate":125000.,"interference":intmeta,
                 "license":"CC0-1.0","generator":"etherglass.synthetic/v1"})
    return {"mixture":mixture,"target":target,"interference":interference,"noise":noise,
            "symbols":symbols,"metadata":meta}
