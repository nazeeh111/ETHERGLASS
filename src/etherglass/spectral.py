"""Shared NumPy/SciPy spectral features for classical and neural estimators."""
from __future__ import annotations

import numpy as np
from scipy.ndimage import uniform_filter1d

FRAME_SIZE=2048
FEATURE_CHANNELS=7


def band_prior(n: int, sps: float = 8.) -> np.ndarray:
    f=np.abs(np.fft.fftshift(np.fft.fftfreq(n)))
    # Full gain in the RRC support; smooth numerical margin outside its edge.
    edge=.675/sps
    return (.5*(1+np.cos(np.pi*np.clip((f-edge)/(.12/sps),0,1)))).astype(np.float32)


def spectral_features(x: np.ndarray, sps: float=8.) -> tuple[np.ndarray,np.ndarray,float]:
    """Features use only mixture IQ and explicit symbol-rate prior, never truth."""
    x=np.asarray(x,dtype=np.complex64)
    n=len(x)
    # Accumulate in float64: finite cf32 captures can have amplitudes whose
    # squared magnitude overflows float32 before normalization.
    scale=float(np.sqrt(np.mean(np.abs(x.astype(np.complex128))**2))+1e-8)
    spec=np.fft.fftshift(np.fft.fft(x/scale,norm="ortho"))
    power=np.abs(spec)**2
    log=np.log1p(power).astype(np.float32)
    floor=np.maximum(uniform_filter1d(power,size=127,mode="wrap"),1e-6)
    whiten=spec/np.sqrt(floor)
    contrast7=log-uniform_filter1d(log,size=7,mode="wrap")
    contrast31=log-uniform_filter1d(log,size=31,mode="wrap")
    contrast127=log-uniform_filter1d(log,size=127,mode="wrap")
    feat=np.stack([np.clip(whiten.real,-8,8),np.clip(whiten.imag,-8,8),log,
                   contrast7,contrast31,contrast127,band_prior(n,sps)]).astype(np.float32)
    return feat,spec.astype(np.complex64),scale
