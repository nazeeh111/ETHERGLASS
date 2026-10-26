"""Prior-conditioned offline target extraction with mixture-consistent residual."""
from __future__ import annotations
from functools import lru_cache
import hashlib
import io
from pathlib import Path
import time
import numpy as np
from scipy.ndimage import median_filter
from .models import FRAME_SIZE, band_prior, spectral_features, make_model
from .synthetic import rrc_taps

_RESOURCE_MODELS=Path(__file__).resolve().parent/"resources"/"models"
_SOURCE_MODELS=Path(__file__).resolve().parents[2]/"models"
# A source checkout's retrained models must take precedence over wheel copies.
# Installed wheels have no adjacent repository model directory and use resources.
DEFAULT_CHECKPOINT=(_SOURCE_MODELS if (_SOURCE_MODELS/"hybrid.pt").is_file() else _RESOURCE_MODELS)/"hybrid.pt"


@lru_cache(maxsize=16)
def _load_model_cached(path: str,mtime_ns:int,size_bytes:int):
    import torch
    payload=Path(path).read_bytes()
    obj=torch.load(io.BytesIO(payload),map_location="cpu",weights_only=True)
    model=make_model(obj["architecture"])
    model.load_state_dict(obj["state_dict"])
    model.eval()
    obj["checkpoint_sha256"]=hashlib.sha256(payload).hexdigest()
    obj["checkpoint_file"]=Path(path).name
    return model,obj


def load_model(path:str):
    """Reload when a checkpoint is replaced by a new local training run."""
    stat=Path(path).stat()
    return _load_model_cached(path,stat.st_mtime_ns,stat.st_size)


def frame_estimate(x: np.ndarray,sps:float=8.,method:str="hybrid",checkpoint=None):
    feat,spec,scale=spectral_features(x,sps)
    n=len(x)
    if method in {"hybrid","unet","pretrained","no_dsp","scratch"}:
        import torch
        path=Path(checkpoint) if checkpoint else DEFAULT_CHECKPOINT.with_name(f"{method}.pt")
        if not path.is_file(): raise FileNotFoundError(f"trained checkpoint unavailable: {path}")
        model,info=load_model(str(path.resolve()))
        with torch.inference_mode(): gain=model(torch.from_numpy(feat[None]))[0].numpy()
    elif method == "bandpass":
        gain=band_prior(n,sps)
    elif method == "matched":
        h=rrc_taps(sps)
        # Zero-phase matched pulse filter. Not an oracle equalizer or decoder.
        gain=np.abs(np.fft.fftshift(np.fft.fft(h,n)))
        gain/=gain.max()+1e-12
    elif method == "wiener":
        power=np.abs(spec)**2
        median=median_filter(power,size=31,mode="wrap")/.6931471805599453
        # Practical robust spectral background estimate, no clean reference.
        gain=np.minimum(1.,median/(power+1e-8))*band_prior(n,sps)
    elif method == "notch":
        power=np.abs(spec)**2
        median=median_filter(power,size=127,mode="wrap")/.6931471805599453
        # Sparse-peak excision. Fixed 10x robust-background threshold, selected
        # a priori; no reference labels or clean components used at inference.
        gain=(power<=10*np.maximum(median,1e-6)).astype(float)*band_prior(n,sps)
    else: raise ValueError(f"unknown method: {method}")
    y=np.fft.ifft(np.fft.ifftshift(spec*gain),norm="ortho")*scale
    return y.astype(np.complex64),gain


def separate(x:np.ndarray,sample_rate:float,target_center:float=0.,
             symbol_rate:float|None=None,method:str="hybrid",checkpoint:Path|None=None)->dict:
    """Extract one PSK-like target using the user's rate/center prior.

    target_center is relative to capture center, in Hz. The second component
    is the exact residual, including interference, noise and target error.
    It is never labelled an independently identified transmitter.
    """
    import torch
    torch.set_num_threads(min(4,torch.get_num_threads()))
    x=np.asarray(x,dtype=np.complex64)
    if x.ndim!=1 or x.size<128 or not np.all(np.isfinite(x)):
        raise ValueError("separation needs at least 128 finite complex samples")
    if not np.isfinite(sample_rate) or sample_rate<=0:
        raise ValueError("sample_rate must be positive")
    if symbol_rate is None:
        raise ValueError("symbol_rate is required: target separation is prior-conditioned")
    if not np.isfinite(symbol_rate) or symbol_rate<=0 or not 4<=sample_rate/symbol_rate<=32:
        raise ValueError("supported samples per symbol: 4..32; training nominal is 8")
    if not np.isfinite(target_center) or abs(target_center)+.8*symbol_rate>=sample_rate/2:
        raise ValueError("target support must fit within the sampled Nyquist band")
    started=time.perf_counter()
    n=len(x); size=FRAME_SIZE; hop=size//2
    phase=np.exp(-2j*np.pi*target_center*np.arange(n)/sample_rate)
    base=x*phase
    padded=np.pad(base,(hop,hop+size),mode="reflect")
    win=np.hanning(size).astype(np.float32)
    out=np.zeros_like(padded); weight=np.zeros(len(padded),dtype=np.float32)
    gains=[]
    # Frames starting after the requested crop cannot contribute to its OLA
    # samples. Avoid computing the two wholly outside trailing frames.
    for start in range(0,hop+n,hop):
        frame=padded[start:start+size]*win
        estimate,gain=frame_estimate(frame,sample_rate/symbol_rate,method,checkpoint)
        out[start:start+size]+=estimate*win
        weight[start:start+size]+=win**2
        gains.append(gain)
    target=(out[hop:hop+n]/np.maximum(weight[hop:hop+n],1e-8)/phase).astype(np.complex64)
    residual=(x-target).astype(np.complex64)
    if not np.all(np.isfinite(target)) or not np.all(np.isfinite(residual)):
        raise ValueError("capture amplitude exceeds representable complex64 reconstruction range")
    ood_rate=not 7.5<=sample_rate/symbol_rate<=8.5
    warnings=[]
    if ood_rate:
        warnings.append("Rate differs from nominal training (8 samples/symbol); performance unvalidated.")
    if n<FRAME_SIZE:
        warnings.append(f"Short record: fewer than {FRAME_SIZE} training-frame samples; reflected padding dominates and separation quality is unvalidated.")
    diagnostics={"method":method,"sample_rate":float(sample_rate),"symbol_rate":float(symbol_rate),
        "target_center_hz":float(target_center),"samples_per_symbol":float(sample_rate/symbol_rate),
        "rrc_rolloff_assumed":.35,
        "runtime_ms":(time.perf_counter()-started)*1000,"frames":len(gains),
        "mixture_consistency_max_abs":float(np.max(np.abs(x-target-residual))),
        "target_power_fraction":float(np.mean(abs(target.astype(np.complex128))**2)/(np.mean(abs(x.astype(np.complex128))**2)+1e-20)),
        "median_spectral_gain":float(np.median(gains)),
        "scope":"Known-rate bandlimited PSK target amid sparse narrowband interference; no message decoding.",
        "warning":" ".join(warnings) or None,"warnings":warnings,
        "short_record_unvalidated":n<FRAME_SIZE,
        "residual_label":"Unassigned residual (interference + noise + reconstruction error)",
        "uncertainty":"Separation quality is unknown without a clean reference; classification is independently gated."}
    modelinfo={"architecture":"classical DSP","trained":False}
    if method in {"hybrid","unet","pretrained","no_dsp","scratch"}:
        path=Path(checkpoint) if checkpoint else DEFAULT_CHECKPOINT.with_name(f"{method}.pt")
        _,info=load_model(str(path.resolve()))
        modelinfo={k:v for k,v in info.items() if k!="state_dict"}
        modelinfo["trained"]=True
    return {"components":[target,residual],"diagnostics":diagnostics,"model":modelinfo}
