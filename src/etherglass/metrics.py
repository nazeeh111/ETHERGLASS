"""Reference-based waveform metrics with explicit alignment conventions."""
from __future__ import annotations
import numpy as np
from scipy.signal import fftconvolve


def waveform_metrics(reference: np.ndarray, estimate: np.ndarray) -> dict:
    """No permutation/delay search. Align one complex scalar (gain and phase).

    SI-SDR is the complex orthogonal projection score; waveform EVM uses
    best scalar alignment of the estimate. These are not symbol EVM or BER.
    """
    r=np.asarray(reference,dtype=np.complex128).reshape(-1)
    e=np.asarray(estimate,dtype=np.complex128).reshape(-1)
    if r.shape!=e.shape or r.size==0 or not np.all(np.isfinite(r)) or not np.all(np.isfinite(e)):
        raise ValueError("reference and estimate must be finite equal-length arrays")
    energy=float(np.vdot(r,r).real)
    if energy<=1e-20: raise ValueError("reference must have nonzero energy")
    alpha=np.vdot(r,e)/(energy+1e-20)
    desired=alpha*r
    error=e-desired
    si_sdr=10*np.log10((np.vdot(desired,desired).real+1e-20)/(np.vdot(error,error).real+1e-20))
    if np.vdot(e,e).real <= 1e-20:
        si_sdr=-120.  # Explicit finite reporting floor for the zero estimate.
    align=np.vdot(e,r)/(np.vdot(e,e).real+1e-20)
    aligned=e*align
    mse=float(np.mean(np.abs(e-r)**2))
    evm=float(np.sqrt(np.mean(np.abs(aligned-r)**2)/(energy/r.size)))
    return {"si_sdr_db":float(si_sdr),"waveform_evm_pct":100*evm,
            "nmse_db":float(10*np.log10((mse+1e-20)/(energy/r.size))),
            "alignment_gain":float(abs(align)),"alignment_phase_rad":float(np.angle(align))}


def synthetic_symbol_metrics(estimate:np.ndarray,symbols:np.ndarray,metadata:dict,
                             already_matched:bool=False)->dict:
    """Oracle-synchronized synthetic symbol metrics, requiring reference symbols.

    Uses generator-truth carrier/timing and one reference-aided complex gain.
    No adaptive equalizer. With multipath, residual ISI remains. These scores
    measure synthetic reconstruction fidelity, not an operational RF decoder.
    """
    from .synthetic import rrc_taps
    sps=int(metadata["samples_per_symbol"])
    family=metadata["family"]
    if family not in {"BPSK","QPSK"}: raise ValueError("unsupported reference constellation")
    e=np.array(estimate,dtype=np.complex128,copy=True)
    t=np.arange(len(e))
    e*=np.exp(-1j*(2*np.pi*metadata["carrier_offset_cycles_per_sample"]*t+metadata["phase_rad"]))
    h=rrc_taps(sps,metadata["rolloff"])
    # The matched-filter comparator has already applied the receive filter.
    # Applying it twice would unfairly increase its intersymbol interference.
    filtered=e if already_matched else fftconvolve(e,h,mode="same")
    times=np.arange(len(symbols))*sps-metadata["symbol_crop_offset_samples"]-metadata["timing_samples"]
    valid=(times>=len(h))&(times<len(e)-len(h))
    times=times[valid]
    refs=np.asarray(symbols)[valid]
    if len(refs)<8: raise ValueError("too few guarded reference symbols")
    received=np.interp(times,t,filtered.real)+1j*np.interp(times,t,filtered.imag)
    gain=np.vdot(received,refs)/(np.vdot(received,received).real+1e-20)
    aligned=received*gain
    order=2 if family=="BPSK" else 4
    offset=0 if order==2 else np.pi/4
    predicted=np.round((np.angle(aligned)-offset)*order/(2*np.pi)).astype(int)%order
    truth=np.asarray(metadata["symbol_labels"])[valid]
    ser=float(np.mean(predicted!=truth))
    pgray=predicted^(predicted>>1);tgray=truth^(truth>>1)
    bits=1 if order==2 else 2
    biterrors=sum(np.count_nonzero(((pgray^tgray)>>b)&1) for b in range(bits))
    return {"oracle_symbol_evm_pct":float(100*np.sqrt(np.mean(abs(aligned-refs)**2))),
            "oracle_ser":ser,"oracle_ber":float(biterrors/(len(refs)*bits)),
            "reference_symbol_count":int(len(refs))}
