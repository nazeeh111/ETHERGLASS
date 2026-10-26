"""One analysis path shared by the CLI and local observatory."""
from __future__ import annotations

import hashlib
import json
import time
import numpy as np

from . import __version__
from .io import Capture
from .dsp import visual_summary, waterfall, occupied_regions, diagnostics


def analyze(capture: Capture, *, start_sample: int = 0, stop_sample: int | None = None,
            target_center: float = 0, symbol_rate: float | None = None,
            method: str | None = None, reference: np.ndarray | None = None) -> dict:
    stop_sample = len(capture.samples) if stop_sample is None else stop_sample
    if not isinstance(start_sample, int) or not isinstance(stop_sample, int) or not 0 <= start_sample < stop_sample <= len(capture.samples):
        raise ValueError("Select a sample interval inside the recording.")
    if stop_sample - start_sample < 64:
        raise ValueError("Select at least 64 samples.")
    if not np.isfinite(target_center) or abs(target_center) >= capture.sample_rate / 2:
        raise ValueError("Target center must lie strictly inside the baseband Nyquist interval.")
    if symbol_rate is not None and (not np.isfinite(symbol_rate) or symbol_rate <= 0 or symbol_rate > capture.sample_rate / 2):
        raise ValueError("Symbol rate must be positive and no greater than half the sample rate.")
    # Never analyze across a retune as if the center frequency were constant.
    segments = capture.metadata.get("captures", [])
    center = capture.center_frequency
    for c in segments:
        boundary = c["core:sample_start"]
        if start_sample < boundary < stop_sample:
            raise ValueError("Selection crosses a tuning boundary. Select samples within a single SigMF capture segment.")
        if boundary <= start_sample:
            center = c.get("core:frequency", center)
    x = capture.samples[start_sample:stop_sample]
    summary = visual_summary(x, capture.sample_rate, symbol_rate)
    glob = capture.metadata.get("global", {})
    # Provenance is descriptive only: it is never passed to a model as labels.
    result = {"version": __version__, "name": capture.name,
              "sample_rate": capture.sample_rate, "center_frequency": center,
              "num_samples": len(x), "original_num_samples": len(capture.samples),
              "duration": len(x) / capture.sample_rate, "source_sha256": capture.source_sha256,
              "selection": {"start_sample": start_sample, "stop_sample": stop_sample,
                            "target_center": target_center, "symbol_rate": symbol_rate},
              "provenance": {"source": glob.get("core:description", "User-provided I/Q"),
                             "license": glob.get("core:license", "unspecified"),
                             "datatype": glob.get("core:datatype", "explicit raw import"),
                             "sha256": capture.source_sha256,
                             "metadata": capture.metadata},
              "warnings": list(capture.warnings), **summary,
              "waterfall": waterfall(x, capture.sample_rate),
              "regions": occupied_regions(summary["spectrum"]), "diagnostics": diagnostics(x),
              "separation": None}
    from .classification import classify
    baseband = x * np.exp(-2j * np.pi * target_center * np.arange(len(x)) / capture.sample_rate)
    result["classification"] = classify(baseband, capture.sample_rate, symbol_rate)
    result["warnings"].append("Occupied regions are heuristic energy estimates, not a count of transmitters.")
    if method:
        if symbol_rate is None:
            raise ValueError("Separation needs an explicit target symbol rate. It is a prior, not an estimated fact.")
        from .separation import separate
        began = time.perf_counter()
        sep = separate(x, capture.sample_rate, target_center, symbol_rate, method=method)
        # Keep full reconstruction arrays internal; plotting payloads are bounded.
        result["separation"] = {"method": method, "components": [],
                                "diagnostics": sep.get("diagnostics", {}),
                                "model": sep.get("model", {}),
                                "limitations": ["Target estimate is conditional on the chosen RRC-PSK prior.",
                                                 "Residual contains interference, noise and reconstruction error.",
                                                 "Unknown cochannel sources may not be identifiable from a single sensor."]}
        arrays = sep["components"]
        if method == "hybrid":
            from .uncertainty import ensemble_indicator
            result["separation"]["uncertainty"] = ensemble_indicator(x, capture.sample_rate, target_center, symbol_rate,
                                                                     estimate=arrays[0])
        for i, component in enumerate(arrays):
            if isinstance(component, dict):
                data = np.asarray(component.get("samples", component.get("signal")))
                name = component.get("name", "target" if i == 0 else "residual")
            else:
                data = np.asarray(component)
                name = "target" if i == 0 else "residual"
            if data.shape != x.shape or not np.isfinite(data).all():
                raise ValueError("Separator returned an invalid reconstruction.")
            item = {"name": name, **visual_summary(data, capture.sample_rate, symbol_rate)}
            if i == 0:
                down = data * np.exp(-2j * np.pi * target_center * np.arange(len(data)) / capture.sample_rate)
                item["classification"] = classify(down, capture.sample_rate, symbol_rate)
            result["separation"]["components"].append(item)
        result["separation"]["diagnostics"]["end_to_end_runtime_ms"] = (time.perf_counter() - began) * 1000
        if reference is not None:
            from .metrics import waveform_metrics
            truth = reference[start_sample:stop_sample]
            target = arrays[0] if not isinstance(arrays[0], dict) else arrays[0].get("samples", arrays[0].get("signal"))
            before, after = waveform_metrics(truth, x), waveform_metrics(truth, target)
            result["separation"]["reference_metrics"] = {"mixture": before, "target": after,
                "si_sdr_improvement_db": after["si_sdr_db"] - before["si_sdr_db"],
                "scope": "Bundled synthetic capture only. Clean received target reference; complex scalar alignment, no delay search. Waveform EVM is not symbol EVM."}
            result["separation"]["diagnostics"]["reference_si_sdr_improvement_db"] = after["si_sdr_db"] - before["si_sdr_db"]
        result["_arrays"] = arrays
    return result


def public_report(result: dict) -> dict:
    return {k: v for k, v in result.items() if not k.startswith("_")}


def demo_capture() -> tuple[Capture, np.ndarray]:
    """A fixed independent seed, never selected by its measured performance."""
    from .synthetic import generate_example
    from .io import encode_sigmf, load_bytes
    ex = generate_example(870001, n=16384, split="test", snr_db=20, sir_db=-6, family="QPSK")
    data, meta = encode_sigmf(ex["mixture"], 1e6, 0,
                             description="Synthetic QPSK + overlapping chirp, fixed seed 870001. No over-the-air messages.",
                             license="https://creativecommons.org/publicdomain/zero/1.0/",
                             provenance=ex["metadata"])
    return load_bytes(data, metadata=meta, name="QPSK × chirp · 870001"), ex["target"]
