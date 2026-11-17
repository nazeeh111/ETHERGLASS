# Estimate a target. Keep the residual.

An actual export/reopen journey from current source `f0ed9e43dd065eff2b91e7e27b1ac4fc8e65958f`, tree `627439d7ba1f8385a59968897bc3a277efce8e80`, version 0.1.1.dev0. The existing fixed synthetic reference uses seed 870001, 16384 complex samples at 1 MHz, known QPSK plus an overlapping chirp and noise. These are CC0 generated samples, not a private or over-the-air recording. No new training, calibration study, seed search or provider call was performed.

## Run and reopen the actual outputs

Follow the [CPU setup](../../README.md#run-it-on-a-cpu). The dependency-light first route is:

```sh
etherglass demo --output bandpass-reference
```

For the figure's actual Hybrid inference with included fixed checkpoints:

```sh
pip install -e '.[all]'
etherglass demo --method hybrid --output hybrid-reference
```

Output paths must be fresh. `hybrid-reference` contains the original mixture SigMF pair, full `analysis.json`, `target.zip` and `residual.zip`. No target symbol-rate or carrier offset is inferred: this demo supplies 0 Hz and 125000 symbols/s. A generic `analyze --method hybrid` call without `--symbol-rate` is refused before creating output.

[Download reference-exports.zip](reference-exports.zip), extract into a fresh directory and verify its SHA256SUMS. Extract each nested target/residual ZIP into separate directories. Each has `capture.sigmf-data`, matching `capture.sigmf-meta`, and the producer's analysis receipt. The root mixture files preserve the original input bytes. For example:

```python
from etherglass.io import load_capture
import numpy as np
mixture = load_capture("mixture.sigmf-data")
target = load_capture("target/capture.sigmf-data")
residual = load_capture("residual/capture.sigmf-data")
assert len(mixture.samples) == len(target.samples) == len(residual.samples) == 16384
error = mixture.samples.astype(np.complex128) - target.samples - residual.samples
print(np.max(np.abs(error)))
```

These particular cf32_le exports have maximum absolute closure error 1.23e-7 and relative RMS error 2.27e-8 from floating-point rounding. That arithmetic identity is not evidence of correct source separation: the residual is defined by subtraction. Source provenance refers to the included original mixture, while each derived SigMF pair carries its own SHA512. Source/model hashes refer to the pinned full repository; checkpoints are explicitly external to this signal-export bundle and unnecessary merely to read samples. The bundle README declares every included/external path. New HTML IDs, request timings and ZIP timestamps can change on rerun; compare source, samples and scientific fields rather than expecting every artifact byte to repeat.

## Read the figure

The three panels are production `etherglass.dsp.waterfall` transforms of the **exported** mixture, target and residual arrays. All use a 256-sample periodic Hann window, 128-sample hop and 127 time frames by 256 frequency bins. Sample rate 1 MHz gives 3.90625 kHz bin spacing; windows are 0.256 ms long and advance 0.128 ms. All share the same time/frequency coordinates. The renderer embeds each native raster grid in the SVG without smoothing or new bins; axes/text remain vector. This is a scientific reconstruction of the exported samples, not an application screenshot.

Color is each bin's power-density level minus the maximum mixture STFT level. Every panel uses this **same reference** and the same display range −55…0 dB, saturating lower values. It is neither independent per-panel normalization nor dBFS, physical power, or a calibrated RF measurement. The full input-unit²/Hz values follow the production window normalization. [figure-data.npz](figure-data.npz) retains all complex arrays, the clean synthetic reference, time/frequency vectors and all 97536 relative-level cells. [provenance.json](provenance.json) pins source, checkpoint, runtime and exact transformations. `python docs/presentation/render.py` regenerates both sizes using NumPy/Matplotlib without inference, training or device access.

## Keep the evidence states separate

For this one preexisting fixed reference:

| Observation | Actual result | Scope |
| --- | ---: | --- |
| Mixture target SI-SDR | -6.16765 dB | known clean synthetic reference |
| Bandpass target SI-SDR | -6.15954 dB | same reference and supplied priors |
| Hybrid target SI-SDR | 15.55349 dB | same fixed reference, not an aggregate score |
| Hybrid waveform EVM | 16.45746% | complex scalar alignment; not symbol EVM or BER |
| Training-seed disagreement | indicator | no calibrated error-bound field for this demo |

SI-SDR aligns one complex scalar, with no permutation or delay search. The reference remains scoring-only; it is not supplied to the separator. Independent recomputation from reopened exported samples matches the report. These numbers illustrate a fixed case; the [retained study](../../README.md#a-falsifiable-experiment) has different aggregate results, failures and comparators. In particular, it does not establish superiority over the U-Net comparator or successful arbitrary cochannel recovery.

The mixture classifier abstains. The target's supported synthetic QPSK label checks only its finite support assumptions, not separation correctness. Training-seed disagreement can be small when all models are wrong. The separately gated [conditional synthetic bound example](../../README.md#conditional-synthetic-bound-example) has a different internally generated distribution/runtime contract. Neither generic imports nor the ordinary reference demo inherit its marginal coverage.

## Actual observatory controls

`etherglass serve` opens no browser and binds loopback; visit its printed URL. Load the reference, retain the explicit 0 Hz / 125 kbaud prior, run Hybrid, switch between target/residual/mixture, and export the matching completed state. The [current capture](observatory-current.png) shows this actual run with Residual selected. Numeric start/stop controls and Apply inspect a sample interval; reconstruction applies the selected method. This task also inspected 2–8 ms, reconstructed 6000 samples with Bandpass, refused a reversed 9–8 ms interval, and reopened the downloaded target SigMF pair. Export identity is tied to the completed analysis, not an uncommitted draft.

All expert controls and current-main model/evidence binding remain unchanged. No new product defect was established in this bounded journey. This is not a full UI/accessibility or hardware validation. See [complete tutorial](../QUICKSTART.md), [model card](../MODEL_CARD.md), [protocol](../../research/PROTOCOL.md), [licenses](../DATA_LICENSES.md) and [current API guide](../../src/etherglass/web/api.html) for the deeper contracts.
