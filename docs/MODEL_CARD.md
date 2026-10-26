# Model card and scientific report

ETHERGLASS v0.1.0 is a compact, prior-conditioned RF interference-suppression study and offline research instrument. It reconstructs one received, root-raised-cosine BPSK/QPSK target from sparse narrowband interference. It is not a blind general-purpose source separator or a real-message decoder.

## Release evidence

The shifted holdout contains 24 independently generated records, evaluated with three training seeds and the exact public overlap-add inference path. Mean hybrid waveform SI-SDR is **10.11 dB**, compared with **-3.29 dB** for matched filtering, **1.32 dB** for robust spectral Wiener, **7.79 dB** for adaptive notch, and **10.19 dB** for the small spectral U-Net. Higher SI-SDR means better target reconstruction under the stated complex-scalar alignment convention.

These are synthetic-population results. The same model is explicitly challenged on cochannel digital mixtures and mismatched symbol rates. External recordings have no clean reference; no real-world separation score is claimed.

## Controlled comparison

| Method | Nominal SI-SDR / dB | Shift SI-SDR / dB | Shift waveform EVM / % | Shift mean inference / ms |
|---|---:|---:|---:|---:|
| mixture | -0.29 | -4.35 | 82.36 | 0.00 |
| bandpass | -0.18 | -4.18 | 81.84 | 2.96 |
| matched | 1.44 | -3.29 | 78.05 | 10.04 |
| wiener | 6.35 | 1.32 | 63.19 | 6.35 |
| notch | 11.43 | 7.79 | 38.85 | 5.76 |
| unet | 10.34 | 10.19 | 30.21 | 70.03 |
| pretrained | 11.53 | 5.17 | 48.84 | 53.36 |
| scratch | 1.36 | 9.89 | 31.34 | 41.92 |
| no_dsp | 11.14 | 10.02 | 30.81 | 48.96 |
| hybrid | 10.90 | 10.11 | 30.57 | 51.19 |

Classical methods run once per generated input; learned results average seeds 0/1/2. Runtime is observed on the recorded CPU environment, with model-load/cache variability retained in raw rows. The U-Net is a local spectral encoder/decoder comparator with the same input features, support prior and training budgets. It is not the RF Challenge paper's time-domain model.

## Hypothesis and ablations

The predefined hypothesis compares stationary-tone pretraining followed by 160 chirp/channel adaptation updates against the identical randomly initialized model given the same 160 adaptation updates. Pretraining adds 320 updates and 1,536 examples, so the comparison matches adaptation budget, not total training cost. The `no_dsp` ablation removes only multiscale log-contrast channels; whitening and the analytic target-support prior remain.

| Shift comparison | Mean SI-SDR difference / dB | Paired-example bootstrap 95% interval / dB |
|---|---:|---|
| hybrid-scratch | +0.22 | [+0.002, +0.520] |
| hybrid-pretrained | +4.94 | [+3.797, +6.052] |
| hybrid-no_dsp | +0.09 | [-0.004, +0.195] |
| hybrid-unet | -0.08 | [-0.221, +0.051] |
| hybrid-wiener | +8.79 | [+7.063, +10.573] |
| hybrid-notch | +2.32 | [+1.705, +3.032] |
| hybrid-matched | +13.40 | [+11.636, +15.031] |

Bootstrap intervals resample held-out example IDs after averaging training seeds. They describe example variation, not the population uncertainty over all possible training runs. Three seeds are a small robustness check.

Adaptation changes nominal-tone SI-SDR from 11.53 to 10.90 dB, while changing shifted-chirp SI-SDR from 5.17 to 10.11 dB. Both sides of this distribution tradeoff are retained.

## Stress tests and independent generator

| Condition | Hybrid SI-SDR / dB | Notch SI-SDR / dB | Small U-Net SI-SDR / dB |
|---|---:|---:|---:|
| torchsig_independent_generator | 10.73 | 10.43 | 10.09 |
| overlapping_digital_stress | -5.06 | -5.07 | -5.05 |
| rate_6_samples_per_symbol | 10.77 | 8.10 | 10.84 |
| rate_10_samples_per_symbol | 12.70 | 10.10 | 12.50 |
| snr_0 | 5.80 | 4.93 | 5.72 |
| sir_-15 | 6.79 | 4.55 | 6.52 |

TorchSig is a different pinned generator implementation, not an independent hardware experiment. All methods receive the same target-center and symbol-rate priors. A single sensor cannot uniquely decompose arbitrary statistically similar cochannel sources. The model's nonnegative spectral gain cannot recreate information that interference has made unidentifiable.

## Metrics and physical assumptions

SI-SDR uses complex projection with one gain/phase factor, without delay or permutation search. Waveform EVM is least-squares complex-scalar-aligned waveform error, not symbol EVM. Raw NMSE is unaligned. Symbol EVM/SER/BER appear only when the original synthetic symbol sequence is valid. They use exact generator carrier/timing plus reference-aided gain; these oracle-synchronized fidelity metrics are not a deployable receiver. Multipath is not oracle equalized. TorchSig and real inputs without valid symbols receive no symbol metrics.

## Trained recognition and abstention

A separate portable physical-feature classifier was trained on 2,000 known and 2,000 unknown examples, with disjoint probability/threshold calibration. Final untouched tests contain 1000 known BPSK/QPSK and 1000 unknown probes. Known accuracy before abstention is 100.00%, known coverage 100.00%, and unknown false acceptance 0.00% (Wilson 95% upper bound 0.383%). Support AUROC is 1.0000.

Known-class Brier score: 2.87793e-29; known-class 10-bin ECE: 2.44662e-16; balanced known/unknown support Brier score: 1.11541e-06. These easy, finite synthetic probes do not establish open-world RF recognition. They include two test-only unknown families (AM and 32-QAM); raw rows and feature arrays are retained. Earlier exploratory seed ranges are excluded and recorded in model metadata.

Inference uses exactly the first 2,048 samples and a supplied symbol-rate prior. The supported calibration rates are 6/8/10/12 samples per symbol, rolloff 0.25–0.45, and SNR 8–30 dB, with shared channel distributions across families. Missing prior, insufficient samples, unsupported rate, insufficient support, or conditional class confidence below 0.8 returns unknown. Confidence is conditional on BPSK/QPSK support; the support probability depends on the balanced probe prior. The exact fit-time classifier source is retained in research/provenance; post-fit runtime artifact-routing/cache corrections leave the physical feature functions, coefficients, calibration and measured test rows unchanged. Fit and runtime hashes are recorded separately.

## Separation uncertainty

Three training seeds provide actual reconstruction disagreement. A separate split-calibration population (96 examples, seeds 900000 onward) sets a nominal 90% upper bound on raw received-target waveform RMS error. On 96 untouched same-distribution examples, observed coverage is **96.88%**. Mean RMS bound is 0.4987 input units; mean actual RMS error 0.2793.

The bound is marginal and conditional on the declared synthetic chirp/channel distribution. It has no coverage guarantee for arbitrary user recordings, unknown families, other rates, or real hardware. Correlated model bias can yield low disagreement while every seed is wrong. The exact calibration rank, checkpoint hashes and per-example outcomes are retained.

## Independent recordings

All 40 RadioPile ITS windows produced finite measured reports. Each has 500 samples, below the 2,048-sample classifier requirement, so this checks replay and short-record abstention only. The continuous CAMRAS excerpt has 65536 samples from one session. The stated 8 samples/symbol analysis prior is a hypothesis, not source ground truth. Its 32 adjacent windows produced 0 accepted classifications. That conditional abstention result is not real modulation accuracy or separation validation.

No real input was used for training, calibration, threshold tuning or source-domain modulation labels. Source IDs do not establish device/session independence; neighboring windows are correlated. Licenses, transformations, original identifiers and checksums are retained.

## Model, compute and reproducibility

The proposed model has 6,865 parameters; the spectral U-Net 27,121. Training uses batch 24, AdamW, gradient clipping at 5, 320 pretraining updates and 160 adaptation updates. Seed 0 is the predetermined release checkpoint. CPU inference uses a mathematically equivalent unfold/GEMM convolution path, with native PyTorch fallback. Output and input/weight/bias-gradient equivalence tests cover dilations 1/2/4/8; floating-point reduction order can differ. Completed native seed 0 hybrid/ablation artifacts were preserved; an incomplete native U-Net attempt was stopped and restarted from its original initialization at full budget using GEMM. The aborted attempt is documented and contributes no selected checkpoint.

Training configuration, exact per-stage logs, source/checkpoint hashes, raw evaluation rows, classifier features, licensed source provenance, runtime environment, scripts and editable figure sources accompany the release. Reproduce in a copy to preserve original artifacts. See [protocol](../research/PROTOCOL.md), [architecture](ARCHITECTURE.md) and [quickstart](QUICKSTART.md).

## Intended use and limits

Use for offline RF interference experiments, reproducible teaching and transparent I/Q inspection. Live SDR acquisition, transmit functions, private-message recovery, arbitrary source identification and universal decoding are absent. Original code/weights are MIT, original synthetic data CC0, and attributed external recordings retain CC-BY-4.0. This first release establishes a narrow synthetic-domain contribution and an actual-input instrument; real-world separation efficacy remains unestablished.
