# Measured separation results

Generated from `research/results.json`; raw paired observations are in `evaluation.csv`.

Denominators: **432 independent generated records across 18 conditions, 24 records per condition**, evaluated on 3 model-training seeds. The raw CSV has **8640 method/run observations**; these are not 8640 independent recordings. Classical methods run once per record and learned methods once per training seed. All records are 4,096 samples. Metrics below are record-level arithmetic means, then training-seed means for learned methods; BER/SER are macro averages over records, not pooled bit counts.

| Method | Nominal SI-SDR dB | Shift SI-SDR dB | Shift waveform EVM % | Shift oracle SER | Shift CPU ms / 4096 samples |
|---|---:|---:|---:|---:|---:|
| Mixture | -0.291 | -4.351 | 82.36 | 0.3445 | 0.00 |
| RRC matched | 1.438 | -3.291 | 78.05 | 0.3437 | 10.04 |
| Robust Wiener | 6.355 | 1.316 | 63.19 | 0.1408 | 6.35 |
| Adaptive notch | 11.434 | 7.787 | 38.85 | 0.0115 | 5.76 |
| Spectral U-Net | 10.337 | 10.193 | 30.21 | 0.0009 | 70.03 |
| Pretrained only | 11.530 | 5.174 | 48.84 | 0.0532 | 53.36 |
| Adapt from scratch | 1.355 | 9.891 | 31.34 | 0.0022 | 41.92 |
| Contrast ablation | 11.140 | 10.018 | 30.81 | 0.0012 | 48.96 |
| ETHERGLASS | 10.902 | 10.111 | 30.57 | 0.0016 | 51.19 |

All learned methods use 3 training seeds: [0, 1, 2]. Seed0 was predetermined for the app. Symbol metrics use actual synthetic reference symbols and oracle timing/carrier/gain; they do not measure real-message decoding. Waveform scores align one complex scalar without delay or permutation search.

## Paired comparisons

| Cell / comparison | Mean SI-SDR delta dB | Paired-example bootstrap 95% CI | Wins / examples |
|---|---:|---|---:|
| nominal:hybrid-scratch | +9.546 | [+7.615, +11.545] | 24/24 |
| shift:hybrid-scratch | +0.221 | [+0.002, +0.520] | 10/24 |
| nominal:hybrid-pretrained | -0.628 | [-1.625, +0.421] | 9/24 |
| shift:hybrid-pretrained | +4.937 | [+3.797, +6.052] | 24/24 |
| nominal:hybrid-no_dsp | -0.238 | [-0.684, +0.255] | 7/24 |
| shift:hybrid-no_dsp | +0.094 | [-0.004, +0.195] | 17/24 |
| nominal:hybrid-unet | +0.565 | [+0.347, +0.790] | 20/24 |
| shift:hybrid-unet | -0.082 | [-0.221, +0.051] | 9/24 |
| nominal:hybrid-wiener | +4.547 | [+3.603, +5.648] | 23/24 |
| shift:hybrid-wiener | +8.795 | [+7.063, +10.573] | 24/24 |
| nominal:hybrid-notch | -0.533 | [-1.417, +0.342] | 10/24 |
| shift:hybrid-notch | +2.324 | [+1.705, +3.032] | 24/24 |
| nominal:hybrid-matched | +9.464 | [+7.228, +11.628] | 23/24 |
| shift:hybrid-matched | +13.402 | [+11.636, +15.031] | 24/24 |

## Falsifiable hypothesis verdicts

- Stationary-tone pretraining improves fixed-budget chirp adaptation: **positive mean effect within this holdout**; paired mean +0.221 dB, interval [+0.002, +0.520] dB.
- Explicit contrast features improve the shifted-condition model: **inconclusive within this holdout**; paired mean +0.094 dB, interval [-0.004, +0.195] dB.

Intervals average training seeds before resampling independent example IDs, so they do not quantify the full uncertainty over training runs. The pretraining comparison wins on only 10/24 records; its small positive mean and near-zero interval boundary should not be read as a consistent per-record gain. These exploratory comparison intervals are not adjusted for multiple comparisons. All negative comparisons remain above. See `PROTOCOL.md` for priors, split ranges, ablations and limits.

## Independent-generator, channel and failure tests

| Condition | Mixture SI-SDR dB | Notch SI-SDR dB | U-Net SI-SDR dB | Hybrid SI-SDR dB | Hybrid waveform EVM % |
|---|---:|---:|---:|---:|---:|
| Flat channel + chirp | -1.092 | 8.965 | 11.555 | 11.593 | 26.36 |
| Multipath + chirp | -1.373 | 9.127 | 11.119 | 11.086 | 27.43 |
| 0.85-sample timing offset | -1.673 | 7.990 | 10.324 | 10.367 | 29.42 |
| 0.004 cycles/sample carrier offset | -2.186 | 8.752 | 10.389 | 10.294 | 29.52 |
| Independent TorchSig PSK generator | -1.567 | 10.425 | 10.088 | 10.733 | 28.87 |
| Overlapping QPSK stress (unidentifiable) | -5.080 | -5.068 | -5.052 | -5.064 | 87.28 |
| 6 samples/symbol (untrained rate) | -2.630 | 8.100 | 10.836 | 10.769 | 28.07 |
| 10 samples/symbol (untrained rate) | -0.273 | 10.102 | 12.498 | 12.704 | 24.78 |

The TorchSig row uses an independent implementation of simulated pulse shaping, not hardware data. Its helper hides transmitted symbols, so symbol EVM/SER/BER are absent. The digital-interferer test deliberately violates the sparse-interference prior and cannot establish unique two-transmitter recovery. The 6/10 sample-rate tests are outside nominal 8 samples/symbol training; the application marks those rates unvalidated.

Figures are generated only from saved actual measurements in `research/figures/`, with editable SVG sources. The main product demo and social artwork are generated separately by `scripts/make_figures.py`.
