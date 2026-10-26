# Research foundations

Research inspected on 2026-10-25. These are foundations and implementation references, not claims that ETHERGLASS reproduces published scores.

## The tractable question

Single-channel recovery needs assumptions. Two arbitrary, fully overlapping sources do not have a unique decomposition from their sum. Begin with a known, pulse-shaped signal of interest and a specified interference/channel envelope. Keep its generated clean waveform and symbols. Recover the target and export the remainder as a **residual**, which also contains noise and estimation error. Do not label that remainder a clean second transmitter.

The falsifiable question is whether learned interference-aware correction improves held-out target reconstruction over matched filtering and practical interference suppression, and whether targeted adaptation improves performance under a declared shift. A positive result needs paired cases and repeated training seeds; a negative result remains useful. Model size, training budget, and inference access to target parameters must accompany every comparison.

## Signal generation

[TorchSig](https://github.com/TorchDSP/torchsig/tree/1197261d4f0d4c93de18757ee2ceb368ac04104c) is an MIT-licensed PyTorch signal-generation and augmentation toolkit. Its [2022 study](https://arxiv.org/abs/2207.09918) introduces Sig53 and motivates broad impairment diversity. That work evaluates classification, not this project's separator.

The inspected `constellation_modulator_baseband` helper accepts a constellation, rectangular/SRRC pulse, sample count, samples per symbol, roll-off, and NumPy generator. Pin a source revision rather than assume a stable interface. The full package brings audio, Lightning, OpenCV, and Numba dependencies; a narrow optional adapter or attributed helper extraction can keep CPU replay lighter. Claim TorchSig-generated training only if the experiment actually used it.

## Separation and fair baselines

[RF Challenge, v3 (2025)](https://arxiv.org/html/2409.08839v3) provides known-target interference-rejection experiments and classical/neural comparisons. Its [challenge description](https://rfchallenge.mit.edu/icassp24-single-channel/) explicitly supplies signal-of-interest generation parameters. ETHERGLASS therefore treats conditioned extraction as a useful starting scope, while leaving blind multi-source identification unproven. The v3 record flags a test-leakage footnote, reinforcing the need to split recordings before windowing. Compact architecture variants are local baselines, not exact published checkpoint reproductions.

## Independent recordings

[RadioPile](https://github.com/xandervanparys/RadioPile/tree/fd6cb8a8fe824509121e3f85d8a7b153b0e9644a) unifies heterogeneous datasets with provenance. Its [per-source record](https://zenodo.org/records/22877108) permits choosing one source without downloading the 117.46 GB aggregate. Real recordings lacking clean source waveforms cannot establish reconstruction error or bit error rate. Their appropriate initial role is actual-input replay, numerical stability, and rejection of unsupported waveforms. Source identity is not a modulation label. License and extraction details are in [DATA_LICENSES.md](DATA_LICENSES.md).

## Metadata and interoperability

[SigMF 1.2.6](https://sigmf.org/) defines interleaved I/Q formats and JSON metadata. Each component's bit width appears in the datatype, so `cf32_le` means two float32 values per complex sample. ETHERGLASS should preserve sample rate, capture frequency, provenance, and licensing, validate supported layouts, and reject unsupported metadata rather than silently reinterpret it.

[GNU Radio](https://github.com/gnuradio/gnuradio) is a mature DSP interoperability reference under GPL-3.0. Ordinary raw/SigMF file exchange does not require importing or copying its implementation. Live hardware support remains outside the initial offline release unless tested explicitly.

## Technical illustrations

[PaperVizAgent](https://github.com/google-research/papervizagent), formerly PaperBanana, is Apache-2.0 and uses planning, styling, generation, and critique stages. Its documented generation path requires externally configured model services. ETHERGLASS has no authorization for separately billed APIs, so this inspection does not authorize running those services. Editable local vector diagrams and plots computed from experiment outputs provide a feasible, verifiable route. Review labels against the actual implementation and retain their sources.

## Scope of evidence

Synthetic holdouts show performance only under the declared simulator. Adaptation data must not reuse test symbols, channel realizations, or recording sessions. With generated symbols, report receiver assumptions beside EVM and BER. With real unknown signals, report what was measured and what was withheld. No universal decoding, transmitter identification, or over-the-air separation claim follows from a small synthetic study.
