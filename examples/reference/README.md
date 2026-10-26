# Fixed reference observation

QPSK+chirp seed870001,16,384complex samples,1Msps,125kbaud,SNR20dB,SIR−6dB. This seed was fixed before evaluating its performance.

Original synthetic data: CC0-1.0. `target` is the clean received waveform; `interference` and `noise` are exact generated components. They sum to `mixture` within float32 precision. Metadata preserves generator/channel parameters and symbols are random, not private messages. Clean references are used only for supervised training/evaluation, never provided to the inference model.

Rebuild with `python scripts/generate_reference.py`. Manifest contains actual SHA256 hashes.
