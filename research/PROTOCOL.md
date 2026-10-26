# Experiment protocol

ETHERGLASS tests one limited and falsifiable hypothesis: **with a fixed 160-update adaptation budget, stationary-tone pretraining improves a compact interference-aware separator on unseen chirp mixtures compared with an identical randomly initialized network.** A second ablation asks whether explicit multiscale spectral contrasts help. Negative results remain in the report.

This is prior-conditioned target extraction, not blind demixing. One sensor observes one root-raised-cosine BPSK/QPSK target, sparse narrowband interference and complex additive white Gaussian noise (AWGN). The supplied target center and symbol rate are available to every method. Fully overlapping statistically similar digital signals are deliberately tested as an identifiability stress condition. The returned residual includes noise and reconstruction error, and is not a recovered second transmitter.

## Data and split discipline

The original local generator uses unit-energy root-raised-cosine (RRC) pulses with rolloff 0.35, guard symbols, fractional timing, carrier offset, optional multipath and measured per-crop SNR/SIR. It preserves received clean target, interference, noise, transmitted synthetic symbols, channel coefficients and random seed. No real recording is used for separation training. Generated training components are CC0-1.0.

Fixed example ranges are disjoint: training 100000–101535; adaptation 200000–200511; development validation 300000 onward; final controlled holdouts 400000 onward; the unselected demo seed 870001. Exact test-cell starts are in `run_experiment.py`. Examples contain independently generated symbols and channels, and are generated as whole records before framing. Test records are 4,096 samples, training frames 2,048. No windows from a generated record are assigned to different splits.

Training is 1,536 stationary-tone examples and 512 chirp/channel adaptation examples. Stationary tones number 1–3. Chirps span 0.006–0.035 cycles/sample across a frame. Target SNR is 10–30 dB and SIR −12 to 8 dB in training/adaptation. Final tests also cover 0 dB SNR and SIR −15/15 dB. Target nominal samples/symbol is 8. Larger timing/carrier offsets and multipath are included in shifted cases. Separate 6/10 samples-per-symbol tests are outside the nominal training rate, and their inference output carries an explicit unvalidated-rate warning.

An independent holdout uses the actual pinned TorchSig constellation-modulator helper extracted under MIT, with provenance retained in `src/etherglass/_vendor/`. Those targets are generated longer and centrally cropped. This tests a different implementation, **not independent hardware measurements**. The helper does not expose transmitted symbols, so symbol EVM/SER/BER are omitted for those examples. External real RadioPile rows are a separate unsupported-input test with no separation ground truth.

## Models and fair comparisons

- **Hybrid:** seven mixture-only spectral features: real/imaginary fast Fourier transform (FFT) after local root-mean-square (RMS) whitening, log power, local log-power contrasts at widths 7/31/127 and an analytic target-support prior. A compact dilated residual Conv1D network predicts a nonnegative spectral gain, constrained by target support. FFT inversion and overlap-add reconstruct the complex waveform.
- **Pretrained:** the same network after stationary-tone training, before chirp adaptation.
- **Scratch:** the same network trained only for the 160 chirp adaptation updates. It matches the adaptation update/data budget, not total compute. Pretraining adds 320 updates and 1,536 examples; this distinction is part of the hypothesis.
- **Contrast-feature ablation (`no_dsp`):** the hybrid with the three explicit contrast channels removed. IQ whitening and support remain. This does not remove all classical DSP.
- **Small spectral U-Net:** a local encoder/decoder comparator using the identical seven mixture-only features, target-support prior, data and optimizer budgets. It is not a reproduction of the RF Challenge paper's time-domain model or published results.
- **Matched filter:** zero-phase RRC matched-pulse filter at the supplied symbol rate.
- **Bandpass:** the same analytic target-support filter used by learned models.
- **Robust spectral Wiener:** local 31-bin median power background divided by instantaneous power, capped at unity and constrained by target support. It is a practical non-oracle shrinkage method.
- **Adaptive notch:** excises spectral bins above 10 times a 127-bin robust median background. Threshold chosen before final holdout evaluation; no clean signal appears at inference.
- **Mixture:** unchanged input.

Three independent training seeds 0/1/2 use the same disjoint datasets and fixed update budget. Seed 0 is the predetermined released checkpoint. AdamW, clipping 5, batch 24, learning rates 0.002 pretraining / 0.0008 adaptation; exact configuration and per-stage loss/time logs are retained. No checkpoint is selected using final holdout results. A Hann window is applied to each training mixture/reference frame. The loss is complex FFT MSE normalized by each clean target's frame energy.

The CPU convolution implementation was optimized after seed 0 started: unfold plus matrix multiplication computes exactly the same convolution with unchanged kernels, weights, parameters and receptive field. Native PyTorch is retained with `--native-conv`. Forward outputs and input/weight/bias gradients are tested for dilations 1/2/4/8 at floating-point tolerance. `training_runtime.json` records the backend for each seed. The completed native seed 0 hybrid/contrast models also predate a float64 RMS-accumulation fix for extreme capture amplitudes. These mathematically equivalent paths can differ by floating-point rounding; bitwise cross-backend reproduction is not claimed. All heldout inference uses the same final public implementation.

Completed seed 0 hybrid/pretraining/contrast-ablation checkpoint and log bytes were preserved when the native CPU bottleneck was diagnosed. Its incomplete U-Net attempt was interrupted and restarted from the same original initialization seed with the full fixed budget on the equivalent GEMM path. The interrupted attempt is retained as `training_logs/aborted_native_attempt_seed0.json` and contributes no checkpoint or benchmark rows. This was an execution repair, with no change to model/data/budget or test-driven selection.

## Inference and metrics

Final evaluation calls the exact public `separate()` overlap-add path, with 2,048-point Hann analysis/synthesis and 50% overlap. Mirror padding protects record edges; synthesized frames are divided by accumulated squared windows. The reconstructed target plus the unassigned residual equals the input to floating-point precision. No clean reference, label or hidden synthetic metadata is supplied to this API.

Waveform scale-invariant signal-to-distortion ratio (SI-SDR) uses complex projection with a single gain/phase factor, with **no delay search or permutation optimization**. Waveform error-vector magnitude (EVM) is reported separately after least-squares gain/phase alignment. Neither is symbol EVM. Raw normalized mean-squared error (NMSE) is also reported without alignment. Zero estimates have a −120 dB SI-SDR reporting floor.

Synthetic symbol EVM, symbol error rate (SER) and bit error rate (BER) require the generator's actual transmitted reference symbols. They use an RRC receive filter, exact generator carrier/timing and one reference-aided complex gain. The matched-filter comparator already supplies the receive-filtered signal and is not filtered a second time. No equalizer removes multipath, so residual ISI is retained. QPSK bit errors use Gray labels. These are **oracle-synchronized synthetic fidelity metrics**, not a deployable communications receiver or an ability to decode real recordings. Real-data and TorchSig examples lacking valid symbols receive no symbol metrics.

Each SNR/SIR/channel cell reports raw rows, mean metrics and training-seed means. Paired bootstrap intervals resample heldout example IDs after averaging training seeds; they estimate example variation, not the full uncertainty over possible model-training runs. Three seeds provide a small robustness check, not a universal statistical guarantee. CPU timing includes the public inference call and all frames; first model load and cached calls are not artificially separated, so raw runtime rows show that variability.

## Limits and success criteria

Success means a trained model, exact-input scientific instrument and independently generated holdout improvement on a stated task. It does not require every comparison to favor the hybrid. No arbitrary cochannel PSK recovery, protocol recognition, real-message decoding or hardware-domain separation performance is claimed. A finite closed-set modulation label is never itself evidence that the target was correctly separated. Classification abstention and model disagreement are reported separately from clean-reference separation quality.
