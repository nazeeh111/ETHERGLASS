#!/usr/bin/env python3
"""Write the release scientific report directly from frozen experiment outputs."""
import json
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]


def main():
    r=json.loads((ROOT/'research'/'results.json').read_text())
    c=json.loads((ROOT/'models'/'classifier_metrics.json').read_text())
    u=json.loads((ROOT/'research'/'uncertainty_results.json').read_text())
    external=json.loads((ROOT/'research'/'external_results.json').read_text())
    def score(cell,method):return r['methods'][cell][method]['si_sdr_db']
    lines=['# Model card and scientific report','',
      'ETHERGLASS v0.1.0 is a compact, prior-conditioned RF interference-suppression study and offline research instrument. It reconstructs one received, root-raised-cosine BPSK/QPSK target from sparse narrowband interference. It is not a blind general-purpose source separator or a real-message decoder.','',
      '## Release evidence','',
      f"The shifted holdout contains {r['examples_per_cell']} independently generated records, evaluated with three training seeds and the exact public overlap-add inference path. Mean hybrid waveform SI-SDR is **{score('shift','hybrid'):.2f} dB**, compared with **{score('shift','matched'):.2f} dB** for matched filtering, **{score('shift','wiener'):.2f} dB** for robust spectral Wiener, **{score('shift','notch'):.2f} dB** for adaptive notch, and **{score('shift','unet'):.2f} dB** for the small spectral U-Net. Higher SI-SDR means better target reconstruction under the stated complex-scalar alignment convention.",'',
      'These are synthetic-population results. The same model is explicitly challenged on cochannel digital mixtures and mismatched symbol rates. External recordings have no clean reference; no real-world separation score is claimed.','',
      '## Controlled comparison','',
      '| Method | Nominal SI-SDR / dB | Shift SI-SDR / dB | Shift waveform EVM / % | Shift mean inference / ms |',
      '|---|---:|---:|---:|---:|']
    for m in ['mixture','bandpass','matched','wiener','notch','unet','pretrained','scratch','no_dsp','hybrid']:
        s=r['methods']['shift'][m]
        lines.append(f"| {m} | {score('nominal',m):.2f} | {s['si_sdr_db']:.2f} | {s['waveform_evm_pct']:.2f} | {s['runtime_ms']:.2f} |")
    lines+=['','Classical methods run once per generated input; learned results average seeds0/1/2. Runtime is observed on the recorded CPU environment, with model-load/cache variability retained in raw rows. The U-Net is a local spectral encoder/decoder comparator with the same input features, support prior and training budgets. It is not the RF Challenge paper\'s time-domain model.','',
      '## Hypothesis and ablations','',
      'The predefined hypothesis compares stationary-tone pretraining followed by160chirp/channel adaptation updates against the identical randomly initialized model given the same160adaptation updates. Pretraining adds320updates and1,536examples, so the comparison matches adaptation budget, not total training cost. The `no_dsp` ablation removes only multiscale log-contrast channels; whitening and the analytic target-support prior remain.','',
      '| Shift comparison | Mean SI-SDR difference / dB | Paired-example bootstrap 95% interval / dB |',
      '|---|---:|---|']
    for key,value in r['comparisons'].items():
        if key.startswith('shift:'):
            lo,hi=value['paired_example_bootstrap_95_ci_db']
            lines.append(f"| {key.split(':')[1]} | {value['mean_si_sdr_delta_db']:+.2f} | [{lo:+.3f}, {hi:+.3f}] |")
    lines+=['','Bootstrap intervals resample held-out example IDs after averaging training seeds. They describe example variation, not the population uncertainty over all possible training runs. Three seeds are a small robustness check.','',
      f"Adaptation changes nominal-tone SI-SDR from {score('nominal','pretrained'):.2f} to {score('nominal','hybrid'):.2f} dB, while changing shifted-chirp SI-SDR from {score('shift','pretrained'):.2f} to {score('shift','hybrid'):.2f} dB. Both sides of this distribution tradeoff are retained.",'',
      '## Stress tests and independent generator','',
      '| Condition | Hybrid SI-SDR / dB | Notch SI-SDR / dB | Small U-Net SI-SDR / dB |',
      '|---|---:|---:|---:|']
    for cell in ['torchsig_independent_generator','overlapping_digital_stress','rate_6_samples_per_symbol','rate_10_samples_per_symbol','snr_0','sir_-15']:
        if cell in r['methods']:
            lines.append(f"| {cell} | {score(cell,'hybrid'):.2f} | {score(cell,'notch'):.2f} | {score(cell,'unet'):.2f} |")
    lines+=['','TorchSig is a different pinned generator implementation, not an independent hardware experiment. All methods receive the same target-center and symbol-rate priors. A single sensor cannot uniquely decompose arbitrary statistically similar cochannel sources. The model\'s nonnegative spectral gain cannot recreate information that interference has made unidentifiable.','',
      '## Metrics and physical assumptions','',
      'SI-SDR uses complex projection with one gain/phase factor, without delay or permutation search. Waveform EVM is least-squares complex-scalar-aligned waveform error, not symbol EVM. Raw NMSE is unaligned. Symbol EVM/SER/BER appear only when the original synthetic symbol sequence is valid. They use exact generator carrier/timing plus reference-aided gain; these oracle-synchronized fidelity metrics are not a deployable receiver. Multipath is not oracle equalized. TorchSig and real inputs without valid symbols receive no symbol metrics.','',
      '## Trained recognition and abstention','',
      f"A separate portable physical-feature classifier was trained on2,000known and2,000unknown examples, with disjoint probability/threshold calibration. Final untouched tests contain {c['known_test_n']}known BPSK/QPSK and {c['unknown_test_n']}unknown probes. Known accuracy before abstention is {100*c['known_class_accuracy_before_abstention']:.2f}%, known coverage {100*c['known_coverage']:.2f}%, and unknown false acceptance {100*c['unknown_false_accept_rate']:.2f}% (Wilson95% upper bound {100*c['unknown_false_accept_wilson95'][1]:.3f}%). Support AUROC is {c['unknown_detection_auroc']:.4f}.",'',
      f"Known-class Brier score: {c['known_brier_score']:.6g}; known-class10-bin ECE: {c['known_binary_ece_10_bins']:.6g}; balanced known/unknown support Brier score: {c['known_support_brier_score']:.6g}. These easy, finite synthetic probes do not establish open-world RF recognition. They include two test-only unknown families (AM and32QAM); raw rows and feature arrays are retained. Earlier exploratory seed ranges are excluded and recorded in model metadata.",'',
      'Inference uses exactly the first2,048samples and a supplied symbol-rate prior. The supported calibration rates are6/8/10/12samples per symbol, rolloff0.25–0.45, and SNR8–30dB, with shared channel distributions across families. Missing prior, insufficient samples, unsupported rate, insufficient support, or conditional class confidence below0.8 returns unknown. Confidence is conditional on BPSK/QPSK support; the support probability depends on the balanced probe prior. The exact fit-time classifier source is retained in research/provenance; post-fit runtime artifact-routing/cache corrections leave the physical feature functions, coefficients, calibration and measured test rows unchanged. Fit and runtime hashes are recorded separately.','',
      '## Separation uncertainty','',
      f"Three training seeds provide actual reconstruction disagreement. A separate split-calibration population ({u['model']['calibration_examples']}examples, seeds900000onward) sets a nominal90% upper bound on raw received-target waveform RMS error. On {u['test_examples']}untouched same-distribution examples, observed coverage is **{100*u['observed_coverage']:.2f}%**. Mean RMS bound is {u['mean_rms_upper']:.4f}input units; mean actual RMS error {u['mean_actual_rms_error']:.4f}.",'',
      'The bound is marginal and conditional on the declared synthetic chirp/channel distribution. It has no coverage guarantee for arbitrary user recordings, unknown families, other rates, or real hardware. Correlated model bias can yield low disagreement while every seed is wrong. The exact calibration rank, checkpoint hashes and per-example outcomes are retained.','',
      '## Independent recordings','',
      f"All {external['radiopile']['windows']}RadioPile ITS windows produced finite measured reports. Each has500samples, below the2,048sample classifier requirement, so this checks replay and short-record abstention only. The continuous CAMRAS excerpt has {external.get('camras',{}).get('samples',0)}samples from one session. The stated8samples/symbol analysis prior is a hypothesis, not source ground truth. Its {len(external.get('camras',{}).get('classification_windows',[]))}adjacent windows produced {external.get('camras',{}).get('accepted_windows','unreported')}accepted classifications. That conditional abstention result is not real modulation accuracy or separation validation.",'',
      'No real input was used for training, calibration, threshold tuning or source-domain modulation labels. Source IDs do not establish device/session independence; neighboring windows are correlated. Licenses, transformations, original identifiers and checksums are retained.','',
      '## Model, compute and reproducibility','',
      'The proposed model has6,865parameters; the spectral U-Net27,121. Training uses batch 24, AdamW, gradient clipping at 5, 320 pretraining updates and 160 adaptation updates. Seed0is the predetermined release checkpoint. CPU inference uses a mathematically equivalent unfold/GEMM convolution path, with native PyTorch fallback. Output and input/weight/bias-gradient equivalence tests cover dilations1/2/4/8; floating-point reduction order can differ. Completed native seed0hybrid/ablation artifacts were preserved; an incomplete native U-Net attempt was stopped and restarted from its original initialization at full budget using GEMM. The aborted attempt is documented and contributes no selected checkpoint.','',
      'Training configuration, exact per-stage logs, source/checkpoint hashes, raw evaluation rows, classifier features, licensed source provenance, runtime environment, scripts and editable figure sources accompany the release. Reproduce in a copy to preserve original artifacts. See [protocol](../research/PROTOCOL.md), [architecture](ARCHITECTURE.md) and [quickstart](QUICKSTART.md).','',
      '## Intended use and limits','',
      'Use for offline RF interference experiments, reproducible teaching and transparent I/Q inspection. Live SDR acquisition, transmit functions, private-message recovery, arbitrary source identification and universal decoding are absent. Original code/weights are MIT, original synthetic data CC0, and attributed external recordings retain CC-BY-4.0. This first release establishes a narrow synthetic-domain contribution and an actual-input instrument; real-world separation efficacy remains unestablished.','']
    import re
    prose='\n'.join(lines)
    replacements={
        'seeds0/1/2':'seeds 0/1/2','by160chirp':'by 160 chirp','same160adaptation':'same 160 adaptation',
        'adds320updates and1,536examples':'adds 320 updates and 1,536 examples',
        'on2,000known and2,000unknown':'on 2,000 known and 2,000 unknown',
        'Wilson95%':'Wilson 95%','known-class10-bin':'known-class 10-bin','and32QAM':'and 32-QAM',
        'first2,048samples':'first 2,048 samples','are6/8/10/12samples':'are 6/8/10/12 samples',
        'rolloff0.25':'rolloff 0.25','SNR8–30dB':'SNR 8–30 dB','below0.8':'below 0.8',
        'seeds900000onward':'seeds 900000 onward','nominal90%':'nominal 90%',
        '40RadioPile':'40 RadioPile','has500samples':'has 500 samples','the2,048sample':'the 2,048-sample',
        'stated8samples/symbol':'stated 8 samples/symbol','has6,865parameters':'has 6,865 parameters',
        'U-Net27,121':'U-Net 27,121','Seed0is':'Seed 0 is','seed0hybrid':'seed 0 hybrid',
        'dilations1/2/4/8':'dilations 1/2/4/8'
    }
    for old,new in replacements.items():prose=prose.replace(old,new)
    prose=re.sub(r'(\d)(examples|untouched|samples|adjacent|accepted|input units|known|unknown)(?=\b)',r'\1 \2',prose)
    (ROOT/'docs'/'MODEL_CARD.md').write_text(prose)

    print('Wrote scientific report from actual measured JSON artifacts.')


if __name__=='__main__':main()
