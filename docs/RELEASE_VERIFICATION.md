# Release verification

Version: 0.1.0. Prepared 2026-10-26.

## Executed local checks

- **51 tests passed** on Python 3.12.15 / macOS arm64. One non-failing Starlette/httpx deprecation warning remains in the test harness.
- Independent review reproduced every published aggregate and all 14 paired bootstrap intervals from 8,640 CSV rows: 432 generated records across 18 conditions. All 15 checkpoint hashes and the configuration hash matched.
- Classifier fit-source snapshots and separate runtime hashes matched. Saved final test features reproduce the reported outcomes.
- Uncertainty calibration uses order statistic 88 of 96. Source/weight hashes matched; untouched same-distribution coverage was 93/96, without a real-RF guarantee.
- A fresh isolated wheel installation executed outside the source checkout. Bundled weights, three ensemble seeds, calibration, web files and the TorchSig license were present. The actual CPU demo improved SI-SDR by 21.721 dB on its fixed synthetic reference.
- Browser checks against the installed wheel passed actual SigMF/sidecar and raw-int16 uploads, source hashes and metadata, reconstruction, trace-specific classification, all three SigMF/report exports, crop/reset, pointer selection and 390px mobile layout. No page errors or horizontal mobile overflow were observed. The editable architecture diagram was rendered and inspected.
- The 20-second launch film passed full FFmpeg decode and visual frame inspection. Its original generated score received level checks; no auditory review is claimed.

`research/ui-verification.json` and `assets/launch/qa/` retain evidence. `ARTIFACTS.json` records exact source, model, recording, evidence and media bytes. Wheel/source builds and all Markdown local-link checks succeeded.

## Public release status

The user directly approved committing and publicly publishing this complete payload to `nazeeh111/ETHERGLASS`, including weights, licensed samples, benchmark evidence, documentation, demo and release `v0.1.0`, on 2026-10-26. An earlier automatic approval rejection was resolved through that scoped confirmation; it was not bypassed.

The [Linux CPU workflow](https://github.com/nazeeh111/ETHERGLASS/actions/workflows/ci.yml) runs installation, regression tests, artifact verification and the actual demo. Final remote-CI, tag, fresh-clone and release-download outcomes are recorded in `verification-results.json` attached to the [versioned release](https://github.com/nazeeh111/ETHERGLASS/releases/tag/v0.1.0) after execution. Configured or unexecuted checks are not counted as passes.
