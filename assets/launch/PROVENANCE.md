# Launch source and provenance

This is an edited launch film with real screenshots and measured analytical graphics. It is not a continuous screen recording.

- Product source: `../../src/etherglass/web/index.html`, `style.css`, `app.js`.
- Analytical source: `composition/assets/reference-vis.json`. Fixed synthetic QPSK + overlapping chirp, seed 870001; 16,384 complex samples; 1 MHz sample rate. Source SHA-256: `68dab3f2d14d693bc000e0c4f6cd627a14fe9b8e250b09a2a395d845eb521186`.
- Model: trained hybrid seed 0, 6,865 parameters. The bundled report describes its known-rate PSK target prior and sparse tone/chirp scope.
- Metric: mixture SI-SDR −6.167652653863637 dB; recovered target 15.553494789961997 dB; difference +21.721147443825636 dB. Clean received target reference with complex scalar alignment; no delay search. One synthetic reference demonstration, not held-out aggregate performance or universal recovery evidence.
- PSD graphics: directly derived from the report's 512 actual frequency/power values. All three plots share −500…+500 kHz horizontal and −105…−30 dB vertical axes. Values below −105 dB are clipped to the bottom display boundary.
- Waterfall: actual 127×256 time/frequency power matrix, colored with the source UI's 4th/99.5th percentile scale and exact six-stop map. The hook uses a visual crop, without quantitative axes.
- UI screenshots: actual running product loaded with the fixed synthetic reference. Export dialog shown is actual product UI. See `qa/verification.md` for final checks.
- Audio: `scripts/generate_assets.py` writes an original deterministic sine-based ambient soundtrack. It is composed sound design, not demodulated or recorded radio. No voice, stock music, bundled music, purchased assets or paid APIs.

## Rebuild

`python3 scripts/generate_assets.py` regenerates the measured graphics and original audio using Python's standard library. The JSON contains their measured source values. Composition is editable in `composition/index.html`. The screenshot files are preserved as source images.

Use a compatible installed HyperFrames runtime, e.g. `python3 /path/to/local.py lint composition --json`, then `check composition --json --samples 9`, then `render composition --output launch-source.mp4 --fps 30 --quality standard --workers 1`. No network access is needed at render time. Root composition dimensions are 1280×720; duration 20 seconds.

The original render is retained as `launch-source.mp4`. `brag.jpg` is extracted from a settled selected hero frame. `brag.mp4` replaces only the first video frame with this poster, preserving audio and duration. See `scripts/finish_video.py` for the exact local FFmpeg commands.

## Rights

Project-created film, graphics, screenshots and scripts follow the repository's MIT license. The original generated `instrument-score.wav` is dedicated to CC0 1.0: https://creativecommons.org/publicdomain/zero/1.0/. Its generator remains MIT. Synthetic source data is CC0. System fonts are requested by name and are not redistributed.

The separately licensed GSAP 3.14.2 browser runtime retains its upstream copyright header and is **not covered by the repository MIT license**. Its upstream license is the [GSAP standard no-charge license](https://gsap.com/standard-license/), as identified by the [official repository](https://github.com/greensock/GSAP) and npm package. Pinned source: `https://cdn.jsdelivr.net/npm/gsap@3.14.2/dist/gsap.min.js`. This dependency is only used by the editable launch-film composition. The rendered film does not execute or embed it.
