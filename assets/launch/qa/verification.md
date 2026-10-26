# Final launch-film verification

Final files: `../brag.mp4`, `../brag.jpg`; public copies `../../launch.mp4`, `../../launch.jpg` are byte-identical. See verification.json for exact SHA-256 hashes.

- 1280×720 landscape, 30 fps, 600 video frames, exactly 20 seconds.
- H.264 video and AAC stereo audio, 48 kHz, yuv420p, fast-start MP4.
- Full FFmpeg decode passed without errors.
- Poster selected at settled hook time 1.8 seconds; baked into frame zero. Mean RGB pixel difference from JPEG poster to decoded frame zero is recorded in verification.json; small encoding differences are expected.
- Current-source HyperFrames lint, runtime, layout and contrast checks pass. Nine settled timestamps checked; additional transition samples were inspected. Zero errors, zero contrast warnings. Nine layout warnings refer only to identical mastheads deliberately overlapping for 125 milliseconds during three 0.45-second crossfades.
- Automated motion module was disabled in the installed runner. No automated motion pass is claimed. Four settled scenes and three rendered crossfade samples were visually inspected in the final MP4. Selected settled rendered frames are retained here.
- Local system-font aliases avoid remote font fallback/download in final check and render. No font binaries are bundled.
- Original composed ambient score is intentionally restrained: mean −33.3 dB, peak −24.5 dB after encoding, no clipping. The score is not RF audio. No auditory review was performed; level and signal checks were performed.
- First UI scene uses the actual loaded reference capture before reconstruction. Last scene shows the actual export dialog after reconstruction. The film is an edited demonstration with still captures and measured plots, not an uninterrupted screen recording.
- Numerical evidence is a single fixed synthetic reference demonstration. It is explicitly labeled on screen; not a held-out benchmark or general claim. Actual source values and alignment conditions are in `../PROVENANCE.md` and `../composition/assets/reference-vis.json`.

Raw successful render is preserved as `../launch-source.mp4`. Failed/interrupted diagnostic runs, temporary transition frames, and raw browser/render logs are retained outside public outputs in the task's `work/launch-qa` directory. No Git commits, remote writes or publication were performed by the launch-film worker.
