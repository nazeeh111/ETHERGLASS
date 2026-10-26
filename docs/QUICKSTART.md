# From samples to evidence

ETHERGLASS runs entirely on a CPU. Python 3.12 is the reference environment. A GPU, an SDR, a network model service, and an API key are unnecessary. The checked-in models were actually trained; opening the observatory does not train them again.

```sh
git clone https://github.com/nazeeh111/ETHERGLASS.git
cd ETHERGLASS
python3.12 -m venv .venv
source .venv/bin/activate
# Linux CPU-only installations can install torch first from its CPU wheel index:
# pip install torch==2.9.1 --index-url https://download.pytorch.org/whl/cpu
pip install -e '.[test]'
pytest
etherglass serve
```

Open the loopback address printed in the terminal. If the port is occupied, choose another port or `--port 0` to let the operating system select a free one. The server binds to `127.0.0.1` and stores up to four captures in memory. Restarting clears the session. It does not upload recordings anywhere.

## The reference observation

1. Select **Load reference capture**. This generates a fixed, independently seeded QPSK target with overlapping chirp interference. The raw traces come from those samples.
2. The target is centered at baseband 0 Hz with a symbol rate of 125,000 baud in a 1 Msps recording. These are explicit simulation priors, not automatically discovered protocol properties.
3. Choose **Hybrid** and **Separate selected interval**. Switch between mixture, recovered target, and residual. Spectrum, complex envelope, and raw I/Q cloud update together.
4. Drag within the waterfall to choose a time interval and target offset. Numeric start/stop controls provide the same operation without a pointer. **Apply** refreshes the selection; **Reset** returns to the complete recording.
5. Inspect the evidence panel. Classification can abstain. The reconstruction log includes runtime, consistency, and reference metrics only for the synthetic reference capture.
6. **Export evidence** downloads a ZIP containing I/Q data, SigMF metadata, and the exact analysis report for the current selection/component.

The scatter plot contains unsynchronized samples. Its shape is not symbol EVM. The residual includes interference, noise, and target reconstruction errors.

## Your own recording

Supply a `.sigmf-data` file with its `.sigmf-meta` sidecar, or raw interleaved complex samples plus an explicit sample rate, center frequency, and encoding. Enter RF center 0 for a baseband simulation. Raw input follows `I0,Q0,I1,Q1,…`; it is not a mono WAV file.

Supported encodings are `cf32_le`, `cf32_be`, `cf64_le`, `cf64_be`, `ci16_le`, `ci16_be`, `ci32_le`, `ci32_be`, `ci8`, and `cu8` through the Python API/CLI. Integer samples are normalized by the signed full-scale magnitude; unsigned 8-bit samples are centered at 128. Floating samples retain amplitude. The observatory exposes the most common encodings.

```sh
etherglass analyze observation.sigmf-data --output result
etherglass analyze observation.iq --datatype ci16_le --sample-rate 1000000 \
  --center-frequency 915000000 --target-center 0 --symbol-rate 125000 \
  --method hybrid --output reconstruction
etherglass demo --output reference-run
```

Output directories must be new. This protects previous reports from accidental replacement. Captures are capped at 64 MiB in the interactive app. For larger recordings, the reusable `iter_iq_blocks` API reads bounded chunks; model separation operates in overlapped frames. Streaming file iteration and offline replay are supported; live SDR acquisition is not implemented in this release.

## Interoperability and limits

SigMF export uses `cf32_le`, a SHA-512 checksum, sample rate, center frequency, optional source license, and optional ETHERGLASS provenance. The original input's SHA-256 and crop bounds accompany the report. Multi-tune captures must be analyzed one tuning segment at a time. Multiple channels, embedded headers, trailing bytes, and nonzero SigMF offsets are rejected explicitly.

Exported data can be opened in SigMF-aware tools or a GNU Radio File Source configured for complex float32 at the supplied sample rate. The API does not claim a tested live GNU Radio or hardware integration.

The separator uses a known pulse-shape/rate prior. Unknown wideband and arbitrary overlapping digital inputs remain outside its validated domain. No messages are recovered. Imported recordings do not acquire redistribution permission through analysis; keep their original license and attribution.

Numerical input support is 1e-9 to 1e15 samples/second. Values outside this broad range are rejected before plotting or duration calculations can overflow. This numeric range does not extend the model's scientifically validated domain.
