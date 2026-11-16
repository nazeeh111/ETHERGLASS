# Current observatory capture

`observatory-current.png` is an unaltered local browser capture of the unreleased
0.1.1.dev0 observatory, recorded on 2026-11-16 at 1280 × 844 pixels.

The reference action generated the bundled synthetic QPSK target with overlapping
chirp interference using seed **870001**: **16,384 samples**, **1 MHz** sample rate,
**0 Hz** target offset and a supplied **125,000-baud** prior. The completed method
was **Hybrid**. These priors were supplied for the simulated example, not inferred
from a physical transmitter. The plots show the actual mixture and reconstruction.

This is one reference observation, not an aggregate benchmark. The ensemble
readout is a disagreement indicator. It does not issue a calibrated error bound
for this 16,384-sample recording or a selected crop. The separate generated-only
bound is restricted to internally drawn full 4,096-sample examples, its fixed
models, exact admitted runtime and declared simulator distribution. Its nominal
90% marginal interpretation depends on exchangeability with that distribution;
it is not a per-frame guarantee, physical RF result or identifiability proof.

## Byte identities

| Artifact | SHA-256 |
| --- | --- |
| [Capture](observatory-current.png) | `3c0ffbafc9b8b9e1cbf4287ef089f70d9b1c403813cbb1cdfa56311cb7fae76e` |
| [UI JavaScript](../src/etherglass/web/app.js) | `aa0eb2e32f3187b9d4df0df2ee3adfd94c60fc555001abff3536b4c57fa46059` |
| [Stylesheet](../src/etherglass/web/style.css) | `375817176a8bd3b72291b6b1e0c437eb528f1e1311ab9c545b598f158454cec6` |
| [Synthetic implementation binding](../src/etherglass/resources/models/synthetic-bound-implementation-v3.json) | `35dfd5aa65e8834cbdef94e171b2354f3d33e811b2f510e60102c891ae8ae433` |

The binding identifies the separately reviewed numerical implementation; it does
not expand the eligibility of the captured reference recording. Historical
captures, models and study records remain under their original identities.
