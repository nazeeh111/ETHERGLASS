# Data, licensing, and provenance

Inspected 2026-10-25. A repository's code license does not replace a recording's license.

| Material | Terms inspected | Use in ETHERGLASS |
|---|---|---|
| Original generated mixtures and clean references | Project's stated synthetic-data license | Deterministic training/evaluation; no private traffic |
| [TorchSig source](https://github.com/TorchDSP/torchsig/blob/1197261d4f0d4c93de18757ee2ceb368ac04104c/LICENSE.md) | MIT | If code is reused, retain the upstream copyright and permission notice |
| [sigmf-python](https://github.com/sigmf/sigmf-python) | LGPL-3.0-or-later | Unmodified installed dependency used for authoritative metadata schema validation; not copied into project source |
| [RadioPile pipeline](https://github.com/xandervanparys/RadioPile/blob/fd6cb8a8fe824509121e3f85d8a7b153b0e9644a/LICENSE) | Apache-2.0 | Research reference; does not license its source IQ |
| [RadioPile aggregate](https://zenodo.org/records/22876704) | CC-BY-NC-SA-4.0 as released | Not downloaded or redistributed |
| [RadioPile ITS band per-source](https://zenodo.org/records/22877108) | CC-BY-4.0, confirmed by the release's `LICENSES.md` | 40-window excerpt included for independent unsupported-input testing |
| RadioPile NIST Wi-Fi per-source | [NIST data terms](https://www.nist.gov/open/license) | No bulk download; cite NIST and mark modifications if used |
| RadioPile remaining five per-source datasets | CC-BY-NC-SA-4.0 | Not included in default training or redistributed |
| [CAMRAS satellite recordings](https://data.camras.nl/satellites/raw/) | CC-BY-4.0 in page and individual SigMF metadata | Permissive fallback for a bounded real replay excerpt |

## RadioPile source distinctions

The aggregate is approximately 117.46 GB. The separate ITS artifact is 388,327,096 bytes. ITS contains over-the-air recordings with added synthetic AWGN, so call it **hybrid real-plus-synthetic**, not untouched field measurements. Its 90,000 windows have 500 complex samples, 25 Msps metadata, 5.9 GHz center frequency, and six technology labels. Per-window modulation and SNR are not known. Observation identifiers keep sibling windows together; they do not necessarily identify independent recording sessions or physical devices.

For any derived subset, preserve its original window indices, labels, source `recording_id`, `file_path`, metadata, DOI, original citation, and a checksum of the distributed bytes. If no trustworthy session/device identifiers exist, say so and avoid splitting windows into purportedly independent training/test sessions. The initial external sample is reserved for testing, not model fitting or threshold tuning.

Source authors: M. Girmay, V. Maglogiannis, D. Naudts, M. Aslam, A. Shahid, and I. Moerman, “Technology recognition and traffic characterization for wireless technologies in ITS band,” *Vehicular Communications* 39 (2023), 100563, [doi:10.1016/j.vehcom.2022.100563](https://doi.org/10.1016/j.vehcom.2022.100563). RadioPile processing: Xander Vanparys, Adnan Shahid, Eli De Poorter, and Mohammad Cheraghinia, [doi:10.5281/zenodo.22877108](https://doi.org/10.5281/zenodo.22877108). No endorsement is implied.

The upstream [source documentation](https://github.com/xandervanparys/RadioPile/blob/fd6cb8a8fe824509121e3f85d8a7b153b0e9644a/docs/DATASETS.md) says the ITS license was confirmed in writing by the source authors. This project relies on the explicit CC-BY-4.0 grant in the published per-source release, not on the upstream GitLab repository having a generic code license.

## Redistribution rules

Keep attribution and the applicable license beside every real recording. Describe cropping, scaling, format conversion, or injected mixtures. Original project code and model licenses do not relicense third-party bytes. No clean separation truth is claimed for a real recording without independently known source components. Do not report BER/EVM as ground-truth metrics on unlabeled real traffic.

## Verified extraction

A public HTTP-range HDF5 reader fetched 3,538,944 bytes, below its strict 4 MiB
cap, and exported 40 windows across five unsupported technologies. The included
NPZ has shape `(40, 2, 500)` and its SHA256 is
`01cb034734bf0f4729fb2650486be5149e9e2e4bca1f83146f4d7e582894256e`.
The manifest records each original row and observation. The full source's published
MD5 is retained but was not verified because the full artifact was not downloaded.
The earlier attempted six-class extraction exceeded its download cap and exported
nothing; the smaller five-class extraction completed successfully.

See [`examples/real/`](../examples/real/README.md) and
[`scripts/fetch_radiopile_subset.py`](../scripts/fetch_radiopile_subset.py).

## Continuous CAMRAS observation

The first65,536complex samples (262,144bytes) of the public Danuri recording
`camras-2022_08_07_08_31_34_2260.830MHz_0.5Msps_ci16_le` are included under
[CC-BY-4.0](https://creativecommons.org/licenses/by/4.0/), explicitly granted by
[the CAMRAS listing](https://data.camras.nl/satellites/raw/) and original sidecar.
Attribution: **Stichting CAMRAS, Dwingeloo Radio Telescope**, observation dated
2022-08-07. Original metadata and a derived manifest are preserved beside it.
The excerpt retains original interleaved int16 sample bytes; its sidecar is
updated to SigMF1.2.6, a crop description, and a newly calculated checksum.

SHA256: `962a45d420c0a05bbf5cdc7db3f4a1f81a200127028ec84cea41ccafc123b65b`.
The source returned HTTP206 with range0–262143 of43,400,000bytes. Retrieval is
reproducible with `scripts/fetch_camras_excerpt.py`; the full recording was not
downloaded. There is one session, and adjacent windows are correlated.
No clean reference, modulation label, or symbol-rate ground truth is available.
Replay and conditional abstention are measured in `research/external_results.json`,
without model fitting or threshold changes.
