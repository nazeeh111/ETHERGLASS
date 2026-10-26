# Independent real-input excerpt

`radiopile-its-40.npz` contains 40 windows (8 each of LTE, 5G NR, ITS-G5,
C-V2X PC5, and WiFi) selected from the RadioPile v1.0 ITS band per-source file.
Each row has 2 × 500 float32 values (I, then Q), 25 Msps, 5.9 GHz center.
Rows are separate windows; **do not concatenate them as a continuous capture**.
The original data are over-the-air recordings augmented with synthetic AWGN.
No clean components, reference symbols, or per-window modulation labels are known.

License: [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/).
Attribution: M. Girmay, V. Maglogiannis, D. Naudts, M. Aslam, A. Shahid,
and I. Moerman, *Technology recognition and traffic characterization for wireless
technologies in ITS band*, Vehicular Communications 39 (2023), 100563,
[doi:10.1016/j.vehcom.2022.100563](https://doi.org/10.1016/j.vehcom.2022.100563).
Processed by Xander Vanparys, Adnan Shahid, Eli De Poorter, and Mohammad
Cheraghinia, RadioPile v1.0, [doi:10.5281/zenodo.22877108](https://doi.org/10.5281/zenodo.22877108).

Modifications: selected a small convenience subset of unit-power-normalized
RadioPile windows and saved them as NPZ. No further signal alteration.
See `radiopile-its-40.manifest.json` for original row/observation identifiers,
source attributes, the exact download boundary and exported SHA256.
`RADIOPILE_LICENSES.md` is the release's original per-source license statement;
only the ITS band source is present in this folder.

Re-extract from the public release, within a strict 4 MiB HTTP-range budget:

```sh
python scripts/fetch_radiopile_subset.py --output examples/real
```

The full source's published MD5 is recorded but was **not verified** by downloading
the entire source. The included subset checksum was verified locally. This is an
external stability/unknown-input convenience test, not real-world separation
validation or a representative technology classifier benchmark. Correlated sibling
windows share an observation ID and must never be split across training and test.

## Continuous CAMRAS excerpt

`camras-danuri-65536.sigmf-data` and its sidecar retain the first65,536complex
int16samples of a public Danuri observation from **Stichting CAMRAS, Dwingeloo
Radio Telescope**, dated2022-08-07. License: **CC-BY-4.0**. Source and original
metadata: [CAMRAS raw satellite recordings](https://data.camras.nl/satellites/raw/).
The original bytes are cropped only; the derived sidecar records the crop and
new checksum. `camras-danuri.manifest.json` and `camras-original-metadata.json`
preserve attribution and provenance. Rebuild with `scripts/fetch_camras_excerpt.py`.

This is one session, not32 independent recordings. The true modulation and symbol
rate are unknown, and there is no clean target reference. Analysis with a stated
rate hypothesis tests numerical stability and conditional abstention. It does
not establish real-world separation quality or modulation-classification accuracy.
