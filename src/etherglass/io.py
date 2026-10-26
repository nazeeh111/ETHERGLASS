"""Bounded, explicit-format complex I/Q and SigMF import/export.

Integer I/Q is normalized to [-1,1); floats retain physical input amplitude.
No file paths from metadata are followed. Multi-capture metadata is preserved.
"""
from __future__ import annotations

import hashlib
import io
import json
import math
import zipfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterator

import numpy as np

MAX_BYTES = 64 * 1024 * 1024
FORMATS = {
    "cf32_le": ("<f4", 1.0, 0.0), "cf32_be": (">f4", 1.0, 0.0),
    "cf64_le": ("<f8", 1.0, 0.0), "cf64_be": (">f8", 1.0, 0.0),
    "ci16_le": ("<i2", 32768.0, 0.0), "ci16_be": (">i2", 32768.0, 0.0),
    "ci32_le": ("<i4", 2147483648.0, 0.0), "ci32_be": (">i4", 2147483648.0, 0.0),
    "ci8": ("i1", 128.0, 0.0), "cu8": ("u1", 128.0, 128.0),
}


@dataclass
class Capture:
    samples: np.ndarray
    sample_rate: float
    center_frequency: float
    name: str = "capture"
    metadata: dict = field(default_factory=dict)
    source_sha256: str = ""
    warnings: list[str] = field(default_factory=list)

    def __post_init__(self):
        self.samples = np.asarray(self.samples, dtype=np.complex64)
        if self.samples.ndim != 1 or len(self.samples) < 64:
            raise ValueError("A capture must contain at least 64 complex samples in one dimension.")
        if not np.isfinite(self.samples).all():
            raise ValueError("I/Q samples contain NaN or infinity.")
        if not math.isfinite(self.sample_rate) or not 1e-9 <= self.sample_rate <= 1e15:
            raise ValueError("Sample rate must be finite and between 1e-9 and 1e15 samples/second (supported numerical range).")
        if not math.isfinite(self.center_frequency) or self.center_frequency < 0:
            raise ValueError("Center frequency must be finite and nonnegative (Hz). Use 0 for baseband.")


def decode_iq(data: bytes, datatype: str) -> np.ndarray:
    if datatype not in FORMATS:
        raise ValueError(f"Unsupported I/Q datatype {datatype!r}. Supported: {', '.join(FORMATS)}")
    if not data or len(data) > MAX_BYTES:
        raise ValueError("Capture size must be between 1 byte and 64 MiB.")
    dtype, scale, bias = FORMATS[datatype]
    d = np.dtype(dtype)
    if len(data) % (2 * d.itemsize):
        raise ValueError("Truncated I/Q pair: data length is not a multiple of the complex sample size.")
    # Convert before subtraction, including unsigned input.
    pairs = (np.frombuffer(data, d).reshape(-1, 2).astype(np.float64) - bias) / scale
    x = (pairs[:, 0] + 1j * pairs[:, 1]).astype(np.complex64)
    if not np.isfinite(x).all():
        raise ValueError("Samples cannot be represented as finite complex64 values.")
    return x


def load_bytes(data: bytes, *, metadata: dict | None = None, datatype: str | None = None,
               sample_rate: float | None = None, center_frequency: float | None = None,
               name: str = "capture") -> Capture:
    meta = metadata if metadata is not None else {}
    if not isinstance(meta, dict):
        raise ValueError("SigMF metadata must be a JSON object.")
    try:
        json.dumps(meta, allow_nan=False)
    except (ValueError, TypeError, RecursionError) as e:
        raise ValueError("Metadata must be finite, JSON-compatible data (no NaN or infinity).") from e
    if metadata is not None:
        from sigmf import SigMFFile
        from jsonschema.exceptions import ValidationError
        try:
            SigMFFile(metadata=meta).validate()
        except ValidationError as e:
            field = "/".join(str(part) for part in e.absolute_path) or "root"
            raise ValueError(f"SigMF schema validation failed at {field}. Supply a valid supported sidecar or use explicit raw metadata.") from e
    glob = meta.get("global", {})
    if not isinstance(glob, dict):
        raise ValueError("SigMF global metadata must be an object.")
    extensions = glob.get("core:extensions", [])
    if not isinstance(extensions, list):
        raise ValueError("SigMF extensions must be a list of declarations.")
    for extension in extensions:
        if (not isinstance(extension, dict) or not isinstance(extension.get("name"), str)
                or not isinstance(extension.get("version"), str)
                or not isinstance(extension.get("optional"), bool)):
            raise ValueError("Each SigMF extension needs name, version and boolean optional fields.")
        supported = extension["name"] == "etherglass" and extension["version"] == "0.1.0"
        if not extension["optional"] and not supported:
            raise ValueError(f"Unsupported required SigMF extension: {extension['name']} {extension['version']}")
    datatype = glob.get("core:datatype", datatype)
    sample_rate = glob.get("core:sample_rate", sample_rate)
    if datatype is None or sample_rate is None:
        raise ValueError("Provide an explicit complex datatype and sample rate, or a SigMF metadata file.")
    x = decode_iq(data, datatype)
    captures = meta.get("captures", [])
    if not isinstance(captures, list) or not all(isinstance(c, dict) for c in captures):
        raise ValueError("SigMF captures must be a list of objects.")
    previous = -1
    for c in captures:
        start = c.get("core:sample_start")
        if isinstance(start, bool) or not isinstance(start, int) or not 0 <= start < len(x) or start <= previous:
            raise ValueError("Capture sample starts must be ordered, unique integers within the data.")
        previous = start
        if "core:frequency" in c and (not np.isfinite(c["core:frequency"]) or c["core:frequency"] < 0):
            raise ValueError("Capture frequencies must be finite and nonnegative.")
    if captures and captures[0]["core:sample_start"] != 0:
        raise ValueError("The first SigMF capture must start at sample 0.")
    if glob.get("core:offset", 0) != 0 or glob.get("core:num_channels", 1) != 1:
        raise ValueError("This release supports single-channel SigMF captures with zero sample offset.")
    if glob.get("core:trailing_bytes", 0) or any(c.get("core:header_bytes", 0) for c in captures):
        raise ValueError("Embedded headers and trailing bytes are unsupported. Export a contiguous I/Q dataset first.")
    for annotation in meta.get("annotations", []):
        if not isinstance(annotation, dict):
            raise ValueError("Each SigMF annotation must be an object.")
        start, count = annotation.get("core:sample_start"), annotation.get("core:sample_count", 0)
        if isinstance(start, bool) or not isinstance(start, int) or not 0 <= start < len(x):
            raise ValueError("Annotation start is outside the capture.")
        if isinstance(count, bool) or not isinstance(count, int) or count < 0 or start + count > len(x):
            raise ValueError("Annotation count is outside the capture.")
    center = captures[0].get("core:frequency", center_frequency) if captures else center_frequency
    if center is None:
        raise ValueError("Provide center frequency in Hz (0 for a baseband recording).")
    warnings = []
    if not glob.get("core:license"):
        warnings.append("No source license supplied. Import grants no redistribution rights.")
    if glob.get("core:sha512") and hashlib.sha512(data).hexdigest() != glob["core:sha512"]:
        raise ValueError("SigMF SHA-512 checksum does not match the uploaded data.")
    if len(captures) > 1:
        warnings.append("Multiple tuning segments: analyses must stay within one capture segment.")
    return Capture(x, float(sample_rate), float(center), Path(name).name, meta,
                   hashlib.sha256(data).hexdigest(), warnings)


def load_capture(path: str | Path, *, metadata_path: str | Path | None = None, **kwargs) -> Capture:
    path = Path(path)
    if path.stat().st_size > MAX_BYTES:
        raise ValueError("Capture exceeds 64 MiB. Use iter_iq_blocks for streaming, or select a smaller recording.")
    if metadata_path is None and path.name.endswith(".sigmf-data"):
        candidate = path.with_suffix(".sigmf-meta")
        if candidate.exists():
            metadata_path = candidate
    metadata = None
    if metadata_path is not None:
        p = Path(metadata_path)
        if p.stat().st_size > 1024 * 1024:
            raise ValueError("Metadata exceeds 1 MiB.")
        metadata = json.loads(p.read_text())
    return load_bytes(path.read_bytes(), metadata=metadata, name=path.name, **kwargs)


def iter_iq_blocks(path: str | Path, datatype: str, block_samples: int = 65536) -> Iterator[np.ndarray]:
    """Read arbitrarily large recordings without loading them in full (no overlap)."""
    if datatype not in FORMATS or not isinstance(block_samples, int) or block_samples < 1 or block_samples > 1_000_000:
        raise ValueError("Use a supported datatype and block_samples between 1 and 1,000,000.")
    sample_bytes = np.dtype(FORMATS[datatype][0]).itemsize * 2
    if Path(path).stat().st_size % sample_bytes:
        raise ValueError("Recording ends in an incomplete complex sample.")
    with open(path, "rb") as f:
        while data := f.read(block_samples * sample_bytes):
            yield decode_iq(data, datatype)


def encode_sigmf(samples: np.ndarray, sample_rate: float, center_frequency: float,
                 *, description: str = "ETHERGLASS analysis", license: str | None = None,
                 provenance: dict | None = None) -> tuple[bytes, dict]:
    cap = Capture(samples, sample_rate, center_frequency)
    pairs = np.column_stack((cap.samples.real, cap.samples.imag)).astype("<f4")
    data = pairs.tobytes()
    glob = {"core:datatype": "cf32_le", "core:sample_rate": sample_rate,
            "core:version": "1.2.6", "core:description": description,
            "core:sha512": hashlib.sha512(data).hexdigest(),
            "core:recorder": "ETHERGLASS 0.1.0"}
    if license:
        glob["core:license"] = license
    # Custom extension metadata is declared optional, never required to read I/Q.
    if provenance:
        glob["core:extensions"] = [{"name": "etherglass", "version": "0.1.0", "optional": True}]
        glob["etherglass:provenance"] = provenance
    return data, {"global": glob, "captures": [{"core:sample_start": 0, "core:frequency": center_frequency}], "annotations": []}


def export_zip(samples: np.ndarray, sample_rate: float, center_frequency: float,
               report: dict, *, license: str | None = None) -> bytes:
    provenance = {k: report[k] for k in ("source_sha256", "selection", "separation_method") if k in report}
    source = report.get("provenance", {})
    if source:
        provenance["source"] = source
    data, meta = encode_sigmf(samples, sample_rate, center_frequency, license=license,
                            description=f"ETHERGLASS derived {report.get('exported_component', 'capture')}; source and transformations retained in provenance",
                            provenance=provenance)
    original_global = source.get("metadata", {}).get("global", {}) if isinstance(source, dict) else {}
    if original_global.get("core:author"):
        meta["global"]["core:author"] = original_global["core:author"]
    out = io.BytesIO()
    with zipfile.ZipFile(out, "w", zipfile.ZIP_DEFLATED) as z:
        z.writestr("capture.sigmf-data", data)
        z.writestr("capture.sigmf-meta", json.dumps(meta, indent=2, allow_nan=False))
        z.writestr("analysis.json", json.dumps(report, indent=2, allow_nan=False))
    return out.getvalue()
