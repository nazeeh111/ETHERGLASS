"""Reproducible offline command line entry point."""
from __future__ import annotations
import argparse
import json
from pathlib import Path


def main(argv=None):
    from . import __version__
    from .separation import METHODS

    parser = argparse.ArgumentParser(prog="etherglass", description="Offline RF I/Q observatory. No radio hardware or network API required.")
    parser.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    sub = parser.add_subparsers(dest="command", required=True)
    serve = sub.add_parser("serve", help="Serve the loopback-only local observatory")
    serve.add_argument("--port", type=int, default=0, help="Loopback port;0 selects a free port (default)")
    demo = sub.add_parser("demo", help="Analyze an independently seeded reference capture")
    demo.add_argument("--output", type=Path, required=True, help="New directory for SigMF, report and reconstruction exports")
    demo.add_argument("--method", choices=METHODS, default="bandpass",
                      help="Prior-conditioned estimator (default: bandpass; neural methods need [neural])")
    bounded = sub.add_parser("synthetic-bound-example", help="Draw one internally generated full-frame synthetic bound example")
    bounded.add_argument("--output", type=Path, required=True, help="New directory for a synthetic bound report")
    inspect = sub.add_parser("analyze", help="Analyze a SigMF or explicitly described raw I/Q capture")
    inspect.add_argument("path", type=Path)
    inspect.add_argument("--metadata", type=Path, help="SigMF sidecar; its supplied fields take precedence over raw fallback fields")
    inspect.add_argument("--datatype", help="Raw complex encoding, for example cf32_le or ci16_le")
    inspect.add_argument("--sample-rate", type=float, help="Raw sample rate in Hz")
    inspect.add_argument("--center-frequency", type=float, help="Raw RF center in Hz; use 0 for baseband")
    inspect.add_argument("--target-center", type=float, default=0, help="Target offset from RF center in Hz (default: 0)")
    inspect.add_argument("--symbol-rate", type=float, help="Supplied target prior in baud, required for reconstruction")
    inspect.add_argument("--method", choices=METHODS, help="Optional prior-conditioned reconstruction")
    inspect.add_argument("--start", type=int, default=0, help="Inclusive sample offset in the original recording")
    inspect.add_argument("--stop", type=int, help="Exclusive sample offset in the original recording")
    inspect.add_argument("--output", type=Path, required=True, help="New directory; existing outputs are preserved")
    args = parser.parse_args(argv)
    if args.command == "serve":
        try:
            import uvicorn
            from .app import app
        except ModuleNotFoundError as error:
            if error.name not in {"uvicorn", "fastapi", "starlette", "pydantic", "multipart"}:
                raise
            parser.error("The local observatory needs optional web dependencies. Install 'etherglass[web]'.")
        uvicorn.run(app, host="127.0.0.1", port=args.port)
        return 0
    if args.output.exists():
        parser.error(f"Output directory already exists: {args.output}. Choose a new directory.")
    if args.command == "synthetic-bound-example":
        from .calibrated_example import generated_example
        try:
            result = generated_example()
            args.output.mkdir(parents=True, exist_ok=False)
            (args.output / "synthetic-bound.json").write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
            print(json.dumps(result, indent=2, allow_nan=False))
            return 0
        except (ValueError, FileExistsError, FileNotFoundError) as e:
            parser.error(str(e))
    from .analysis import analyze, demo_capture, public_report
    from .io import load_capture, encode_sigmf, export_zip
    try:
        if args.command == "demo":
            capture, reference = demo_capture()
            result = analyze(capture, symbol_rate=125000, method=args.method, reference=reference)
            data, meta = encode_sigmf(capture.samples, capture.sample_rate, capture.center_frequency,
                                     license=capture.metadata["global"]["core:license"],
                                     provenance=capture.metadata["global"].get("etherglass:provenance"))

        else:
            capture = load_capture(args.path, metadata_path=args.metadata, datatype=args.datatype,
                                   sample_rate=args.sample_rate, center_frequency=args.center_frequency)
            result = analyze(capture, start_sample=args.start, stop_sample=args.stop,
                             target_center=args.target_center, symbol_rate=args.symbol_rate, method=args.method)
        report = public_report(result)
        args.output.mkdir(parents=True, exist_ok=False)
        if args.command == "demo":
            (args.output / "mixture.sigmf-data").write_bytes(data)
            (args.output / "mixture.sigmf-meta").write_text(json.dumps(meta, indent=2))
        (args.output / "analysis.json").write_text(json.dumps(report, indent=2, allow_nan=False))
        if result["separation"]:
            for i, component in enumerate(result["_arrays"]):
                name = "target" if i == 0 else "residual"
                if isinstance(component, dict):
                    component = component.get("samples", component.get("signal"))
                (args.output / f"{name}.zip").write_bytes(export_zip(component, capture.sample_rate,
                    result["center_frequency"], {**report, "exported_component": name},
                    license=capture.metadata.get("global", {}).get("core:license")))
        print(json.dumps({"output": str(args.output.resolve()), "samples": result["num_samples"],
                          "classification": result["classification"],
                          "separation": result["separation"]["diagnostics"] if result["separation"] else None}, indent=2))
        return 0
    except (ValueError, FileExistsError, FileNotFoundError) as e:
        parser.error(str(e))


if __name__ == "__main__":
    main()
