"""Reproducible offline command line entry point."""
from __future__ import annotations
import argparse
import json
from pathlib import Path


def main(argv=None):
    parser = argparse.ArgumentParser(prog="etherglass", description="Offline RF I/Q observatory. No radio hardware or network API required.")
    sub = parser.add_subparsers(dest="command", required=True)
    serve = sub.add_parser("serve", help="Open the loopback-only local observatory")
    serve.add_argument("--port", type=int, default=0, help="Loopback port;0 selects a free port (default)")
    demo = sub.add_parser("demo", help="Analyze an independently seeded reference capture")
    demo.add_argument("--output", type=Path, required=True, help="New directory for SigMF, report and reconstruction exports")
    demo.add_argument("--method", default="hybrid")
    inspect = sub.add_parser("analyze", help="Analyze a SigMF or explicitly described raw I/Q capture")
    inspect.add_argument("path", type=Path)
    inspect.add_argument("--metadata", type=Path)
    inspect.add_argument("--datatype")
    inspect.add_argument("--sample-rate", type=float)
    inspect.add_argument("--center-frequency", type=float)
    inspect.add_argument("--target-center", type=float, default=0)
    inspect.add_argument("--symbol-rate", type=float)
    inspect.add_argument("--method", help="Optional prior-conditioned reconstruction: hybrid, matched, wiener, unet")
    inspect.add_argument("--start", type=int, default=0)
    inspect.add_argument("--stop", type=int)
    inspect.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.command == "serve":
        import uvicorn
        uvicorn.run("etherglass.app:app", host="127.0.0.1", port=args.port)
        return 0
    from .analysis import analyze, demo_capture, public_report
    from .io import load_capture, encode_sigmf, export_zip
    try:
        args.output.mkdir(parents=True, exist_ok=False)
        if args.command == "demo":
            capture, reference = demo_capture()
            result = analyze(capture, symbol_rate=125000, method=args.method, reference=reference)
            data, meta = encode_sigmf(capture.samples, capture.sample_rate, capture.center_frequency,
                                     license=capture.metadata["global"]["core:license"],
                                     provenance=capture.metadata["global"].get("etherglass:provenance"))
            (args.output / "mixture.sigmf-data").write_bytes(data)
            (args.output / "mixture.sigmf-meta").write_text(json.dumps(meta, indent=2))
        else:
            capture = load_capture(args.path, metadata_path=args.metadata, datatype=args.datatype,
                                   sample_rate=args.sample_rate, center_frequency=args.center_frequency)
            result = analyze(capture, start_sample=args.start, stop_sample=args.stop,
                             target_center=args.target_center, symbol_rate=args.symbol_rate, method=args.method)
        report = public_report(result)
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
