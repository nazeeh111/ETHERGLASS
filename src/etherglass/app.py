"""Loopback-only offline observatory. No accounts, telemetry, radio or paid APIs."""
from __future__ import annotations

import json
import threading
import uuid
from collections import OrderedDict
from pathlib import Path

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.exceptions import RequestValidationError
from fastapi.responses import FileResponse, Response, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, ConfigDict, Field
from starlette.middleware.trustedhost import TrustedHostMiddleware

from .analysis import analyze, demo_capture, public_report
from .io import Capture, MAX_BYTES, export_zip, load_bytes

WEB = Path(__file__).parent / "web"
app = FastAPI(title="ETHERGLASS", version="0.1.0", docs_url=None, redoc_url=None)
app.add_middleware(TrustedHostMiddleware, allowed_hosts=["127.0.0.1", "localhost", "[::1]", "testserver"])
store: OrderedDict[str, dict] = OrderedDict()
lock = threading.RLock()


@app.exception_handler(RequestValidationError)
async def validation_error_handler(request, exc):
    # Invalid NaN/Inf must not break JSON serialization of the error response.
    errors = [{"loc": list(e["loc"]), "msg": e["msg"], "type": e["type"]} for e in exc.errors()]
    return JSONResponse(status_code=422, content={"detail": errors})


def remember(capture: Capture, reference=None) -> str:
    with lock:
        key = uuid.uuid4().hex
        store[key] = {"capture": capture, "reference": reference, "result": None}
        while len(store) > 4:
            store.popitem(last=False)
        return key


def lookup(key: str) -> dict:
    with lock:
        if key not in store:
            raise HTTPException(404, "Capture expired or is unknown. Import it again.")
        store.move_to_end(key)
        return store[key]


def run_analysis(key, **kwargs):
    item = lookup(key)
    try:
        # Store and export one coherent result even when users issue concurrent requests.
        with lock:
            result = analyze(item["capture"], reference=item["reference"], **kwargs)
            result["capture_id"] = key
            item["result"] = result
            return public_report(result)
    except (ValueError, TypeError, KeyError) as e:
        raise HTTPException(422, str(e)) from e
    except FileNotFoundError as e:
        raise HTTPException(503, "Trained model artifact is missing. See the reproducible setup instructions.") from e


@app.get("/")
def index():
    return FileResponse(WEB / "index.html")


@app.get("/api/health")
def health():
    return {"status": "ok", "version": "0.1.0", "mode": "offline-replay"}


@app.get("/api/demo")
def demo():
    capture, reference = demo_capture()
    key = remember(capture, reference)
    return run_analysis(key, symbol_rate=125000)


@app.post("/api/import")
async def import_capture(data_file: UploadFile = File(...), meta_file: UploadFile | None = File(None),
                         sample_rate: float | None = Form(None), center_frequency: float | None = Form(None),
                         datatype: str | None = Form(None)):
    data = await data_file.read(MAX_BYTES + 1)
    if len(data) > MAX_BYTES:
        raise HTTPException(413, "Capture exceeds 64 MiB. Select a smaller recording; the Python API supports streaming blocks.")
    metadata = None
    try:
        if meta_file is not None:
            raw = await meta_file.read(1024 * 1024 + 1)
            if len(raw) > 1024 * 1024:
                raise ValueError("Metadata exceeds 1 MiB.")
            metadata = json.loads(raw)
        capture = load_bytes(data, metadata=metadata, sample_rate=sample_rate,
                             center_frequency=center_frequency, datatype=datatype,
                             name=data_file.filename or "capture")
        key = remember(capture)
        # A multi-tune file starts at its first segment; selection may inspect later ones.
        captures = capture.metadata.get("captures", [])
        stop = captures[1]["core:sample_start"] if len(captures) > 1 else None
        return run_analysis(key, stop_sample=stop)
    except (ValueError, TypeError, KeyError, UnicodeDecodeError) as e:
        raise HTTPException(422, str(e)) from e
    finally:
        await data_file.close()
        if meta_file:
            await meta_file.close()


class AnalyzeRequest(BaseModel):
    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)
    capture_id: str = Field(min_length=1, max_length=64)
    start_sample: int = Field(default=0, ge=0)
    stop_sample: int | None = Field(default=None, ge=64)
    target_center: float = 0
    symbol_rate: float | None = None
    method: str = "hybrid"
    inspect_only: bool = False


@app.post("/api/analyze")
def reanalyze(request: AnalyzeRequest):
    args = request.model_dump(exclude={"capture_id", "inspect_only"})
    if request.inspect_only:
        args["method"] = None
    return run_analysis(request.capture_id, **args)


@app.get("/api/export/{capture_id}")
def export(capture_id: str, component: str = "mixture"):
    if component not in {"mixture", "target", "residual"}:
        raise HTTPException(422, "Choose mixture, target or residual.")
    item = lookup(capture_id)
    with lock:
        result = item["result"]
        if result is None:
            raise HTTPException(409, "Analyze the capture before export.")
        cap = item["capture"]
        selection = result["selection"]
        if component == "mixture":
            x = cap.samples[selection["start_sample"]:selection["stop_sample"]]
        else:
            if "_arrays" not in result:
                raise HTTPException(409, "Separate this selection before exporting a reconstructed component.")
            x = result["_arrays"][0 if component == "target" else 1]
            if isinstance(x, dict):
                x = x.get("samples", x.get("signal"))
        report = public_report(result)
        report["exported_component"] = component
        report["separation_method"] = result["separation"]["method"] if result["separation"] else None
        blob = export_zip(x, cap.sample_rate, result["center_frequency"], report,
                          license=cap.metadata.get("global", {}).get("core:license"))
    return Response(blob, media_type="application/zip",
                    headers={"Content-Disposition": f'attachment; filename="etherglass-{component}.zip"'})


app.mount("/static", StaticFiles(directory=WEB), name="static")
