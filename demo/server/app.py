"""HTTP API of the Solv3R demo, and the built web client.

    uvicorn server.app:app --port 7860            (from the demo/ folder; see run.ps1 / run.sh)
"""
from __future__ import annotations

import asyncio
import logging
import os
from concurrent.futures import Future
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Literal

from fastapi import FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.responses import FileResponse, JSONResponse, Response, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field, field_validator
from starlette.concurrency import run_in_threadpool

from . import packing
from .engine import DEMO_DIR, SAMPLES, Engine, UserError

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")

engine = Engine(checkpoint=os.environ.get("SOLV3R_CHECKPOINT") or None)
WEB_DIST = DEMO_DIR / "web" / "dist"


@asynccontextmanager
async def lifespan(_app: FastAPI):
    engine.start(asyncio.get_running_loop())
    yield


app = FastAPI(title="Solv3R demo", lifespan=lifespan)


@app.exception_handler(UserError)
async def _user_error(_req: Request, exc: UserError):
    return JSONResponse({"detail": str(exc)}, status_code=400)


def session(sid: str):
    try:
        return engine.get(sid)
    except KeyError:
        raise HTTPException(404, "This session has expired. Load the input again.") from None


# ---------------------------------------------------------------------------------------------- info
@app.get("/api/info")
def info():
    return engine.info()


@app.get("/api/samples/{name}/cover.jpg")
def sample_cover(name: str):
    if name not in SAMPLES:
        raise HTTPException(404)
    from solv3r import list_images
    from PIL import Image
    import numpy as np
    paths = list_images(str(SAMPLES[name]["path"]))
    im = Image.open(paths[len(paths) // 2]).convert("RGB")
    return Response(packing.jpeg(np.array(im).transpose(2, 0, 1), long_side=480, quality=82), media_type="image/jpeg",
                    headers={"Cache-Control": "max-age=86400"})


# ---------------------------------------------------------------------------------------------- sessions
@app.post("/api/sessions")
async def create_session(sample: str | None = Form(None), live: bool = Form(False), stride: int = Form(1),
                         max_images: int | None = Form(None), frames: int | None = Form(None),
                         sharpest: bool = Form(True), files: list[UploadFile] | None = File(None)):
    """A session from a sample (loaded before the answer), the webcam, or uploads: images or one video. Uploads are
    answered once saved, with `preparing` set; the pool is filled in the background (prepare / input events). For a
    video, `sharpest` takes the sharpest frame of each stretch instead of evenly spaced ones."""
    stride = max(1, stride)
    max_images = max_images if max_images and max_images > 0 else None
    frames = frames if frames and frames > 0 else None
    if live:
        s = engine.create_live()
    elif sample:
        s = await run_in_threadpool(engine.create_from_sample, sample, stride, max_images)
    elif files:
        for f in files:                                  # spooled to disk by the multipart parser
            f.file.seek(0)
        uploads = [(f.filename or "file", f.file) for f in files]
        s = await run_in_threadpool(engine.create_from_uploads, uploads, stride, max_images, frames, sharpest)
    else:
        raise HTTPException(400, "Pick a sample, or drop images or a video.")
    return s.summary()


class VideoParams(BaseModel):
    frames: int | None = Field(None, ge=2, le=2000)
    sharpest: bool = True


@app.post("/api/sessions/{sid}/video")
def reextract_video(sid: str, p: VideoParams):
    """A new session with another number of frames of this session's video, without uploading it again."""
    return engine.create_from_video_of(session(sid), p.frames, p.sharpest).summary()


@app.get("/api/sessions/{sid}")
def get_session(sid: str):
    return session(sid).summary()


@app.delete("/api/sessions/{sid}")
def delete_session(sid: str):
    engine.delete(sid)
    return {"ok": True}


@app.post("/api/sessions/{sid}/images")
async def add_images(sid: str, files: list[UploadFile] = File(...)):
    s = session(sid)
    data = [(f.filename or "image", await f.read()) for f in files]
    idx = await run_in_threadpool(engine.add_uploads, s, data)
    return {"indices": idx, "session": s.summary()}


@app.get("/api/sessions/{sid}/image/{i}.jpg")
def image(sid: str, i: int, size: int | None = None):
    s = session(sid)
    if s.images is None or not (0 <= i < len(s.images)):
        raise HTTPException(404)
    return Response(packing.jpeg(s.images[i], long_side=size, quality=85 if size else 92), media_type="image/jpeg",
                    headers={"Cache-Control": "max-age=3600"})


# ---------------------------------------------------------------------------------------------- jobs
class RunParams(BaseModel):
    mode: Literal["offline", "incremental", "any_order", "stream", "continue", "add_views", "perturb"]
    calls: int = Field(16, ge=1, le=512)
    adaptive: float | None = Field(None, gt=0, lt=1)
    max_calls: int = Field(64, ge=1, le=512)
    subset: int | None = Field(None, ge=1)
    preview: int = Field(1, ge=0, le=64)
    budget: int | None = Field(None, ge=10_000, le=20_000_000)
    sky: bool = False
    seed: int = 0
    initial: int = Field(8, ge=2)
    group: int = Field(4, ge=1)
    config: Literal["L8", "L4", "S4"] = "L8"
    live: bool = False
    indices: list[int] | None = None
    how: Literal["reset", "noise", "swap"] = "reset"
    count: int | None = Field(None, ge=1)
    sigma: float = Field(1.0, gt=0, le=10)
    views: list[int] | None = None


@app.post("/api/sessions/{sid}/run")
def run(sid: str, p: RunParams):
    s = session(sid)
    params = p.model_dump(exclude_none=True)
    mode = params.pop("mode")
    if mode == "stream" and p.live:
        from .engine import LiveFeed
        s.live = LiveFeed()
    job = engine.submit(s, mode, params)
    return {"job": job.id}


class SplatParams(BaseModel):
    init: Literal["sharp", "points"] = "sharp"
    keyframes: int = Field(8, ge=1, le=48)
    align: Literal["depth", "scale"] = "depth"
    steps: int = Field(3000, ge=0, le=30000)
    refine_cameras: bool = True
    exposure: bool = True
    holdout: bool = True
    resolution: int = Field(1008, ge=252, le=4096)       # long side, never above the source frames
    max_splats: int = Field(2_000_000, ge=0, le=8_000_000)   # 0: from the training size (splats.auto_budget)
    sh_degree: int = Field(3, ge=0, le=3)
    sky: bool | None = None
    coarse: bool = True                                     # coarse to fine: the first half at lower sizes

    @field_validator("max_splats")
    @classmethod
    def _budget(cls, v: int) -> int:
        if 0 < v < 100_000:
            raise ValueError("max_splats is 0 (automatic) or at least 100,000")
        return v


@app.post("/api/sessions/{sid}/splats")
def run_splats(sid: str, p: SplatParams):
    job = engine.submit(session(sid), "splats", p.model_dump())
    return {"job": job.id}


@app.get("/api/sessions/{sid}/splats/{rid}/{name:path}")
def splat_file(sid: str, rid: str, name: str):
    s = session(sid)
    try:
        path = engine.splat_file(s, rid, name)
    except KeyError:
        raise HTTPException(404, "No such splat file (the run may still be going, or it was replaced).") from None
    if name == "scene.ply":
        stem = s.source.split(":")[-1].split(".")[0].replace(" ", "_") or "solv3r"
        return FileResponse(path, filename=f"splats_{stem}_{rid}.ply", media_type="application/octet-stream")
    media = "image/jpeg" if name.endswith(".jpg") else "application/octet-stream"
    return FileResponse(path, media_type=media, headers={"Cache-Control": "no-store"})


@app.delete("/api/sessions/{sid}/splats/{rid}")
def delete_splats(sid: str, rid: str):
    engine.delete_splats(session(sid), rid)
    return {"ok": True}


@app.post("/api/sessions/{sid}/cancel")
def cancel(sid: str):
    s = session(sid)
    if s.job is not None:
        s.job.cancel.set()
    if s.live is not None:
        s.live.q.put(None)
    return {"ok": True}


@app.post("/api/sessions/{sid}/live/frame")
async def live_frame(sid: str, request: Request):
    """One webcam frame (JPEG body). Returns once the solver has answered it, so the client sends at the solver's rate."""
    s = session(sid)
    if s.live is None or not s.live.open or s.job is None:
        raise HTTPException(409, "The live stream is not running.")
    body = await request.body()
    idx = await run_in_threadpool(engine.add_live_frame, s, body)
    fut: Future = Future()
    s.live.q.put((idx, fut))
    try:
        return await asyncio.wait_for(asyncio.wrap_future(fut), timeout=60)
    except asyncio.TimeoutError:
        raise HTTPException(504, "The solver did not answer the frame in time.") from None


@app.post("/api/sessions/{sid}/live/stop")
def live_stop(sid: str):
    s = session(sid)
    if s.live is not None:
        s.live.q.put(None)
    return {"ok": True}


@app.get("/api/sessions/{sid}/events")
async def events(sid: str, request: Request, since: int = 0):
    s = session(sid)
    last = request.headers.get("last-event-id")
    if last and last.isdigit():                          # a reconnecting EventSource resumes after its last event
        since = max(since, int(last) + 1)

    async def stream():
        seq = since
        yield "retry: 1500\n\n"
        while True:
            if await request.is_disconnected():
                return
            ev = s.changed
            for n, data in s.events_since(seq):
                yield f"id: {n}\ndata: {data}\n\n"
                seq = n + 1
            try:
                await asyncio.wait_for(ev.wait(), timeout=15)
            except asyncio.TimeoutError:
                yield ": keepalive\n\n"

    return StreamingResponse(stream(), media_type="text/event-stream",
                             headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})


@app.get("/api/sessions/{sid}/blob/{key}")
def blob(sid: str, key: str):
    s = session(sid)
    data = s.blobs.get(key)
    if data is None:                                     # superseded by a newer payload (only the latest few are kept)
        return Response(status_code=204, headers={"Cache-Control": "no-store"})
    return Response(data, media_type="application/octet-stream", headers={"Cache-Control": "no-store"})


@app.get("/api/sessions/{sid}/maps/{pos}/{kind}.png")
def depth_map(sid: str, pos: int, kind: Literal["depth", "conf"]):
    s = session(sid)
    try:
        return Response(engine.depth_png(s, pos, kind), media_type="image/png", headers={"Cache-Control": "no-store"})
    except KeyError:
        raise HTTPException(404, "No outputs for this view yet.") from None


@app.get("/api/sessions/{sid}/export/{fmt}")
async def export(sid: str, fmt: Literal["ply", "npz", "html"], keep: float = 0.5, sky: bool = False):
    s = session(sid)
    out: Path = await run_in_threadpool(engine.export, s, keep, sky)
    name = {"ply": "solve.ply", "npz": "solve.npz", "html": "solve.html"}[fmt]
    stem = s.source.split(":")[-1].split(".")[0].replace(" ", "_") or "solv3r"
    return FileResponse(out / name, filename=f"solv3r_{stem}.{fmt}")


# ---------------------------------------------------------------------------------------------- web client
if WEB_DIST.is_dir():
    app.mount("/", StaticFiles(directory=WEB_DIST, html=True), name="web")
else:
    @app.get("/")
    def no_client():
        return Response("<p>The web client is not built: run <code>npm install && npm run build</code> in "
                        "demo/web, or start the Vite dev server with <code>npm run dev</code>.</p>",
                        media_type="text/html")
