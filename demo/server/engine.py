"""The Solv3R demo engine: the model, the sessions, and the jobs that run on the GPU.

All GPU work runs on one worker thread, one job at a time. A session holds an input pool (images at the model size)
and, after a solve, the solver state z with its context, so a solve can be continued, extended with new views or
perturbed and solved again: the properties of a looped solver that has no step index.

Jobs report through session events (JSON, sent to the browser over server-sent events) and binary point payloads
(see packing.py) that the browser fetches by key.
"""
from __future__ import annotations

import asyncio
import dataclasses
import json
import logging
import math
import os
import queue
import re
import shutil
import subprocess
import threading
import time
import traceback
import uuid
from concurrent.futures import Future, ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import torch

from solv3r import PRESETS, Solv3R, StreamingSolver, list_images, load_images
from solv3r.export import confident_points, unproject, write_html, write_ply
from solv3r.images import EXTENSIONS, VIDEO_EXTENSIONS, _ffmpeg, model_size, resize
from solv3r.model import Context, autocast, cameras_from_pose_enc
from solv3r.network import skyseg

from . import packing

log = logging.getLogger("solv3r.demo")

DEMO_DIR = Path(__file__).resolve().parents[1]
REPO_DIR = DEMO_DIR.parent / "solv3r"
CACHE_DIR = DEMO_DIR / "cache"
# sessions do not survive a restart: the server clears this folder when it starts, so a second server (a test
# instance on another port) needs its own (SOLV3R_SESSIONS_DIR) or it deletes the first one's session files
SESSIONS_DIR = Path(os.environ.get("SOLV3R_SESSIONS_DIR") or CACHE_DIR / "sessions")
skyseg.SKYSEG_ONNX_PATH = str(CACHE_DIR / "skyseg.onnx")   # not the working directory

MAX_SESSIONS = 6
MAX_EVENTS = 50000
CLOUD_BLOBS_KEPT = 3

SAMPLES = {
    "park": dict(title="Park", path=REPO_DIR / "examples/data/park", blurb="39 drone photos of a park",
                 mode="offline", params=dict(sky=True)),
    "desk": dict(title="Desk", path=REPO_DIR / "examples/data/desk", blurb="289 frames of a handheld desk video",
                 mode="stream", params=dict(config="L8")),
}


class Cancelled(Exception):
    pass


class UserError(Exception):
    """An error caused by the request, shown to the user as is."""


@dataclasses.dataclass
class Job:
    id: str
    mode: str
    params: dict
    cancel: threading.Event = dataclasses.field(default_factory=threading.Event)
    state: str = "queued"

    def check(self):
        if self.cancel.is_set():
            raise Cancelled()


class LiveFeed:
    """Frames posted by the browser (webcam) for a live streaming job. Each item is (pool index, Future)."""

    def __init__(self):
        self.q: queue.Queue = queue.Queue()
        self.open = True


def _sync():
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def _round(a, nd=5):
    return np.round(np.asarray(a, np.float64), nd).tolist()


def cameras_json(c2w, K):
    """c2w (S, 4, 4), K (S, 3, 3) -> compact JSON: the top 3 rows of c2w (12 numbers) and fx, fy, cx, cy per view."""
    c2w = np.asarray(c2w, np.float64)
    K = np.asarray(K, np.float64)
    return dict(c2w=_round(c2w[:, :3, :4].reshape(len(c2w), 12)),
                K=_round(np.stack([K[:, 0, 0], K[:, 1, 1], K[:, 0, 2], K[:, 1, 2]], 1), 3))


class Session:
    def __init__(self, sid: str, loop: asyncio.AbstractEventLoop):
        self.id = sid
        self.loop = loop
        self.created = self.touched = time.time()
        self.dir = SESSIONS_DIR / sid
        self.dir.mkdir(parents=True, exist_ok=True)
        self.lock = threading.RLock()
        # the input pool
        self.names: list[str] = []
        self.paths: list[str | None] = []                # source file of each pool image (full size), if any
        self.images: torch.Tensor | None = None          # (N, 3, H, W) uint8, at the model size
        self.hw: tuple[int, int] | None = None
        self.source_hw: tuple[int, int] | None = None    # size of the source images (photos, video frames) if known
        self.source = ""
        self.warnings: list[str] = []
        self.sky: dict[int, np.ndarray] = {}            # pool index -> (H, W) bool, True on sky
        self.video: dict | None = None                   # an uploaded video: its file and probe, the frames taken
        self.prep: dict | None = None                    # while the pool is being filled: label, done, total
        self.prep_t = 0.0
        self.closed = threading.Event()                  # set when the session is deleted (stops a preparation)
        # the solve
        self.kind: str | None = None                     # "solve" (z is kept) or "stream"
        self.order: list[int] = []                       # pool index of each view of the solve, in solve order
        self.stage_of: list[int] = []
        self.ctx: Context | None = None
        self.z: torch.Tensor | None = None
        self.calls = 0
        self.version = 0                                 # bumped whenever the decoded outputs change
        self.pred: dict | None = None                    # last decoded outputs on the CPU (depth, conf, c2w, K)
        self.splat_runs: list[dict] = []                 # Gaussian splat runs on the solve (splats.py)
        self.sel: dict = {}
        self.run_id = 0
        # events and payloads
        self.events: list[tuple[int, str]] = []
        self.base = 0
        self.changed = asyncio.Event()
        self.blobs: dict[str, bytes] = {}
        self.cloud_keys: list[str] = []
        self.job: Job | None = None
        self.live: LiveFeed | None = None
        self.png_cache: dict = {}
        self.exports: dict = {}

    # ------------------------------------------------------------------ events
    def emit(self, type: str, **data):
        with self.lock:
            seq = self.base + len(self.events)
            self.events.append((seq, json.dumps({"seq": seq, "type": type, "t": round(time.time(), 3), **data})))
            if len(self.events) > MAX_EVENTS:
                drop = len(self.events) - MAX_EVENTS
                self.events = self.events[drop:]
                self.base += drop
        self.loop.call_soon_threadsafe(self._notify)

    def _notify(self):
        old, self.changed = self.changed, asyncio.Event()
        old.set()

    def events_since(self, seq: int):
        with self.lock:
            i = max(0, seq - self.base)
            return self.events[i:]

    # ------------------------------------------------------------------ payloads
    def put_cloud(self, data: bytes) -> str:
        key = f"c{self.run_id}_{self.version}"
        with self.lock:
            self.blobs[key] = data
            self.cloud_keys.append(key)
            while len(self.cloud_keys) > CLOUD_BLOBS_KEPT:
                self.blobs.pop(self.cloud_keys.pop(0), None)
        return key

    def put_blob(self, key: str, data: bytes):
        with self.lock:
            self.blobs[key] = data

    def reset_solve(self, kind: str):
        with self.lock:
            self.kind, self.order, self.stage_of = kind, [], []
            self.ctx = self.z = None
            self.calls = 0
            self.pred = None
            self.run_id += 1
            self.version += 1
            self.blobs.clear()
            self.cloud_keys.clear()
            self.png_cache.clear()
            self.exports.clear()
            self.splat_runs = []                         # they belong to the previous reconstruction
        shutil.rmtree(self.dir / "splats", ignore_errors=True)

    # ------------------------------------------------------------------ pool
    def add_images(self, imgs_u8: torch.Tensor, names: list[str], paths: list[str] | None = None) -> list[int]:
        with self.lock:
            start = 0 if self.images is None else len(self.images)
            self.images = imgs_u8 if self.images is None else torch.cat([self.images, imgs_u8])
            self.names += names
            self.paths += list(paths) if paths is not None else [None] * len(names)
            return list(range(start, start + len(names)))

    def summary(self) -> dict:
        with self.lock:
            return dict(
                id=self.id, source=self.source, n=0 if self.images is None else len(self.images), names=self.names,
                hw=self.hw, source_hw=self.source_hw, warnings=self.warnings, kind=self.kind, order=self.order, stage_of=self.stage_of,
                calls=self.calls, version=self.version, has_sky=bool(self.sky), live=self.live is not None,
                preparing=self.prep,
                video=None if self.video is None else {k: v for k, v in self.video.items() if k != "path"},
                job=None if self.job is None else dict(id=self.job.id, mode=self.job.mode, state=self.job.state),
                splats=list(self.splat_runs), next_seq=self.base + len(self.events))

    def free(self):
        with self.lock:
            self.ctx = self.z = None
            self.blobs.clear()
            self.images = None
        shutil.rmtree(self.dir, ignore_errors=True)


def _natural(name: str):
    """File-name order with numbers compared by value (frame_2 before frame_10)."""
    return [int(t) if t.isdigit() else t.lower() for t in re.split(r"(\d+)", name)]


def _to_u8(images: torch.Tensor) -> torch.Tensor:
    return (images.clamp(0, 1) * 255).round().to(torch.uint8)


def _file_hw(path: str) -> tuple[int, int] | None:
    """(height, width) of an image file as shown (after its EXIF rotation), from its header."""
    from PIL import Image
    try:
        with Image.open(path) as im:
            w, h = im.size
            if im.getexif().get(0x0112) in (5, 6, 7, 8):
                w, h = h, w
            return h, w
    except (OSError, ValueError):
        return None


def _read_image(path: str, size) -> tuple[torch.Tensor, bool]:
    """One image as (3, h, w) uint8 at the model size (its own unless size is given), preprocessed as
    solv3r.load_images does; and whether its aspect ratio differs from size (it is stretched)."""
    from PIL import Image, ImageOps
    try:
        im = ImageOps.exif_transpose(Image.open(path)).convert("RGB")
    except (OSError, ValueError) as e:
        raise UserError(f"could not read the image {Path(path).name}: {e}") from None
    t = torch.from_numpy(np.array(im)).permute(2, 0, 1).float() / 255
    own = model_size(*t.shape[-2:])
    size = tuple(size or own)
    return _to_u8(resize(t[None], size))[0], own != size


# ---------------------------------------------------------------------------------------------- video
VIDEO_FRAME_SIDE = 3840          # long side of the extracted frames at most (4K: the splats train on them at full
                                 # size); the pool resizes them to the model size
VIDEO_SCAN_SIDE = 960            # sharpest frames: the sharpness of every frame is measured at this long side ...
VIDEO_SCAN_PART = 0.7            # ... and the frame taken in each stretch is the sharpest of its central 70 %
VIDEO_AUTO_RATE = 3.0            # automatic frame count: 3 frames per second of video ...
VIDEO_AUTO_RANGE = (24, 240)     # ... and at least 24, at most 240 (longer videos get sparser)
VIDEO_MAX_FRAMES = 2000


# ffmpeg runs detached from the server's console and standard input: a server whose launching shell has gone away
# (for example a background shell that was stopped) can hand its children unusable handles
_FFMPEG_KW = dict(stdin=subprocess.DEVNULL, creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))


def probe_video(path: Path) -> dict:
    """Codec, display size (after the rotation a phone records), frame rate, duration and the estimated frame count
    of a video, read from ffmpeg's description of the file (ffprobe is not always installed)."""
    exe = _ffmpeg()
    try:
        r = subprocess.run([exe, "-hide_banner", "-nostdin", "-i", str(path)], capture_output=True, timeout=60,
                           **_FFMPEG_KW)
    except (OSError, subprocess.TimeoutExpired) as e:
        raise UserError(f"could not run ffmpeg ({exe}): {e}") from None
    text = r.stderr.decode(errors="ignore")
    stream = re.search(r"Stream #\S+.*?: Video: (\w+)(.*)", text)
    if not stream:
        log.warning("no video stream in %s: ffmpeg %s exited %s: %s", path.name, exe, r.returncode, text[-800:])
        if "Input #" in text:
            raise UserError("ffmpeg found no video stream in this file")
        last = text.strip().splitlines()[-1][-200:] if text.strip() else f"no output, exit code {r.returncode}"
        raise UserError(f"ffmpeg could not open this file: {last}")
    rest = stream.group(2)
    size = re.search(r"\b(\d{2,5})x(\d{2,5})\b", rest)
    rate = re.search(r"([\d.]+)(k?) fps", rest) or re.search(r"([\d.]+)(k?) tbr", rest)
    dur = re.search(r"Duration: (\d+):(\d+):(\d+(?:\.\d+)?)", text)
    rot = re.search(r"rotation of (-?[\d.]+) degrees", text, re.I)
    w, h = (int(size.group(1)), int(size.group(2))) if size else (0, 0)
    if rot and round(abs(float(rot.group(1)))) % 180 == 90:
        w, h = h, w
    fps = float(rate.group(1)) * (1000 if rate.group(2) else 1) if rate else 0.0
    duration = int(dur.group(1)) * 3600 + int(dur.group(2)) * 60 + float(dur.group(3)) if dur else 0.0
    return dict(codec=stream.group(1), width=w, height=h, fps=round(fps, 3), duration=round(duration, 3),
                frames=int(round(duration * fps)))


def auto_frames(meta: dict) -> int:
    """The default number of frames for a video: VIDEO_AUTO_RATE per second, within VIDEO_AUTO_RANGE."""
    lo, hi = VIDEO_AUTO_RANGE
    n = min(max(round(meta["duration"] * VIDEO_AUTO_RATE), lo), hi) if meta["duration"] else hi
    return min(n, meta["frames"]) if meta["frames"] else n


def sample_rate(meta: dict, frames: int) -> float:
    """Frames per second of video that give frames frames over the whole video (frame k is at k / rate seconds)."""
    rate = frames / meta["duration"] if meta["duration"] else VIDEO_AUTO_RATE
    return min(rate, meta["fps"]) if meta["fps"] else rate


def _clock(t: float) -> str:
    return f"{int(t // 60)}:{t % 60:05.2f}"


def _scale_to(L: int) -> str:
    """ffmpeg scale filter: the long side at most L (never enlarged), even sizes."""
    return (f"scale=w='trunc(iw*min(1\\,{L}/max(iw\\,ih))/2)*2'"
            f":h='trunc(ih*min(1\\,{L}/max(iw\\,ih))/2)*2'")


def scan_sharpness(video: Path, meta: dict, progress=None, stop=None) -> np.ndarray:
    """The sharpness of every frame of the video, in decoding order: the variance of the Laplacian of its grey
    image at VIDEO_SCAN_SIDE (motion blur and focus blur lower it). One decoding pass: ffmpeg pipes the small grey
    frames, numpy measures them in a few threads."""
    w, h = meta["width"] or 1280, meta["height"] or 720
    r = min(1.0, VIDEO_SCAN_SIDE / max(w, h))
    sw, sh = max(16, int(w * r) // 2 * 2), max(16, int(h * r) // 2 * 2)
    cmd = [_ffmpeg(), "-hide_banner", "-nostdin", "-loglevel", "error", "-i", str(video), "-an", "-sn", "-dn",
           "-vf", f"scale={sw}:{sh}:flags=area,format=gray", "-fps_mode", "passthrough", "-f", "rawvideo",
           "-pix_fmt", "gray", "pipe:1"]
    size = sw * sh

    def measure(buf: bytes) -> float:
        a = np.frombuffer(buf, np.uint8).reshape(sh, sw).astype(np.int16)
        lap = 4 * a[1:-1, 1:-1] - a[:-2, 1:-1] - a[2:, 1:-1] - a[1:-1, :-2] - a[1:-1, 2:]
        return float(lap.astype(np.float32).var())

    futures = []
    total = meta["frames"] or None
    p = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, bufsize=size * 4, **_FFMPEG_KW)
    try:
        with ThreadPoolExecutor(4, thread_name_prefix="solv3r-sharp") as pool:
            while True:
                if stop is not None and stop.is_set():
                    raise Cancelled()
                buf = p.stdout.read(size)
                if len(buf) < size:
                    break
                futures.append(pool.submit(measure, buf))
                if progress and len(futures) % 30 == 0:
                    progress(len(futures), max(total or 0, len(futures)))
            p.wait()
            values = np.array([f.result() for f in futures], np.float64)
    finally:
        if p.poll() is None:
            p.kill()
            p.wait()
    if p.returncode != 0 or len(values) == 0:
        raise UserError("ffmpeg could not decode the video to find the sharpest frames")
    return values


def pick_sharpest(sharp: np.ndarray, n: int) -> list[int]:
    """n frame indices spread over the video: the sharpest frame of the central VIDEO_SCAN_PART of each of n equal
    stretches (so two picks are never closer than ~30 % of the spacing)."""
    N = len(sharp)
    n = max(1, min(n, N))
    out = []
    for k in range(n):
        a, b = k * N / n, (k + 1) * N / n
        c, half = (a + b) / 2, (b - a) * VIDEO_SCAN_PART / 2
        lo = max(0, math.ceil(c - half - 0.5))
        hi = min(N - 1, max(lo, math.floor(c + half - 0.5)))
        out.append(lo + int(np.argmax(sharp[lo:hi + 1])))
    return sorted(set(out))


def _ffmpeg_frames(cmd: list[str], out_dir: Path, total: int, progress, stop) -> list[str]:
    """Runs an ffmpeg command that writes %06d.jpg frames into out_dir, reporting -progress lines."""
    err_path = out_dir.parent / "ffmpeg.log"
    with open(err_path, "wb") as err:                    # a file, not a pipe: a damaged video can log a lot
        p = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=err, text=True, **_FFMPEG_KW)
        try:
            for line in p.stdout:
                if stop is not None and stop.is_set():
                    raise Cancelled()
                key, _, value = line.strip().partition("=")
                if key == "frame" and value.isdigit() and progress:
                    progress(min(int(value), total), total)
            p.wait()
        finally:
            if p.poll() is None:
                p.kill()
                p.wait()
    paths = sorted(str(q) for q in out_dir.glob("*.jpg"))
    if p.returncode != 0 or not paths:
        tail = err_path.read_text(errors="ignore").strip().splitlines()[-3:]
        raise UserError("ffmpeg could not read the video" + (f": {' '.join(tail)[-300:]}" if tail else ""))
    return paths


def extract_video(video: Path, out_dir: Path, meta: dict, frames: int, progress=None, stop=None,
                  sharpest=False, scan_progress=None) -> tuple[list[str], list[float]]:
    """frames frames over the whole video, as .jpg files with the long side at most VIDEO_FRAME_SIDE, and the time
    of each in seconds. Evenly spaced, or with `sharpest` the sharpest frame of each stretch (scan_sharpness first:
    one more decoding pass). ffmpeg decodes the whole video (sequential decoding beats seeking for a few hundred
    frames) and applies the recorded rotation; progress(done, total) is called as frames are written; stop (an
    Event) kills ffmpeg."""
    out_dir.mkdir(parents=True, exist_ok=True)
    head = [_ffmpeg(), "-hide_banner", "-nostdin", "-loglevel", "error", "-nostats", "-progress", "pipe:1",
            "-i", str(video), "-an", "-sn", "-dn"]
    tail = ["-q:v", "2", str(out_dir / "%06d.jpg")]
    if sharpest:
        sharp = scan_sharpness(video, meta, scan_progress, stop)
        idx = pick_sharpest(sharp, frames)
        select = "+".join(f"eq(n\\,{i})" for i in idx)
        cmd = head + ["-vf", f"select='{select}',{_scale_to(VIDEO_FRAME_SIDE)}", "-fps_mode", "passthrough"] + tail
        paths = _ffmpeg_frames(cmd, out_dir, len(idx), progress, stop)
        per_frame = meta["duration"] / len(sharp) if meta["duration"] else 1 / (meta["fps"] or 30)
        times = [i * per_frame for i in idx[:len(paths)]]
        log.info("sharpest frames: %d of %d decoded frames, median sharpness %.0f (all frames %.0f)", len(idx),
                 len(sharp), float(np.median(sharp[idx])), float(np.median(sharp)))
        return paths, times
    rate = sample_rate(meta, frames)
    cmd = head + ["-vf", f"fps={rate:.6f},{_scale_to(VIDEO_FRAME_SIDE)}", "-frames:v", str(int(frames))] + tail
    paths = _ffmpeg_frames(cmd, out_dir, frames, progress, stop)
    return paths, [k / rate for k in range(len(paths))]


class Engine:
    def __init__(self, checkpoint: str | None = None, device: str = "cuda"):
        self.checkpoint = checkpoint
        self.device = device
        self.model: Solv3R | None = None
        self.ready = threading.Event()
        self.load_error: str | None = None
        self.load_seconds = 0.0
        self.gpu = ThreadPoolExecutor(1, thread_name_prefix="solv3r-gpu")
        self.cpu = ThreadPoolExecutor(4, thread_name_prefix="solv3r-io")
        self.sessions: dict[str, Session] = {}
        self.loop: asyncio.AbstractEventLoop | None = None
        self.sessions_lock = threading.Lock()

    # ------------------------------------------------------------------ lifecycle
    def start(self, loop):
        self.loop = loop
        SESSIONS_DIR.mkdir(parents=True, exist_ok=True)
        for d in SESSIONS_DIR.iterdir():                 # sessions do not survive a restart
            shutil.rmtree(d, ignore_errors=True)
        self.gpu.submit(self._load)
        self.cpu.submit(self._fetch_skyseg)

    def _fetch_skyseg(self):
        """Downloads the sky segmentation model (176 MB) once, in the background, so the first sky mask does not wait."""
        try:
            if not os.path.exists(skyseg.SKYSEG_ONNX_PATH):
                import urllib.request
                log.info("downloading the sky segmentation model ...")
                tmp = skyseg.SKYSEG_ONNX_PATH + ".part"
                urllib.request.urlretrieve(skyseg.SKYSEG_URL, tmp)
                os.replace(tmp, skyseg.SKYSEG_ONNX_PATH)
                log.info("sky segmentation model ready")
        except Exception:                                # noqa: BLE001
            log.exception("could not download the sky segmentation model (it is fetched again when needed)")

    def _load(self):
        try:
            t = time.time()
            log.info("loading Solv3R weights ...")
            self.model = Solv3R.from_pretrained(self.checkpoint, device=self.device)
            log.info("warming up the GPU kernels ...")
            with torch.no_grad():
                for hw in ((336, 504), (378, 504)):
                    ctx = self.model.encode(torch.rand(3, 3, *hw))
                    z = self.model.solve(ctx, 2)
                    self.model.decode(z, ctx)
                    self.model.decode_cameras(z, ctx)
            _sync()
            torch.cuda.empty_cache()
            self.load_seconds = time.time() - t
            log.info("model ready in %.1fs", self.load_seconds)
            self.ready.set()
        except Exception as e:                           # noqa: BLE001
            self.load_error = f"{type(e).__name__}: {e}"
            log.exception("could not load the model")

    def info(self) -> dict:
        gpu = None
        if torch.cuda.is_available():
            p = torch.cuda.get_device_properties(0)
            gpu = dict(name=p.name, total_gib=round(p.total_memory / 2**30, 1), capability=f"{p.major}.{p.minor}")
        return dict(ready=self.ready.is_set(), error=self.load_error, load_seconds=round(self.load_seconds, 1),
                    gpu=gpu, torch=torch.__version__,
                    presets={k: dataclasses.asdict(v) for k, v in PRESETS.items()},
                    samples=[dict(id=k, title=v["title"], blurb=v["blurb"], mode=v["mode"], params=v["params"],
                                  n=len(list_images(str(v["path"])))) for k, v in SAMPLES.items() if v["path"].is_dir()])

    # ------------------------------------------------------------------ sessions
    def get(self, sid: str) -> Session:
        s = self.sessions.get(sid)
        if s is None:
            raise KeyError(sid)
        s.touched = time.time()
        return s

    def _new_session(self) -> Session:
        with self.sessions_lock:
            idle = sorted((s for s in self.sessions.values() if s.job is None), key=lambda s: s.touched)
            while len(self.sessions) >= MAX_SESSIONS and idle:
                old = idle.pop(0)
                self.sessions.pop(old.id, None)
                old.closed.set()
                self.gpu.submit(old.free)              # GPU tensors are released on the GPU thread
            s = Session(uuid.uuid4().hex[:10], self.loop)
            self.sessions[s.id] = s
            return s

    def delete(self, sid: str):
        with self.sessions_lock:
            s = self.sessions.pop(sid, None)
        if s is not None:
            s.closed.set()
            if s.job is not None:
                s.job.cancel.set()
            if s.live is not None:
                s.live.q.put(None)
            self.gpu.submit(s.free)

    def create_from_paths(self, paths: list[str], source: str, stride=1, max_images=None) -> Session:
        paths = paths[::max(1, int(stride))][:max_images or None]
        if not paths:
            raise UserError("no images to load")
        s = self._new_session()
        s.source = source
        try:
            self._load_into(s, paths)
        except Exception:
            self.delete(s.id)
            raise
        return s

    def create_from_sample(self, name: str, stride=1, max_images=None) -> Session:
        if name not in SAMPLES:
            raise UserError(f"unknown sample {name}")
        return self.create_from_paths(list_images(str(SAMPLES[name]["path"])), f"sample:{name}", stride, max_images)

    def create_from_uploads(self, files: list[tuple[str, object]], stride=1, max_images=None,
                            frames=None, sharpest=True) -> Session:
        """Uploaded images, or one video. files are (name, binary file object). The files are saved and checked here;
        the pool is filled in the background (frame extraction, image decoding), reported by prepare events and
        finished by an input event (or input_error). For a video, frames is the number of frames to take (None: auto)
        and max_images caps it; stride applies to images."""
        s = self._new_session()
        try:
            inputs = s.dir / "inputs"
            inputs.mkdir(parents=True, exist_ok=True)
            videos, images = [], []
            for i, (name, f) in enumerate(files):
                safe = f"{i:05d}_{Path(name).name}"
                with open(inputs / safe, "wb") as out:
                    shutil.copyfileobj(f, out, 1 << 20)
                if safe.lower().endswith(VIDEO_EXTENSIONS):
                    videos.append(inputs / safe)
                elif safe.lower().endswith(EXTENSIONS):
                    images.append(str(inputs / safe))
            if videos and images:
                raise UserError("drop either images or one video, not both")
            if len(videos) > 1:
                raise UserError("drop one video at a time")
            if videos:
                name = videos[0].name[6:]
                s.source = f"video:{name}"
                if max_images:
                    frames = min(frames or max_images, max_images)
                self._prepare(s, self._prepare_video, videos[0], name, frames, sharpest)
            else:
                if not images:
                    raise UserError(f"no images found: use {', '.join(EXTENSIONS)} files or a video "
                                    f"({', '.join(VIDEO_EXTENSIONS)})")
                paths = sorted(images, key=lambda p: _natural(Path(p).name[6:]))   # by original name
                paths = paths[::max(1, int(stride))][:max_images or None]
                s.source = f"upload:{len(images)} images"
                self._prepare(s, self._prepare_images, paths)
        except Exception:
            self.delete(s.id)
            raise
        return s

    def create_from_video_of(self, src: Session, frames: int | None, sharpest: bool = True) -> Session:
        """A new session with other frames of the video of src (no new upload)."""
        if src.video is None:
            raise UserError("this input is not a video")
        video = Path(src.video["path"])
        s = self._new_session()
        try:
            inputs = s.dir / "inputs"
            inputs.mkdir(parents=True, exist_ok=True)
            dst = inputs / video.name
            try:
                os.link(video, dst)                      # same disk: no copy
            except OSError:
                shutil.copyfile(video, dst)
            s.source = src.source
            self._prepare(s, self._prepare_video, dst, src.video["name"], frames, sharpest)
        except FileNotFoundError:
            self.delete(s.id)
            raise UserError("the video is gone from the server: load it again") from None
        except Exception:
            self.delete(s.id)
            raise
        return s

    def create_live(self) -> Session:
        s = self._new_session()
        s.source = "live"
        return s

    # ------------------------------------------------------------------ filling the pool in the background
    def _prepare(self, s: Session, fn, *args):
        s.prep = dict(label="Starting", done=0, total=None)
        self.cpu.submit(self._prepare_run, s, fn, args)

    def _prepare_run(self, s: Session, fn, args):
        error = None
        try:
            fn(s, *args)
        except Cancelled:
            shutil.rmtree(s.dir, ignore_errors=True)     # the session was deleted while it was loading
            return
        except UserError as e:
            error = str(e)
        except Exception as e:                           # noqa: BLE001
            log.exception("could not load the input")
            error = f"{type(e).__name__}: {e}"
        with s.lock:
            s.prep = None
        if error:
            s.emit("input_error", error=error)
        else:
            s.emit("input", n=len(s.images), hw=s.hw)

    def _progress(self, s: Session, label: str, done: int, total: int | None):
        if s.closed.is_set():
            raise Cancelled()
        now = time.time()
        with s.lock:
            same = s.prep is not None and s.prep["label"] == label
            s.prep = dict(label=label, done=done, total=total)
        if not same or done == total or now - s.prep_t > 0.1:   # at most ~10 events a second
            s.prep_t = now
            s.emit("prepare", label=label, done=done, total=total)

    def _prepare_video(self, s: Session, path: Path, name: str, frames: int | None, sharpest: bool = True):
        self._progress(s, f"Reading {name}", 0, None)
        meta = probe_video(path)
        auto = auto_frames(meta)
        n = max(2, min(int(frames or auto), VIDEO_MAX_FRAMES, meta["frames"] or VIDEO_MAX_FRAMES))
        s.video = dict(name=name, path=str(path), **meta, auto=auto, requested=n, extracted=0, sharpest=sharpest)
        log.info("video %s: %s %dx%d %.2f fps %.1fs, taking %d frames%s", name, meta["codec"], meta["width"],
                 meta["height"], meta["fps"], meta["duration"], n, " (the sharpest of each stretch)" if sharpest else "")
        label = "Extracting the sharpest frames" if sharpest else "Extracting frames"
        self._progress(s, "Measuring the sharpness of every frame" if sharpest else label, 0, None if sharpest else n)
        paths, times = extract_video(path, s.dir / "frames", meta, n, stop=s.closed, sharpest=sharpest,
                                     progress=lambda d, t: self._progress(s, label, d, t),
                                     scan_progress=lambda d, t: self._progress(s, "Measuring the sharpness of every "
                                                                                "frame", d, t))
        s.video["extracted"] = len(paths)
        s.video["spacing"] = round(meta["duration"] / len(paths), 4) if meta["duration"] else round(1 / sample_rate(meta, n), 4)
        stem = Path(name).stem
        names = [f"{stem} @ {_clock(t)}" for t in times]
        self._load_into(s, paths, names=names,
                        progress=lambda d, t: self._progress(s, "Resizing frames to the model size", d, t))

    def _prepare_images(self, s: Session, paths: list[str]):
        self._load_into(s, paths, progress=lambda d, t: self._progress(s, "Reading images", d, t))

    def _load_into(self, s: Session, paths: list[str], progress=None, names: list[str] | None = None):
        """Reads images into the pool at the model size of the first one (or of the session), decoding in parallel
        (the same preprocessing as solv3r.load_images)."""
        if not paths:
            raise UserError("no images to load")
        names = names or [Path(p).name[6:] if Path(p).parent.name == "inputs" else Path(p).name for p in paths]
        out: list[torch.Tensor | None] = [None] * len(paths)
        stretched = []
        start = 0
        size = s.hw
        if size is None:                                 # the first image sets the size
            out[0], _ = _read_image(paths[0], None)
            size = tuple(out[0].shape[-2:])
            start = 1
        done = start
        with ThreadPoolExecutor(min(8, os.cpu_count() or 4), thread_name_prefix="solv3r-read") as pool:
            futures = {pool.submit(_read_image, paths[i], size): i for i in range(start, len(paths))}
            try:
                for f in as_completed(futures):
                    i = futures[f]
                    out[i], flag = f.result()
                    if flag:
                        stretched.append(names[i])
                    done += 1
                    if progress:
                        progress(done, len(paths))
            except BaseException:
                for f in futures:
                    f.cancel()
                raise
        if s.source_hw is None:
            s.source_hw = _file_hw(paths[0])
        if stretched:
            stretched.sort(key=_natural)
            s.warnings.append(f"{len(stretched)} images have another aspect ratio and are stretched to {size}: "
                              f"{', '.join(stretched[:3])}{', ...' if len(stretched) > 3 else ''}")
        s.hw = size
        return s.add_images(torch.stack(out), names, paths)

    def add_uploads(self, s: Session, files: list[tuple[str, bytes]]) -> list[int]:
        if s.hw is None:
            raise UserError("the session has no images yet")
        inputs = s.dir / "added"
        inputs.mkdir(parents=True, exist_ok=True)
        paths = []
        for name, data in files:
            if not name.lower().endswith(EXTENSIONS):
                continue
            p = inputs / f"{uuid.uuid4().hex[:6]}_{Path(name).name}"
            p.write_bytes(data)
            paths.append(str(p))
        if not paths:
            raise UserError(f"no images found: use {', '.join(EXTENSIONS)} files")
        imgs = load_images(paths, size=s.hw)
        return s.add_images(_to_u8(imgs), [Path(p).name[7:] for p in paths], paths)

    def add_live_frame(self, s: Session, jpeg_bytes: bytes) -> int:
        import io
        from PIL import Image, ImageOps
        im = ImageOps.exif_transpose(Image.open(io.BytesIO(jpeg_bytes))).convert("RGB")
        t = torch.from_numpy(np.array(im)).permute(2, 0, 1).float() / 255
        if s.hw is None:
            s.hw = model_size(*t.shape[-2:])
        idx = s.add_images(_to_u8(resize(t[None], s.hw)), [f"live_{len(s.names):05d}.jpg"])
        return idx[0]

    # ------------------------------------------------------------------ jobs
    def submit(self, s: Session, mode: str, params: dict) -> Job:
        runners = dict(offline=self._offline, incremental=self._staged, any_order=self._staged, stream=self._stream,
                       continue_=self._continue, add_views=self._add_views, perturb=self._perturb,
                       splats=self._splats)
        key = "continue_" if mode == "continue" else mode
        if key not in runners:
            raise UserError(f"unknown mode {mode}")
        with s.lock:
            if s.job is not None:
                raise UserError("a job is already running in this session")
            if s.prep is not None:
                raise UserError("the input is still loading")
            if s.images is None and not (mode == "stream" and params.get("live")):
                raise UserError("load images first")
            if mode == "splats" and s.pred is None:
                raise UserError("solve or stream first: the splats start from a reconstruction")
            job = Job(uuid.uuid4().hex[:8], mode, params)
            s.job = job
        s.emit("job", id=job.id, mode=mode, state="queued", params=params)
        self.gpu.submit(self._run, s, job, runners[key])
        return job

    def _run(self, s: Session, job: Job, fn):
        if self.model is None:
            with s.lock:
                s.job = None
            s.emit("job", id=job.id, mode=job.mode, state="error",
                   error=f"the model is not loaded: {self.load_error or 'still loading'}")
            return
        job.state = "running"
        s.emit("job", id=job.id, mode=job.mode, state="running")
        torch.cuda.reset_peak_memory_stats()
        t0 = time.time()
        state, error = "done", None
        try:
            with torch.no_grad():
                fn(s, job, job.params)
        except Cancelled:
            state = "cancelled"
            try:
                if s.kind == "solve" and s.z is not None and job.mode != "splats":
                    with torch.no_grad():
                        self._emit_cloud(s, final=True, budget=job.params.get("budget"))
            except Exception:                            # noqa: BLE001
                log.exception("final decode after cancel")
        except torch.cuda.OutOfMemoryError:
            state, error = "error", "Out of GPU memory. Try fewer views (stride), subset updates, or a lower point density."
        except UserError as e:
            state, error = "error", str(e)
        except Exception as e:                           # noqa: BLE001
            log.error("job %s failed:\n%s", job.mode, traceback.format_exc())
            state, error = "error", f"{type(e).__name__}: {e}"
        finally:
            if state == "error":
                torch.cuda.empty_cache()
            if s.live is not None and job.mode == "stream":
                s.live.open = False
                self._drain_live(s)
            peak = torch.cuda.max_memory_allocated() / 2**30 if torch.cuda.is_available() else 0.0
            with s.lock:
                s.job = None
            s.emit("job", id=job.id, mode=job.mode, state=state, error=error, seconds=round(time.time() - t0, 2),
                   peak_gib=round(peak, 2), calls=s.calls)

    # ------------------------------------------------------------------ helpers
    def _floats(self, s: Session, idx) -> torch.Tensor:
        return s.images[idx].float() / 255

    def _budget_per_view(self, s: Session, budget, n_views) -> int:
        H, W = s.hw
        budget = int(budget or 1_500_000)
        return int(min(H * W, max(1000, budget // max(1, n_views))))

    def _selection(self, s: Session, pool_idx: int, m: int) -> torch.Tensor:
        key = (pool_idx, m)
        if key not in s.sel:
            H, W = s.hw
            s.sel[key] = packing.pixel_subset(H * W, m, seed=1000 + pool_idx)
        return s.sel[key]

    def _prepare_sky(self, s: Session, idx: list[int], want: bool):
        if not want:
            return
        todo = [i for i in idx if i not in s.sky]
        if not todo:
            return
        if not os.path.exists(skyseg.SKYSEG_ONNX_PATH):
            s.emit("progress", label="Downloading the sky segmentation model (176 MB, once)", frac=None)
        s.emit("progress", label=f"Segmenting sky in {len(todo)} images", frac=0.0)
        session = self._skyseg_session()
        H, W = s.hw
        for k, i in enumerate(todo):
            bgr = np.ascontiguousarray(s.images[i].permute(1, 2, 0).numpy()[..., ::-1])   # the model takes BGR uint8
            mask = skyseg._non_sky_mask(bgr, session)                                       # 255 on non-sky
            if mask.shape != (H, W):
                import cv2
                mask = cv2.resize(mask, (W, H))
            s.sky[i] = ~(mask > 0.1)                                                        # as solv3r.export.sky_mask
            if k % 8 == 7 or k == len(todo) - 1:
                s.emit("progress", label=f"Segmenting sky in {len(todo)} images", frac=(k + 1) / len(todo))

    def _skyseg_session(self):
        """The sky segmentation ONNX session, loaded once (solv3r's helper loads it on every call)."""
        if getattr(self, "_skyseg", None) is None:
            import warnings
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore")
                self._skyseg = skyseg._load_session(intra_op_num_threads=max(1, (os.cpu_count() or 4) - 2))
        return self._skyseg

    def _sky_for(self, s: Session, idx: list[int]):
        if idx and all(i in s.sky for i in idx):
            return np.stack([s.sky[i] for i in idx])
        return None

    def _cams(self, s: Session):
        pe = self.model.decode_cameras(s.z, s.ctx)
        c2w, K = cameras_from_pose_enc(pe, s.hw)
        return c2w.numpy(), K.numpy()

    def _emit_cloud(self, s: Session, final: bool, budget=None, planned=None, label=None):
        """Decodes the whole state, stores the outputs and sends the point payload."""
        _sync()
        t = time.time()
        pred = self.model.decode(s.z, s.ctx)
        c2w, K = cameras_from_pose_enc(pred["pose_enc"], s.hw)
        S = s.ctx.S
        m = self._budget_per_view(s, budget, max(S, planned or 0))
        sel = torch.stack([self._selection(s, i, m) for i in s.order])
        data, n = packing.pack_solve(pred["world_points"], pred["depth_conf"], s.images[s.order], sel,
                                     self._sky_for(s, s.order))
        with s.lock:
            s.version += 1
            s.pred = dict(depth=pred["depth"][..., 0].float().cpu().numpy().astype(np.float16),
                          conf=pred["depth_conf"].float().cpu().numpy().astype(np.float16),
                          c2w=c2w.numpy(), K=K.numpy())
            s.png_cache.clear()
        key = s.put_cloud(data)
        del pred
        _sync()
        s.emit("cloud", key=key, n=n, views=S, final=final, call=s.calls, version=s.version,
               order=s.order, stage_of=s.stage_of, ms=round(1000 * (time.time() - t), 1), label=label,
               has_sky=self._sky_for(s, s.order) is not None, **cameras_json(c2w.numpy(), K.numpy()))

    def _solve(self, s: Session, job: Job, calls: int, *, adaptive=None, min_calls=4, subset=None, preview=1,
               budget=None, planned=None, final=True, seed=0):
        """`calls` solver calls on the session state, with per-call events and point previews every `preview`
        calls (0: only at the end). adaptive: stop once a call changes the state by less than this fraction.
        subset: update only this many random views per call; the others are held through the key / value cache."""
        model, ctx = self.model, s.ctx
        S = ctx.S
        rng = np.random.default_rng(seed + s.calls)
        subset = int(subset) if subset and 0 < int(subset) < S else None
        for i in range(int(calls)):
            job.check()
            _sync()
            t = time.time()
            z = s.z
            updated = None
            with autocast():
                if subset:
                    idx = np.sort(rng.choice(S, subset, replace=False))
                    held = np.setdiff1d(np.arange(S), idx)
                    cache = model.cache(z[held], ctx.subset(held))
                    zn = z.clone()
                    zn[idx] = model.step_cached(z[idx], ctx.subset(idx), cache).to(z.dtype)
                    updated = idx.tolist()
                    del cache
                else:
                    zn = model.step(z, ctx)
            change = float((zn.float() - z.float()).norm() / z.float().norm().clamp_min(1e-9))
            s.z = zn
            s.calls += 1
            _sync()
            ms = 1000 * (time.time() - t)
            c2w, K = self._cams(s)
            last = (i == int(calls) - 1) or (adaptive and i + 1 >= min_calls and change < adaptive)
            s.emit("call", call=s.calls, residual=change, ms=round(ms, 2), views=S, updated=updated,
                   **cameras_json(c2w, K))
            if last:
                break
            if preview and (i + 1) % int(preview) == 0:
                self._emit_cloud(s, final=False, budget=budget, planned=planned)
        self._emit_cloud(s, final=final, budget=budget, planned=planned)

    def _pool_indices(self, s: Session, p: dict) -> list[int]:
        n = len(s.images)
        idx = p.get("indices") or list(range(n))
        idx = [int(i) for i in idx if 0 <= int(i) < n]
        if not idx:
            raise UserError("no views selected")
        return idx

    def _encode(self, s: Session, idx: list[int], reference_first=True):
        _sync()
        t = time.time()
        ctx = self.model.encode(self._floats(s, idx), reference_first=reference_first)
        _sync()
        return ctx, 1000 * (time.time() - t)

    # ------------------------------------------------------------------ modes
    def _offline(self, s: Session, job: Job, p: dict):
        """All views solved together (examples/offline.py), with live previews of the state after each call."""
        idx = self._pool_indices(s, p)
        self._prepare_sky(s, idx, p.get("sky", False))
        s.reset_solve("solve")
        s.order, s.stage_of = list(idx), [0] * len(idx)
        s.emit("progress", label=f"Encoding {len(idx)} views", frac=0.0)
        s.ctx, ms = self._encode(s, idx)
        s.z = self.model.start(s.ctx)
        s.emit("encoded", views=len(idx), ms=round(ms, 1), hw=s.hw, order=s.order, stage_of=s.stage_of)
        adaptive = p.get("adaptive")
        calls = int(p.get("max_calls", 64)) if adaptive else int(p.get("calls", 16))
        if p.get("preview", 1):
            self._emit_cloud(s, final=False, budget=p.get("budget"), label="start state (the encodings)")
        self._solve(s, job, calls, adaptive=adaptive, subset=p.get("subset"), preview=p.get("preview", 1),
                    budget=p.get("budget"), seed=int(p.get("seed", 0)))

    def _staged(self, s: Session, job: Job, p: dict):
        """Incremental (file order) or any-order (random order) solves: groups of views join the ongoing solve
        (examples/incremental.py, examples/any_order.py)."""
        idx = self._pool_indices(s, p)
        n = len(idx)
        initial, group, calls = max(2, int(p.get("initial", 8))), max(1, int(p.get("group", 4))), int(p.get("calls", 8))
        if job.mode == "incremental":
            stages = [list(range(min(initial, n)))] + [list(range(a, min(a + group, n))) for a in range(initial, n, group)]
        else:
            first = sorted(set(np.linspace(0, n - 1, min(initial, n)).round().astype(int).tolist()))
            rest = np.random.default_rng(int(p.get("seed", 0))).permutation([i for i in range(n) if i not in first]).tolist()
            stages = [first] + [rest[a:a + group] for a in range(0, len(rest), group)]
        stages = [[idx[j] for j in st] for st in stages if st]
        self._prepare_sky(s, idx, p.get("sky", False))
        s.reset_solve("solve")
        for k, ids in enumerate(stages):
            job.check()
            if k == 0:
                s.ctx, ms = self._encode(s, ids)
                s.z = self.model.start(s.ctx)
            else:
                _sync()
                t = time.time()
                s.z, s.ctx = self.model.add_views(s.z, s.ctx, self._floats(s, ids))
                _sync()
                ms = 1000 * (time.time() - t)
            s.order += ids
            s.stage_of += [k] * len(ids)
            s.emit("stage", stage=k, stages=len(stages), added=ids, order=s.order, stage_of=s.stage_of,
                   ms_encode=round(ms, 1), call=s.calls,
                   label=f"Stage {k + 1}/{len(stages)}: {len(ids)} views join, {len(s.order)} in the solve")
            self._solve(s, job, calls, preview=p.get("preview", 0), budget=p.get("budget"), planned=n,
                        final=(k == len(stages) - 1))

    def _continue(self, s: Session, job: Job, p: dict):
        """More calls from the current state: the solve has no step index, so it simply continues."""
        self._need_solve(s)
        calls = int(p.get("calls", 8))
        s.emit("marker", call=s.calls, kind="continue", label=f"continue +{calls}")
        self._solve(s, job, calls, adaptive=p.get("adaptive"), subset=p.get("subset"), preview=p.get("preview", 1),
                    budget=p.get("budget"), seed=int(p.get("seed", 0)))

    def _add_views(self, s: Session, job: Job, p: dict):
        """New views join the ongoing solve; all views keep being updated."""
        self._need_solve(s)
        ids = [int(i) for i in p.get("indices", []) if 0 <= int(i) < len(s.images) and int(i) not in s.order]
        if not ids:
            raise UserError("no new views to add")
        self._prepare_sky(s, ids, bool(s.sky))
        _sync()
        t = time.time()
        s.z, s.ctx = self.model.add_views(s.z, s.ctx, self._floats(s, ids))
        _sync()
        stage = (max(s.stage_of) + 1) if s.stage_of else 0
        s.order += ids
        s.stage_of += [stage] * len(ids)
        s.emit("stage", stage=stage, stages=stage + 1, added=ids, order=s.order, stage_of=s.stage_of,
               ms_encode=round(1000 * (time.time() - t), 1), call=s.calls,
               label=f"{len(ids)} views join the solve, {len(s.order)} in total")
        s.emit("marker", call=s.calls, kind="add", label=f"+{len(ids)} views")
        self._solve(s, job, int(p.get("calls", 8)), preview=p.get("preview", 1), budget=p.get("budget"))

    def _perturb(self, s: Session, job: Job, p: dict):
        """Damages part of the state, shows it, and solves again: the solver corrects views from any state."""
        self._need_solve(s)
        S = s.ctx.S
        how = p.get("how", "reset")
        views = [int(v) for v in p.get("views", []) if 0 <= int(v) < S]
        if not views:
            k = max(1, min(int(p.get("count", max(1, S // 4))), S - 1))
            views = sorted(np.random.default_rng(int(p.get("seed", s.calls))).choice(np.arange(1, S), k, replace=False).tolist())
        z = s.z.clone()
        if how == "reset":
            z[views] = self.model.start(s.ctx)[views].to(z.dtype)
            label = f"reset {len(views)} views"
        elif how == "noise":
            sigma = float(p.get("sigma", 1.0))
            rms = z.float().pow(2).mean().sqrt()
            z[views] = (z[views].float() + sigma * rms * torch.randn_like(z[views].float())).to(z.dtype)
            label = f"noise σ={sigma:g} on {len(views)} views"
        elif how == "swap":
            if len(views) < 2:
                raise UserError("swapping needs two views")
            perm = views[1:] + views[:1]
            z[views] = s.z[perm]
            label = f"shuffled the states of {len(views)} views"
        else:
            raise UserError(f"unknown perturbation {how}")
        s.z = z
        s.emit("marker", call=s.calls, kind="perturb", label=label, views=views)
        self._emit_cloud(s, final=False, budget=p.get("budget"), label=f"after: {label}")
        c2w, K = self._cams(s)
        s.emit("cameras", call=s.calls, **cameras_json(c2w, K))
        self._solve(s, job, int(p.get("calls", 16)), preview=p.get("preview", 1), budget=p.get("budget"))

    def _splats(self, s: Session, job: Job, p: dict):
        """Gaussian splats on the current reconstruction (SHARP key frames or Solv3R points, gsplat training)."""
        from . import splats
        try:
            splats.run(s, job, p, CACHE_DIR, lambda label, frac: s.emit("progress", label=label, frac=frac))
        except splats.SplatError as e:
            raise UserError(str(e)) from None

    def splat_file(self, s: Session, rid: str, name: str) -> Path:
        """A file of a splat run: init.splat / final.splat (viewer), scene.ply, compare/<k>_<gt|init|final>.jpg."""
        if not re.fullmatch(r"[0-9a-f]{6}", rid) or not re.fullmatch(r"(init|final)\.splat|scene\.ply|"
                                                                     r"compare/\d+_(gt|init|final)\.jpg", name):
            raise KeyError(name)
        path = s.dir / "splats" / rid / name
        if not path.is_file():
            raise KeyError(name)
        return path

    def delete_splats(self, s: Session, rid: str):
        with s.lock:
            if s.job is not None and s.job.mode == "splats" and s.splat_runs and s.splat_runs[-1]["id"] == rid:
                raise UserError("this run is still going: cancel it first")
            s.splat_runs = [r for r in s.splat_runs if r["id"] != rid]
        shutil.rmtree(s.dir / "splats" / rid, ignore_errors=True)

    def _need_solve(self, s: Session):
        if s.kind != "solve" or s.z is None:
            raise UserError("run an offline, incremental or any-order solve first")

    def _stream(self, s: Session, job: Job, p: dict):
        """Streaming (examples/streaming.py): frames are pushed one at a time and each is answered on arrival."""
        live = bool(p.get("live"))
        cfg = PRESETS[p.get("config", "L8")]
        if live:
            T = None
            frames = self._live_frames(s, job)
            m_total = 1200
        else:
            idx = self._pool_indices(s, p)
            T = len(idx)
            cfg = dataclasses.replace(cfg, memory_stride=max(cfg.memory_stride, -(-T // cfg.memory)))
            self._prepare_sky(s, idx, p.get("sky", False))
            frames = ((i, None) for i in idx)
            m_total = T
        s.reset_solve("stream")
        solver = StreamingSolver(self.model, cfg, depth=True)
        budget = int(p.get("budget") or 3_000_000)
        s.emit("stream_start", config=dataclasses.asdict(cfg), frames=T, live=live, hw=s.hw)
        poses, Ks = {}, {}
        for t, (i, fut) in enumerate(frames):
            job.check()
            m = self._budget_per_view(s, budget, m_total)
            frame = self._floats(s, [i])[0]
            with s.lock:
                s.order.append(i)
                s.stage_of.append(0)
            _sync()
            t0 = time.time()
            answered = solver.push(frame)
            _sync()
            ms_total = 1000 * (time.time() - t0)
            out = []
            parts = []
            for f in answered:
                c2w = np.eye(4)
                c2w[:3, :3], c2w[:3, 3] = solver.R[f].cpu().numpy(), solver.C[f].cpu().numpy()
                K = solver.K[f].cpu().numpy()
                poses[f], Ks[f] = c2w, K
                pi = s.order[f]
                parts.append(packing.frame_points(solver.depths[f], solver.confs[f], c2w, K, s.images[pi].numpy(),
                                                  self._selection(s, pi, m), f, s.sky.get(pi)))
                out.append(f)
            payload = dict(t=t, answered=out, ms=round(solver.ms[-1], 2) if solver.ms and answered else None,
                           ms_total=round(ms_total, 1), refreshed=bool(solver.refreshed[-1]) if answered else False)
            if out:
                data, n = packing.concat(parts)
                key = f"s{s.run_id}_{t}"
                s.put_blob(key, data)
                payload.update(key=key, n=n, **cameras_json(np.stack([poses[f] for f in out]),
                                                            np.stack([Ks[f] for f in out])))
            else:
                payload.update(label=f"Collecting the {cfg.anchors} anchor frames ({t + 1}/{cfg.anchors})")
            s.emit("frame", **payload)
            if fut is not None:
                fut.set_result(dict(t=t, answered=out))
        if solver.t + 1 < cfg.anchors:
            raise UserError(f"streaming needs at least {cfg.anchors} frames (the anchors)")
        res = solver.result()
        with s.lock:
            s.version += 1
            s.pred = dict(depth=res.depth.astype(np.float16), conf=res.depth_conf.astype(np.float16),
                          c2w=res.c2w, c2w_exit=res.c2w_exit, K=res.intrinsics)
        s.emit("stream_done", frames=solver.t + 1, ms_per_frame=round(res.ms_per_frame, 2),
               refreshed=[i for i, r in enumerate(res.refreshed) if r], version=s.version,
               exit=_round(res.c2w_exit[:, :3, :4].reshape(len(res.c2w_exit), 12)))

    def _live_frames(self, s: Session, job: Job):
        while True:
            job.check()
            try:
                item = s.live.q.get(timeout=0.25)
            except queue.Empty:
                continue
            if item is None:
                return
            yield item

    def _drain_live(self, s: Session):
        while True:
            try:
                item = s.live.q.get_nowait()
            except queue.Empty:
                return
            if item is not None and not item[1].done():
                item[1].set_result(dict(t=None, answered=[], closed=True))

    # ------------------------------------------------------------------ outputs
    def depth_png(self, s: Session, pos: int, kind: str) -> bytes:
        key = (s.version, pos, kind)
        with s.lock:
            if key in s.png_cache:
                return s.png_cache[key]
            pred = s.pred
        if pred is None or not (0 <= pos < len(pred["depth"])):
            raise KeyError(pos)
        if kind == "depth":
            img = packing.colorize(pred["depth"][pos].astype(np.float32), salient_low=True)
        elif kind == "conf":
            img = packing.colorize(pred["conf"][pos].astype(np.float32), salient_low=False)
        else:
            raise KeyError(kind)
        data = packing.png(img)
        with s.lock:
            s.png_cache[key] = data
        return data

    def export(self, s: Session, keep: float, sky: bool) -> Path:
        """Writes <dir>/solve.{npz,ply,html} (the files of the Solv3R examples) for the current outputs."""
        keep = float(min(1.0, max(0.01, keep)))
        key = (s.version, round(keep, 3), bool(sky))
        if key in s.exports and s.exports[key].exists():
            return s.exports[key]
        out = s.dir / f"export_v{s.version}_k{int(keep * 1000)}_s{int(bool(sky))}"
        out.mkdir(parents=True, exist_ok=True)
        if s.kind == "solve" and s.z is not None:
            fut = self.gpu.submit(self._export_solve, s, out, keep, sky)
        elif s.kind == "stream" and s.pred is not None:
            fut = self.cpu.submit(self._export_stream, s, out, keep, sky)
        else:
            raise UserError("nothing to export yet")
        fut.result()
        s.exports[key] = out
        return out

    def _export_solve(self, s: Session, out: Path, keep: float, sky: bool):
        with torch.no_grad():
            pred = self.model.decode(s.z, s.ctx)
        c2w, K = cameras_from_pose_enc(pred["pose_enc"], s.hw)
        depth = pred["depth"][..., 0].float().cpu().numpy()
        conf = pred["depth_conf"].float().cpu().numpy()
        names = [s.names[i] for i in s.order]
        np.savez_compressed(out / "solve.npz", c2w=c2w.numpy(), K=K.numpy(), depth=depth.astype(np.float16),
                            depth_conf=conf.astype(np.float16), names=np.array(names))
        masks = self._sky_for(s, s.order) if sky else None
        images = self._floats(s, s.order)
        xyz, rgb = confident_points(pred["world_points"].float().cpu().numpy(), conf, images, keep,
                                    None if masks is None else ~masks)
        del pred
        write_ply(out / "solve.ply", xyz, rgb)
        write_html(out / "solve.html", xyz, rgb, c2w.numpy(), title="Solv3R solve")

    def _export_stream(self, s: Session, out: Path, keep: float, sky: bool):
        pred = s.pred
        T = len(pred["depth"])
        H, W = s.hw
        np.savez_compressed(out / "solve.npz", c2w=pred["c2w"], c2w_exit=pred["c2w_exit"], K=pred["K"],
                            depth=pred["depth"], depth_conf=pred["conf"],
                            names=np.array([s.names[i] for i in s.order[:T]]))
        every = max(1, int(-(-T * H * W * keep // 20e6)))
        idx = list(range(0, T, every))
        frames = self._floats(s, [s.order[i] for i in idx])
        points = unproject(pred["depth"][idx].astype(np.float32), pred["c2w"][idx], pred["K"][idx])
        masks = self._sky_for(s, [s.order[i] for i in idx]) if sky else None
        xyz, rgb = confident_points(points, pred["conf"][idx].astype(np.float32), frames, keep,
                                    None if masks is None else ~masks)
        write_ply(out / "solve.ply", xyz, rgb)
        write_html(out / "solve.html", xyz, rgb, pred["c2w"], title="Solv3R stream")
