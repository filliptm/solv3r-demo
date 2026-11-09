"""Gaussian splats from a Solv3R reconstruction.

Where the splats start from (`init`):
- "sharp": SHARP (Apple, arXiv:2512.10685) predicts ~1.2M Gaussians from one image, in that image's camera, with a
  metric scale. It runs on a few key frames. Each frame's Gaussians are placed with Solv3R's camera of the frame and
  matched to Solv3R's depth of the frame: one scale ("scale"), or that scale and a smooth per-pixel correction
  ("depth"), which bends SHARP's single-image geometry onto the multi-view one. The key frames are then merged:
  a surface seen by several key frames keeps the Gaussians of the frame that sees it closest (the finest), and
  Gaussians that float in front of a surface another key frame sees are removed.
- "points": Solv3R's confident pixels of the training views, as small round Gaussians (what the point cloud shows).

Training (`steps` > 0) fits the splats to the frames with gsplat (L1 + SSIM, MCMC densification with a cap),
at up to `resolution` on the long side (never above the source frames), optionally refining the cameras (pose and
focal length) and a per-frame exposure. Coarse to fine (`coarse`): the first half of the steps renders at a half,
a quarter ... of the size (down to ~500 px), the second half at the full size; this is faster and lets the
camera refinement settle before the fine detail. `max_splats` 0 picks the cap from the training size.

Scoring: with enough views every 8th view is held out of the training (and of the key frames). Each held-out view
is rendered after a short test-time alignment of its camera and exposure (50 steps with the splats frozen, as in
InstantSplat; 20 more at each later score; at most ~2K px, then scored at the full size) and scored with PSNR and
SSIM, at the start (step 0: SHARP or the points as they are) and during the training.

SHARP's weights are licensed for research only (LICENSE_MODEL in apple/ml-sharp).
"""
from __future__ import annotations

import copy
import functools
import importlib.util
import logging
import math
import os
import shutil
import sys
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

log = logging.getLogger("solv3r.demo.splats")

SHARP_URL = "https://ml-site.cdn-apple.com/models/sharp/sharp_2572gikvuh.pt"
SHARP_SIDE = 1536                     # SHARP's input resolution (it resizes every image to this square)
C0 = 0.28209479177387814              # degree-0 spherical harmonic
EVAL_VIEWS = 12                       # held-out views scored (spread over the held-out set)
ALIGN_STEPS = 50                      # test-time alignment of a held-out view's camera and exposure (first score)
ALIGN_MORE = 20                       # ... and at each later score of the run (continuing from the previous one)
MAX_RUNS = 6
IMAGE_EXT = (".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff")
ALIGN_SIDE = 2048                     # test-time alignment at most this long side (then scored at the full size)
COMPARE_SIDE = 1008                   # runs trained above this size are also scored at it (psnr_compare)
COARSE_SIDE = 480                     # coarse to fine: the coarsest level is at least this long side ...
COARSE_LEVELS = 4                     # ... with at most this many levels (4: 1/8, 1/4, 1/2, 1)
COARSE_PART = 0.5                     # ... spread over this part of the steps (the rest is at the full size)

DEFAULTS = dict(init="sharp", keyframes=8, align="depth", steps=3000, refine_cameras=True, exposure=True,
                holdout=True, resolution=1008, max_splats=2_000_000, sh_degree=3, sky=None, coarse=True)


def auto_budget(w: int, h: int) -> int:
    """The splat cap for `max_splats` 0: 2M up to ~0.6 MP of training size, growing with the square root of the
    pixel count, at most 6M (what a browser shows smoothly)."""
    mp = w * h / 1e6
    return int(round(min(6e6, max(2e6, 2e6 * math.sqrt(mp / 0.6))) / 1e5) * 1e5)


def coarse_levels(long_side: int, on: bool) -> list[int]:
    """The divisors of the training size, coarsest first: [8, 4, 2, 1] for 4K, [2, 1] for 1008 px, [1] off."""
    k = 0
    while on and k < COARSE_LEVELS - 1 and long_side / 2 ** (k + 1) >= COARSE_SIDE:
        k += 1
    return [2 ** i for i in range(k, -1, -1)]


def level_at(step: int, steps: int, levels: list[int]) -> int:
    """The level (index into levels) of a step: the coarse levels share the first COARSE_PART of the steps."""
    m = len(levels) - 1
    if m == 0:
        return 0
    return min(m, int((step - 1) / max(1, steps) / (COARSE_PART / m)))


class SplatError(Exception):
    """A problem with the request, shown to the user as is."""


# ---------------------------------------------------------------------------------------------- gsplat setup
_gsplat = None


def gsplat():
    """gsplat, with its CUDA kernels compiled on first use (about a minute once, cached by torch afterwards)."""
    global _gsplat
    if _gsplat is None:
        _cuda_env()
        _patch_gsplat()
        import gsplat as g
        from gsplat.cuda._backend import _C
        if _C is None:
            raise RuntimeError("gsplat has no CUDA kernels: the CUDA toolkit was not found (see the server log)")
        _gsplat = g
    return _gsplat


def _cuda_env():
    """The JIT build needs ninja (next to python), MSVC (found by torch) and the CUDA toolkit that matches torch
    (CUDA_PATH may point at an older one that does not know this GPU)."""
    scripts = os.path.dirname(sys.executable)
    if scripts not in os.environ.get("PATH", "").split(os.pathsep):
        os.environ["PATH"] = scripts + os.pathsep + os.environ.get("PATH", "")
    if torch.cuda.is_available():
        major, minor = torch.cuda.get_device_capability()
        os.environ.setdefault("TORCH_CUDA_ARCH_LIST", f"{major}.{minor}")
    if os.name == "nt" and torch.version.cuda:
        home = (Path(os.environ.get("ProgramFiles", r"C:\Program Files")) / "NVIDIA GPU Computing Toolkit" / "CUDA"
                / f"v{torch.version.cuda}")
        if (home / "bin" / "nvcc.exe").exists():
            os.environ["CUDA_HOME"] = os.environ["CUDA_PATH"] = str(home)
            if "torch.utils.cpp_extension" in sys.modules:
                sys.modules["torch.utils.cpp_extension"].CUDA_HOME = str(home)


def _patch_gsplat():
    """Two fixes gsplat 1.5.3 needs to compile with MSVC (applied once to the installed package): cl rejects a GCC
    flag, and rpcndr.h's `#define small char` breaks a torch header that IntersectTile.cu includes."""
    if os.name != "nt":
        return
    spec = importlib.util.find_spec("gsplat")
    if spec is None or not spec.submodule_search_locations:
        return
    root = Path(next(iter(spec.submodule_search_locations)))
    backend = root / "cuda" / "_backend.py"
    src = backend.read_text(encoding="utf-8")
    old = '        extra_cflags = [opt_level, "-Wno-attributes"]\n'
    if old in src and 'os.name == "nt"' not in src:
        backend.write_text(src.replace(old, old + '        if os.name == "nt":  # MSVC (cl) rejects the GCC flags above\n'
                                                  '            extra_cflags = ["/Od" if FAST_COMPILE else "/O2"]\n'),
                           encoding="utf-8")
    cu = root / "cuda" / "csrc" / "IntersectTile.cu"
    src = cu.read_text(encoding="utf-8")
    inc = "#include <c10/cuda/CUDACachingAllocator.h>"
    if inc in src and "#undef small" not in src:
        cu.write_text(src.replace(inc, "#ifdef small  // Windows: rpcndr.h defines small as char\n#undef small\n"
                                       "#endif\n" + inc, 1), encoding="utf-8")


# ---------------------------------------------------------------------------------------------- SHARP
_sharp = None


def sharp_model(cache_dir: Path, device="cuda"):
    """SHARP's predictor, downloaded once (2.8 GB) and kept on the GPU."""
    global _sharp
    if _sharp is None:
        from sharp.models import PredictorParams, create_predictor
        state = torch.hub.load_state_dict_from_url(SHARP_URL, model_dir=str(cache_dir), progress=False,
                                                   map_location="cpu")
        net = create_predictor(PredictorParams())
        net.load_state_dict(state)
        _sharp = net.eval().to(device)
    return _sharp


def sharp_ready(cache_dir: Path) -> bool:
    return (cache_dir / SHARP_URL.rsplit("/", 1)[-1]).exists()


@torch.no_grad()
def sharp_predict(net, img: torch.Tensor, f_px: float):
    """img (3, H, W) float on the GPU -> Gaussians in the camera of the image (OpenCV axes, metres): means (N, 3),
    shape factors L (N, 3, 3, covariance L Lᵀ), rgb (N, 3, sRGB), opacity (N,). As sharp.cli.predict, with the
    unprojection on the GPU (SHARP's own decomposes 1.2M covariances on the CPU, ~5 s); the factors are decomposed
    once, after the key frames are merged."""
    from sharp.utils.color_space import linearRGB2sRGB
    from sharp.utils.gaussians import get_unprojection_matrix
    H, W = img.shape[-2:]
    dev = img.device
    x = F.interpolate(img[None], size=(SHARP_SIDE, SHARP_SIDE), mode="bilinear", align_corners=True)
    g = net(x, torch.tensor([f_px / W], device=dev, dtype=torch.float32))
    K = torch.tensor([[f_px, 0, W / 2, 0], [0, f_px, H / 2, 0], [0, 0, 1, 0], [0, 0, 0, 1]], device=dev,
                     dtype=torch.float32)
    K[0] *= SHARP_SIDE / W
    K[1] *= SHARP_SIDE / H
    M = get_unprojection_matrix(torch.eye(4, device=dev), K, (SHARP_SIDE, SHARP_SIDE))[:3]
    A, b = M[:, :3], M[:, 3]
    means = g.mean_vectors[0] @ A.T + b
    L = A @ (quat_to_rot(g.quaternions[0]) * g.singular_values[0][:, None, :])
    rgb = linearRGB2sRGB(g.colors[0]).clamp(0, 1)
    return means, L, rgb, g.opacities[0].clamp(0, 1)


# ---------------------------------------------------------------------------------------------- geometry
def quat_to_rot(q: torch.Tensor) -> torch.Tensor:
    q = F.normalize(q, dim=-1)
    w, x, y, z = q.unbind(-1)
    return torch.stack([1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y),
                        2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x),
                        2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)], -1).view(*q.shape[:-1], 3, 3)


def rot_to_quat(R: torch.Tensor) -> torch.Tensor:
    """Rotation matrices -> unit quaternions (wxyz), from the largest of the four candidate components."""
    m = R
    t = m[..., 0, 0] + m[..., 1, 1] + m[..., 2, 2]
    cands = []
    s = (1 + t).clamp_min(1e-12).sqrt() * 2
    cands.append(torch.stack([s / 4, (m[..., 2, 1] - m[..., 1, 2]) / s, (m[..., 0, 2] - m[..., 2, 0]) / s,
                              (m[..., 1, 0] - m[..., 0, 1]) / s], -1))
    s = (1 + m[..., 0, 0] - m[..., 1, 1] - m[..., 2, 2]).clamp_min(1e-12).sqrt() * 2
    cands.append(torch.stack([(m[..., 2, 1] - m[..., 1, 2]) / s, s / 4, (m[..., 0, 1] + m[..., 1, 0]) / s,
                              (m[..., 0, 2] + m[..., 2, 0]) / s], -1))
    s = (1 + m[..., 1, 1] - m[..., 0, 0] - m[..., 2, 2]).clamp_min(1e-12).sqrt() * 2
    cands.append(torch.stack([(m[..., 0, 2] - m[..., 2, 0]) / s, (m[..., 0, 1] + m[..., 1, 0]) / s, s / 4,
                              (m[..., 1, 2] + m[..., 2, 1]) / s], -1))
    s = (1 + m[..., 2, 2] - m[..., 0, 0] - m[..., 1, 1]).clamp_min(1e-12).sqrt() * 2
    cands.append(torch.stack([(m[..., 1, 0] - m[..., 0, 1]) / s, (m[..., 0, 2] + m[..., 2, 0]) / s,
                              (m[..., 1, 2] + m[..., 2, 1]) / s, s / 4], -1))
    cands = torch.stack(cands, -2)                                           # (..., 4, 4)
    pick = torch.stack([t, m[..., 0, 0], m[..., 1, 1], m[..., 2, 2]], -1).argmax(-1)
    q = torch.gather(cands, -2, pick[..., None, None].expand(*pick.shape, 1, 4))[..., 0, :]
    return F.normalize(q, dim=-1)


def decompose_factor(L: torch.Tensor, chunk=1 << 20):
    """Shape factors (N, 3, 3), covariance L Lᵀ -> scales (N, 3) and rotations (N, 4): L = U Σ Vᵀ gives the
    covariance U Σ² Uᵀ. (A batched SVD of the factor: better conditioned than the covariance, and cuSOLVER's batched
    symmetric eigensolver fails on this torch / GPU.)"""
    scales, quats = [], []
    for a in range(0, len(L), chunk):
        U, S, _ = torch.linalg.svd(L[a:a + chunk].float())
        U = U.clone()
        U[torch.linalg.det(U) < 0, :, 2] *= -1
        scales.append(S.clamp_min(1e-10))
        quats.append(rot_to_quat(U))
    return torch.cat(scales), torch.cat(quats)


def axis_angle_to_rot(w: torch.Tensor) -> torch.Tensor:
    """(..., 3) -> (..., 3, 3), exact and differentiable at 0."""
    z = torch.zeros_like(w[..., 0])
    S = torch.stack([z, -w[..., 2], w[..., 1], w[..., 2], z, -w[..., 0], -w[..., 1], w[..., 0], z], -1)
    return torch.linalg.matrix_exp(S.view(*w.shape[:-1], 3, 3))


def delta_pose(rot: torch.Tensor, trans: torch.Tensor) -> torch.Tensor:
    """A small camera correction as a 4x4 (applied on the camera side: viewmat' = delta @ viewmat)."""
    D = torch.eye(4, device=rot.device, dtype=rot.dtype).expand(*rot.shape[:-1], 4, 4).clone()
    D[..., :3, :3] = axis_angle_to_rot(rot)
    D[..., :3, 3] = trans
    return D


def refined_view(w2c: torch.Tensor, rot=None, trans=None, focal=None) -> torch.Tensor:
    """A camera with a pose correction (rot, trans) and a focal length correction (log fx, log fy). gsplat has no
    gradient for the intrinsics, so the focal correction scales the camera's x and y instead: fx·(s·x)/z is the
    projection of a camera with the focal length s·fx (covariances and the view direction of the colours follow)."""
    vm = delta_pose(rot, trans) @ w2c if rot is not None else w2c
    if focal is not None:
        s = torch.cat([torch.exp(focal), torch.ones(2, device=focal.device, dtype=focal.dtype)])
        vm = s[:, None] * vm
    return vm


def project(X: torch.Tensor, w2c: torch.Tensor, K: torch.Tensor):
    """World points (N, 3) -> camera depth z (N,) and pixel coordinates u, v (N,)."""
    Xc = X @ w2c[:3, :3].T + w2c[:3, 3]
    z = Xc[:, 2]
    zs = z.clamp_min(1e-9)
    return z, K[0, 0] * Xc[:, 0] / zs + K[0, 2], K[1, 1] * Xc[:, 1] / zs + K[1, 2]


def sample_map(m: torch.Tensor, u: torch.Tensor, v: torch.Tensor):
    """Nearest pixel of a (H, W) map at u, v (clamped to the image)."""
    H, W = m.shape
    return m[v.long().clamp(0, H - 1), u.long().clamp(0, W - 1)]


# ---------------------------------------------------------------------------------------------- metrics
_win = {}


def _window(device):
    """The 11-tap Gaussian (sigma 1.5) of 3DGS's SSIM as a vertical and a horizontal kernel for the 15 channels of
    ssim's stack: the 11x11 window is separable (zero padding too), so two 1D passes give the same result."""
    if device not in _win:
        x = torch.arange(11, device=device, dtype=torch.float32) - 5
        g = torch.exp(-x ** 2 / (2 * 1.5 ** 2))
        g = g / g.sum()
        _win[device] = (g.view(1, 1, 11, 1).repeat(15, 1, 1, 1), g.view(1, 1, 1, 11).repeat(15, 1, 1, 1))
    return _win[device]


def ssim(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """SSIM of two (H, W, 3) images in [0, 1] (11x11 Gaussian window, as in 3DGS)."""
    a = a.permute(2, 0, 1)[None]
    b = b.permute(2, 0, 1)[None]
    wv, wh = _window(a.device)
    x = torch.cat([a, b, a * a, b * b, a * b], 1)
    m = F.conv2d(F.conv2d(x, wv, padding=(5, 0), groups=15), wh, padding=(0, 5), groups=15)
    mu_a, mu_b, aa, bb, ab = m.split(3, 1)
    s_aa = aa - mu_a ** 2
    s_bb = bb - mu_b ** 2
    s_ab = ab - mu_a * mu_b
    c1, c2 = 0.01 ** 2, 0.03 ** 2
    return (((2 * mu_a * mu_b + c1) * (2 * s_ab + c2)) / ((mu_a ** 2 + mu_b ** 2 + c1) * (s_aa + s_bb + c2))).mean()


def psnr_t(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    return -10 * torch.log10(F.mse_loss(a, b).clamp_min(1e-10))


def psnr(a: torch.Tensor, b: torch.Tensor) -> float:
    return float(psnr_t(a, b))


# ---------------------------------------------------------------------------------------------- the scene
class Scene:
    """The views of a reconstruction: Solv3R's cameras, depth and confidence (model size) and the frames at the
    training size, all on the GPU."""

    def __init__(self, s, p: dict, device="cuda", progress=None):
        pred = s.pred
        V = len(pred["depth"])
        self.order = list(s.order[:V])
        self.names = [s.names[i] for i in self.order]
        self.hm, self.wm = Hm, Wm = s.hw
        self.device = device
        self.c2w = torch.from_numpy(np.asarray(pred["c2w"], np.float32)).to(device)
        self.w2c = torch.linalg.inv(self.c2w)
        self.Km = torch.from_numpy(np.asarray(pred["K"], np.float32)).to(device)
        self.depth = torch.from_numpy(np.asarray(pred["depth"], np.float32)).to(device)
        self.conf = torch.from_numpy(np.asarray(pred["conf"], np.float32)).to(device)
        self.sky = None
        if p["sky"] and all(i in s.sky for i in self.order):
            self.sky = torch.from_numpy(np.stack([s.sky[i] for i in self.order])).to(device)
        # the training size: up to `resolution` on the long side, never above the source frames. Above the model
        # size it keeps the source's aspect ratio: the model size rounds each side to 14 px and the pool images are
        # stretched to it (no crop), so the intrinsics scale per axis and the frames are not stretched back
        self.paths = [s.paths[i] if i < len(s.paths) else None for i in self.order]
        src_hw = _source_hw(self.paths[0])
        src = max(src_hw) if src_hw else max(Hm, Wm)
        long_t = int(max(max(Hm, Wm), min(int(p["resolution"]), src)))
        if src_hw and long_t > max(Hm, Wm):
            r = long_t / src
            self.ht, self.wt = round(src_hw[0] * r), round(src_hw[1] * r)
        else:
            r = long_t / max(Hm, Wm)
            self.ht, self.wt = round(Hm * r), round(Wm * r)
        self.Kt = self.Km.clone()
        self.Kt[:, 0] *= self.wt / Wm
        self.Kt[:, 1] *= self.ht / Hm
        self.src_long = src
        self.pool_images = s.images
        # every 8th view is held out (with enough views)
        V = len(self.order)
        self.test = [q for q in range(V) if q % 8 == 4] if p["holdout"] and V >= 12 else []
        held = set(self.test)
        self.train = [q for q in range(V) if q not in held]
        self.eval = [self.test[round(k)] for k in np.linspace(0, len(self.test) - 1, min(EVAL_VIEWS, len(self.test)))] \
            if self.test else []
        self.images = self._load(range(V), (self.ht, self.wt), progress)
        centers = self.c2w[:, :3, 3]
        extent = float((centers - centers.mean(0)).norm(dim=1).max()) * 1.1
        self.median_depth = float(self.depth[::max(1, V // 16)].median())
        self.scale = max(extent, 0.5 * self.median_depth)           # sets the learning rates of positions
        self.near = 0.01 * self.median_depth
        self._levels: dict[int, tuple] = {}

    def level(self, d: int):
        """The training size divided by d: (H, W, intrinsics of every view, frames (V, H, W, 3) uint8). The frames
        are area-averaged from the full-size ones once per level."""
        if d == 1:
            return self.ht, self.wt, self.Kt, self.images
        if d not in self._levels:
            H, W = max(8, round(self.ht / d)), max(8, round(self.wt / d))
            K = self.Kt.clone()
            K[:, 0] *= W / self.wt
            K[:, 1] *= H / self.ht
            out = torch.empty(len(self.images), H, W, 3, dtype=torch.uint8, device=self.device)
            for a in range(0, len(self.images), 16):
                x = self.images[a:a + 16].permute(0, 3, 1, 2).float()
                out[a:a + 16] = F.interpolate(x, size=(H, W), mode="area").round().byte().permute(0, 2, 3, 1)
            self._levels[d] = (H, W, K, out)
        return self._levels[d]

    def _load(self, positions, size, progress=None) -> torch.Tensor:
        positions = list(positions)
        out = [None] * len(positions)
        with ThreadPoolExecutor(min(8, os.cpu_count() or 4), thread_name_prefix="splat-read") as pool:
            futs = {pool.submit(self.frame, q, size): k for k, q in enumerate(positions)}
            for n, f in enumerate(futs):
                out[futs[f]] = f.result()
                if progress and (n % 8 == 7 or n == len(positions) - 1):
                    progress(f"Loading {len(positions)} frames at {size[1]}×{size[0]}", (n + 1) / len(positions))
        return torch.from_numpy(np.stack(out)).to(self.device)

    def frame(self, q: int, size) -> np.ndarray:
        """View q as (H, W, 3) uint8 at `size`: from its source file (photo, video frame), else the model-size image."""
        from PIL import Image, ImageOps
        H, W = size
        path = self.paths[q]
        if path and os.path.isfile(path) and path.lower().endswith(IMAGE_EXT):
            im = ImageOps.exif_transpose(Image.open(path)).convert("RGB")
            if im.size != (W, H):
                im = im.resize((W, H), Image.LANCZOS)
            return np.array(im)                                # writable (torch.from_numpy)
        t = self.pool_images[self.order[q]].float()[None] / 255
        if t.shape[-2:] != (H, W):
            t = F.interpolate(t, size=(H, W), mode="bilinear", align_corners=False, antialias=True)
        return (t[0].clamp(0, 1) * 255).round().byte().permute(1, 2, 0).numpy()


def _source_hw(path) -> tuple[int, int] | None:
    """(height, width) of a source image as shown (after its EXIF rotation), from its header."""
    if not path or not os.path.isfile(path) or not path.lower().endswith(IMAGE_EXT):
        return None
    try:
        from PIL import Image
        with Image.open(path) as im:
            w, h = im.size
            if im.getexif().get(0x0112) in (5, 6, 7, 8):
                w, h = h, w
            return h, w
    except OSError:
        return None


def pick_keyframes(sc: Scene, k: int) -> list[int]:
    """k training views spread over the capture: farthest-point sampling on camera position and direction."""
    cand = sc.train
    if k >= len(cand):
        return list(cand)
    C = sc.c2w[cand, :3, 3]
    d = sc.c2w[cand, :3, 2]
    ext = float((C - C.mean(0)).norm(dim=1).max().clamp_min(1e-9))
    dist = lambda a: (C - C[a]).norm(dim=1) / ext + 0.5 * (1 - d @ d[a])   # noqa: E731
    chosen = [0]
    best = dist(0)
    while len(chosen) < k:
        j = int(best.argmax())
        chosen.append(j)
        best = torch.minimum(best, dist(j))
    return sorted(cand[j] for j in chosen)


# ---------------------------------------------------------------------------------------------- starting splats
def sharp_init(g, sc: Scene, keys: list[int], p: dict, net, check, progress):
    """SHARP on the key frames, each placed with Solv3R's camera and depth, then merged (see the module docstring).
    Returns the Gaussians and per key frame statistics."""
    dev = sc.device
    side = int(min(SHARP_SIDE, max(sc.src_long, max(sc.hm, sc.wm))))
    r = side / max(sc.hm, sc.wm)
    size_s = (round(sc.hm * r), round(sc.wm * r))
    parts, stats = [], []
    t_net = 0.0
    for k, q in enumerate(keys):
        check()
        progress(f"SHARP on key frame {k + 1} of {len(keys)} ({sc.names[q]})", k / len(keys))
        img = torch.from_numpy(sc.frame(q, size_s)).to(dev).permute(2, 0, 1).float() / 255
        torch.cuda.synchronize()
        t = time.time()
        means, L, rgb, op = sharp_predict(net, img, float(sc.Km[q, 0, 0]) * size_s[1] / sc.wm)
        torch.cuda.synchronize()
        t_net += time.time() - t
        f, st = _align(g, sc, q, means, L, rgb, op, p["align"])
        st.update(view=q, name=sc.names[q])
        stats.append(st)
        keep = (op > 0.02) & (means[:, 2] > 0)
        z, u, v = project(means, torch.eye(4, device=dev), sc.Km[q])
        if sc.sky is not None:
            keep &= ~sample_map(sc.sky[q], u, v)
        if p.get("support", bool(p["steps"])):
            # for training, only where Solv3R's depth of the frame is confident: elsewhere the depth fit has nothing
            # to go on and SHARP's single-image depth puts the splats in the wrong place for the other views (better
            # after training, measured on the Desk video; without training these guesses still help: they fill gaps)
            keep &= sample_map(sc.conf[q], u, v) >= torch.quantile(sc.conf[q].flatten()[::4], 0.3)
        if p.get("layers", "both") == "front":
            keep[len(keep) // 2:] = False                      # SHARP's output is layer-major: front layer first
        means, L, rgb, op, f = (a[keep] for a in (means, L, rgb, op, f))
        Rw, tw = sc.c2w[q, :3, :3], sc.c2w[q, :3, 3]
        cam = means * f[:, None]
        parts.append(dict(means=cam @ Rw.T + tw, L=(Rw @ L) * f[:, None, None], rgb=rgb, op=op,
                          foot=cam[:, 2] / sc.Km[q, 0, 0], key=q))
        del img
    progress("Merging the key frames", None)
    total = sum(len(a["means"]) for a in parts)
    kept = _merge(sc, parts)
    G = {k: torch.cat([a[k][m] for a, m in zip(parts, kept)]) for k in ("means", "L", "rgb", "op")}
    del parts
    merged = len(G["means"])
    start = int(p["max_splats"]) // 2 if p["steps"] else int(p["max_splats"])   # as the points: room to densify
    if merged > start:
        sel = torch.randperm(merged, device=dev)[:start]
        G = {k: v[sel] for k, v in G.items()}
    G["scales"], G["quats"] = decompose_factor(G.pop("L"))
    for st, m in zip(stats, kept):
        st["kept"] = int(m.sum())
    return G, dict(keyframes=stats, predicted=total, merged=merged, sharp_seconds=round(t_net, 2),
                   sharp_size=[size_s[1], size_s[0]])


def _align(g, sc: Scene, q: int, means, L, rgb, op, mode: str):
    """Per-Gaussian factor (N,) that brings SHARP's depth of frame q onto Solv3R's: exp(global log scale + smooth
    correction). The correction is a weighted average of the log depth ratio in cells of ~24 px, smoothed and
    interpolated; cells without confident support fall back to the global scale."""
    dev = sc.device
    Hm, Wm = sc.hm, sc.wm
    n = len(means)
    out, alpha, _ = g.rasterization(means, torch.zeros(n, 4, device=dev), torch.zeros(n, 3, device=dev), op, rgb,
                                    torch.eye(4, device=dev)[None], sc.Km[q][None], Wm, Hm, near_plane=1e-3,
                                    render_mode="ED", packed=False, covars=L @ L.transpose(1, 2))
    zs, a = out[0, ..., 0], alpha[0, ..., 0]
    D, C = sc.depth[q], sc.conf[q]
    valid = (a > 0.5) & (zs > 0) & (D > 0) & (C >= torch.quantile(C.flatten()[::4], 0.3))
    if sc.sky is not None:
        valid &= ~sc.sky[q]
    if int(valid.sum()) < 500:
        valid = (a > 0.5) & (zs > 0) & (D > 0)
    logr = torch.log(D.clamp_min(1e-9)) - torch.log(zs.clamp_min(1e-9))
    g0 = logr[valid].median()
    resid = (logr - g0)[valid]
    field = torch.zeros_like(D)
    if mode == "depth":
        w = valid.float()
        r = ((logr - g0).clamp(-0.7, 0.7) * w)[None, None]
        gh, gw = max(4, round(Hm / 24)), max(4, round(Wm / 24))
        rs = F.adaptive_avg_pool2d(r, (gh, gw))
        ws = F.adaptive_avg_pool2d(w[None, None], (gh, gw))
        k = torch.tensor([[1., 2., 1.], [2., 4., 2.], [1., 2., 1.]], device=dev)[None, None] / 16
        rs = F.conv2d(F.pad(rs, (1, 1, 1, 1), mode="replicate"), k)
        ws = F.conv2d(F.pad(ws, (1, 1, 1, 1), mode="replicate"), k)
        cell = torch.where(ws > 0.05, rs / ws.clamp_min(1e-6), torch.zeros_like(rs))
        field = F.interpolate(cell, size=(Hm, Wm), mode="bilinear", align_corners=False)[0, 0]
    z, u, v = project(means, torch.eye(4, device=dev), sc.Km[q])
    grid = torch.stack([u / Wm * 2 - 1, v / Hm * 2 - 1], -1)[None, None]
    corr = F.grid_sample(field[None, None], grid, mode="bilinear", padding_mode="border", align_corners=False)[0, 0, 0]
    after = (resid - (field[valid] if mode == "depth" else 0)).abs()
    return torch.exp(g0 + corr), dict(scale=round(float(torch.exp(g0)), 5),
                                      agree_before=round(float(resid.abs().median()), 4),
                                      agree_after=round(float(after.median()), 4),
                                      coverage=round(float(valid.float().mean()), 3))


def _merge(sc: Scene, parts, tau_surface=0.05, tau_free=0.12):
    """Which Gaussians of each key frame to keep: drop those on a surface another key frame sees closer (finer), and
    those in front of a surface another key frame sees (free space), where that frame's depth is confident."""
    keys = [a["key"] for a in parts]
    thr = {q: torch.quantile(sc.conf[q].flatten()[::4], 0.3) for q in keys}
    kept = []
    for a in parts:
        drop = torch.zeros(len(a["means"]), dtype=torch.bool, device=sc.device)
        for q in keys:
            if q == a["key"]:
                continue
            z, u, v = project(a["means"], sc.w2c[q], sc.Km[q])
            inb = (z > 0) & (u >= 0) & (u < sc.wm) & (v >= 0) & (v < sc.hm)
            D = sample_map(sc.depth[q], u, v)
            ok = inb & (sample_map(sc.conf[q], u, v) >= thr[q]) & (D > 0)
            if sc.sky is not None:
                ok &= ~sample_map(sc.sky[q], u, v)
            surface = ok & ((z - D).abs() < tau_surface * D)
            drop |= surface & (z / sc.Km[q, 0, 0] < a["foot"])
            drop |= ok & (z < D * (1 - tau_free))
        kept.append(~drop)
    return kept


def points_init(sc: Scene, p: dict):
    """Solv3R's confident pixels of the training views (the top 60 % by confidence per view) as round Gaussians the
    size of the pixel footprint (scaled up for the subsampling)."""
    dev = sc.device
    Hm, Wm = sc.hm, sc.wm
    vv, uu = torch.meshgrid(torch.arange(Hm, device=dev, dtype=torch.float32) + 0.5,
                            torch.arange(Wm, device=dev, dtype=torch.float32) + 0.5, indexing="ij")
    xs, cs, fs = [], [], []
    for q in sc.train:
        D, C = sc.depth[q], sc.conf[q]
        m = (C >= torch.quantile(C.flatten()[::4], 0.4)) & (D > 0)
        if sc.sky is not None:
            m &= ~sc.sky[q]
        K = sc.Km[q]
        d = D[m]
        Xc = torch.stack([(uu[m] - K[0, 2]) / K[0, 0] * d, (vv[m] - K[1, 2]) / K[1, 1] * d, d], -1)
        xs.append(Xc @ sc.c2w[q, :3, :3].T + sc.c2w[q, :3, 3])
        cs.append(sc.pool_images[sc.order[q]].to(dev).permute(1, 2, 0)[m].float() / 255)
        fs.append(d / K[0, 0])
    X, rgb, foot = torch.cat(xs), torch.cat(cs), torch.cat(fs)
    total = len(X)
    n = int(min(total, p["max_splats"] // 2))
    sel = torch.randperm(total, device=dev)[:n]
    s = (foot[sel] * math.sqrt(total / max(n, 1)) * 0.8)[:, None].expand(n, 3)
    return dict(means=X[sel], quats=torch.tensor([1., 0, 0, 0], device=dev).expand(n, 4).clone(), scales=s.clone(),
                rgb=rgb[sel], op=torch.full((n,), 0.5, device=dev)), dict(points=total)


# ---------------------------------------------------------------------------------------------- training
def to_params(G: dict, sh_degree: int) -> torch.nn.ParameterDict:
    n = len(G["means"])
    k = (sh_degree + 1) ** 2 - 1
    P = dict(means=G["means"], scales=torch.log(G["scales"].clamp_min(1e-9)), quats=F.normalize(G["quats"], dim=-1),
             opacities=torch.logit(G["op"].clamp(1e-4, 1 - 1e-4)), sh0=((G["rgb"] - 0.5) / C0)[:, None, :],
             shN=torch.zeros(n, k, 3, device=G["means"].device))
    return torch.nn.ParameterDict({key: torch.nn.Parameter(v.float().contiguous()) for key, v in P.items()})


def render(g, P, viewmats, Ks, W, H, sh_degree, near, detach=False):
    get = (lambda t: t.detach()) if detach else (lambda t: t)    # noqa: E731
    colors = torch.cat([get(P["sh0"]), get(P["shN"])], 1)
    img, alpha, info = g.rasterization(get(P["means"]), get(P["quats"]), torch.exp(get(P["scales"])),
                                       torch.sigmoid(get(P["opacities"])), colors, viewmats, Ks, W, H,
                                       sh_degree=sh_degree, near_plane=near, packed=False)
    return img, alpha, info


def evaluate(g, P, sc: Scene, sh_degree: int, keep_images=False, align: dict | None = None, steps=ALIGN_STEPS,
             focal=False):
    """Held-out views: camera (and with `focal` its focal length) and exposure aligned for `steps` steps (splats
    frozen), then rendered at the training size. `align` keeps each view's alignment between calls, so later scores
    of a run continue from the earlier ones (fewer steps). The alignment runs at the training size up to ALIGN_SIDE
    (gsplat is no faster at half of ~1000 px, more splats per tile, but is above that). Returns mean PSNR, SSIM,
    per view PSNR and (optionally) the renders."""
    if not sc.eval:
        return None
    H, W = sc.ht, sc.wt
    d = 1
    while max(H, W) / d > ALIGN_SIDE:
        d *= 2
    Ha, Wa, Ka, imgs_a = sc.level(d)
    align = {} if align is None else align
    out_psnr, out_ssim, out_cmp, images = [], [], [], []
    for q in sc.eval:
        gt = sc.images[q].float() / 255
        gt_a = imgs_a[q].float() / 255
        prev = align.get(q)
        rot = (prev[0].clone() if prev else torch.zeros(3, device=sc.device)).requires_grad_()
        tr = (prev[1].clone() if prev else torch.zeros(3, device=sc.device)).requires_grad_()
        ex = (prev[2].clone() if prev else torch.zeros(6, device=sc.device)).requires_grad_()
        fo = (prev[3].clone() if prev else torch.zeros(2, device=sc.device)).requires_grad_()
        groups = [dict(params=[rot], lr=5e-4), dict(params=[tr], lr=5e-4 * sc.scale), dict(params=[ex], lr=1e-2)]
        if focal:
            groups.append(dict(params=[fo], lr=2e-4))
        opt = torch.optim.Adam(groups)
        with torch.enable_grad():
            for _ in range(steps):
                vm = refined_view(sc.w2c[q], rot, tr, fo if focal else None)
                img, _, _ = render(g, P, vm[None], Ka[q][None], Wa, Ha, sh_degree, sc.near, detach=True)
                img = img[0] * (1 + ex[:3]) + ex[3:]
                loss = (img - gt_a).abs().mean()
                opt.zero_grad(set_to_none=True)
                loss.backward()
                opt.step()
        align[q] = (rot.detach(), tr.detach(), ex.detach(), fo.detach())
        with torch.no_grad():
            vm = refined_view(sc.w2c[q], rot, tr, fo if focal else None)
            img, _, _ = render(g, P, vm[None], sc.Kt[q][None], W, H, sh_degree, sc.near, detach=True)
            img = (img[0] * (1 + ex[:3]) + ex[3:]).clamp(0, 1)
            out_psnr.append(psnr(img, gt))
            out_ssim.append(float(ssim(img, gt)))
            if max(H, W) > COMPARE_SIDE:
                # the same render and frame at COMPARE_SIDE: comparable with runs trained at that size (a larger
                # size scores lower for the same splats: every missing fine detail counts)
                hw = (round(H * COMPARE_SIDE / max(H, W)), round(W * COMPARE_SIDE / max(H, W)))
                small = lambda t: F.interpolate(t.permute(2, 0, 1)[None], size=hw, mode="area")[0].permute(1, 2, 0)  # noqa: E731
                out_cmp.append(psnr(small(img), small(gt)))
            if keep_images:
                images.append((img * 255).round().byte().cpu().numpy())
    out = dict(psnr=round(float(np.mean(out_psnr)), 3), ssim=round(float(np.mean(out_ssim)), 4),
               views=[round(v, 2) for v in out_psnr])
    if out_cmp:
        out["psnr_compare"] = round(float(np.mean(out_cmp)), 3)
        out["compare_side"] = COMPARE_SIDE
    return out, images


def train(g, P, sc: Scene, p: dict, check, on_step, on_eval):
    """gsplat training on the training views (one random view per step), coarse to fine with `coarse`.
    on_step(step, steps, loss, train_psnr, splats, ms_per_step, level) every 25 steps, where level is a dict of
    the size being trained (index, count, width, height); on_eval(step) at 25, 50 and 75 %."""
    from gsplat.strategy import MCMCStrategy
    steps = int(p["steps"])
    sh_max = int(p["sh_degree"])
    S = sc.scale
    lrs = dict(means=1.6e-4 * S, scales=5e-3, quats=1e-3, opacities=5e-2, sh0=2.5e-3, shN=2.5e-3 / 20)
    opts = {k: torch.optim.Adam([P[k]], lr=lrs[k], eps=1e-15) for k in P.keys()}
    sched = torch.optim.lr_scheduler.ExponentialLR(opts["means"], gamma=0.01 ** (1 / max(1, steps)))
    cap = int(max(p["max_splats"], len(P["means"])))
    strategy = MCMCStrategy(cap_max=cap, refine_start_iter=min(500, steps // 10), refine_stop_iter=int(steps * 0.8),
                            refine_every=100, min_opacity=0.005)
    strategy.check_sanity(P, opts)
    state = strategy.initialize_state()
    T = len(sc.train)
    dev = sc.device
    cam_rot = torch.zeros(T, 3, device=dev, requires_grad=True)
    cam_tr = torch.zeros(T, 3, device=dev, requires_grad=True)
    cam_f = torch.zeros(T, 2, device=dev, requires_grad=True)      # log fx, log fy corrections
    expo = torch.zeros(T, 6, device=dev, requires_grad=True)
    extra = []
    if p["refine_cameras"]:
        extra += [dict(params=[cam_rot], lr=1e-4), dict(params=[cam_tr], lr=1e-4 * S), dict(params=[cam_f], lr=1e-4)]
    if p["exposure"]:
        extra += [dict(params=[expo], lr=5e-3)]
    opt_extra = torch.optim.Adam(extra) if extra else None
    # the camera corrections settle: their step size falls to a tenth over the run
    sched_extra = torch.optim.lr_scheduler.ExponentialLR(opt_extra, gamma=0.1 ** (1 / max(1, steps))) \
        if opt_extra is not None else None
    first = sc.train.index(0) if 0 in sc.train else -1             # the reference camera stays put (the gauge)
    sh_every = max(1, int(0.5 * steps / (sh_max + 1)))
    evals = sorted({int(steps * f) for f in (0.25, 0.5, 0.75)} - {0})
    levels = coarse_levels(max(sc.ht, sc.wt), bool(p.get("coarse", True)))
    rng = np.random.default_rng(0)
    ema = None
    t0 = time.time()
    t_eval = 0.0
    for step in range(1, steps + 1):
        check()
        li = level_at(step, steps, levels)
        H, W, Ks, imgs = sc.level(levels[li])
        j = int(rng.integers(T))
        q = sc.train[j]
        gt = imgs[q].float() / 255
        vm = sc.w2c[q]
        if p["refine_cameras"]:
            vm = refined_view(vm, cam_rot[j], cam_tr[j], cam_f[j]) if j != first else refined_view(vm, focal=cam_f[j])
        deg = min(sh_max, (step - 1) // sh_every)
        img, _, info = render(g, P, vm[None], Ks[q][None], W, H, deg, sc.near)
        img = img[0]
        if p["exposure"]:
            img = img * (1 + expo[j, :3]) + expo[j, 3:]
        l1 = (img - gt).abs().mean()
        loss = 0.8 * l1 + 0.2 * (1 - ssim(img, gt))
        loss = loss + 0.01 * torch.sigmoid(P["opacities"]).mean() + 0.01 * torch.exp(P["scales"]).mean()
        loss.backward()
        for o in opts.values():
            o.step()
            o.zero_grad(set_to_none=True)
        if opt_extra is not None:
            opt_extra.step()
            opt_extra.zero_grad(set_to_none=True)
            sched_extra.step()
        sched.step()
        strategy.step_post_backward(P, opts, state, step, info, lr=sched.get_last_lr()[0])
        with torch.no_grad():                                      # on the GPU: no wait for it at every step
            b = psnr_t(img.detach().clamp(0, 1), gt)
            ema = b if ema is None else 0.9 * ema + 0.1 * b
        if step % 25 == 0 or step == steps:
            on_step(step, steps, float(loss.detach()), float(ema), len(P["means"]), (time.time() - t0 - t_eval) / step,
                    dict(index=li, count=len(levels), width=W, height=H))
        if step in evals:
            t_e = time.time()
            on_eval(step)
            t_eval += time.time() - t_e
    return dict(cameras=_camera_shift(sc, cam_rot, cam_tr, cam_f) if p["refine_cameras"] else None)


ETA_EXP = 0.1          # the time of a step grows about as (pixel count)^ETA_EXP from one level to the next and in
                       # proportion to the splat count (measured on 185 photos with 5.4M splats: 68, 80, 82 ms at
                       # 648, 1297, 2594 px: the splats, not the pixels, set the time)


@functools.lru_cache(maxsize=8)
def _level_spans(steps: int, n: int) -> tuple:
    """(first step, last step) of each of n levels over steps steps (as level_at)."""
    spans = [[0, -1] for _ in range(n)]
    levels = list(range(n))
    for st in range(1, steps + 1):
        li = level_at(st, steps, levels)
        if spans[li][0] == 0:
            spans[li][0] = st
        spans[li][1] = st
    return tuple(tuple(x) for x in spans)


def _eta(ms, step, steps, level, levels, count, cap) -> str:
    """' · about N min left': the current level's time per step for the remaining steps, scaled to the splat cap
    (MCMC reaches it early) and, for the later levels, by their pixel count (scoring not included)."""
    if ms is None:
        return ""
    cur = levels[level["index"]]
    ms = ms * max(1.0, cap / max(1, count))
    left = 0.0
    for li, (a, b) in enumerate(_level_spans(steps, len(levels))):
        if li < level["index"] or b < a:
            continue
        n = b - max(step, a - 1)
        left += max(0, n) * ms * (cur / levels[li]) ** (2 * ETA_EXP)
    if left < 60:
        return " · under a minute left"
    return f" · about {round(left / 60)} min left"


def _camera_shift(sc: Scene, rot, tr, focal=None):
    with torch.no_grad():
        ang = rot.norm(dim=1) * 180 / math.pi
        out = dict(rotation_deg=round(float(ang.median()), 4), rotation_deg_max=round(float(ang.max()), 4),
                   translation=round(float(tr.norm(dim=1).median() / sc.median_depth), 5))
        if focal is not None:
            pct = (torch.exp(focal) - 1).abs().max(dim=1).values * 100
            out.update(focal_pct=round(float(pct.median()), 4), focal_pct_max=round(float(pct.max()), 4))
        return out


# ---------------------------------------------------------------------------------------------- files
def write_splat(path: Path, P) -> int:
    """The .splat format (32 bytes a Gaussian: position, scale, RGBA, rotation) for the browser: no view-dependent
    colour, a quarter of the PLY's size."""
    with torch.no_grad():
        n = len(P["means"])
        buf = np.empty(n, dtype=[("p", "<f4", 3), ("s", "<f4", 3), ("c", "u1", 4), ("r", "u1", 4)])
        buf["p"] = P["means"].float().cpu().numpy()
        buf["s"] = torch.exp(P["scales"]).float().cpu().numpy()
        rgba = torch.cat([0.5 + C0 * P["sh0"][:, 0], torch.sigmoid(P["opacities"])[:, None]], 1)
        buf["c"] = (rgba.clamp(0, 1) * 255).round().byte().cpu().numpy()
        q = F.normalize(P["quats"], dim=-1)
        buf["r"] = (q * 128 + 128).round().clamp(0, 255).byte().cpu().numpy()
    tmp = path.with_suffix(".part")
    buf.tofile(tmp)
    os.replace(tmp, path)
    return n


def write_ply(path: Path, P):
    """The standard 3DGS PLY (INRIA layout), with the spherical harmonics."""
    with torch.no_grad():
        n = len(P["means"])
        rest = P["shN"].transpose(1, 2).reshape(n, -1)
        cols = [P["means"], torch.zeros_like(P["means"]), P["sh0"][:, 0], rest, P["opacities"][:, None],
                P["scales"], F.normalize(P["quats"], dim=-1)]
        data = torch.cat([c.float() for c in cols], 1).cpu().numpy().astype("<f4")
    names = (["x", "y", "z", "nx", "ny", "nz", "f_dc_0", "f_dc_1", "f_dc_2"]
             + [f"f_rest_{i}" for i in range(rest.shape[1])] + ["opacity", "scale_0", "scale_1", "scale_2"]
             + [f"rot_{i}" for i in range(4)])
    header = ("ply\nformat binary_little_endian 1.0\n" + f"element vertex {n}\n"
              + "".join(f"property float {k}\n" for k in names) + "end_header\n")
    tmp = path.with_suffix(".part")
    with open(tmp, "wb") as f:
        f.write(header.encode("ascii"))
        f.write(data.tobytes())
    os.replace(tmp, path)


def write_jpeg(path: Path, img: np.ndarray, quality=90):
    from PIL import Image
    Image.fromarray(img).save(path, quality=quality)


# ---------------------------------------------------------------------------------------------- the job
def run(s, job, p: dict, cache_dir: Path, emit_progress):
    """One splat run in session s (on the GPU worker). Fills s.splat_runs[-1] and emits `splat` (the run, at each
    milestone) and `splat_step` (the training) events."""
    p = {**DEFAULTS, **{k: v for k, v in p.items() if v is not None}}
    if p["sky"] is None:
        p["sky"] = bool(s.sky)
    if s.pred is None:
        raise SplatError("solve or stream first: the splats start from a reconstruction")
    if p["init"] == "sharp" and not sharp_ready(cache_dir):
        emit_progress("Downloading SHARP (2.8 GB, once)", None)
    rid = uuid.uuid4().hex[:6]
    out = s.dir / "splats" / rid
    (out / "compare").mkdir(parents=True, exist_ok=True)
    label = (f"SHARP × {p['keyframes']}" if p["init"] == "sharp" else "Solv3R points") + \
        (f" + {p['steps']:,} steps" if p["steps"] else "")
    R = dict(id=rid, label=label, params=p, state="running", created=round(time.time(), 3), stage=None,
             init=None, final=None, curve=[], seconds={}, error=None)

    def send():
        """Publishes a copy of the run (the summary and the event never see it change)."""
        snap = copy.deepcopy(R)
        with s.lock:
            for k, r in enumerate(s.splat_runs):
                if r["id"] == rid:
                    s.splat_runs[k] = snap
                    break
            else:
                s.splat_runs.append(snap)
                while len(s.splat_runs) > MAX_RUNS:
                    shutil.rmtree(s.dir / "splats" / s.splat_runs.pop(0)["id"], ignore_errors=True)
        s.emit("splat", run=snap)

    send()
    t_all = time.time()
    try:
        if _gsplat is None:
            emit_progress("Loading gsplat (it compiles its CUDA kernels on the first run, about a minute)", None)
        g = gsplat()
        dev = "cuda"
        sc = Scene(s, p, dev, progress=emit_progress)
        budget = int(p["max_splats"]) or auto_budget(sc.wt, sc.ht)
        p = {**p, "max_splats": budget}                       # R["params"] keeps the request (0: automatic)
        levels = coarse_levels(max(sc.ht, sc.wt), bool(p["coarse"])) if p["steps"] else [1]
        R["label"] = f"{label} · {max(sc.wt, sc.ht)} px"
        R.update(size=[sc.wt, sc.ht], source_long=sc.src_long, budget=budget,
                 levels=[[round(sc.wt / d), round(sc.ht / d)] for d in levels],
                 views=dict(total=len(sc.order), train=len(sc.train), test=len(sc.test), scored=len(sc.eval)),
                 eval_views=[dict(view=q, pool=sc.order[q], name=sc.names[q]) for q in sc.eval])
        if p["holdout"] and not sc.eval:
            R["note"] = "fewer than 12 views: nothing is held out, so there is no score"
        job.check()
        t = time.time()
        if p["init"] == "sharp":
            emit_progress("Loading SHARP", None)
            net = sharp_model(cache_dir, dev)
            keys = pick_keyframes(sc, int(p["keyframes"]))
            G, info = sharp_init(g, sc, keys, p, net, job.check, emit_progress)
        else:
            G, info = points_init(sc, p)
        R["seconds"]["init"] = round(time.time() - t, 2)
        R["start"] = info
        P = to_params(G, int(p["sh_degree"]))
        del G
        n = write_splat(out / "init.splat", P)
        emit_progress("Scoring the held-out views", None)
        t = time.time()
        aligned: dict = {}
        focal = bool(p["refine_cameras"])                    # the held-out cameras get a focal alignment too
        ev = evaluate(g, P, sc, 0, keep_images=True, align=aligned, focal=focal)
        R["seconds"]["score"] = round(time.time() - t, 2)
        R["init"] = dict(n=n, **(ev[0] if ev else {}))
        if ev:
            R["curve"].append(dict(step=0, psnr=ev[0]["psnr"], ssim=ev[0]["ssim"]))
            for k, (q, img) in enumerate(zip(sc.eval, ev[1])):
                write_jpeg(out / "compare" / f"{k}_gt.jpg", sc.images[q].cpu().numpy())
                write_jpeg(out / "compare" / f"{k}_init.jpg", img)
        R["stage"] = "init"
        send()
        if not p["steps"]:
            write_ply(out / "scene.ply", P)
            R["state"] = "done"
            return
        t = time.time()
        rate = dict(level=-1, t=0.0, step=0, ms=None)          # the time per step of the current level

        def on_step(step, steps, loss, train_psnr, count, sec, level):
            now = time.time() - scoring[0]
            if level["index"] != rate["level"]:
                rate.update(level=level["index"], t=now, step=step, ms=None)
            elif step - rate["step"] >= 50:
                rate["ms"] = (now - rate["t"]) / (step - rate["step"])
            s.emit("splat_step", run=rid, step=step, steps=steps, loss=round(loss, 5), psnr=round(train_psnr, 3),
                   n=count, ms=round(1000 * sec, 2), level=level["index"], w=level["width"], h=level["height"])
            size = f"{level['width']}×{level['height']}"
            stage = f" ({level['index'] + 1} of {level['count']})" if level["count"] > 1 else ""
            left = _eta(rate["ms"], step, steps, level, levels, count, budget)
            emit_progress(f"Training at {size}{stage} · step {step:,} of {steps:,} · {count / 1e6:.1f}M splats{left}",
                          step / steps)

        scoring = [0.0]

        def on_eval(step):
            emit_progress(f"Scoring the held-out views at step {step:,}", None)
            t_e = time.time()
            e = evaluate(g, P, sc, int(p["sh_degree"]), align=aligned, steps=ALIGN_MORE, focal=focal)
            scoring[0] += time.time() - t_e
            if e:
                R["curve"].append(dict(step=step, psnr=e[0]["psnr"], ssim=e[0]["ssim"]))
                send()

        with torch.enable_grad():
            tr = train(g, P, sc, p, job.check, on_step, on_eval)
        R["seconds"]["train"] = round(time.time() - t - scoring[0], 2)
        R["seconds"]["score"] = round(R["seconds"]["score"] + scoring[0], 2)
        R["cameras"] = tr["cameras"]
        emit_progress("Scoring the held-out views", None)
        t = time.time()
        ev = evaluate(g, P, sc, int(p["sh_degree"]), keep_images=True, align=aligned, steps=ALIGN_MORE, focal=focal)
        R["seconds"]["score"] = round(R["seconds"]["score"] + time.time() - t, 2)
        n = write_splat(out / "final.splat", P)
        write_ply(out / "scene.ply", P)
        R["final"] = dict(n=n, **(ev[0] if ev else {}))
        if ev:
            R["curve"].append(dict(step=int(p["steps"]), psnr=ev[0]["psnr"], ssim=ev[0]["ssim"]))
            for k, img in enumerate(ev[1]):
                write_jpeg(out / "compare" / f"{k}_final.jpg", img)
        R["stage"] = "final"
        R["state"] = "done"
    except BaseException as e:
        from .engine import Cancelled
        R["state"] = "cancelled" if isinstance(e, Cancelled) else "error"
        R["error"] = None if isinstance(e, Cancelled) else (str(e) if isinstance(e, SplatError) else f"{type(e).__name__}: {e}")
        raise
    finally:
        R["seconds"]["total"] = round(time.time() - t_all, 2)
        send()
        torch.cuda.empty_cache()
