"""Binary point payloads, colour ramps and image encodings for the web client.

A point payload is one little-endian buffer of n points, laid out as consecutive arrays so the client can view it
without copying:

    xyz    float32 (n, 3)   world position, in the frame of the first view (OpenCV axes: x right, y down, z forward)
    view   uint16  (n,)     index of the view in the solve (or the frame index in a stream)
    rgb    uint8   (n, 3)   colour of the source pixel
    conf   uint8   (n,)     confidence rank of the point within its view, 0 (least) .. 255 (most confident)
    flags  uint8   (n,)     bit 0: the pixel is sky

19 bytes per point. The confidence rank lets the client keep "the most confident k % of each view", the rule the
Solv3R exporter uses, without asking the server again.
"""
from __future__ import annotations

import io

import numpy as np
import torch
from PIL import Image

BYTES_PER_POINT = 19

# Sequential single-hue ramp (light -> dark), the reference palette of the dataviz guide. Index 0 is the lightest.
RAMP = ["#cde2fb", "#b7d3f6", "#9ec5f4", "#86b6ef", "#6da7ec", "#5598e7", "#3987e5", "#2a78d6", "#256abf",
        "#1c5cab", "#184f95", "#104281", "#0d366b"]


def _lut(colors, n=256):
    c = np.array([[int(h[i:i + 2], 16) for i in (1, 3, 5)] for h in colors], np.float32)
    x, t = np.linspace(0, 1, len(c)), np.linspace(0, 1, n)
    return np.stack([np.interp(t, x, c[:, k]) for k in range(3)], 1).round().astype(np.uint8)


LUT = _lut(RAMP)


def pack(xyz, view, rgb, conf, flags) -> bytes:
    return b"".join([np.ascontiguousarray(xyz, "<f4").tobytes(), np.ascontiguousarray(view, "<u2").tobytes(),
                     np.ascontiguousarray(rgb, np.uint8).tobytes(), np.ascontiguousarray(conf, np.uint8).tobytes(),
                     np.ascontiguousarray(flags, np.uint8).tobytes()])


def pixel_subset(n_pix: int, m: int, seed: int) -> torch.Tensor:
    """A fixed random subset of m of the n_pix pixels of an image (the same for every update of that image, so the
    point cloud does not flicker between solver calls)."""
    if m >= n_pix:
        return torch.arange(n_pix)
    g = torch.Generator().manual_seed(int(seed))
    return torch.randperm(n_pix, generator=g)[:m]


def _rank_u8(c: torch.Tensor) -> torch.Tensor:
    """Per-row rank of c (rows, m) scaled to 0..255."""
    m = c.shape[-1]
    rank = c.argsort(-1).argsort(-1).float()
    return (rank * (255.0 / max(m - 1, 1))).round().to(torch.uint8)


def pack_solve(points, conf, rgb, sel, sky=None):
    """points (S, H, W, 3) and conf (S, H, W) on any device, rgb (S, 3, H, W) uint8 on the CPU, sel (S, m) pixel
    indices, sky (S, H, W) bool numpy or None -> (payload bytes, number of points)."""
    S = points.shape[0]
    dev = points.device
    sel_d = sel.to(dev)
    P = points.reshape(S, -1, 3).float()
    p = torch.gather(P, 1, sel_d[..., None].expand(-1, -1, 3))
    c = _rank_u8(torch.gather(conf.reshape(S, -1).float(), 1, sel_d))
    ok = torch.isfinite(p).all(-1)
    p, c, ok = p.cpu().numpy(), c.cpu().numpy(), ok.cpu().numpy()
    sel_c = sel.cpu()
    col = torch.gather(rgb.reshape(S, 3, -1), 2, sel_c[:, None, :].expand(-1, 3, -1)).permute(0, 2, 1).numpy()
    view = np.broadcast_to(np.arange(S, dtype=np.uint16)[:, None], ok.shape)
    if sky is not None:
        flags = np.take_along_axis(sky.reshape(S, -1), sel_c.numpy(), 1).astype(np.uint8)
    else:
        flags = np.zeros(ok.shape, np.uint8)
    return pack(p[ok], view[ok], col[ok], c[ok], flags[ok]), int(ok.sum())


def frame_points(depth, conf, c2w, K, rgb, sel, view, sky=None):
    """One frame of a stream: depth / conf (h, w) numpy, c2w (4, 4), K (3, 3), rgb (3, h, w) uint8, sel (m,) pixel
    indices -> the arrays of pack() for that frame's points."""
    h, w = depth.shape
    sel = sel.numpy() if isinstance(sel, torch.Tensor) else sel
    v, u = np.divmod(sel, w)
    d = depth.reshape(-1)[sel].astype(np.float64)
    x = (u - K[0, 2]) / K[0, 0] * d
    y = (v - K[1, 2]) / K[1, 1] * d
    xyz = np.stack([x, y, d], 1) @ c2w[:3, :3].T + c2w[:3, 3]
    c = conf.reshape(-1)[sel].astype(np.float32)
    rank = np.empty(len(c), np.float32)
    rank[np.argsort(c, kind="stable")] = np.arange(len(c))
    c8 = np.round(rank * (255.0 / max(len(c) - 1, 1))).astype(np.uint8)
    col = rgb.reshape(3, -1)[:, sel].T
    flags = sky.reshape(-1)[sel].astype(np.uint8) if sky is not None else np.zeros(len(sel), np.uint8)
    ok = np.isfinite(xyz).all(1) & (d > 0)
    return (xyz[ok].astype(np.float32), np.full(int(ok.sum()), view, np.uint16), col[ok], c8[ok], flags[ok])


def concat(parts):
    """Joins the arrays of several frame_points() results into one payload."""
    if not parts:
        return pack(np.zeros((0, 3), np.float32), np.zeros(0, np.uint16), np.zeros((0, 3), np.uint8),
                    np.zeros(0, np.uint8), np.zeros(0, np.uint8)), 0
    cols = list(zip(*parts))
    arrays = [np.concatenate(c) for c in cols]
    return pack(*arrays), len(arrays[0])


# ---------------------------------------------------------------------------------------------- images
def jpeg(rgb_u8, long_side=None, quality=88) -> bytes:
    """rgb (3, h, w) uint8 tensor / array -> JPEG bytes, optionally downscaled so the long side is long_side."""
    a = rgb_u8.numpy() if isinstance(rgb_u8, torch.Tensor) else rgb_u8
    im = Image.fromarray(np.ascontiguousarray(a.transpose(1, 2, 0)))
    if long_side and max(im.size) > long_side:
        s = long_side / max(im.size)
        im = im.resize((max(1, round(im.width * s)), max(1, round(im.height * s))), Image.LANCZOS)
    buf = io.BytesIO()
    im.save(buf, "JPEG", quality=quality)
    return buf.getvalue()


def png(rgb_hw3) -> bytes:
    buf = io.BytesIO()
    Image.fromarray(np.ascontiguousarray(rgb_hw3)).save(buf, "PNG", optimize=False)
    return buf.getvalue()


def colorize(values, salient_low=True, valid=None):
    """A scalar map (h, w) -> (h, w, 3) uint8 on the sequential ramp, the salient end light. salient_low: low values
    (near depth) are light; otherwise high values (high confidence) are light.

    Values are mapped by their rank (histogram equalisation): monotonic, so near stays lighter than far, and the
    structure stays visible when the range is narrow (aerial views) or has outliers (sky)."""
    v = np.asarray(values, np.float32)
    ok = np.isfinite(v) if valid is None else (valid & np.isfinite(v))
    if not ok.any():
        return np.zeros(v.shape + (3,), np.uint8)
    vals = v[ok]
    t = np.empty(len(vals), np.float32)
    t[np.argsort(vals, kind="stable")] = np.linspace(0, 1, len(vals), dtype=np.float32)
    if not salient_low:
        t = 1 - t
    idx = np.full(v.shape, 255, np.uint8)
    idx[ok] = np.round(t * 255).astype(np.uint8)
    return LUT[idx]
