# Solv3R web demo

An interactive web demo of [Solv3R: 3D Looped Transformer as a Multi-view Solver](https://ubc-vision.github.io/solv3r/)
([paper](https://arxiv.org/abs/2611.02857), [code](https://github.com/ubc-vision/solv3r),
[weights](https://huggingface.co/jgkwak/solv3r)). It runs the released model on your GPU and shows a solve as it
happens: cameras move after every solver call, the point cloud converges, and the change of the state per call is
plotted live.

```
.\demo\setup.ps1      # once: Python env with CUDA PyTorch, solv3r, server deps, web client
.\demo\run.ps1        # then open http://127.0.0.1:7860
```

## What it does

**Inputs:** the two samples of the Solv3R repo (Park: 39 drone photos, Desk: 289 video frames), your own photos
(JPG / PNG), a video, or a live webcam.

**Video:** drop an MP4 (or MOV, MKV, WebM, AVI, M4V) and the frames are taken for you over the whole video, 3 per
second by default (at least 24, at most 240), upright as the phone recorded them. By default each frame is the
sharpest of its stretch of the video (the variance of the Laplacian of every frame at 960 px, one extra decoding
pass; the least blurred frame of the central 70 % of each stretch), else evenly spaced (*Sharpest video frames* in
the loading options). Frames are kept at up to 3840 px for the splats; the solver gets them at its 504 px. The sidebar shows the
upload and the extraction as they progress, then the video's length, size, frame rate and codec; *Extract again* takes
another number of frames from the uploaded file without uploading it again. Each frame is named by its time in the
video. Videos of more than 120 frames select the streaming mode, shorter ones the offline solve. A 58 s 4K HEVC clip
gives its 173 frames in about 5 s here.

**Modes**, matching the scripts in `solv3r/examples`:

| Mode | What happens |
|---|---|
| Offline | All views are solved together for a fixed number of calls (16 in the paper) or until a call changes the state by less than a tolerance (adaptive, 0.02). The point cloud can be decoded after every call to watch it converge. |
| Incremental | A first group is solved, then groups join the ongoing solve in file order (`Solv3R.add_views`); earlier views keep being refined. |
| Any order | Views spread over the capture first, then the rest in a random order. |
| Streaming | `StreamingSolver` with the L8 / L4 / S4 presets of paper Table 8: each frame is answered on arrival. From loaded frames or live from the webcam. |
| Subset updates | Offline option: each call updates k random views, the others enter only through the key / value cache (`Solv3R.cache` + `step_cached`), as in the paper's stochastic subset solve. |

**Solver lab:** the state `z` stays on the GPU after a solve, so you can

- *continue* it for more calls (the update has no step index),
- *add views* to it (upload more photos; they join the ongoing solve),
- *perturb* it: reset some views to their encoding, add noise to their state, or shuffle states between views, see
  the damaged reconstruction, and watch the solver repair it. The chart marks every action.

**Viewer:** colour by image, confidence or view order / time; keep the most confident k % of each view (the exporter's
rule) without re-fetching; point and camera size; camera frustums with image planes; trajectories (on arrival and
refined, for streams); sky hiding; click a camera to select it, isolate its points or look through it; replay a
stream frame by frame. Below: the convergence chart (or solver time per frame for streams, with memory refreshes),
per-view depth and confidence maps, and a log.

**Downloads:** the same files as the Solv3R examples: `.ply` point cloud (full resolution, confidence filtered, sky
removed if masked), `.npz` cameras and depth (`c2w`, `K`, `depth`, `depth_conf`, plus `c2w_exit` for streams), and the
standalone `.html` viewer; and a screenshot.

**Splats:** Gaussian splats on a finished solve or stream (sidebar *Splats*, `server/splats.py`). Solv3R gives cameras
and per-pixel depth, not splats; this turns them into a splat scene, two ways, and scores both:

- *SHARP key frames*: [SHARP](https://github.com/apple/ml-sharp) (Apple) predicts ~1.2M Gaussians from one image in
  under a second. It runs on N key frames spread over the capture (farthest-point sampling of the cameras). Each
  frame's Gaussians are placed with Solv3R's camera and fitted to Solv3R's depth of that frame: one scale (SHARP is
  metric, Solv3R is not), or that scale and a smooth per-pixel correction, which bends SHARP's single-image geometry
  onto the multi-view depth. Then the frames are merged: a surface several key frames see keeps the Gaussians of
  the frame that sees it closest, and Gaussians in front of a surface another key frame sees are removed.
- *Solv3R points*: the confident pixels of the training views as small round Gaussians.

Either start can then be trained with [gsplat](https://github.com/nerfstudio-project/gsplat) against the frames, at
up to the full size of the photos or video frames (4096 px at most): L1 + SSIM, MCMC densification with a cap,
spherical harmonics up to degree 3, optional camera refinement (pose and focal length of each camera; the reference
camera's pose stays fixed) and per-frame exposure. Two presets: *Preview* (3,000 steps at up to 1008 px, 2M splats)
and *High* (30,000 steps at the full size, the splat cap from the training size: 2M up to ~0.6 MP, at most 6M).
Coarse to fine, the first half of the steps trains at a half, a quarter … of the size (down to ~500 px), so the
cameras settle before the fine detail. *Make splats after each solve* starts a run with the current settings when
a solve or stream started on the page finishes. With 12 or more views every 8th view is held out of training and of the key
frames; up to 12 of them are scored (PSNR, SSIM) after a short test-time alignment of their camera and exposure
(splats frozen, as in InstantSplat), before training and during it. The *Splats* tab below lists the runs side by
side and shows each held-out frame next to its renders; the viewer switches between points and splats (rendered
with [Spark](https://sparkjs.dev), loaded on first use), and a run downloads as a standard 3DGS `.ply`.

The first run compiles gsplat's CUDA kernels (about a minute, cached in `%LOCALAPPDATA%\torch_extensions`): it needs
the CUDA toolkit that matches torch (12.8 here; the server picks `CUDA\v12.8` even if `CUDA_PATH` points elsewhere)
and, on Windows, Visual Studio's C++ tools. `splats.py` patches two MSVC incompatibilities of gsplat 1.5.3 in the
installed package before the build. SHARP's checkpoint (2.8 GB) downloads to `cache/` on first use. SHARP's weights
are under Apple's research licence (research use only).

## Setup

Requirements: an NVIDIA GPU with a recent driver, [uv](https://docs.astral.sh/uv/), Node.js 18+, git. ffmpeg is
optional (`imageio-ffmpeg` provides one for video input).

`setup.ps1` clones the Solv3R and SHARP repos next to this folder if needed (they are not part of this repo), creates `.venv` (Python 3.11), and installs PyTorch
from the CUDA 12.8 wheels (`-Cuda cu126` etc. for other builds; cu128 is required for Blackwell / RTX 50xx), `solv3r`
in editable mode, the server dependencies, and the web client. Manually:

```
uv venv .venv --python 3.11
uv pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128
uv pip install -e ./solv3r -r demo/requirements.txt
git clone --depth 1 https://github.com/apple/ml-sharp.git && uv pip install --no-deps -e ./ml-sharp
cd demo/web && npm install && npm run build
```

The weights (470 MB) are downloaded from Hugging Face on the first start, and the sky segmentation model (176 MB) in
the background. Set `SOLV3R_CHECKPOINT` to use a local `.safetensors` file. Sessions live in `cache/sessions`, which the server clears when it starts: a second server on another port needs its own folder (`SOLV3R_SESSIONS_DIR`).

## Running

```
.\demo\run.ps1                     # Windows, http://127.0.0.1:7860
.\demo\run.ps1 -BindHost 0.0.0.0   # reachable from other machines
./demo/run.sh                      # Linux / macOS / Git Bash (HOST, PORT)
```

The browser's camera API only works on `localhost` or over HTTPS, so the webcam mode needs the browser on the same
machine (or a TLS proxy in front of the server).

For frontend development, run the server, then `npm run dev` in `demo/web` (Vite on :5173, proxying `/api`).

## Layout

```
demo/
  server/
    app.py       FastAPI routes: sessions, jobs, server-sent events, payloads, depth maps, exports, the built client
    engine.py    model loading and warm-up, sessions (input pool + solver state), one GPU worker thread, all modes
    packing.py   binary point payloads (19 bytes / point), colour ramps, image encodings
    splats.py    Gaussian splats: SHARP key frames placed with Solv3R's cameras and depth, gsplat training, scoring
  web/           React + TypeScript + Tailwind + Base UI + three.js (Vite)
    src/viewer/Viewer.ts   point cloud renderer (shader-side confidence / colour / isolation / sky / timeline)
    src/lib/store.ts       app state and the event stream from the server
    src/components/        sidebar, viewer overlays, charts, views inspector
  cache/         sessions, the sky model, logs (created at run time)
```

Uploads are answered once saved; the server then fills the session's input pool in the background (video probe and
frame extraction with ffmpeg, parallel image decoding) and reports it with `prepare` events, then `input` (or
`input_error`); the upload's form field `sharpest` (default true) picks the sharpest frames of a video.
`POST /api/sessions/{id}/video {"frames": n, "sharpest": true}` makes a new session from the same video. A job emits JSON events (`call` with the residual and all cameras, `cloud` with a payload key, `frame` for streams,
`stage`, `marker`, `progress`, `job`) over `GET /api/sessions/{id}/events`; point payloads are fetched from
`/api/sessions/{id}/blob/{key}` and parsed in place (float32 xyz, uint16 view, uint8 rgb, confidence rank, flags).
Only the latest few solve payloads are kept, stream payloads for the current run; sessions are in memory (up to 6).
`POST /api/sessions/{id}/splats` starts a splat run (`init`, `keyframes`, `align`, `steps`, `refine_cameras`,
`exposure`, `holdout`, `resolution` up to 4096, `max_splats` (0: automatic), `sh_degree`, `coarse`); it reports `splat` events (the run, at each
milestone) and `splat_step` events (training), and its files are `/api/sessions/{id}/splats/{run}/init.splat`,
`final.splat` (viewer), `scene.ply` and `compare/{k}_{gt|init|final}.jpg`. A new solve drops the runs.

## Measured here

RTX PRO 6000 Blackwell (96 GB), torch 2.11 + cu128, the Park sample (39 views at 504 × 336): encode ≈ 0.1 s, one
solver call ≈ 50–70 ms, a dense decode of all views ≈ 0.25–0.4 s, peak memory 2.3 GB. Relative change after 16 calls:
0.021. Subset updates (8 of 39 per call) reach the same cameras as the full solve (0.3 % centre, 0.13° rotation
difference). The Desk stream (288 frames, L8) runs at ≈ 105 ms per frame for the solver; a webcam stream at ≈ 4 fps
end to end.

Splats (held-out PSNR / SSIM after the test-time alignment; one run each, so differences under ~0.5 dB are noise):

| Scene | Start | Before training | After training |
|---|---|---|---|
| Park, 39 photos at 504 × 336 (34 trained, 5 scored), 2000 steps | SHARP × 8 | 15.8 dB | 21.8 dB / 0.71 |
| | Solv3R points | 15.5 dB | 21.2 dB / 0.68 |
| Desk video, 48 frames at 1008 × 756 (42 trained, 6 scored), 1500 steps | SHARP × 8 | 12.6 dB | 17.7 dB / 0.77 |
| | Solv3R points | 12.7 dB | 17.7 dB / 0.77 |

Full size (Mip-NeRF 360 *garden*, 185 photos at 2594 × 1681, 162 trained, 12 scored, SHARP × 8; "at 1008" scores the
same renders at 1008 px, so runs of different sizes compare):

| Training | Splats | Held-out PSNR at the run's size | At 1008 px | Time |
|---|---|---|---|---|
| Preview: 3,000 steps at 1008 px | 2M | 23.6 dB / 0.71 | 23.6 dB | 1.5 min |
| 3,000 steps at 2594 px | 5.4M | 22.9 dB / 0.63 | – | 4 min |
| 30,000 steps at 1008 px | 2M | 27.4 dB / 0.86 | 27.4 dB | 14 min |
| High: 30,000 steps at 2594 px | 5.4M | 25.6 dB / 0.77 | 28.6 dB | 42 min |

Most of the gain is the longer training (+3.7 dB at 1008 px); the full size adds 1.2 dB there and the detail a
1008 px run does not have (wood grain, cracks at 2594 px). The High run's held-out score is flat from 22,500 to
30,000 steps (25.56 dB both). The time of a step follows the splat count more than the size: 68, 80 and 82 ms at
648, 1297 and 2594 px with 5.4M splats. Coarse to fine at 1008 px: 23.6 dB against 23.4 dB without, same time.
Focal length refinement corrected Solv3R's focal lengths by 0.4 % (median, up to 1.2 %): about 10 px at the edge
of a 2594 px frame. Sharpest video frames: on a 47 s 640 × 360 phone clip the median sharpness of the 141 frames
taken rose 14 % and that of the blurriest tenth 31 %, for 0.7 s more extraction (2.9 against 1.3 s for 20 s of 4K).

SHARP's network takes 0.76 s per key frame at 1536², and a key frame's single-image depth agrees with Solv3R's to
1–14 % (median, by frame) after the global scale and to 0.5–1.2 % after the per-pixel fit. Without training SHARP is the better
start (it also fills what the points leave empty: +0.9 dB on Desk); after training the two are close. Camera
refinement adds about 0.9 dB on Desk (3000 steps: 18.8 against 17.9 dB). A training step takes ≈ 20 ms at 504 px
and ≈ 20–30 ms at 1008 px with 1–2M splats when the GPU is otherwise idle; another program using the GPU (here,
ComfyUI) made it 3–5 × slower. The Desk video is the 504 px sample scaled up, so it shows no gain from the higher
resolution.

## License

The demo code follows the Solv3R code license (Apache-2.0, [LICENSE](../LICENSE)). The model weights are under the
[NVIDIA License](https://github.com/ubc-vision/solv3r/blob/main/LICENSES/NVIDIA-LICENSE.txt): non-commercial,
research and evaluation use only. SHARP's
weights are under the [Apple Machine Learning Research Model License](https://github.com/apple/ml-sharp/blob/main/LICENSE_MODEL)
(research purposes only, no commercial use); gsplat is Apache-2.0, Spark MIT.
