export type Mode = "offline" | "incremental" | "any_order" | "stream";
export type JobMode = Mode | "continue" | "add_views" | "perturb" | "splats";
export type JobState = "queued" | "running" | "done" | "error" | "cancelled";

export interface Info {
  ready: boolean;
  error: string | null;
  load_seconds: number;
  gpu: { name: string; total_gib: number; capability: string } | null;
  torch: string;
  presets: Record<string, StreamPreset>;
  samples: Sample[];
}

export interface StreamPreset {
  anchors: number;
  recent: number;
  memory: number;
  memory_stride: number;
  calls: number;
  anchor_calls: number;
  refresh_tau: number;
  refresh_check: number;
}

export interface Sample {
  id: string;
  title: string;
  blurb: string;
  mode: Mode;
  params: Record<string, unknown>;
  n: number;
}

export interface SessionSummary {
  id: string;
  source: string;
  n: number;
  names: string[];
  hw: [number, number] | null;
  /** [height, width] of the source images or video frames (full size), if known */
  source_hw?: [number, number] | null;
  warnings: string[];
  kind: "solve" | "stream" | null;
  order: number[];
  stage_of: number[];
  calls: number;
  version: number;
  has_sky: boolean;
  live: boolean;
  job: { id: string; mode: JobMode; state: JobState } | null;
  next_seq: number;
  /** Set while the server fills the input pool (frame extraction, image decoding). */
  preparing: Preparing | null;
  video: VideoMeta | null;
  /** Gaussian splat runs on the current reconstruction, oldest first. */
  splats: SplatRun[];
}

export interface SplatParams {
  init: "sharp" | "points";
  keyframes: number;
  align: "depth" | "scale";
  steps: number;
  refine_cameras: boolean;
  exposure: boolean;
  holdout: boolean;
  resolution: number;
  /** 0: from the training size (2M up to ~0.6 MP, at most 6M) */
  max_splats: number;
  sh_degree: number;
  sky?: boolean | null;
  /** coarse to fine: the first half of the steps at lower sizes */
  coarse: boolean;
}

export interface SplatScore {
  n: number;
  psnr?: number;
  ssim?: number;
  /** PSNR of each scored held-out view */
  views?: number[];
  /** for runs trained above compare_side (1008 px): the PSNR of the same renders at that size */
  psnr_compare?: number;
  compare_side?: number;
}

export interface SplatKeyframe {
  view: number;
  name: string;
  /** SHARP's metric depth times this gives Solv3R's */
  scale: number;
  /** median |log depth ratio| to Solv3R's depth after the global scale, and after the per-pixel correction */
  agree_before: number;
  agree_after: number;
  coverage: number;
  kept?: number;
}

export interface SplatRun {
  id: string;
  label: string;
  params: SplatParams;
  state: "running" | "done" | "error" | "cancelled";
  created: number;
  /** the latest splats ready to view */
  stage: "init" | "final" | null;
  /** [width, height] the splats are trained and scored at */
  size?: [number, number];
  /** long side of the source frames */
  source_long?: number;
  /** the splat cap used (max_splats, or the automatic one) */
  budget?: number;
  /** [width, height] of each training size, coarse to fine */
  levels?: [number, number][];
  views?: { total: number; train: number; test: number; scored: number };
  eval_views?: { view: number; pool: number; name: string }[];
  note?: string;
  start?: {
    keyframes?: SplatKeyframe[];
    predicted?: number;
    merged?: number;
    sharp_seconds?: number;
    sharp_size?: [number, number];
    points?: number;
  };
  init: SplatScore | null;
  final: SplatScore | null;
  curve: { step: number; psnr: number; ssim: number }[];
  cameras?: { rotation_deg: number; rotation_deg_max: number; translation: number; focal_pct?: number; focal_pct_max?: number } | null;
  seconds: { init?: number; score?: number; train?: number; total?: number };
  error: string | null;
}

export interface SplatStep {
  step: number;
  steps: number;
  loss: number;
  /** PSNR of the training views (smoothed) */
  psnr: number;
  n: number;
  ms: number;
  /** the training size of the step (coarse to fine) */
  level?: number;
  w?: number;
  h?: number;
}

export interface Preparing {
  label: string;
  done: number;
  total: number | null;
}

/** An uploaded video as probed by the server, and the frames taken from it. */
export interface VideoMeta {
  name: string;
  codec: string;
  width: number;
  height: number;
  fps: number;
  duration: number;
  /** estimated frame count of the video */
  frames: number;
  /** the automatic number of frames for this video */
  auto: number;
  requested: number;
  extracted: number;
  /** seconds between the frames taken (on average) */
  spacing?: number;
  /** the sharpest frame of each stretch was taken (else evenly spaced frames) */
  sharpest?: boolean;
}

/** Cameras as sent by the server: the top 3 rows of camera-to-world (12 numbers) and fx, fy, cx, cy. */
export interface CameraSet {
  c2w: number[][];
  K: number[][];
}

export interface CallPoint {
  call: number;
  residual: number;
  ms: number;
  views: number;
  updated: number[] | null;
}

export interface Marker {
  call: number;
  kind: "continue" | "add" | "perturb" | "stage";
  label: string;
}

export interface FramePoint {
  t: number;
  ms: number | null;
  ms_total: number;
  refreshed: boolean;
}

export interface CloudInfo {
  n: number;
  views: number;
  call: number;
  final: boolean;
  label: string | null;
  ms: number;
}

export type ColorMode = "rgb" | "confidence" | "order";

export interface RunParams {
  mode: JobMode;
  calls?: number;
  adaptive?: number | null;
  max_calls?: number;
  subset?: number | null;
  preview?: number;
  budget?: number;
  sky?: boolean;
  seed?: number;
  initial?: number;
  group?: number;
  config?: "L8" | "L4" | "S4";
  live?: boolean;
  indices?: number[];
  how?: "reset" | "noise" | "swap";
  count?: number;
  sigma?: number;
  views?: number[];
}
