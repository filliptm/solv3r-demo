import { create } from "zustand";
import { api, ApiError } from "./api";
import { fmtBytes } from "./cn";
import type {
  CallPoint, CameraSet, CloudInfo, FramePoint, Info, JobMode, JobState, Marker, Mode, RunParams, SessionSummary,
  SplatParams, SplatRun, SplatStep,
} from "./types";
import { DEFAULT_SETTINGS, type SplatState, type Viewer, type ViewerSettings } from "../viewer/Viewer";

// ---------------------------------------------------------------------------------------------- viewer handle
let viewer: Viewer | null = null;
export function attachViewer(v: Viewer | null) {
  viewer = v;
  if (v) {
    v.apply(useApp.getState().view);
    v.onSelect = (i) => useApp.getState().select(i);
    v.imageUrl = (i, size) => {
      const s = useApp.getState();
      const pool = s.order[i];
      return s.session && pool !== undefined ? api.imageUrl(s.session.id, pool, size) : null;
    };
    v.onSplatState = (st) => useApp.setState({ splatLoad: st });
    const shown = useApp.getState().splatShown;
    const sid = useApp.getState().session?.id;
    if (shown && sid) void v.setSplats(api.splatUrl(sid, shown.run, shown.stage));
  }
}
export const getViewer = () => viewer;

// ---------------------------------------------------------------------------------------------- state
export interface JobInfo {
  id: string;
  mode: JobMode;
  state: JobState;
  error?: string | null;
  seconds?: number;
  peak_gib?: number;
}

export interface LogLine {
  t: number;
  text: string;
  tone?: "error" | "muted";
}

export type PreviewChoice = "every" | "4" | "end";

export interface Params {
  offline: { calls: number; adaptive: boolean; tol: number; maxCalls: number; subsetOn: boolean; subset: number; preview: PreviewChoice };
  staged: { initial: number; group: number; calls: number; seed: number; previewEachCall: boolean };
  stream: { config: "L8" | "L4" | "S4"; source: "frames" | "live" };
  sky: boolean;
  density: number;
  lab: { calls: number; how: "reset" | "noise" | "swap"; count: number; sigma: number };
}

interface AppState {
  info: Info | null;
  infoError: string | null;
  session: SessionSummary | null;
  inputBusy: boolean;
  inputError: string | null;
  /** the upload of files, before the server has a session for them */
  inputProgress: { label: string; frac: number | null } | null;
  mode: Mode;
  /** the mode was picked for the input (a sample's or a video's recommendation), not by the user */
  modeAuto: boolean;
  params: Params;
  job: JobInfo | null;
  runError: string | null;
  labError: string | null;
  progress: { label: string; frac: number | null } | null;
  runKind: "solve" | "stream" | null;
  runMode: JobMode | null;
  calls: CallPoint[];
  markers: Marker[];
  frames: FramePoint[];
  order: number[];
  stageOf: number[];
  cloud: CloudInfo | null;
  points: number;
  cameraCount: number;
  hasSky: boolean;
  encodeMs: number | null;
  streamDone: { ms_per_frame: number; refreshed: number[]; frames: number } | null;
  version: number;
  view: ViewerSettings;
  log: LogLine[];
  bottomTab: "convergence" | "views" | "splats" | "log";
  live: { active: boolean; error: string | null; sent: number; fps: number };
  /** Gaussian splat runs of the current reconstruction, oldest first */
  splats: SplatRun[];
  splatSteps: Record<string, SplatStep[]>;
  /** the run and stage the viewer shows (or loads) */
  splatShown: { run: string; stage: "init" | "final" } | null;
  splatLoad: SplatState | null;
  splatParams: SplatParams;
  splatError: string | null;
  /** make splats (with splatParams) when a solve or stream started here finishes */
  splatAuto: boolean;

  loadInfo: () => Promise<void>;
  restore: () => Promise<void>;
  loadSample: (id: string) => Promise<void>;
  loadFiles: (files: File[], opts: { stride: number; maxImages?: number; frames?: number; sharpest?: boolean }) => Promise<void>;
  reextract: (frames: number | null, sharpest?: boolean) => Promise<void>;
  cancelInput: () => void;
  setMode: (m: Mode) => void;
  setParams: (fn: (p: Params) => void) => void;
  run: () => Promise<void>;
  runLab: (kind: "continue" | "perturb") => Promise<void>;
  addViews: (files: File[]) => Promise<void>;
  cancel: () => Promise<void>;
  setView: (v: Partial<ViewerSettings>) => void;
  select: (i: number | null) => void;
  setBottomTab: (t: AppState["bottomTab"]) => void;
  startLive: (video: HTMLVideoElement) => Promise<void>;
  stopLive: () => Promise<void>;
  setSplatParams: (p: Partial<SplatParams>) => void;
  setSplatAuto: (on: boolean) => void;
  runSplats: () => Promise<void>;
  showSplats: (run: string, stage: "init" | "final") => void;
  deleteSplats: (run: string) => Promise<void>;
}

const BASE_MODES: JobMode[] = ["offline", "incremental", "any_order", "stream"];

export const SPLAT_DEFAULTS: SplatParams = {
  init: "sharp",
  keyframes: 8,
  align: "depth",
  steps: 3000,
  refine_cameras: true,
  exposure: true,
  holdout: true,
  resolution: 1008,
  max_splats: 2_000_000,
  sh_degree: 3,
  coarse: true,
};

/** The quality presets: the training settings they set (the start, SHARP or points, stays as chosen). */
export const SPLAT_PRESETS = {
  preview: { steps: 3000, resolution: 1008, max_splats: 2_000_000, sh_degree: 3, coarse: true, refine_cameras: true, exposure: true },
  high: { steps: 30000, resolution: 4096, max_splats: 0, sh_degree: 3, coarse: true, refine_cameras: true, exposure: true },
} satisfies Record<string, Partial<SplatParams>>;
export type SplatPreset = keyof typeof SPLAT_PRESETS;

export function splatPreset(p: SplatParams): SplatPreset | null {
  for (const [k, v] of Object.entries(SPLAT_PRESETS)) {
    if (Object.entries(v).every(([key, val]) => p[key as keyof SplatParams] === val)) return k as SplatPreset;
  }
  return null;
}

/** The splat cap of max_splats 0 for a training size (as splats.auto_budget on the server). */
export function autoBudget(w: number, h: number) {
  const mp = (w * h) / 1e6;
  return Math.round(Math.min(6e6, Math.max(2e6, 2e6 * Math.sqrt(mp / 0.6))) / 1e5) * 1e5;
}

// the splat settings are remembered in this browser
const SPLAT_KEY = "solv3r.splats";
function loadSplatSettings(): { params: SplatParams; auto: boolean } {
  try {
    const raw = JSON.parse(localStorage.getItem(SPLAT_KEY) ?? "null");
    if (raw && typeof raw === "object") {
      const params = { ...SPLAT_DEFAULTS };
      for (const k of Object.keys(SPLAT_DEFAULTS) as (keyof SplatParams)[]) {
        if (raw.params && typeof raw.params[k] === typeof SPLAT_DEFAULTS[k]) (params as Record<string, unknown>)[k] = raw.params[k];
      }
      return { params, auto: Boolean(raw.auto) };
    }
  } catch {
    /* no settings */
  }
  return { params: { ...SPLAT_DEFAULTS }, auto: false };
}
function saveSplatSettings(params: SplatParams, auto: boolean) {
  try {
    localStorage.setItem(SPLAT_KEY, JSON.stringify({ params, auto }));
  } catch {
    /* private mode */
  }
}
const savedSplats = loadSplatSettings();
/** the solve or stream job (started here) whose end starts a splat run */
let autoSplatJob: string | null = null;

const initialRun = {
  progress: null,
  runKind: null,
  runMode: null,
  calls: [] as CallPoint[],
  markers: [] as Marker[],
  frames: [] as FramePoint[],
  order: [] as number[],
  stageOf: [] as number[],
  cloud: null,
  points: 0,
  cameraCount: 0,
  hasSky: false,
  encodeMs: null,
  streamDone: null,
  // splat runs belong to a reconstruction: a new solve drops them (the server too)
  splats: [] as SplatRun[],
  splatSteps: {} as Record<string, SplatStep[]>,
  splatShown: null,
  splatLoad: null,
  splatError: null,
};

export const useApp = create<AppState>((set, get) => ({
  info: null,
  infoError: null,
  session: null,
  inputBusy: false,
  inputError: null,
  inputProgress: null,
  mode: "offline",
  modeAuto: true,
  params: {
    offline: { calls: 16, adaptive: false, tol: 0.02, maxCalls: 64, subsetOn: false, subset: 8, preview: "every" },
    staged: { initial: 8, group: 4, calls: 8, seed: 0, previewEachCall: false },
    stream: { config: "L8", source: "frames" },
    sky: false,
    density: 1_500_000,
    lab: { calls: 16, how: "reset", count: 8, sigma: 1 },
  },
  job: null,
  runError: null,
  labError: null,
  ...initialRun,
  version: 0,
  view: { ...DEFAULT_SETTINGS },
  log: [],
  bottomTab: "convergence",
  live: { active: false, error: null, sent: 0, fps: 0 },
  splatParams: savedSplats.params,
  splatAuto: savedSplats.auto,

  loadInfo: async () => {
    for (;;) {
      try {
        const info = await api.info();
        set({ info, infoError: info.error });
        if (info.ready || info.error) return;
      } catch (e) {
        set({ infoError: `The demo server is not reachable (${(e as Error).message}). Is it running?` });
      }
      await new Promise((r) => setTimeout(r, 1500));
    }
  },

  restore: async () => {
    const sid = sessionStorage.getItem("solv3r.session");
    if (!sid) return;
    try {
      const s = await api.session(sid);
      adoptSession(s, true);
    } catch {
      sessionStorage.removeItem("solv3r.session");
    }
  },

  loadSample: async (id) => {
    set({ inputBusy: true, inputError: null });
    try {
      // the sample's recommended setup is applied at click time, so a mode picked while it loads is kept
      const sample = get().info?.samples.find((x) => x.id === id);
      if (sample) {
        set((st) => ({
          mode: sample.mode,
          modeAuto: true,
          params: {
            ...st.params,
            sky: Boolean(sample.params.sky),
            stream: { ...st.params.stream, source: "frames", config: (sample.params.config as "L8") ?? st.params.stream.config },
          },
        }));
      }
      const s = await api.createFromSample(id);
      adoptSession(s, false);
    } catch (e) {
      set({ inputError: (e as Error).message });
    } finally {
      set({ inputBusy: false });
    }
  },

  loadFiles: async (files, opts) => {
    const total = files.reduce((a, f) => a + f.size, 0);
    const what = files.length === 1 ? files[0].name : `${files.length} files`;
    set({ inputBusy: true, inputError: null, inputProgress: { label: `Uploading ${what}`, frac: 0 } });
    try {
      const s = await api.createFromFiles(files, opts, (loaded, size) =>
        set({ inputProgress: { label: `Uploading ${what} · ${fmtBytes(loaded)} of ${fmtBytes(size || total)}`, frac: loaded / (size || total) } }),
      );
      adoptSession(s, false);
    } catch (e) {
      set({ inputError: (e as Error).message });
    } finally {
      set({ inputBusy: false, inputProgress: null });
    }
  },

  reextract: async (frames, sharpest = true) => {
    const old = get().session;
    if (!old?.video) return;
    set({ inputBusy: true, inputError: null });
    try {
      adoptSession(await api.reextract(old.id, frames, sharpest), false);
    } catch (e) {
      set({ inputError: (e as Error).message });
    } finally {
      set({ inputBusy: false });
    }
  },

  cancelInput: () => {
    const s = get().session;
    if (!s?.preparing) return;
    dropSession(s.id);
    log("Loading cancelled", "muted");
  },

  setMode: (m) => set({ mode: m, modeAuto: false, runError: null }),
  setParams: (fn) =>
    set((st) => {
      const p = structuredClone(st.params);
      fn(p);
      return { params: p };
    }),

  run: async () => {
    const st = get();
    const s = st.session;
    if (!s) return;
    set({ runError: null });
    const p = st.params;
    let params: RunParams;
    if (st.mode === "offline") {
      const o = p.offline;
      const preview = o.preview === "every" ? 1 : o.preview === "4" ? 4 : 0;
      // with subset updates the slider counts updates per view: N / k calls update every view once on average
      const n = s.n;
      const k = Math.min(o.subset, Math.max(1, n - 1));
      const perView = o.subsetOn && k < n ? n / k : 1;
      params = {
        mode: "offline",
        calls: Math.min(512, Math.ceil(o.calls * perView)),
        adaptive: o.adaptive ? o.tol : null,
        max_calls: Math.min(512, Math.ceil(o.maxCalls * perView)),
        subset: o.subsetOn && k < n ? k : null,
        preview,
        budget: p.density,
        sky: p.sky,
      };
    } else if (st.mode === "incremental" || st.mode === "any_order") {
      const g = p.staged;
      params = {
        mode: st.mode,
        initial: g.initial,
        group: g.group,
        calls: g.calls,
        seed: g.seed,
        preview: g.previewEachCall ? 1 : 0,
        budget: p.density,
        sky: p.sky,
      };
    } else {
      params = { mode: "stream", config: p.stream.config, budget: Math.max(p.density * 2, 1_000_000), sky: p.sky };
    }
    try {
      const { job } = await api.run(s.id, params);
      autoSplatJob = get().splatAuto ? job : null;
    } catch (e) {
      set({ runError: (e as Error).message });
    }
  },

  runLab: async (kind) => {
    const st = get();
    const s = st.session;
    if (!s) return;
    set({ labError: null });
    const l = st.params.lab;
    const params: RunParams =
      kind === "continue"
        ? { mode: "continue", calls: l.calls, preview: 1, budget: st.params.density }
        : { mode: "perturb", how: l.how, count: l.count, sigma: l.sigma, calls: l.calls, preview: 1, budget: st.params.density };
    try {
      await api.run(s.id, params);
    } catch (e) {
      set({ labError: (e as Error).message });
    }
  },

  addViews: async (files) => {
    const st = get();
    const s = st.session;
    if (!s) return;
    set({ labError: null });
    try {
      const r = await api.addImages(s.id, files);
      set({ session: r.session });
      await api.run(s.id, {
        mode: "add_views",
        indices: r.indices,
        calls: st.params.lab.calls,
        preview: 1,
        budget: st.params.density,
      });
    } catch (e) {
      set({ labError: (e as Error).message });
    }
  },

  cancel: async () => {
    const s = get().session;
    if (!s) return;
    if (get().live.active) {
      await get().stopLive();
      return;
    }
    await api.cancel(s.id).catch(() => undefined);
  },

  setView: (v) => {
    set((st) => ({ view: { ...st.view, ...v } }));
    viewer?.apply(get().view);
  },

  select: (i) => {
    set((st) => ({ view: { ...st.view, selected: i } }));
    viewer?.apply(get().view);
  },

  setBottomTab: (t) => set({ bottomTab: t }),

  startLive: async (video) => {
    set({ live: { active: true, error: null, sent: 0, fps: 0 }, runError: null });
    let s: SessionSummary;
    try {
      s = await api.createLive();
      adoptSession(s, false);
      await api.run(s.id, { mode: "stream", live: true, config: get().params.stream.config, budget: 3_000_000 });
    } catch (e) {
      set({ live: { active: false, error: (e as Error).message, sent: 0, fps: 0 } });
      return;
    }
    const canvas = document.createElement("canvas");
    const t0 = performance.now();
    let sent = 0;
    while (get().live.active) {
      const w = video.videoWidth;
      const h = video.videoHeight;
      if (!w || !h) {
        await new Promise((r) => setTimeout(r, 50));
        continue;
      }
      const scale = Math.min(1, 640 / Math.max(w, h));
      canvas.width = Math.round(w * scale);
      canvas.height = Math.round(h * scale);
      canvas.getContext("2d")!.drawImage(video, 0, 0, canvas.width, canvas.height);
      const blob = await new Promise<Blob | null>((r) => canvas.toBlob(r, "image/jpeg", 0.85));
      if (!blob) continue;
      try {
        const r = await api.liveFrame(s.id, blob);
        if (r.closed) break;
        sent += 1;
        set((st) => ({ live: { ...st.live, sent, fps: (1000 * sent) / (performance.now() - t0) } }));
      } catch (e) {
        if (get().live.active) set((st) => ({ live: { ...st.live, error: (e as Error).message } }));
        break;
      }
    }
    set((st) => ({ live: { ...st.live, active: false } }));
    await api.liveStop(s.id).catch(() => undefined);
  },

  stopLive: async () => {
    const s = get().session;
    set((st) => ({ live: { ...st.live, active: false } }));
    if (s) await api.liveStop(s.id).catch(() => undefined);
  },

  setSplatParams: (p) => {
    set((st) => ({ splatParams: { ...st.splatParams, ...p } }));
    saveSplatSettings(get().splatParams, get().splatAuto);
  },
  setSplatAuto: (on) => {
    set({ splatAuto: on });
    saveSplatSettings(get().splatParams, on);
  },

  runSplats: async () => {
    const st = get();
    const s = st.session;
    if (!s) return;
    set({ splatError: null });
    try {
      await api.runSplats(s.id, { ...st.splatParams, sky: st.hasSky ? st.view.hideSky : false });
    } catch (e) {
      set({ splatError: (e as Error).message });
    }
  },

  showSplats: (run, stage) => {
    const s = get().session;
    if (!s) return;
    set((st) => ({ splatShown: { run, stage }, view: { ...st.view, display: "splats" } }));
    viewer?.apply(get().view);
    void viewer?.setSplats(api.splatUrl(s.id, run, stage));
  },

  deleteSplats: async (run) => {
    const s = get().session;
    if (!s) return;
    try {
      await api.deleteSplats(s.id, run);
    } catch (e) {
      set({ splatError: (e as Error).message });
      return;
    }
    splatDeleted.add(run);
    set((st) => {
      const shown = st.splatShown?.run === run;
      return {
        splats: st.splats.filter((r) => r.id !== run),
        splatShown: shown ? null : st.splatShown,
        view: shown ? { ...st.view, display: "points" as const } : st.view,
      };
    });
    if (get().splatShown === null) {
      void viewer?.setSplats(null);
      viewer?.apply(get().view);
    }
  },
}));

// ---------------------------------------------------------------------------------------------- events
let source: EventSource | null = null;
let lastSeq = -1;
let cloudWanted: { sid: string; key: string; n: number } | null = null;
let cloudBusy = false;
const frameQueue: { sid: string; key: string; n: number }[] = [];
let frameBusy = false;

function log(text: string, tone?: LogLine["tone"]) {
  useApp.setState((st) => ({ log: [...st.log.slice(-400), { t: Date.now(), text, tone }] }));
}

function adoptSession(s: SessionSummary, replay: boolean) {
  const prev = useApp.getState().session;
  // the replaced session can not be reached from this tab any more: free its memory on the server now
  if (!replay && prev && prev.id !== s.id) api.deleteSession(prev.id);
  source?.close();
  lastSeq = -1;
  cloudWanted = null;
  frameQueue.length = 0;
  splatKnown = { since: Date.now() / 1000, ids: new Set((s.splats ?? []).map((r) => r.id)) };
  splatDeleted.clear();
  sessionStorage.setItem("solv3r.session", s.id);
  useApp.setState((st) => ({
    session: s,
    ...initialRun,
    job: null,
    runError: null,
    labError: null,
    log: replay ? [] : st.log,
    view: { ...st.view, selected: null, viewMax: Infinity, display: "points" },
    bottomTab: st.bottomTab === "splats" ? "convergence" : st.bottomTab,
  }));
  viewer?.reset(s.hw ?? undefined);
  viewer?.apply(useApp.getState().view);
  if (!replay && s.preparing)
    log(s.source.startsWith("video:") ? `Taking frames from ${s.source.slice(6)}` : "Reading the uploaded images", "muted");
  else if (!replay) logLoaded(s);
  source = new EventSource(`/api/sessions/${s.id}/events?since=0`);
  source.onmessage = (m) => {
    const ev = JSON.parse(m.data);
    if (ev.seq <= lastSeq) return;
    lastSeq = ev.seq;
    handleEvent(s.id, ev);
  };
  source.onerror = () => {
    api.session(s.id).catch((e) => {
      if (e instanceof ApiError && e.status === 404) {
        source?.close();
        useApp.setState({ inputError: "The server restarted and this session is gone. Load the input again." });
      }
    });
  };
}

function logLoaded(s: SessionSummary) {
  const v = s.video;
  const size = s.hw ? `, model size ${s.hw[1]}×${s.hw[0]}` : "";
  if (v) log(`Loaded ${s.n} frames of ${v.name} (one every ${(v.spacing ?? v.duration / v.extracted).toFixed(2)} s of the video)${size}`);
  else log(`Loaded ${s.n} ${s.source === "live" ? "live frames" : "images"} (${describeSource(s.source)})${size}`);
  s.warnings.forEach((w) => log(w, "muted"));
}

/** Forgets a session that is gone or unusable (and deletes it on the server). */
function dropSession(sid: string) {
  if (useApp.getState().session?.id !== sid) return;
  source?.close();
  source = null;
  sessionStorage.removeItem("solv3r.session");
  api.deleteSession(sid);
  useApp.setState({ session: null });
}

function describeSource(src: string) {
  if (src.startsWith("sample:")) return `sample “${src.slice(7)}”`;
  if (src.startsWith("video:")) return `frames of ${src.slice(6)}`;
  if (src === "live") return "webcam";
  return "upload";
}

async function refreshSession(sid: string): Promise<SessionSummary | null> {
  try {
    const s = await api.session(sid);
    if (useApp.getState().session?.id !== sid) return null;
    useApp.setState({ session: s });
    return s;
  } catch {
    return null; // gone
  }
}

async function pumpCloud() {
  if (cloudBusy) return;
  cloudBusy = true;
  try {
    while (cloudWanted) {
      const w = cloudWanted;
      cloudWanted = null;
      const buf = await api.blob(w.sid, w.key).catch(() => null);
      if (buf && !cloudWanted && useApp.getState().session?.id === w.sid) {
        viewer?.setCloud(buf, w.n);
        useApp.setState({ points: w.n });
      }
    }
  } finally {
    cloudBusy = false;
  }
}

async function pumpFrames() {
  if (frameBusy) return;
  frameBusy = true;
  try {
    while (frameQueue.length) {
      const w = frameQueue.shift()!;
      const buf = await api.blob(w.sid, w.key).catch(() => null);
      if (buf && useApp.getState().session?.id === w.sid) {
        viewer?.appendCloud(buf, w.n);
        useApp.setState((st) => ({ points: st.points + w.n }));
      }
    }
  } finally {
    frameBusy = false;
  }
}

const MODE_LABEL: Record<JobMode, string> = {
  offline: "Offline solve",
  incremental: "Incremental solve",
  any_order: "Any-order solve",
  stream: "Streaming",
  continue: "Continue",
  add_views: "Add views",
  perturb: "Perturb and re-solve",
  splats: "Splats",
};

/** Runs the session no longer has (deleted, or replaced by a new solve): their replayed events are ignored. */
let splatKnown: { since: number; ids: Set<string> } | null = null;
const splatDeleted = new Set<string>();

/** A run changed. When its next stage is ready (the start, then the trained splats) the viewer shows it. */
function onSplatRun(run: SplatRun) {
  if (splatDeleted.has(run.id)) return;
  if (splatKnown && run.created < splatKnown.since && !splatKnown.ids.has(run.id)) return;
  const st = useApp.getState();
  const prev = st.splats.find((r) => r.id === run.id);
  const splats = prev ? st.splats.map((r) => (r.id === run.id ? run : r)) : [...st.splats, run];
  useApp.setState({ splats });
  if (!prev) log(`Splats: ${run.label}`, "muted");
  if (run.stage && run.stage !== prev?.stage) {
    if (splats[splats.length - 1]?.id === run.id) st.showSplats(run.id, run.stage);
    const sc = run.stage === "final" ? run.final : run.init;
    if (sc) {
      const what = run.stage === "final" ? `trained ${run.params.steps.toLocaleString()} steps`
        : run.params.init === "sharp" ? "from SHARP" : "from Solv3R's points";
      const score = sc.psnr !== undefined ? ` · held-out PSNR ${sc.psnr.toFixed(2)} dB, SSIM ${sc.ssim?.toFixed(3)}` : "";
      log(`Splats ${what}: ${sc.n.toLocaleString()} splats${score}`);
    }
  }
  if (run.state === "error" && run.error && prev?.state !== "error") useApp.setState({ splatError: run.error });
}

// eslint-disable-next-line @typescript-eslint/no-explicit-any
function handleEvent(sid: string, ev: any) {
  const set = useApp.setState;
  switch (ev.type) {
    case "prepare":
      // only while the pool is being filled (a restored session replays the events of a finished preparation)
      set((st) =>
        st.session?.id === sid && st.session.preparing
          ? { session: { ...st.session, preparing: { label: ev.label, done: ev.done, total: ev.total } } }
          : {},
      );
      break;
    case "input": {
      const cur = useApp.getState().session;
      if (cur?.id !== sid || !cur.preparing) break;
      set({ session: { ...cur, preparing: null, n: ev.n, hw: ev.hw } });
      viewer?.reset(ev.hw ?? undefined);
      viewer?.apply(useApp.getState().view);
      refreshSession(sid).then((s) => {
        if (!s) return;
        logLoaded(s);
        // frames go to the streaming mode, not the webcam; a long video is streamed, a short one solved offline,
        // unless the user picked another mode
        const st = useApp.getState();
        const params = { ...st.params, stream: { ...st.params.stream, source: "frames" as const } };
        if (s.video && (st.modeAuto || st.mode === "offline")) set({ mode: s.n > 120 ? "stream" : "offline", modeAuto: true, params });
        else set({ params });
      });
      break;
    }
    case "input_error":
      log(`Could not load the input: ${ev.error}`, "error");
      set({ inputError: ev.error });
      dropSession(sid);
      break;
    case "job": {
      const job: JobInfo = { id: ev.id, mode: ev.mode, state: ev.state, error: ev.error, seconds: ev.seconds, peak_gib: ev.peak_gib };
      if (ev.state === "queued" && BASE_MODES.includes(ev.mode)) {
        set((st) => ({ ...initialRun, runMode: ev.mode, view: { ...st.view, selected: null, viewMax: Infinity, display: "points" } }));
        viewer?.reset(useApp.getState().session?.hw ?? undefined);
        viewer?.apply(useApp.getState().view);
        set({ bottomTab: useApp.getState().bottomTab === "log" ? "log" : "convergence" });
      }
      if (ev.state === "queued" && ev.mode === "splats") set({ splatError: null, bottomTab: "splats" });
      if (ev.state === "queued") log(`${MODE_LABEL[ev.mode as JobMode]} queued`, "muted");
      if (ev.state === "done") log(`${MODE_LABEL[ev.mode as JobMode]} finished in ${ev.seconds}s · peak GPU memory ${ev.peak_gib} GiB`);
      if (ev.state === "cancelled") log(`${MODE_LABEL[ev.mode as JobMode]} cancelled`, "muted");
      if (ev.state === "error") {
        log(`${MODE_LABEL[ev.mode as JobMode]} failed: ${ev.error}`, "error");
        if (["continue", "add_views", "perturb"].includes(ev.mode)) set({ labError: ev.error });
        else if (ev.mode === "splats") set({ splatError: ev.error });
        else set({ runError: ev.error });
      }
      set({ job, progress: ev.state === "running" || ev.state === "queued" ? useApp.getState().progress : null });
      if (["done", "error", "cancelled"].includes(ev.state) && ev.id === autoSplatJob) {
        autoSplatJob = null;
        // the solve this page started has finished: its splats (as set up in the Splats section) follow
        if (ev.state === "done" && useApp.getState().splatAuto && useApp.getState().session?.id === sid) {
          log("Making splats (set to start after each solve)", "muted");
          setTimeout(() => void useApp.getState().runSplats(), 0);
        }
      }
      if (["done", "error", "cancelled"].includes(ev.state)) {
        refreshSession(sid);
        if (ev.mode === "stream") setTimeout(() => viewer?.refit(), 600); // after the last chunks are in
      }
      break;
    }
    case "progress":
      set({ progress: { label: ev.label, frac: ev.frac ?? null } });
      break;
    case "encoded":
      set({ encodeMs: ev.ms, order: ev.order, stageOf: ev.stage_of, runKind: "solve" });
      viewer?.setImageSize(ev.hw);
      log(`Encoded ${ev.views} views in ${ev.ms} ms`);
      break;
    case "stage":
      set((st) => ({
        order: ev.order,
        stageOf: ev.stage_of,
        runKind: "solve",
        markers: ev.stage > 0 && ev.stages > 1 && st.job?.mode !== "add_views"
          ? [...st.markers, { call: ev.call, kind: "stage", label: `stage ${ev.stage + 1}` }]
          : st.markers,
        progress: { label: ev.label, frac: ev.stages > 1 ? ev.stage / ev.stages : null },
      }));
      log(ev.label);
      break;
    case "call": {
      const c: CallPoint = { call: ev.call, residual: ev.residual, ms: ev.ms, views: ev.views, updated: ev.updated };
      set((st) => ({
        calls: [...st.calls, c],
        cameraCount: ev.views,
        progress: { label: `Call ${ev.call} · change ${ev.residual.toFixed(4)}`, frac: st.progress?.frac ?? null },
      }));
      viewer?.setCameras({ c2w: ev.c2w, K: ev.K } as CameraSet, ev.updated);
      viewer?.setViewCount(ev.views);
      break;
    }
    case "cameras":
      viewer?.setCameras({ c2w: ev.c2w, K: ev.K } as CameraSet);
      break;
    case "marker":
      set((st) => ({ markers: [...st.markers, { call: ev.call, kind: ev.kind, label: ev.label }] }));
      log(`Marker at call ${ev.call}: ${ev.label}`, "muted");
      break;
    case "cloud": {
      const info: CloudInfo = { n: ev.n, views: ev.views, call: ev.call, final: ev.final, label: ev.label, ms: ev.ms };
      set({ cloud: info, order: ev.order, stageOf: ev.stage_of, hasSky: ev.has_sky, version: ev.version, cameraCount: ev.views, runKind: "solve" });
      viewer?.setCameras({ c2w: ev.c2w, K: ev.K } as CameraSet);
      viewer?.setViewCount(ev.views);
      cloudWanted = { sid, key: ev.key, n: ev.n };
      setTimeout(pumpCloud, 20); // a burst of events (a replay) fetches only the latest cloud
      if (ev.final) log(`Decoded ${ev.views} views at call ${ev.call}: ${ev.n.toLocaleString()} points in ${ev.ms} ms`);
      break;
    }
    case "stream_start":
      set({ runKind: "stream", frames: [], streamDone: null });
      if (ev.hw) viewer?.setImageSize(ev.hw);
      log(`Streaming with anchors ${ev.config.anchors}, recent ${ev.config.recent}, memory ${ev.config.memory}, ${ev.config.calls} calls per frame`);
      break;
    case "frame": {
      const f: FramePoint = { t: ev.t, ms: ev.ms, ms_total: ev.ms_total, refreshed: ev.refreshed };
      set((st) => ({
        frames: [...st.frames, f],
        // a stream pushes the input pool in order: frame t is pool image t
        order: Array.from({ length: ev.t + 1 }, (_, i) => st.order[i] ?? i),
        progress: { label: ev.label ?? `Frame ${ev.t + 1}${ev.ms ? ` · ${ev.ms.toFixed(0)} ms` : ""}${ev.refreshed ? " · memory refreshed" : ""}`, frac: null },
        cameraCount: ev.answered?.length ? Math.max(st.cameraCount, ev.t + 1) : st.cameraCount,
      }));
      if (ev.key) {
        viewer?.addCameras(ev.answered, { c2w: ev.c2w, K: ev.K } as CameraSet);
        viewer?.setViewCount(ev.t + 1);
        frameQueue.push({ sid, key: ev.key, n: ev.n });
        pumpFrames();
      }
      break;
    }
    case "splat":
      onSplatRun(ev.run as SplatRun);
      break;
    case "splat_step": {
      const p: SplatStep = { step: ev.step, steps: ev.steps, loss: ev.loss, psnr: ev.psnr, n: ev.n, ms: ev.ms, level: ev.level, w: ev.w, h: ev.h };
      set((st) => ({ splatSteps: { ...st.splatSteps, [ev.run]: [...(st.splatSteps[ev.run] ?? []), p] } }));
      break;
    }
    case "stream_done":
      set({ streamDone: { ms_per_frame: ev.ms_per_frame, refreshed: ev.refreshed, frames: ev.frames }, version: ev.version });
      viewer?.setExitTrajectory(ev.exit);
      log(`Stream of ${ev.frames} frames: ${ev.ms_per_frame} ms per frame (solver), memory refreshed at ${ev.refreshed.length} frames`);
      break;
  }
}
