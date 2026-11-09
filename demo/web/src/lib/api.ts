import type { Info, RunParams, SessionSummary, SplatParams } from "./types";

export class ApiError extends Error {
  constructor(message: string, readonly status: number) {
    super(message);
  }
}

async function handle<T>(r: Response): Promise<T> {
  if (!r.ok) {
    let msg = `${r.status} ${r.statusText}`;
    try {
      const j = await r.json();
      if (typeof j.detail === "string") msg = j.detail;
      else if (Array.isArray(j.detail)) msg = j.detail.map((d: { msg: string }) => d.msg).join("; ");
    } catch {
      /* not JSON */
    }
    throw new ApiError(msg, r.status);
  }
  return r.json() as Promise<T>;
}

export const api = {
  info: () => fetch("/api/info").then((r) => handle<Info>(r)),

  session: (sid: string) => fetch(`/api/sessions/${sid}`).then((r) => handle<SessionSummary>(r)),

  createFromSample: (sample: string, stride = 1, maxImages?: number) => {
    const f = new FormData();
    f.set("sample", sample);
    f.set("stride", String(stride));
    if (maxImages) f.set("max_images", String(maxImages));
    return fetch("/api/sessions", { method: "POST", body: f }).then((r) => handle<SessionSummary>(r));
  },

  /** Uploads images or one video; XHR rather than fetch for the upload progress. The answer comes once the files are
   *  saved: the session then fills its pool in the background (`preparing`, prepare / input events). */
  createFromFiles: (
    files: File[],
    opts: { stride?: number; maxImages?: number; frames?: number; sharpest?: boolean },
    onProgress?: (loaded: number, total: number) => void,
  ) =>
    new Promise<SessionSummary>((resolve, reject) => {
      const f = new FormData();
      files.forEach((file) => f.append("files", file, file.name));
      f.set("stride", String(opts.stride ?? 1));
      if (opts.maxImages) f.set("max_images", String(opts.maxImages));
      if (opts.frames) f.set("frames", String(opts.frames));
      f.set("sharpest", String(opts.sharpest ?? true));
      const x = new XMLHttpRequest();
      x.open("POST", "/api/sessions");
      x.upload.onprogress = (e) => e.lengthComputable && onProgress?.(e.loaded, e.total);
      x.onload = () =>
        handle<SessionSummary>(
          new Response(x.responseText, {
            status: x.status,
            statusText: x.statusText,
            headers: { "Content-Type": x.getResponseHeader("Content-Type") ?? "application/json" },
          }),
        ).then(resolve, reject);
      x.onerror = () => reject(new ApiError("The upload failed: the server is not reachable.", 0));
      x.send(f);
    }),

  /** A new session with another number of frames of a session's video (null: automatic). */
  reextract: (sid: string, frames: number | null, sharpest = true) =>
    fetch(`/api/sessions/${sid}/video`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ frames, sharpest }),
    }).then((r) => handle<SessionSummary>(r)),

  deleteSession: (sid: string) => fetch(`/api/sessions/${sid}`, { method: "DELETE" }).catch(() => undefined),

  createLive: () => {
    const f = new FormData();
    f.set("live", "true");
    return fetch("/api/sessions", { method: "POST", body: f }).then((r) => handle<SessionSummary>(r));
  },

  addImages: (sid: string, files: File[]) => {
    const f = new FormData();
    files.forEach((file) => f.append("files", file, file.name));
    return fetch(`/api/sessions/${sid}/images`, { method: "POST", body: f }).then((r) =>
      handle<{ indices: number[]; session: SessionSummary }>(r),
    );
  },

  run: (sid: string, params: RunParams) =>
    fetch(`/api/sessions/${sid}/run`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(params),
    }).then((r) => handle<{ job: string }>(r)),

  runSplats: (sid: string, params: SplatParams) =>
    fetch(`/api/sessions/${sid}/splats`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(params),
    }).then((r) => handle<{ job: string }>(r)),

  deleteSplats: (sid: string, rid: string) =>
    fetch(`/api/sessions/${sid}/splats/${rid}`, { method: "DELETE" }).then((r) => handle(r)),

  /** viewer file (.splat), the full PLY, or a held-out comparison image */
  splatUrl: (sid: string, rid: string, stage: "init" | "final") => `/api/sessions/${sid}/splats/${rid}/${stage}.splat`,
  splatPlyUrl: (sid: string, rid: string) => `/api/sessions/${sid}/splats/${rid}/scene.ply`,
  compareUrl: (sid: string, rid: string, k: number, which: "gt" | "init" | "final") =>
    `/api/sessions/${sid}/splats/${rid}/compare/${k}_${which}.jpg`,

  cancel: (sid: string) => fetch(`/api/sessions/${sid}/cancel`, { method: "POST" }).then((r) => handle(r)),

  liveFrame: (sid: string, jpeg: Blob) =>
    fetch(`/api/sessions/${sid}/live/frame`, { method: "POST", body: jpeg }).then((r) =>
      handle<{ t: number | null; answered: number[]; closed?: boolean }>(r),
    ),

  liveStop: (sid: string) => fetch(`/api/sessions/${sid}/live/stop`, { method: "POST" }).then((r) => handle(r)),

  blob: async (sid: string, key: string): Promise<ArrayBuffer | null> => {
    const r = await fetch(`/api/sessions/${sid}/blob/${key}`);
    if (r.status === 204 || r.status === 404) return null; // superseded or gone
    if (!r.ok) throw new ApiError(`payload ${key}: ${r.status}`, r.status);
    return r.arrayBuffer();
  },

  imageUrl: (sid: string, pool: number, size?: number) =>
    `/api/sessions/${sid}/image/${pool}.jpg${size ? `?size=${size}` : ""}`,

  mapUrl: (sid: string, pos: number, kind: "depth" | "conf", version: number) =>
    `/api/sessions/${sid}/maps/${pos}/${kind}.png?v=${version}`,

  exportUrl: (sid: string, fmt: "ply" | "npz" | "html", keep: number, sky: boolean) =>
    `/api/sessions/${sid}/export/${fmt}?keep=${keep.toFixed(3)}&sky=${sky ? 1 : 0}`,

  coverUrl: (sample: string) => `/api/samples/${sample}/cover.jpg`,
};
