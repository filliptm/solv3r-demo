import * as React from "react";
import { Download, Sparkles, Square, Trash2 } from "lucide-react";
import { api } from "../lib/api";
import { cn, fmtCompact } from "../lib/cn";
import { autoBudget, SPLAT_DEFAULTS, SPLAT_PRESETS, splatPreset, useApp, type SplatPreset } from "../lib/store";
import type { SplatRun } from "../lib/types";
import { Button, Confirm, ErrorText, Hint, NumberField, ProgressBar, Section, Segmented, Select, Switch, Tooltip } from "./ui";

const RESOLUTION = [
  { value: "504", label: "504 px", description: "The model size: fastest" },
  { value: "1008", label: "1008 px", description: "Twice the model size" },
  { value: "1512", label: "1512 px" },
  { value: "2016", label: "2016 px" },
  { value: "4096", label: "Full size", description: "The size of the photos or video frames (up to 4096 px)" },
];
const BUDGET = [
  { value: "0", label: "Auto", description: "From the training size: 2M up to ~0.6 MP, at most 6M" },
  { value: "1000000", label: "1M" },
  { value: "2000000", label: "2M" },
  { value: "4000000", label: "4M", description: "Slower to train and to show in the browser" },
  { value: "6000000", label: "6M", description: "Slower to train and to show in the browser" },
];
const PRESETS: { value: SplatPreset; label: string; title: string }[] = [
  { value: "preview", label: "Preview", title: "3,000 steps at up to 1008 px, 2M splats: a few minutes" },
  { value: "high", label: "High", title: "30,000 steps at the full size of the frames, splats from the size: tens of minutes" },
];

/** The size the splats will train at: the long side `resolution`, never above the source nor below the model size. */
function trainingSize(hw: [number, number] | null, source: [number, number] | null | undefined, resolution: number) {
  if (!hw) return null;
  const [h, w] = hw;
  const model = Math.max(h, w);
  const src = source ? Math.max(source[0], source[1]) : model;
  const long = Math.max(model, Math.min(resolution, src));
  // above the model size the source's aspect ratio (as the server: the model size rounds each side to 14 px)
  const [bh, bw] = source && long > model ? source : hw;
  const r = long / Math.max(bh, bw);
  return { w: Math.round(bw * r), h: Math.round(bh * r), src };
}
const SH = [
  { value: "0", label: "0", description: "One colour per splat" },
  { value: "1", label: "1" },
  { value: "3", label: "3", description: "View-dependent colour, as in 3DGS" },
];

/** Gaussian splats on the reconstruction: SHARP key frames or Solv3R's points, then gsplat training. */
export function SplatSection() {
  const session = useApp((s) => s.session);
  const info = useApp((s) => s.info);
  const job = useApp((s) => s.job);
  const progress = useApp((s) => s.progress);
  const cloud = useApp((s) => s.cloud);
  const streamDone = useApp((s) => s.streamDone);
  const runs = useApp((s) => s.splats);
  const p = useApp((s) => s.splatParams);
  const set = useApp((s) => s.setSplatParams);
  const runSplats = useApp((s) => s.runSplats);
  const cancel = useApp((s) => s.cancel);
  const error = useApp((s) => s.splatError);
  const order = useApp((s) => s.order);
  const auto = useApp((s) => s.splatAuto);
  const setAuto = useApp((s) => s.setSplatAuto);
  const [more, setMore] = React.useState(false);
  const preset = splatPreset(p);
  const size = trainingSize(session?.hw ?? null, session?.source_hw, p.resolution);
  const budget = p.max_splats || (size ? autoBudget(size.w, size.h) : 2_000_000);

  const active = job?.state === "running" || job?.state === "queued";
  const splatJob = active && job?.mode === "splats";
  const reconstructed = Boolean(cloud?.final || streamDone);
  const canRun = Boolean(info?.ready && session && reconstructed && !active);
  const views = order.length;
  const heldOut = p.holdout && views >= 12 ? Math.floor((views - 5) / 8) + 1 : 0; // views 4, 12, 20, ... (as the server)

  return (
    <Section title="Splats">
      <Hint>
        Gaussian splats on this reconstruction. SHARP (Apple) predicts splats from single key frames; they are placed with
        Solv3R’s cameras and fitted to its depth, then merged. Training then fits the splats to the frames with gsplat.
      </Hint>
      <Segmented
        label="Quality"
        value={(preset ?? "") as SplatPreset}
        onChange={(v) => set(SPLAT_PRESETS[v])}
        items={PRESETS}
      />
      <Hint className="tabular-nums">
        {preset === null ? "Custom: " : ""}
        {p.steps ? `${p.steps.toLocaleString()} steps` : "No training"}
        {size ? ` at ${size.w}×${size.h}` : p.resolution >= 4096 ? " at the full size" : ` at up to ${p.resolution} px`}
        {` · up to ${fmtCompact(budget)} splats`}
        {p.steps && p.coarse && size && Math.max(size.w, size.h) >= 960 ? " · coarse to fine" : ""}
        {size && p.resolution < 4096 && p.resolution > size.src ? ` (the frames are ${size.src} px)` : ""}
      </Hint>
      <Segmented
        label="Start from"
        value={p.init}
        onChange={(v) => set({ init: v })}
        items={[
          { value: "sharp", label: "SHARP key frames", title: "Single-image splats on frames spread over the capture" },
          { value: "points", label: "Solv3R points", title: "The confident points of the solve, as small round splats" },
        ]}
      />
      <div className="grid grid-cols-2 gap-3">
        {p.init === "sharp" ? (
          <NumberField label="Key frames" value={p.keyframes} min={1} max={48} onChange={(v) => set({ keyframes: v })} hint={`${fmtCompact(p.keyframes * 1.18e6)} splats`} />
        ) : (
          <div />
        )}
        <NumberField label="Training steps" value={p.steps} min={0} max={30000} step={500} onChange={(v) => set({ steps: v })} hint={p.steps ? undefined : "no training"} />
      </div>
      {p.init === "sharp" && (
        <Segmented
          label="Fit SHARP to Solv3R’s depth"
          size="sm"
          value={p.align}
          onChange={(v) => set({ align: v })}
          items={[
            { value: "depth", label: "Per pixel", title: "One scale and a smooth per-pixel correction: SHARP's shape bent onto Solv3R's depth" },
            { value: "scale", label: "One scale", title: "Only SHARP's overall scale is matched" },
          ]}
        />
      )}
      <button
        type="button"
        className="self-start text-xs text-stone-500 underline-offset-2 hover:text-stone-300 hover:underline"
        onClick={() => setMore((v) => !v)}
        aria-expanded={more}
      >
        {more ? "Fewer options" : "More options"}
      </button>
      {more && (
        <div className="flex flex-col gap-3">
          <div className="grid grid-cols-2 gap-x-3 gap-y-2">
            <Switch label="Refine cameras" checked={p.refine_cameras} onChange={(v) => set({ refine_cameras: v })} disabled={!p.steps} />
            <Switch label="Exposure per frame" checked={p.exposure} onChange={(v) => set({ exposure: v })} disabled={!p.steps} />
            <Switch label="Hold out views" checked={p.holdout} onChange={(v) => set({ holdout: v })} />
            <Switch label="Coarse to fine" checked={p.coarse} onChange={(v) => set({ coarse: v })} disabled={!p.steps} />
          </div>
          <div className="grid grid-cols-3 gap-3">
            <Select label="Resolution" value={String(p.resolution)} items={RESOLUTION} onChange={(v) => set({ resolution: Number(v) })} />
            <Select label="Max splats" value={String(p.max_splats)} items={BUDGET} onChange={(v) => set({ max_splats: Number(v) })} />
            <Select label="SH degree" value={String(p.sh_degree)} items={SH} onChange={(v) => set({ sh_degree: Number(v) })} />
          </div>
          <Hint>
            Every 8th view is left out of training and of the key frames, then scored (PSNR, SSIM) after a short alignment of
            its camera and exposure. Resolution is the long side the splats are trained and scored at, never above the
            source frames. Refine cameras corrects each camera’s pose and focal length. Coarse to fine trains the first half
            of the steps at a half, a quarter … of the size: faster, and the cameras settle before the fine detail.
          </Hint>
          <Button size="sm" variant="ghost" className="self-start" onClick={() => set({ ...SPLAT_DEFAULTS, init: p.init })}>
            Reset options
          </Button>
        </div>
      )}

      <div className="flex flex-col gap-2">
        {splatJob ? (
          <Button variant="danger" onClick={cancel} className="w-full">
            <Square className="size-3.5" aria-hidden />
            {job?.state === "queued" ? "Cancel (queued)" : "Stop"}
          </Button>
        ) : (
          <Button variant="primary" onClick={runSplats} disabled={!canRun} className="w-full">
            <Sparkles className="size-4" aria-hidden />
            {p.init === "sharp" ? `Make splats from ${p.keyframes} key frames` : "Make splats from the points"}
            {p.steps ? ` + ${p.steps.toLocaleString()} steps` : ""}
          </Button>
        )}
        {splatJob && progress && (
          <div className="flex flex-col gap-1.5">
            <ProgressBar value={progress.frac} label={progress.label} />
            <p className="text-xs text-pretty text-stone-400 tabular-nums" aria-live="polite">
              {progress.label}
            </p>
          </div>
        )}
        <Switch label="Make splats after each solve" checked={auto} onChange={setAuto} />
        {!reconstructed && !active && (
          <Hint>{auto ? "Run a solve or a stream: the splats follow when it finishes." : "Run a solve or a stream first: the splats start from its cameras and depth."}</Hint>
        )}
        {reconstructed && views > 0 && (
          <Hint className="tabular-nums">
            {views} views{heldOut ? `: ${views - heldOut} to train on, ${heldOut} held out for scoring` : p.holdout ? ": too few to hold any out (12 needed), so no score" : ", none held out"}.
          </Hint>
        )}
        <ErrorText>{error}</ErrorText>
      </div>

      {runs.length > 0 && <RunList runs={runs} />}
      <p className="text-xs text-pretty text-stone-600">SHARP weights: Apple research licence (research use only). gsplat: Apache-2.0.</p>
    </Section>
  );
}

function RunList({ runs }: { runs: SplatRun[] }) {
  const session = useApp((s) => s.session);
  const shown = useApp((s) => s.splatShown);
  const display = useApp((s) => s.view.display);
  const show = useApp((s) => s.showSplats);
  const remove = useApp((s) => s.deleteSplats);
  const setTab = useApp((s) => s.setBottomTab);
  const [pending, setPending] = React.useState<SplatRun | null>(null);
  return (
    <>
    <Confirm
      open={pending !== null}
      onOpenChange={(v) => !v && setPending(null)}
      title="Delete this splat run?"
      description={`“${pending?.label ?? ""}” and its files (splats, PLY, comparison images) are removed from the server.`}
      confirmLabel="Delete run"
      onConfirm={() => {
        if (pending) void remove(pending.id);
        setPending(null);
      }}
    />
    <ul className="flex flex-col gap-1.5" aria-label="Splat runs">
      {[...runs].reverse().map((r) => {
        const isShown = shown?.run === r.id && display === "splats";
        const best = r.final ?? r.init;
        return (
          <li key={r.id} className={cn("flex flex-col gap-1 rounded-md border p-2", isShown ? "border-blue-500/60 bg-blue-950/20" : "border-stone-800 bg-stone-900/50")}>
            <div className="flex items-baseline justify-between gap-2">
              <button
                type="button"
                className="min-w-0 truncate text-left text-sm font-medium text-stone-100 hover:underline disabled:no-underline"
                disabled={!r.stage}
                onClick={() => {
                  show(r.id, r.stage!);
                  setTab("splats");
                }}
              >
                {r.label}
              </button>
              <span className={cn("shrink-0 text-xs tabular-nums", r.state === "error" ? "text-red-300" : "text-stone-400")}>
                {r.state === "running" ? "running" : r.state === "error" ? "failed" : r.state === "cancelled" ? "stopped" : best?.psnr !== undefined ? `${best.psnr.toFixed(2)} dB` : "done"}
              </span>
            </div>
            {(r.init || r.final) && (
              <p className="text-xs text-stone-400 tabular-nums">
                {r.init?.psnr !== undefined && <>start {r.init.psnr.toFixed(2)} dB</>}
                {r.final?.psnr !== undefined && <> → trained {r.final.psnr.toFixed(2)} dB</>}
                {best && <> · {fmtCompact(best.n)} splats</>}
                {r.seconds.total !== undefined && r.state !== "running" && <> · {Math.round(r.seconds.total)} s</>}
              </p>
            )}
            {r.error && <p className="text-xs text-pretty text-red-300">{r.error}</p>}
            {session && r.stage && (
              <div className="flex items-center gap-1">
                <Segmented
                  label="Stage to show"
                  size="sm"
                  className="flex-1"
                  value={shown?.run === r.id ? shown.stage : r.stage}
                  onChange={(v) => show(r.id, v)}
                  items={[
                    { value: "init", label: "Start", title: r.params.init === "sharp" ? "SHARP's merged key frames, before training" : "Solv3R's points as splats, before training" },
                    ...(r.final ? [{ value: "final" as const, label: "Trained", title: "After training" }] : []),
                  ]}
                />
                <Tooltip content="Download the splats (.ply, standard 3DGS layout)">
                  <a
                    href={api.splatPlyUrl(session.id, r.id)}
                    download
                    aria-label="Download the splats as a PLY file"
                    className={cn("inline-flex size-7 items-center justify-center rounded-md text-stone-300 hover:bg-stone-800 hover:text-stone-50 focus-visible:outline-2 focus-visible:outline-blue-400", r.state === "running" && "pointer-events-none opacity-40")}
                  >
                    <Download className="size-4" />
                  </a>
                </Tooltip>
                <Tooltip content="Delete this run">
                  <Button size="icon" variant="ghost" aria-label="Delete this splat run" onClick={() => setPending(r)} disabled={r.state === "running"}>
                    <Trash2 className="size-4" />
                  </Button>
                </Tooltip>
              </div>
            )}
          </li>
        );
      })}
    </ul>
    </>
  );
}
