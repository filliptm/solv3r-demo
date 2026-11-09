import * as React from "react";
import { Crosshair, Download, Focus, Maximize, X } from "lucide-react";
import { api } from "../lib/api";
import { fmtCompact, fmtMs } from "../lib/cn";
import { liveVideo } from "../lib/live";
import { attachViewer, getViewer, useApp } from "../lib/store";
import { Viewer } from "../viewer/Viewer";
import { Button, DropdownMenu, Segmented, Slider, Switch, Tooltip } from "./ui";

export function ViewerPane() {
  const ref = React.useRef<HTMLDivElement>(null);

  React.useEffect(() => {
    const v = new Viewer(ref.current!);
    attachViewer(v);
    return () => {
      attachViewer(null);
      v.dispose();
    };
  }, []);

  return (
    <div className="relative min-h-0 flex-1 overflow-hidden bg-viz-plane">
      <div ref={ref} className="absolute inset-0" aria-label="3D reconstruction viewer" role="img" />
      <Toolbar />
      <ViewActions />
      <Status />
      <Timeline />
      <EmptyState />
      <LivePreview />
    </div>
  );
}

function Toolbar() {
  const view = useApp((s) => s.view);
  const setView = useApp((s) => s.setView);
  const hasSky = useApp((s) => s.hasSky);
  const runKind = useApp((s) => s.runKind);
  const streamDone = useApp((s) => s.streamDone);
  const points = useApp((s) => s.points);
  const splatShown = useApp((s) => s.splatShown);
  const splatLoad = useApp((s) => s.splatLoad);
  const runs = useApp((s) => s.splats);
  const showSplats = useApp((s) => s.showSplats);
  if (!points && !runKind) return null;
  const run = runs.find((r) => r.id === splatShown?.run);
  const ready = runs.filter((r) => r.stage);
  const splats = view.display === "splats";
  return (
    <div className="absolute top-3 left-3 z-10 flex w-64 flex-col gap-3 rounded-lg border border-stone-800 bg-stone-900/90 p-3 shadow-lg">
      {ready.length > 0 && (
        <Segmented
          label="Show"
          size="sm"
          value={view.display}
          onChange={(v) => {
            if (v === "splats" && !splatShown) {
              const last = ready[ready.length - 1];
              showSplats(last.id, last.stage!);
            } else setView({ display: v });
          }}
          items={[
            { value: "points", label: "Points", title: "Solv3R's point cloud" },
            { value: "splats", label: "Splats", title: "Gaussian splats of a splat run" },
          ]}
        />
      )}
      {splats ? (
        <div className="flex flex-col gap-1 text-xs" aria-live="polite">
          <span className="truncate font-medium text-stone-200">{run?.label ?? "Splats"}</span>
          <span className="text-stone-400 tabular-nums">
            {splatLoad?.state === "loading"
              ? "Loading the splats…"
              : splatLoad?.state === "error"
                ? <span className="text-red-300">Could not show the splats: {splatLoad.detail}</span>
                : run
                  ? `${splatShown?.stage === "final" ? "Trained" : run.params.init === "sharp" ? "SHARP, before training" : "Points, before training"} · ${fmtCompact((splatShown?.stage === "final" ? run.final?.n : run.init?.n) ?? 0)} splats`
                  : ""}
          </span>
          <span className="text-stone-500">Look through a held-out camera (Splats tab below) to compare with its frame.</span>
        </div>
      ) : (
        <>
          <Segmented
            label="Point colour"
            size="sm"
            value={view.colorMode}
            onChange={(v) => setView({ colorMode: v })}
            items={[
              { value: "rgb", label: "Colour" },
              { value: "confidence", label: "Confidence", title: "Light: most confident" },
              { value: "order", label: runKind === "stream" ? "Time" : "View order", title: "Light: latest views" },
            ]}
          />
          <Slider label="Most confident points" value={Math.round(view.keep * 100)} min={5} max={100} format={(v) => `${v} %`} onChange={(v) => setView({ keep: v / 100 })} />
        </>
      )}
      <div className="grid grid-cols-2 gap-3">
        {!splats && <Slider label="Point size" value={view.size} min={0.25} max={4} step={0.05} format={(v) => `${v.toFixed(2)}×`} onChange={(v) => setView({ size: v })} />}
        <Slider label="Camera size" value={view.camSize} min={0.2} max={4} step={0.1} format={(v) => `${v.toFixed(1)}×`} onChange={(v) => setView({ camSize: v })} />
      </div>
      <div className="grid grid-cols-2 gap-x-3 gap-y-2">
        <Switch label="Cameras" checked={view.showCameras} onChange={(v) => setView({ showCameras: v })} />
        <Switch label="Images" checked={view.showImages} onChange={(v) => setView({ showImages: v })} disabled={!view.showCameras} />
        <Switch label="Trajectory" checked={view.showTrajectory} onChange={(v) => setView({ showTrajectory: v })} />
        {hasSky && <Switch label="Hide sky" checked={view.hideSky} onChange={(v) => setView({ hideSky: v })} />}
      </div>
      {runKind === "stream" && streamDone && view.showTrajectory && (
        <Segmented
          label="Trajectory"
          size="sm"
          value={view.trajectory}
          onChange={(v) => setView({ trajectory: v })}
          items={[
            { value: "arrival", label: "On arrival", title: "Each frame's pose when it arrived" },
            { value: "refined", label: "Refined", title: "Each frame's pose when it left the recent window" },
          ]}
        />
      )}
    </div>
  );
}

function ViewActions() {
  const session = useApp((s) => s.session);
  const selected = useApp((s) => s.view.selected);
  const isolate = useApp((s) => s.view.isolate);
  const keep = useApp((s) => s.view.keep);
  const hideSky = useApp((s) => s.view.hideSky);
  const hasSky = useApp((s) => s.hasSky);
  const job = useApp((s) => s.job);
  const points = useApp((s) => s.points);
  const runKind = useApp((s) => s.runKind);
  const streamDone = useApp((s) => s.streamDone);
  const setView = useApp((s) => s.setView);
  const select = useApp((s) => s.select);
  const [exporting, setExporting] = React.useState<string | null>(null);
  if (!points) return null;
  const active = job?.state === "running" || job?.state === "queued";
  const exportable = session && !active && (runKind === "solve" || (runKind === "stream" && streamDone));

  const download = async (fmt: "ply" | "npz" | "html") => {
    if (!session) return;
    setExporting(fmt);
    try {
      const a = document.createElement("a");
      a.href = api.exportUrl(session.id, fmt, keep, hasSky && hideSky);
      a.download = "";
      document.body.appendChild(a);
      a.click();
      a.remove();
    } finally {
      setTimeout(() => setExporting(null), 1500);
    }
  };

  const shot = () => {
    const url = getViewer()?.screenshot();
    if (!url) return;
    const a = document.createElement("a");
    a.href = url;
    a.download = "solv3r.png";
    a.click();
  };

  return (
    <div className="absolute top-3 right-3 z-10 flex items-center gap-1.5">
      {selected !== null && (
        <div className="flex items-center gap-1 rounded-lg border border-stone-800 bg-stone-900/90 py-1 pr-1 pl-2.5 shadow-lg">
          <span className="text-xs text-stone-300 tabular-nums">View {selected + 1}</span>
          <Tooltip content="Look through this camera">
            <Button size="icon" variant="ghost" aria-label="Look through the selected camera" onClick={() => getViewer()?.goToCamera(selected)}>
              <Crosshair className="size-4" />
            </Button>
          </Tooltip>
          <Tooltip content={isolate ? "Show all views" : "Show only this view's points"}>
            <Button size="icon" variant="ghost" aria-label="Isolate the selected view's points" aria-pressed={isolate} onClick={() => setView({ isolate: !isolate })} className={isolate ? "bg-stone-700 text-stone-50" : undefined}>
              <Focus className="size-4" />
            </Button>
          </Tooltip>
          <Tooltip content="Clear the selection (Esc)">
            <Button size="icon" variant="ghost" aria-label="Clear the selection" onClick={() => { select(null); setView({ isolate: false }); }}>
              <X className="size-4" />
            </Button>
          </Tooltip>
        </div>
      )}
      <div className="flex items-center gap-1 rounded-lg border border-stone-800 bg-stone-900/90 p-1 shadow-lg">
        <Tooltip content="Frame the whole scene (F)">
          <Button size="icon" variant="ghost" aria-label="Frame the whole scene" onClick={() => getViewer()?.refit(true)}>
            <Maximize className="size-4" />
          </Button>
        </Tooltip>
        <DropdownMenu
          trigger={
            <Button size="sm" variant="ghost" aria-label="Download">
              <Download className="size-4" aria-hidden />
              {exporting ? "Preparing…" : "Download"}
            </Button>
          }
          items={[
            { label: "Point cloud (.ply)", description: `Most confident ${Math.round(keep * 100)} % per view, full resolution`, onSelect: () => download("ply"), disabled: !exportable },
            { label: "Cameras and depth (.npz)", description: "c2w, K, depth, confidence per view", onSelect: () => download("npz"), disabled: !exportable },
            { label: "Standalone viewer (.html)", description: "Opens in any browser", onSelect: () => download("html"), disabled: !exportable },
            { label: "Screenshot (.png)", description: "This view", onSelect: shot },
          ]}
        />
      </div>
    </div>
  );
}

function Status() {
  const points = useApp((s) => s.points);
  const cams = useApp((s) => s.cameraCount);
  const cloud = useApp((s) => s.cloud);
  const runKind = useApp((s) => s.runKind);
  const frames = useApp((s) => s.frames.length);
  const splats = useApp((s) => s.view.display === "splats");
  if (!points && !cams) return null;
  return (
    <div className="pointer-events-none absolute bottom-3 left-3 z-10 rounded-md bg-stone-950/70 px-2 py-1 text-xs text-stone-300 tabular-nums">
      {splats ? "Gaussian splats" : `${fmtCompact(points)} points`} · {cams} {runKind === "stream" ? `of ${frames} frames answered` : "views"}
      {runKind === "solve" && cloud && !splats && (
        <>
          {" "}· {cloud.label ?? `state after call ${cloud.call}`} · decoded in {fmtMs(cloud.ms)}
        </>
      )}
      <span className="text-stone-500"> · drag to orbit, right-drag to pan, scroll to zoom, click a camera</span>
    </div>
  );
}

function Timeline() {
  const runKind = useApp((s) => s.runKind);
  const frames = useApp((s) => s.frames.length);
  const viewMax = useApp((s) => s.view.viewMax);
  const setView = useApp((s) => s.setView);
  const done = useApp((s) => s.streamDone);
  if (runKind !== "stream" || !done || frames < 2) return null;
  const v = Number.isFinite(viewMax) ? viewMax : frames - 1;
  return (
    <div className="absolute right-3 bottom-3 z-10 flex w-80 items-end gap-2 rounded-lg border border-stone-800 bg-stone-900/90 p-3 shadow-lg">
      <Slider className="flex-1" label="Replay the stream" value={v} min={0} max={frames - 1} format={(x) => `frame ${x + 1} / ${frames}`} onChange={(x) => setView({ viewMax: x >= frames - 1 ? Infinity : x })} />
    </div>
  );
}

function EmptyState() {
  const session = useApp((s) => s.session);
  const info = useApp((s) => s.info);
  const infoError = useApp((s) => s.infoError);
  const points = useApp((s) => s.points);
  const cams = useApp((s) => s.cameraCount);
  const job = useApp((s) => s.job);
  const progress = useApp((s) => s.progress);
  const busy = useApp((s) => s.inputBusy);
  const uploading = useApp((s) => s.inputProgress);
  const loadSample = useApp((s) => s.loadSample);
  const run = useApp((s) => s.run);
  const mode = useApp((s) => s.mode);
  if (points || cams) return null;
  const active = job?.state === "running" || job?.state === "queued";

  let title: string;
  let body: string;
  let action: React.ReactNode = null;
  if (infoError) {
    title = "The model could not start";
    body = infoError;
  } else if (!info?.ready) {
    title = "Loading Solv3R";
    body = "Downloading the weights (470 MB, once) and warming up the GPU. This takes a few seconds after the first run.";
  } else if (uploading || session?.preparing) {
    const video = uploading ? /\.(mp4|mov|avi|mkv|webm|m4v)/i.test(uploading.label) : session?.source.startsWith("video:");
    title = uploading ? "Uploading" : (session?.preparing?.label ?? "Loading the input");
    body = video
      ? "Frames are taken at even intervals over the whole video, then resized to the model size. The solve can start once they are in."
      : "The images are read and resized to the model size. The solve can start once they are in.";
  } else if (!session) {
    title = "Reconstruct a scene from photos or video";
    body = "Solv3R solves for cameras, depth and a point cloud by applying one learned Transformer block over and over. Load a sample, or drop your own photos or a video in the sidebar.";
    action = (
      <Button variant="primary" onClick={() => loadSample("park")} disabled={busy}>
        {busy ? "Loading…" : "Try the Park sample"}
      </Button>
    );
  } else if (active) {
    title = progress?.label ?? "Starting";
    body = "Encoding the views. The cameras appear after the first solver call.";
  } else if (session.source === "live") {
    title = "Webcam stream";
    body = "Start the stream in the sidebar. Frames appear here as the solver answers them.";
  } else {
    title = `${session.n} ${session.video ? "frames" : "images"} loaded`;
    body = "Pick a mode in the sidebar and run it.";
    action = (
      <Button variant="primary" onClick={() => run()}>
        Run {mode === "stream" ? "the stream" : "the solve"}
      </Button>
    );
  }
  return (
    <div className="absolute inset-0 flex items-center justify-center p-8">
      <div className="flex max-w-md flex-col items-center gap-3 text-center">
        <h2 className="text-lg font-semibold text-balance text-stone-100">{title}</h2>
        <p className="text-sm text-pretty text-stone-400">{body}</p>
        {action}
      </div>
    </div>
  );
}

function LivePreview() {
  const active = useApp((s) => s.live.active);
  const ref = React.useRef<HTMLDivElement>(null);
  React.useEffect(() => {
    if (active && ref.current) ref.current.appendChild(liveVideo());
  }, [active]);
  if (!active) return null;
  return (
    <div className="absolute right-3 bottom-3 z-10 w-56 overflow-hidden rounded-lg border border-stone-700 bg-black shadow-lg">
      <div ref={ref} className="aspect-[4/3]" />
      <p className="px-2 py-1 text-xs text-stone-400">Camera input</p>
    </div>
  );
}
