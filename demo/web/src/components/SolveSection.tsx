import * as React from "react";
import { Camera, Play, Square } from "lucide-react";
import { closeCamera, openCamera } from "../lib/live";
import { useApp, type Params } from "../lib/store";
import type { Mode } from "../lib/types";
import { Button, ErrorText, Hint, NumberField, ProgressBar, Section, Segmented, Select, Slider, Switch, Tabs } from "./ui";

const MODES: { value: Mode; label: string; blurb: string }[] = [
  { value: "offline", label: "Offline", blurb: "All views are solved together: the same block runs for a number of calls, and you watch the state converge." },
  { value: "incremental", label: "Incremental", blurb: "A first group is solved, then groups join the ongoing solve in file order. Views solved earlier keep being refined." },
  { value: "any_order", label: "Any order", blurb: "Views spread over the capture are solved first, then the others join in a random order." },
  { value: "stream", label: "Streaming", blurb: "Frames arrive one at a time and each is answered on arrival, in the frame of the first frame." },
];

const DENSITY = [
  { value: "500000", label: "Light", description: "0.5M points: fast updates" },
  { value: "1500000", label: "Standard", description: "1.5M points" },
  { value: "4000000", label: "Dense", description: "4M points: needs a strong GPU in the browser" },
];

export function SolveSection() {
  const mode = useApp((s) => s.mode);
  const setMode = useApp((s) => s.setMode);
  const session = useApp((s) => s.session);
  const info = useApp((s) => s.info);
  const job = useApp((s) => s.job);
  const progress = useApp((s) => s.progress);
  const runError = useApp((s) => s.runError);
  const live = useApp((s) => s.live);
  const params = useApp((s) => s.params);
  const setParams = useApp((s) => s.setParams);
  const run = useApp((s) => s.run);
  const cancel = useApp((s) => s.cancel);
  const startLive = useApp((s) => s.startLive);
  const [camError, setCamError] = React.useState<string | null>(null);

  const busy = job?.state === "running" || job?.state === "queued";
  const active = busy && job?.mode !== "splats"; // a splat run has its own Stop and progress in the Splats section
  const ready = Boolean(info?.ready);
  const n = session?.source === "live" ? 0 : (session?.n ?? 0);
  const isLive = mode === "stream" && params.stream.source === "live";
  const preparing = Boolean(session?.preparing);
  const canRun = ready && !busy && !preparing && (isLive || n > 0);

  const onRun = async () => {
    setCamError(null);
    if (isLive) {
      try {
        const video = await openCamera();
        await startLive(video);
      } catch (e) {
        setCamError((e as Error).message);
      } finally {
        closeCamera();
      }
      return;
    }
    run();
  };

  const runLabel = preparing ? "Loading the input…" : isLive ? "Start webcam stream" : mode === "stream" ? `Stream ${n || ""} frames` : `Solve ${n || ""} views`;
  const tooFew = !isLive && n > 0 && ((mode === "stream" && n < (info?.presets[params.stream.config]?.anchors ?? 8)) || n < 2);

  return (
    <Section title="Solve">
      <Tabs.Root value={mode} onValueChange={(v) => setMode(v as Mode)}>
        <Tabs.List>
          {MODES.map((m) => (
            <Tabs.Tab key={m.value} value={m.value} disabled={active}>
              {m.label}
            </Tabs.Tab>
          ))}
        </Tabs.List>
        {MODES.map((m) => (
          <Tabs.Panel key={m.value} value={m.value} className="flex flex-col gap-3 pt-3">
            <Hint>{m.blurb}</Hint>
            {m.value === "offline" && <OfflineParams n={n} />}
            {(m.value === "incremental" || m.value === "any_order") && <StagedParams anyOrder={m.value === "any_order"} n={n} />}
            {m.value === "stream" && <StreamParams />}
          </Tabs.Panel>
        ))}
      </Tabs.Root>

      <div className="grid grid-cols-2 items-end gap-3">
        <Select
          label="Point density"
          value={String(params.density)}
          items={DENSITY}
          onChange={(v) => setParams((p) => void (p.density = Number(v)))}
        />
        <div className="pb-1.5">
          <Switch label="Sky mask" checked={params.sky} onChange={(v) => setParams((p) => void (p.sky = v))} />
        </div>
      </div>
      {params.sky && <Hint>Segments the sky (onnxruntime) so it can be hidden in the viewer and left out of exports. For outdoor captures.</Hint>}

      <div className="flex flex-col gap-2">
        {active || live.active ? (
          <Button variant="danger" onClick={cancel} className="w-full">
            <Square className="size-3.5" aria-hidden />
            {live.active ? "Stop webcam stream" : job?.state === "queued" ? "Cancel (queued)" : "Stop"}
          </Button>
        ) : (
          <Button variant="primary" onClick={onRun} disabled={!canRun || tooFew} className="w-full">
            {isLive ? <Camera className="size-4" aria-hidden /> : <Play className="size-4" aria-hidden />}
            {runLabel}
          </Button>
        )}
        {(active || live.active) && progress && (
          <div className="flex flex-col gap-1.5">
            <ProgressBar value={progress.frac} label={progress.label} />
            <p className="truncate text-xs text-stone-400 tabular-nums" aria-live="polite">
              {progress.label}
            </p>
          </div>
        )}
        {live.active && (
          <p className="text-xs text-stone-400 tabular-nums">
            {live.sent} frames sent · {live.fps.toFixed(1)} fps (the camera sends a frame once the previous one is answered)
          </p>
        )}
        {!ready && <Hint>The model is loading. The button is enabled when it is ready.</Hint>}
        {tooFew && <Hint>{mode === "stream" ? "Streaming needs at least the anchor frames." : "A solve needs at least 2 views."}</Hint>}
        <ErrorText>{runError ?? camError ?? live.error}</ErrorText>
      </div>
    </Section>
  );
}

function OfflineParams({ n }: { n: number }) {
  const o = useApp((s) => s.params.offline);
  const setParams = useApp((s) => s.setParams);
  const set = (fn: (o: Params["offline"]) => void) => setParams((p) => fn(p.offline));
  return (
    <>
      <Segmented
        label="Stopping rule"
        value={o.adaptive ? "adaptive" : "fixed"}
        onChange={(v) => set((x) => void (x.adaptive = v === "adaptive"))}
        items={[
          { value: "fixed", label: "Fixed calls" },
          { value: "adaptive", label: "Adaptive stop" },
        ]}
      />
      {o.adaptive ? (
        <div className="grid grid-cols-2 gap-3">
          <Slider label="Stop below" value={o.tol} min={0.005} max={0.1} step={0.005} format={(v) => v.toFixed(3)} onChange={(v) => set((x) => void (x.tol = v))} />
          <NumberField label="At most" value={o.maxCalls} min={4} max={512} hint="calls" onChange={(v) => set((x) => void (x.maxCalls = v))} />
        </div>
      ) : (
        <Slider
          label={o.subsetOn ? "Updates per view" : "Solver calls"}
          value={o.calls}
          min={1}
          max={128}
          format={(v) => (o.subsetOn && n > 1 ? `${v} → ${Math.ceil((v * n) / Math.min(o.subset, n - 1))} calls` : `${v}${v === 16 ? " (paper)" : ""}`)}
          onChange={(v) => set((x) => void (x.calls = v))}
        />
      )}
      {o.adaptive && <Hint>Stops once one call changes the state by less than this fraction (the paper's adaptive setting uses 0.02).</Hint>}
      <Switch
        label="Update a random subset per call"
        checked={o.subsetOn}
        onChange={(v) => set((x) => void (x.subsetOn = v))}
        description="Each call updates k random views; the others enter only through the key/value cache, as fixed constraints. Uses more calls, less memory per call."
      />
      {o.subsetOn && (
        <Slider label="Views updated per call (k)" value={Math.min(o.subset, Math.max(1, n - 1))} min={1} max={Math.max(2, n - 1)} format={(v) => `${v} of ${n}`} onChange={(v) => set((x) => void (x.subset = v))} />
      )}
      <Select
        label="Live point cloud"
        value={o.preview}
        onChange={(v) => set((x) => void (x.preview = v))}
        items={[
          { value: "every", label: "After every call", description: "Watch the geometry converge" },
          { value: "4", label: "Every 4 calls" },
          { value: "end", label: "Only at the end", description: "Fastest; cameras still move every call" },
        ]}
      />
    </>
  );
}

function StagedParams({ anyOrder, n }: { anyOrder: boolean; n: number }) {
  const g = useApp((s) => s.params.staged);
  const setParams = useApp((s) => s.setParams);
  const stages = n ? 1 + Math.ceil(Math.max(0, n - g.initial) / g.group) : 0;
  return (
    <>
      <div className="grid grid-cols-3 gap-2">
        <NumberField label="First group" value={g.initial} min={2} max={256} onChange={(v) => setParams((p) => void (p.staged.initial = v))} />
        <NumberField label="Then join" value={g.group} min={1} max={128} onChange={(v) => setParams((p) => void (p.staged.group = v))} />
        <NumberField label="Calls each" value={g.calls} min={1} max={64} onChange={(v) => setParams((p) => void (p.staged.calls = v))} />
      </div>
      {anyOrder && <NumberField label="Random seed" value={g.seed} min={0} max={9999} onChange={(v) => setParams((p) => void (p.staged.seed = v))} />}
      {n > 0 && <Hint>{stages} stages, {stages * g.calls} calls in total.</Hint>}
      <Switch label="Point cloud after every call" checked={g.previewEachCall} onChange={(v) => setParams((p) => void (p.staged.previewEachCall = v))} description="Otherwise it updates once per stage." />
    </>
  );
}

function StreamParams() {
  const st = useApp((s) => s.params.stream);
  const info = useApp((s) => s.info);
  const setParams = useApp((s) => s.setParams);
  const presets = info?.presets ?? {};
  const desc = (k: string) => {
    const p = presets[k];
    return p ? `${p.anchors} anchors · ${p.recent} recent · ${p.memory} memory · ${p.calls} calls/frame` : "";
  };
  return (
    <>
      <Segmented
        label="Frame source"
        value={st.source}
        onChange={(v) => setParams((p) => void (p.stream.source = v))}
        items={[
          { value: "frames", label: "Loaded frames" },
          { value: "live", label: "Webcam (live)" },
        ]}
      />
      <Select
        label="Configuration (paper Table 8)"
        value={st.config}
        onChange={(v) => setParams((p) => void (p.stream.config = v))}
        items={[
          { value: "L8", label: "L8 (default)", description: desc("L8") },
          { value: "L4", label: "L4", description: desc("L4") },
          { value: "S4", label: "S4 (lightest)", description: desc("S4") },
        ]}
      />
      {st.source === "live" && (
        <Hint>
          Opens your camera and streams frames to the solver as fast as it answers them. Move slowly around a static scene. The first {presets[st.config]?.anchors ?? 8} frames are the anchors and are solved together.
        </Hint>
      )}
    </>
  );
}
