import * as React from "react";
import { Crosshair } from "lucide-react";
import { api } from "../lib/api";
import { cn, fmtCompact, fmtMs } from "../lib/cn";
import { getViewer, useApp } from "../lib/store";
import { LineChart } from "./LineChart";
import { SplatsPanel } from "./SplatsPanel";
import { Button, Hint, Segmented, Tabs } from "./ui";

export function BottomPanel() {
  const tab = useApp((s) => s.bottomTab);
  const setTab = useApp((s) => s.setBottomTab);
  const logCount = useApp((s) => s.log.filter((l) => l.tone === "error").length);
  const splatRuns = useApp((s) => s.splats.length);
  return (
    <Tabs.Root value={tab} onValueChange={(v) => setTab(v as typeof tab)} className="flex h-72 shrink-0 flex-col border-t border-stone-800 bg-viz-surface">
      <Tabs.List className="px-3">
        <Tabs.Tab value="convergence">Convergence</Tabs.Tab>
        <Tabs.Tab value="views">Views</Tabs.Tab>
        <Tabs.Tab value="splats">
          Splats
          {splatRuns > 0 && <span className="rounded-full bg-stone-800 px-1.5 text-xs text-stone-300 tabular-nums">{splatRuns}</span>}
        </Tabs.Tab>
        <Tabs.Tab value="log">
          Log
          {logCount > 0 && <span className="rounded-full bg-red-900/60 px-1.5 text-xs text-red-200 tabular-nums">{logCount}</span>}
        </Tabs.Tab>
      </Tabs.List>
      <Tabs.Panel value="convergence" className="min-h-0 flex-1">
        <ConvergencePanel />
      </Tabs.Panel>
      <Tabs.Panel value="views" className="min-h-0 flex-1">
        <ViewsPanel />
      </Tabs.Panel>
      <Tabs.Panel value="splats" className="min-h-0 flex-1">
        <SplatsPanel />
      </Tabs.Panel>
      <Tabs.Panel value="log" className="min-h-0 flex-1">
        <LogPanel />
      </Tabs.Panel>
    </Tabs.Root>
  );
}

// ---------------------------------------------------------------------------------------------- convergence
function Stat({ label, value, sub }: { label: string; value: React.ReactNode; sub?: React.ReactNode }) {
  return (
    <div className="flex flex-col gap-0.5">
      <span className="text-xs text-stone-500">{label}</span>
      <span className="text-base font-semibold text-stone-100">{value}</span>
      {sub && <span className="text-xs text-stone-500">{sub}</span>}
    </div>
  );
}

function median(a: number[]) {
  if (!a.length) return null;
  const s = [...a].sort((p, q) => p - q);
  return s[Math.floor(s.length / 2)];
}

function ConvergencePanel() {
  const runKind = useApp((s) => s.runKind);
  const calls = useApp((s) => s.calls);
  const frames = useApp((s) => s.frames);
  const markers = useApp((s) => s.markers);
  const job = useApp((s) => s.job);
  const encodeMs = useApp((s) => s.encodeMs);
  const points = useApp((s) => s.points);
  const cams = useApp((s) => s.cameraCount);
  const streamDone = useApp((s) => s.streamDone);
  const offline = useApp((s) => s.params.offline);
  const runMode = useApp((s) => s.runMode);
  const [asTable, setAsTable] = React.useState<"chart" | "table">("chart");

  if (!runKind && !calls.length) {
    return (
      <div className="flex h-full items-center justify-center p-6">
        <Hint className="max-w-md text-center">
          Each solver call applies the same block to every view. This panel plots how much one call changes the state (‖z′ − z‖ / ‖z‖): it falls as the solve converges, and jumps when views join or the state is perturbed.
        </Hint>
      </div>
    );
  }

  const peak = job && (job.state === "done" || job.state === "cancelled") ? job.peak_gib : undefined;

  if (runKind === "stream") {
    const solved = frames.filter((f) => f.ms !== null);
    const refreshed = frames.filter((f) => f.refreshed).map((f) => f.t + 1);
    return (
      <div className="flex h-full min-h-0">
        <div className="grid w-72 shrink-0 grid-cols-2 content-start gap-x-4 gap-y-3 border-r border-stone-800 p-4">
          <Stat label="Frames answered" value={`${cams} / ${frames.length}`} />
          <Stat label="Solver time per frame" value={fmtMs(streamDone?.ms_per_frame ?? median(solved.map((f) => f.ms!)))} sub={streamDone ? "mean" : "median so far"} />
          <Stat label="Memory refreshes" value={refreshed.length} sub="full-window calls" />
          <Stat label="Points" value={fmtCompact(points)} sub={peak !== undefined ? `peak GPU memory ${peak} GiB` : undefined} />
        </div>
        <div className="flex min-w-0 flex-1 flex-col gap-1 p-3 pb-2">
          <ChartHeader title="Solver time per frame" subtitle="Calls on the recent window against the cached anchors and memory. Baseline ticks: the held frames were refreshed." view={asTable} setView={setAsTable} />
          <div className="min-h-0 flex-1">
            {asTable === "chart" ? (
              <LineChart
                ariaLabel="Solver time per frame, in milliseconds"
                points={solved.map((f) => ({ x: f.t + 1, y: f.ms! }))}
                formatY={(v) => `${Math.round(v)}`}
                xLabel="frame"
                baseTicks={refreshed}
                tooltip={(i) => {
                  const f = solved[i];
                  return (
                    <>
                      <div className="font-semibold text-stone-50 tabular-nums">{fmtMs(f.ms)}</div>
                      <div className="text-stone-400 tabular-nums">frame {f.t + 1}{f.refreshed ? " · memory refreshed" : ""}</div>
                    </>
                  );
                }}
              />
            ) : (
              <DataTable head={["Frame", "Solver (ms)", "Total (ms)", "Refreshed"]} rows={frames.map((f) => [f.t + 1, f.ms?.toFixed(1) ?? "anchor wait", f.ms_total.toFixed(1), f.refreshed ? "yes" : ""])} />
            )}
          </div>
        </div>
      </div>
    );
  }

  const last = calls[calls.length - 1];
  const updated = last?.updated?.length;
  return (
    <div className="flex h-full min-h-0">
      <div className="grid w-72 shrink-0 grid-cols-2 content-start gap-x-4 gap-y-3 border-r border-stone-800 p-4">
        <Stat label="Solver calls" value={calls.length} sub={updated ? `${updated} of ${last.views} views per call` : `${cams} views each`} />
        <Stat label="Last change" value={last ? last.residual.toFixed(4) : "–"} sub="relative, per call" />
        <Stat label="Time per call" value={fmtMs(median(calls.map((c) => c.ms)))} sub={encodeMs !== null ? `encode ${fmtMs(encodeMs)}` : undefined} />
        <Stat label="Points" value={fmtCompact(points)} sub={peak !== undefined ? `peak GPU memory ${peak} GiB` : undefined} />
      </div>
      <div className="flex min-w-0 flex-1 flex-col gap-1 p-3 pb-2">
        <ChartHeader title="Relative change of the state per call" subtitle="‖z′ − z‖ / ‖z‖ on a log scale. It falls as the solve converges; vertical lines mark views joining, continuations and perturbations." view={asTable} setView={setAsTable} />
        <div className="min-h-0 flex-1">
          {asTable === "chart" ? (
            <LineChart
              ariaLabel="Relative change of the solver state per call, log scale"
              points={calls.map((c) => ({ x: c.call, y: c.residual }))}
              yLog
              formatY={(v) => String(+v.toPrecision(1))}
              formatValue={(v) => v.toFixed(v >= 0.1 ? 2 : 4)}
              xLabel="call"
              markers={markers.map((m) => ({ x: m.call + 0.5, label: m.label }))}
              refLine={runMode === "offline" && offline.adaptive ? { y: offline.tol, label: `stop ${offline.tol}` } : null}
              tooltip={(i) => {
                const c = calls[i];
                return (
                  <>
                    <div className="font-semibold text-stone-50 tabular-nums">{c.residual.toFixed(4)}</div>
                    <div className="text-stone-400 tabular-nums">
                      call {c.call} · {fmtMs(c.ms)} · {c.updated ? `${c.updated.length} of ${c.views} views updated` : `${c.views} views`}
                    </div>
                  </>
                );
              }}
            />
          ) : (
            <DataTable head={["Call", "Relative change", "Time (ms)", "Views updated"]} rows={calls.map((c) => [c.call, c.residual.toFixed(5), c.ms.toFixed(1), c.updated ? `${c.updated.length} / ${c.views}` : `${c.views}`])} />
          )}
        </div>
      </div>
    </div>
  );
}

function ChartHeader({ title, subtitle, view, setView }: { title: string; subtitle: string; view: "chart" | "table"; setView: (v: "chart" | "table") => void }) {
  return (
    <div className="flex items-start justify-between gap-4">
      <div className="min-w-0">
        <h3 className="text-sm font-medium text-balance text-stone-100">{title}</h3>
        <p className="line-clamp-1 text-xs text-pretty text-stone-500" title={subtitle}>
          {subtitle}
        </p>
      </div>
      <Segmented label="Chart or table" size="sm" value={view} onChange={setView} items={[{ value: "chart", label: "Chart" }, { value: "table", label: "Table" }]} className="w-36 shrink-0" />
    </div>
  );
}

function DataTable({ head, rows }: { head: string[]; rows: (string | number)[][] }) {
  return (
    <div className="scroll-thin h-full overflow-auto rounded-md border border-stone-800">
      <table className="w-full text-xs tabular-nums">
        <thead className="sticky top-0 bg-stone-900 text-stone-400">
          <tr>
            {head.map((h) => (
              <th key={h} className="px-3 py-1.5 text-left font-medium">
                {h}
              </th>
            ))}
          </tr>
        </thead>
        <tbody>
          {rows.map((r, i) => (
            <tr key={i} className="border-t border-stone-800/70 text-stone-300">
              {r.map((c, j) => (
                <td key={j} className="px-3 py-1">
                  {c}
                </td>
              ))}
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}

// ---------------------------------------------------------------------------------------------- views
function ViewsPanel() {
  const session = useApp((s) => s.session);
  const order = useApp((s) => s.order);
  const stageOf = useApp((s) => s.stageOf);
  const selected = useApp((s) => s.view.selected);
  const select = useApp((s) => s.select);
  const version = useApp((s) => s.version);
  const runKind = useApp((s) => s.runKind);
  const streamDone = useApp((s) => s.streamDone);
  const job = useApp((s) => s.job);
  const stripRef = React.useRef<HTMLDivElement>(null);

  React.useEffect(() => {
    if (selected === null) return;
    stripRef.current?.querySelector(`[data-view="${selected}"]`)?.scrollIntoView({ block: "nearest", inline: "nearest" });
  }, [selected]);

  if (!session || !order.length) {
    return (
      <div className="flex h-full items-center justify-center p-6">
        <Hint>The views of the solve appear here once it starts.</Hint>
      </div>
    );
  }
  const stages = new Set(stageOf).size;
  const pool = selected !== null ? order[selected] : undefined;
  const mapsReady = runKind === "solve" || (runKind === "stream" && streamDone);
  const active = job?.state === "running" || job?.state === "queued";

  return (
    <div className="flex h-full min-h-0 flex-col gap-2 p-3">
      <div ref={stripRef} className="scroll-thin flex shrink-0 gap-1.5 overflow-x-auto pb-1" role="listbox" aria-label="Views of the solve">
        {order.map((p, i) => (
          <button
            key={`${i}-${p}`}
            type="button"
            role="option"
            aria-selected={selected === i}
            data-view={i}
            onClick={() => select(selected === i ? null : i)}
            className={cn(
              "relative shrink-0 overflow-hidden rounded-sm border-2 focus-visible:outline-2 focus-visible:outline-offset-1 focus-visible:outline-blue-400",
              selected === i ? "border-blue-500" : "border-transparent hover:border-stone-600",
            )}
          >
            <img src={api.imageUrl(session.id, p, 160)} alt={session.names[p] ?? `view ${i + 1}`} loading="lazy" className="h-14 w-auto bg-stone-800" />
            <span className="absolute bottom-0 left-0 bg-stone-950/80 px-1 text-xs text-stone-200 tabular-nums">
              {i + 1}
              {stages > 1 ? ` · s${stageOf[i] + 1}` : ""}
            </span>
          </button>
        ))}
      </div>
      {selected === null || pool === undefined ? (
        <div className="flex flex-1 items-center justify-center">
          <Hint>Click a thumbnail, or a camera in the 3D view, to see its depth and confidence.</Hint>
        </div>
      ) : (
        <div className="flex min-h-0 flex-1 gap-3">
          <MapImage label="Input" src={api.imageUrl(session.id, pool, 640)} />
          <MapImage label="Depth (light = near)" src={mapsReady ? api.mapUrl(session.id, selected, "depth", version) : null} pending={active} />
          <MapImage label="Confidence (light = high)" src={mapsReady ? api.mapUrl(session.id, selected, "conf", version) : null} pending={active} />
          <div className="flex w-48 shrink-0 flex-col gap-2 text-xs">
            <span className="truncate font-medium text-stone-200" title={session.names[pool]}>
              {session.names[pool]}
            </span>
            <span className="text-stone-500 tabular-nums">
              view {selected + 1} of {order.length}
              {stages > 1 ? ` · joined in stage ${stageOf[selected] + 1}` : ""}
            </span>
            {selected === 0 && <span className="text-stone-500">Reference view: it defines the world frame.</span>}
            <Button size="sm" onClick={() => getViewer()?.goToCamera(selected)}>
              <Crosshair className="size-3.5" aria-hidden />
              Look through it
            </Button>
          </div>
        </div>
      )}
    </div>
  );
}

function MapImage({ label, src, pending }: { label: string; src: string | null; pending?: boolean }) {
  const [failed, setFailed] = React.useState(false);
  React.useEffect(() => {
    setFailed(false);
  }, [src]);
  return (
    <figure className="flex min-w-0 flex-1 flex-col gap-1">
      <div className="flex min-h-0 flex-1 items-center justify-center overflow-hidden rounded-md bg-stone-900">
        {src && !failed ? (
          <img src={src} alt={label} className="max-h-full max-w-full object-contain" onError={() => setFailed(true)} />
        ) : (
          <span className="px-3 text-center text-xs text-stone-500">{pending ? "Available when the job finishes" : "Not decoded yet"}</span>
        )}
      </div>
      <figcaption className="text-xs text-stone-400">{label}</figcaption>
    </figure>
  );
}

// ---------------------------------------------------------------------------------------------- log
function LogPanel() {
  const log = useApp((s) => s.log);
  const end = React.useRef<HTMLDivElement>(null);
  React.useEffect(() => {
    end.current?.scrollIntoView({ block: "end" }); // returns a promise in recent browsers: not a cleanup
  }, [log.length]);
  return (
    <div className="scroll-thin h-full overflow-y-auto px-4 py-2 font-mono text-xs">
      {log.length === 0 && <p className="text-stone-500">Nothing yet.</p>}
      {log.map((l, i) => (
        <div key={i} className={cn("flex gap-3 py-0.5", l.tone === "error" ? "text-red-300" : l.tone === "muted" ? "text-stone-500" : "text-stone-300")}>
          <span className="shrink-0 text-stone-600 tabular-nums">{new Date(l.t).toLocaleTimeString()}</span>
          <span className="text-pretty">{l.text}</span>
        </div>
      ))}
      <div ref={end} />
    </div>
  );
}
