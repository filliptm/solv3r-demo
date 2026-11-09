import * as React from "react";
import { Crosshair } from "lucide-react";
import { api } from "../lib/api";
import { cn, fmtCompact } from "../lib/cn";
import { getViewer, useApp } from "../lib/store";
import type { SplatRun } from "../lib/types";
import { LineChart } from "./LineChart";
import { Button, Hint, Segmented } from "./ui";

/** The splat runs side by side (held-out scores), and for the selected run its training and its held-out views. */
export function SplatsPanel() {
  const runs = useApp((s) => s.splats);
  const shown = useApp((s) => s.splatShown);
  const show = useApp((s) => s.showSplats);
  const [tab, setTab] = React.useState<"views" | "training">("views");
  const run = runs.find((r) => r.id === shown?.run) ?? runs[runs.length - 1];

  if (!runs.length) {
    return (
      <div className="flex h-full items-center justify-center p-6">
        <Hint className="max-w-lg text-center">
          Make splats in the sidebar. Each run is scored on views left out of its training: rendered from their camera and
          compared with the real frame. Runs appear here side by side, with the held-out frames next to their renders.
        </Hint>
      </div>
    );
  }

  return (
    <div className="flex h-full min-h-0">
      <div className="flex w-[31rem] shrink-0 flex-col gap-1 border-r border-stone-800 p-3">
        <h3 className="text-sm font-medium text-stone-100">Held-out views</h3>
        <p className="text-xs text-pretty text-stone-500">
          PSNR (dB, higher is better) and SSIM of views left out of training, at the run’s size. Start: before training. At
          1008: the trained splats scored at 1008 px, to compare runs of different sizes (a larger size scores lower).
        </p>
        <div className="scroll-thin min-h-0 flex-1 overflow-auto rounded-md border border-stone-800">
          <table className="w-full text-xs tabular-nums">
            <thead className="sticky top-0 bg-stone-900 text-stone-400">
              <tr>
                {["Run", "Start", "Trained", "At 1008", "SSIM", "Splats", "Time"].map((h, i) => (
                  <th key={h} className={cn("px-2 py-1.5 font-medium", i ? "text-right" : "text-left")}>
                    {h}
                  </th>
                ))}
              </tr>
            </thead>
            <tbody>
              {runs.map((r) => {
                const sel = r.id === run?.id;
                const best = r.final ?? r.init;
                return (
                  <tr
                    key={r.id}
                    className={cn("cursor-pointer border-t border-stone-800/70 text-stone-300 hover:bg-stone-800/50", sel && "bg-blue-950/30 text-stone-100")}
                    onClick={() => r.stage && show(r.id, r.stage)}
                    aria-selected={sel}
                  >
                    <td className="max-w-40 truncate px-2 py-1" title={r.label}>
                      {r.label}
                      {r.state === "running" && <span className="text-stone-500"> · running</span>}
                      {r.state === "error" && <span className="text-red-300"> · failed</span>}
                      {r.state === "cancelled" && <span className="text-stone-500"> · stopped</span>}
                    </td>
                    <td className="px-2 py-1 text-right">{r.init?.psnr?.toFixed(2) ?? "–"}</td>
                    <td className="px-2 py-1 text-right font-semibold">{r.final?.psnr?.toFixed(2) ?? "–"}</td>
                    <td className="px-2 py-1 text-right">{compareScore(r)?.toFixed(2) ?? "–"}</td>
                    <td className="px-2 py-1 text-right">{best?.ssim?.toFixed(3) ?? "–"}</td>
                    <td className="px-2 py-1 text-right">{best ? fmtCompact(best.n) : "–"}</td>
                    <td className="px-2 py-1 text-right">{r.seconds.total !== undefined ? fmtSeconds(r.seconds.total) : "–"}</td>
                  </tr>
                );
              })}
            </tbody>
          </table>
        </div>
      </div>
      {run && (
        <div className="flex min-w-0 flex-1 flex-col gap-2 p-3 pb-2">
          <div className="flex items-start justify-between gap-4">
            <div className="min-w-0">
              <h3 className="truncate text-sm font-medium text-stone-100">{run.label}</h3>
              <RunFacts run={run} />
            </div>
            <Segmented
              label="Held-out views or training"
              size="sm"
              value={tab}
              onChange={setTab}
              items={[
                { value: "views", label: "Held-out views" },
                { value: "training", label: "Training" },
              ]}
              className="w-56 shrink-0"
            />
          </div>
          <div className="min-h-0 flex-1">{tab === "views" ? <HeldOut run={run} /> : <Training run={run} />}</div>
        </div>
      )}
    </div>
  );
}

/** The trained splats' PSNR at 1008 px: scored there by the server for larger runs, the run's own score up to it. */
function compareScore(r: SplatRun) {
  if (!r.final) return undefined;
  if (r.final.psnr_compare !== undefined) return r.final.psnr_compare;
  return r.size && Math.max(...r.size) <= 1008 ? r.final.psnr : undefined;
}

function fmtSeconds(s: number) {
  return s < 120 ? `${Math.round(s)} s` : `${Math.round(s / 60)} min`;
}

function RunFacts({ run }: { run: SplatRun }) {
  const facts: string[] = [];
  if (run.size) facts.push(`${run.size[0]}×${run.size[1]}`);
  if (run.levels && run.levels.length > 1) facts.push(`coarse to fine from ${run.levels[0][0]}×${run.levels[0][1]}`);
  if (run.budget && !run.params.max_splats) facts.push(`${fmtCompact(run.budget)} splats at most (auto)`);
  if (run.views) facts.push(`${run.views.train} training views, ${run.views.scored} scored`);
  const k = run.start?.keyframes;
  if (k?.length) {
    const med = (a: number[]) => [...a].sort((p, q) => p - q)[Math.floor(a.length / 2)];
    const before = med(k.map((x) => x.agree_before));
    const after = med(k.map((x) => x.agree_after));
    const pct = (v: number) => `${(100 * (Math.exp(v) - 1)).toFixed(1)} %`;
    facts.push(`SHARP depth vs Solv3R: ${pct(before)} off with one scale` + (run.params.align === "depth" ? `, ${pct(after)} per pixel` : ""));
  }
  if (run.start?.merged !== undefined && run.start.predicted) facts.push(`${fmtCompact(run.start.merged)} of ${fmtCompact(run.start.predicted)} SHARP splats kept after merging`);
  if (run.cameras) {
    facts.push(
      `cameras moved ${run.cameras.rotation_deg.toFixed(3)}° (median)` +
        (run.cameras.focal_pct !== undefined ? `, focal lengths ${run.cameras.focal_pct.toFixed(2)} %` : ""),
    );
  }
  if (run.note) facts.push(run.note);
  return (
    <p className="line-clamp-2 text-xs text-pretty text-stone-500 tabular-nums" title={facts.join(" · ")}>
      {facts.join(" · ")}
    </p>
  );
}

function HeldOut({ run }: { run: SplatRun }) {
  const session = useApp((s) => s.session);
  const [k, setK] = React.useState(0);
  const views = run.eval_views ?? [];
  React.useEffect(() => {
    setK(0);
  }, [run.id]);
  if (!session) return null;
  if (!views.length) {
    return (
      <div className="flex h-full items-center justify-center">
        <Hint>{run.note ?? (run.state === "running" ? "Scores appear once the starting splats are ready." : "No views were held out in this run.")}</Hint>
      </div>
    );
  }
  const i = Math.min(k, views.length - 1);
  const v = views[i];
  const tiles: { which: "gt" | "init" | "final"; label: string; psnr?: number }[] = [
    { which: "gt", label: `Real frame · ${v.name}` },
    ...(run.init ? [{ which: "init" as const, label: run.params.init === "sharp" ? "SHARP, before training" : "Points, before training", psnr: run.init.views?.[i] }] : []),
    ...(run.final ? [{ which: "final" as const, label: `Trained, ${run.params.steps.toLocaleString()} steps`, psnr: run.final.views?.[i] }] : []),
  ];
  return (
    <div className="flex h-full min-h-0 gap-3">
      <div className="scroll-thin flex w-28 shrink-0 flex-col gap-1 overflow-y-auto" role="listbox" aria-label="Held-out views">
        {views.map((x, j) => {
          const d = run.final?.views?.[j] ?? run.init?.views?.[j];
          return (
            <button
              key={x.view}
              type="button"
              role="option"
              aria-selected={j === i}
              onClick={() => setK(j)}
              className={cn(
                "flex items-baseline justify-between gap-1 rounded-sm px-1.5 py-0.5 text-left text-xs tabular-nums focus-visible:outline-2 focus-visible:outline-blue-400",
                j === i ? "bg-stone-800 text-stone-100" : "text-stone-400 hover:bg-stone-800/60",
              )}
            >
              <span>view {x.view + 1}</span>
              {d !== undefined && <span className="text-stone-500">{d.toFixed(1)}</span>}
            </button>
          );
        })}
        <Button size="sm" className="mt-1" onClick={() => getViewer()?.goToCamera(v.view)} title="Look through this view's camera in the 3D view">
          <Crosshair className="size-3.5" aria-hidden />
          Look
        </Button>
      </div>
      {tiles.map((t) => (
        <figure key={t.which} className="flex min-w-0 flex-1 flex-col gap-1">
          <div className="flex min-h-0 flex-1 items-center justify-center overflow-hidden rounded-md bg-stone-900">
            <img src={api.compareUrl(session.id, run.id, i, t.which)} alt={t.label} className="max-h-full max-w-full object-contain" />
          </div>
          <figcaption className="flex items-baseline justify-between gap-2 text-xs text-stone-400">
            <span className="truncate">{t.label}</span>
            {t.psnr !== undefined && <span className="shrink-0 font-semibold text-stone-200 tabular-nums">{t.psnr.toFixed(2)} dB</span>}
          </figcaption>
        </figure>
      ))}
    </div>
  );
}

function Training({ run }: { run: SplatRun }) {
  const steps = useApp((s) => s.splatSteps[run.id]) ?? [];
  if (!run.params.steps) {
    return (
      <div className="flex h-full items-center justify-center">
        <Hint>This run has no training: the scores are those of the starting splats.</Hint>
      </div>
    );
  }
  const last = steps[steps.length - 1];
  // coarse to fine: a marker where the training size changes (the training PSNR drops at a finer size)
  const markers = steps
    .filter((p, i) => i > 0 && p.level !== undefined && p.level !== steps[i - 1].level)
    .map((p) => ({ x: p.step, label: `${p.w} px` }));
  return (
    <div className="flex h-full min-h-0 gap-4">
      <div className="flex min-w-0 flex-1 flex-col gap-1">
        <div className="flex items-center gap-4 text-xs text-stone-400" aria-hidden>
          <span className="flex items-center gap-1.5">
            <span className="h-0.5 w-4 rounded-full bg-[var(--color-viz-series)]" />
            Training views (smoothed)
          </span>
          <span className="flex items-center gap-1.5">
            <span className="size-2 rounded-full bg-[var(--color-viz-ink)]" />
            Held-out views
          </span>
        </div>
        <div className="min-h-0 flex-1">
          <LineChart
            ariaLabel="PSNR of the training views per step, with the held-out scores"
            points={steps.map((p) => ({ x: p.step, y: p.psnr }))}
            dots={run.curve.map((c) => ({ x: c.step, y: c.psnr, label: c.psnr.toFixed(1) }))}
            markers={markers}
            yFrom="data"
            formatY={(v) => `${Math.round(v)}`}
            formatValue={(v) => `${v.toFixed(1)} dB`}
            xLabel="step"
            tooltip={(i) => {
              const p = steps[i];
              return (
                <>
                  <div className="font-semibold text-stone-50 tabular-nums">{p.psnr.toFixed(2)} dB</div>
                  <div className="text-stone-400 tabular-nums">
                    step {p.step.toLocaleString()} · loss {p.loss.toFixed(4)} · {fmtCompact(p.n)} splats
                  </div>
                  {p.w && p.h ? <div className="text-stone-400 tabular-nums">at {p.w}×{p.h}</div> : null}
                </>
              );
            }}
          />
        </div>
      </div>
      <dl className="grid w-44 shrink-0 content-start gap-y-2 text-xs">
        <Fact label="Step" value={last ? `${last.step.toLocaleString()} / ${last.steps.toLocaleString()}` : "–"} />
        {last?.w && last.h ? <Fact label="Size" value={`${last.w}×${last.h}`} /> : null}
        <Fact label="Time per step (average)" value={last ? `${last.ms.toFixed(1)} ms` : "–"} />
        <Fact label="Splats" value={last ? fmtCompact(last.n) : "–"} />
        <Fact label="Training time" value={run.seconds.train !== undefined ? `${Math.round(run.seconds.train)} s` : "–"} />
      </dl>
    </div>
  );
}

function Fact({ label, value }: { label: string; value: string }) {
  return (
    <div className="flex flex-col">
      <dt className="text-stone-500">{label}</dt>
      <dd className="font-semibold text-stone-100 tabular-nums">{value}</dd>
    </div>
  );
}
