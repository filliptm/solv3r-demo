import * as React from "react";

export interface ChartPoint {
  x: number;
  y: number;
}

interface Props {
  points: ChartPoint[];
  yLog?: boolean;
  formatY: (v: number) => string;
  /** Format of the direct end label (defaults to formatY). */
  formatValue?: (v: number) => string;
  xLabel: string;
  /** Vertical event lines (stage boundaries, perturbations ...). */
  markers?: { x: number; label: string }[];
  /** A horizontal reference line (a stopping tolerance). */
  refLine?: { y: number; label: string } | null;
  /** Small ticks on the baseline at these x (stream memory refreshes). */
  baseTicks?: number[];
  baseTickLabel?: string;
  tooltip: (index: number) => React.ReactNode;
  ariaLabel: string;
  /** Linear axis: start at zero (magnitudes, the default) or at the data (a quantity like PSNR, in dB). */
  yFrom?: "zero" | "data";
  /** Extra values the linear axis must include (reference points drawn as markers). */
  yInclude?: number[];
  /** Points drawn as ringed dots with a direct label, on the same axes (e.g. held-out scores). */
  dots?: { x: number; y: number; label: string }[];
}

const M = { left: 46, right: 64, top: 20, bottom: 24 };

function niceStep(span: number, target: number) {
  const raw = span / Math.max(1, target);
  const mag = 10 ** Math.floor(Math.log10(raw));
  for (const m of [1, 2, 5, 10]) if (raw <= m * mag) return m * mag;
  return 10 * mag;
}

function useSize(ref: React.RefObject<HTMLDivElement | null>) {
  const [size, setSize] = React.useState({ w: 0, h: 0 });
  React.useLayoutEffect(() => {
    const el = ref.current;
    if (!el) return;
    const ro = new ResizeObserver(() => setSize({ w: el.clientWidth, h: el.clientHeight }));
    ro.observe(el);
    setSize({ w: el.clientWidth, h: el.clientHeight });
    return () => ro.disconnect();
  }, [ref]);
  return size;
}

/** A single-series line chart in SVG: hairline grid, 2px line, ringed end dot with a direct label, a crosshair that
 * snaps to the nearest x, and the same readout on keyboard focus (arrow keys). */
export function LineChart({ points, yLog, formatY, formatValue = formatY, xLabel, markers = [], refLine, baseTicks = [], baseTickLabel, tooltip, ariaLabel, yFrom = "zero", yInclude = [], dots = [] }: Props) {
  const box = React.useRef<HTMLDivElement>(null);
  const { w, h } = useSize(box);
  const [hover, setHover] = React.useState<number | null>(null);

  const valid = points.filter((p) => Number.isFinite(p.y) && (!yLog || p.y > 0));
  const iw = Math.max(10, w - M.left - M.right);
  const ih = Math.max(10, h - M.top - M.bottom);

  const xs = valid.map((p) => p.x).concat(dots.map((d) => d.x));
  const x0 = xs.length ? Math.min(...xs, ...markers.map((m) => m.x)) : 0;
  const x1 = xs.length ? Math.max(...xs, x0 + 4) : 1;
  const ys = valid.map((p) => p.y).concat(refLine ? [refLine.y] : [], yInclude, dots.map((d) => d.y));

  let y0: number, y1: number, yTicks: number[];
  const tf = (v: number) => (yLog ? Math.log10(v) : v);
  if (yLog) {
    const lo = ys.length ? Math.floor(Math.log10(Math.min(...ys))) : -3;
    const hi = ys.length ? Math.ceil(Math.log10(Math.max(...ys))) : 0;
    y0 = Math.min(lo, hi - 1);
    y1 = hi;
    yTicks = [];
    for (let e = y0; e <= y1; e++) yTicks.push(10 ** e);
  } else if (yFrom === "data") {
    const lo = ys.length ? Math.min(...ys) : 0;
    const hi = ys.length ? Math.max(...ys) : 1;
    const step = niceStep(Math.max(hi - lo, 1e-6), 4);
    y0 = Math.floor(lo / step) * step;
    y1 = Math.max(Math.ceil(hi / step) * step, y0 + step);
    yTicks = [];
    for (let v = y0; v <= y1 + 1e-9; v += step) yTicks.push(+v.toFixed(6));
  } else {
    const hi = ys.length ? Math.max(...ys) : 1;
    const step = niceStep(hi, 4);
    y0 = 0;
    y1 = Math.ceil(hi / step) * step || step;
    yTicks = [];
    for (let v = 0; v <= y1 + 1e-9; v += step) yTicks.push(v);
  }
  const sx = (x: number) => M.left + ((x - x0) / Math.max(1e-9, x1 - x0)) * iw;
  const sy = (v: number) => M.top + ih - ((tf(v) - y0) / Math.max(1e-9, y1 - y0)) * ih;

  const xStep = Math.max(1, niceStep(x1 - x0, Math.max(2, Math.floor(iw / 70))));
  const xTicks: number[] = [];
  for (let x = Math.ceil(x0 / xStep) * xStep; x <= x1; x += xStep) xTicks.push(x);

  const path = valid.map((p, i) => `${i ? "L" : "M"}${sx(p.x).toFixed(1)},${sy(p.y).toFixed(1)}`).join("");
  const last = valid[valid.length - 1];

  const nearest = (px: number) => {
    let best = 0;
    let bd = Infinity;
    valid.forEach((p, i) => {
      const d = Math.abs(sx(p.x) - px);
      if (d < bd) {
        bd = d;
        best = i;
      }
    });
    return best;
  };

  // marker labels only where they do not collide
  let lastLabelX = -Infinity;
  const markerEls = markers.map((m, k) => {
    const x = sx(m.x);
    const showLabel = x - lastLabelX > 72 && x < M.left + iw - 10;
    if (showLabel) lastLabelX = x;
    return (
      <g key={`${m.x}-${k}`}>
        <line x1={x} x2={x} y1={M.top} y2={M.top + ih} stroke="var(--color-viz-axis)" strokeWidth={1} />
        {showLabel && (
          <text x={x + 4} y={M.top - 6} fill="var(--color-viz-muted)" fontSize={11}>
            {m.label.length > 18 ? `${m.label.slice(0, 17)}…` : m.label}
          </text>
        )}
      </g>
    );
  });

  const hv = hover !== null && valid[hover] ? valid[hover] : null;
  const hoverMarkers = hv ? markers.filter((m) => m.x === hv.x || m.x === hv.x - 1) : [];

  return (
    <div
      ref={box}
      className="relative h-full w-full outline-none focus-visible:outline-2 focus-visible:outline-offset-2 focus-visible:outline-blue-400"
      tabIndex={valid.length ? 0 : -1}
      aria-label={ariaLabel}
      role="figure"
      onPointerMove={(e) => {
        if (!valid.length) return;
        const r = e.currentTarget.getBoundingClientRect();
        setHover(nearest(e.clientX - r.left));
      }}
      onPointerLeave={() => setHover(null)}
      onFocus={() => valid.length && setHover(valid.length - 1)}
      onBlur={() => setHover(null)}
      onKeyDown={(e) => {
        if (!valid.length) return;
        if (e.key === "ArrowLeft") setHover((i) => Math.max(0, (i ?? valid.length) - 1));
        else if (e.key === "ArrowRight") setHover((i) => Math.min(valid.length - 1, (i ?? -1) + 1));
        else if (e.key === "Home") setHover(0);
        else if (e.key === "End") setHover(valid.length - 1);
        else return;
        e.preventDefault();
      }}
    >
      {w > 0 && h > 0 && (
        <svg width={w} height={h} className="block" aria-hidden>
          {yTicks.map((v) => (
            <g key={v}>
              <line x1={M.left} x2={M.left + iw} y1={sy(v)} y2={sy(v)} stroke="var(--color-viz-grid)" strokeWidth={1} />
              <text x={M.left - 8} y={sy(v)} dy="0.32em" textAnchor="end" fill="var(--color-viz-muted)" fontSize={11} style={{ fontVariantNumeric: "tabular-nums" }}>
                {formatY(v)}
              </text>
            </g>
          ))}
          <line x1={M.left} x2={M.left + iw} y1={M.top + ih} y2={M.top + ih} stroke="var(--color-viz-axis)" strokeWidth={1} />
          {xTicks.map((x) => (
            <text key={x} x={sx(x)} y={M.top + ih + 16} textAnchor="middle" fill="var(--color-viz-muted)" fontSize={11} style={{ fontVariantNumeric: "tabular-nums" }}>
              {x}
            </text>
          ))}
          <text x={M.left + iw} y={M.top + ih + 16} dx={20} fill="var(--color-viz-muted)" fontSize={11}>
            {xLabel}
          </text>
          {markerEls}
          {refLine && (
            <g>
              <line x1={M.left} x2={M.left + iw} y1={sy(refLine.y)} y2={sy(refLine.y)} stroke="var(--color-viz-ink-2)" strokeWidth={1} opacity={0.6} />
              <text x={M.left + iw + 6} y={sy(refLine.y)} dy="0.32em" fill="var(--color-viz-ink-2)" fontSize={11}>
                {refLine.label}
              </text>
            </g>
          )}
          {baseTicks.map((x) => (
            <line key={`b${x}`} x1={sx(x)} x2={sx(x)} y1={M.top + ih - 6} y2={M.top + ih} stroke="var(--color-viz-ink-2)" strokeWidth={1} />
          ))}
          {baseTicks.length > 0 && baseTickLabel && (
            <text x={M.left + 4} y={M.top + ih - 10} fill="var(--color-viz-muted)" fontSize={11}>
              {baseTickLabel}
            </text>
          )}
          <path d={path} fill="none" stroke="var(--color-viz-series)" strokeWidth={2} strokeLinejoin="round" strokeLinecap="round" />
          {dots.map((d, k) => {
            // a dot at the end of the line would collide with the line's end label: its label goes below
            const nearEnd = last && Math.abs(sx(d.x) - sx(last.x)) < 40;
            return (
              <g key={`d${k}`}>
                <circle cx={sx(d.x)} cy={sy(d.y)} r={4} fill="var(--color-viz-ink)" stroke="var(--color-viz-surface)" strokeWidth={2} />
                <text x={sx(d.x)} y={sy(d.y)} dy={nearEnd ? 18 : -10} textAnchor={k === 0 ? "start" : nearEnd ? "end" : "middle"} fill="var(--color-viz-ink-2)" fontSize={11} style={{ fontVariantNumeric: "tabular-nums" }}>
                  {d.label}
                </text>
              </g>
            );
          })}
          {last && (
            <g>
              <circle cx={sx(last.x)} cy={sy(last.y)} r={4} fill="var(--color-viz-series)" stroke="var(--color-viz-surface)" strokeWidth={2} />
              <text x={sx(last.x) + 8} y={sy(last.y)} dy="0.32em" fill="var(--color-viz-ink)" fontSize={12} fontWeight={600} style={{ fontVariantNumeric: "tabular-nums" }}>
                {formatValue(last.y)}
              </text>
            </g>
          )}
          {hv && (
            <g>
              <line x1={sx(hv.x)} x2={sx(hv.x)} y1={M.top} y2={M.top + ih} stroke="var(--color-viz-ink-2)" strokeWidth={1} />
              <circle cx={sx(hv.x)} cy={sy(hv.y)} r={4} fill="var(--color-viz-series)" stroke="var(--color-viz-surface)" strokeWidth={2} />
            </g>
          )}
        </svg>
      )}
      {hv && hover !== null && (
        <div
          className="pointer-events-none absolute z-20 min-w-36 rounded-md border border-stone-700 bg-stone-900 px-2.5 py-1.5 text-xs shadow-lg"
          style={{
            left: Math.min(sx(hv.x) + 12, w - 180),
            top: Math.max(4, Math.min(sy(hv.y) - 20, h - 90)),
          }}
        >
          {tooltip(hover)}
          {hoverMarkers.map((m) => (
            <div key={m.label} className="mt-1 text-stone-400">
              {m.label}
            </div>
          ))}
        </div>
      )}
    </div>
  );
}
