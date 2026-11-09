import { clsx, type ClassValue } from "clsx";
import { twMerge } from "tailwind-merge";

export function cn(...inputs: ClassValue[]) {
  return twMerge(clsx(inputs));
}

export function fmtInt(n: number) {
  return Math.round(n).toLocaleString();
}

export function fmtCompact(n: number) {
  if (n >= 1e6) return `${(n / 1e6).toFixed(n >= 1e7 ? 0 : 1)}M`;
  if (n >= 1e4) return `${Math.round(n / 1e3)}K`;
  if (n >= 1e3) return `${(n / 1e3).toFixed(1)}K`;
  return String(Math.round(n));
}

export function fmtMs(ms: number | null | undefined) {
  if (ms === null || ms === undefined) return "–";
  if (ms >= 1000) return `${(ms / 1000).toFixed(2)} s`;
  return `${ms < 10 ? ms.toFixed(1) : Math.round(ms)} ms`;
}

export function fmtBytes(n: number) {
  if (n >= 2 ** 30) return `${(n / 2 ** 30).toFixed(1)} GB`;
  if (n >= 2 ** 20) return `${(n / 2 ** 20).toFixed(n >= 100 * 2 ** 20 ? 0 : 1)} MB`;
  return `${Math.max(1, Math.round(n / 2 ** 10))} KB`;
}

/** Seconds as m:ss (or m:ss.s under a minute when precise). */
export function fmtDuration(s: number, precise = false) {
  const m = Math.floor(s / 60);
  const r = s - m * 60;
  return `${m}:${precise ? r.toFixed(1).padStart(4, "0") : String(Math.floor(r)).padStart(2, "0")}`;
}
