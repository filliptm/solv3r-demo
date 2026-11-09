import * as React from "react";
import { FileVideo, Images, Upload } from "lucide-react";
import { api } from "../lib/api";
import { cn, fmtDuration } from "../lib/cn";
import { useApp } from "../lib/store";
import type { VideoMeta } from "../lib/types";
import { Button, Confirm, ErrorText, Hint, NumberField, ProgressBar, Section, Switch } from "./ui";

const ACCEPT = "image/jpeg,image/png,.jpg,.jpeg,.png,video/*,.mp4,.mov,.avi,.mkv,.webm,.m4v";

export function InputSection() {
  const session = useApp((s) => s.session);
  const info = useApp((s) => s.info);
  const busy = useApp((s) => s.inputBusy);
  const uploading = useApp((s) => s.inputProgress);
  const error = useApp((s) => s.inputError);
  const hasRun = useApp((s) => s.runKind !== null);
  const jobActive = useApp((s) => s.job?.state === "running" || s.job?.state === "queued");
  const loadSample = useApp((s) => s.loadSample);
  const loadFiles = useApp((s) => s.loadFiles);
  const [choosing, setChoosing] = React.useState(false);
  const [pending, setPending] = React.useState<null | (() => void)>(null);
  const [opts, setOpts] = React.useState({ stride: 1, maxImages: 0, frames: 0, sharpest: true });
  const [showOpts, setShowOpts] = React.useState(false);
  const [drag, setDrag] = React.useState(false);
  const fileRef = React.useRef<HTMLInputElement>(null);

  const guard = (fn: () => void) => {
    if (hasRun) setPending(() => fn);
    else fn();
  };
  const useFiles = (list: FileList | File[] | null) => {
    const files = Array.from(list ?? []);
    if (fileRef.current) fileRef.current.value = ""; // the same file can be picked again
    if (!files.length) return;
    guard(() => {
      setChoosing(false);
      loadFiles(files, { stride: opts.stride, maxImages: opts.maxImages || undefined, frames: opts.frames || undefined, sharpest: opts.sharpest });
    });
  };

  const loading = uploading !== null || Boolean(session?.preparing);
  const showChooser = !loading && (!session || session.source === "live" || choosing);

  return (
    <Section
      title="Input"
      aside={
        loading ? null : session && !showChooser ? (
          <Button size="sm" variant="ghost" onClick={() => setChoosing(true)} disabled={jobActive}>
            Change input
          </Button>
        ) : session && choosing ? (
          <Button size="sm" variant="ghost" onClick={() => setChoosing(false)}>
            Cancel
          </Button>
        ) : null
      }
    >
      {loading && <LoadingCard />}
      {!loading && session && !showChooser && <SessionSummaryCard guard={guard} />}

      {showChooser && (
        <>
          <div className="grid grid-cols-2 gap-2">
            {(info?.samples ?? []).map((s) => (
              <button
                key={s.id}
                type="button"
                disabled={busy || jobActive}
                onClick={() => guard(() => { setChoosing(false); loadSample(s.id); })}
                className="group flex flex-col overflow-hidden rounded-md border border-stone-800 bg-stone-900 text-left hover:enabled:border-stone-600 focus-visible:outline-2 focus-visible:outline-offset-2 focus-visible:outline-blue-400 disabled:opacity-50"
              >
                <img src={api.coverUrl(s.id)} alt="" className="aspect-[4/3] w-full bg-stone-800 object-cover" loading="lazy" />
                <span className="flex flex-col gap-0.5 p-2">
                  <span className="text-sm font-medium text-stone-100">{s.title}</span>
                  <span className="text-xs text-pretty text-stone-500">{s.blurb}</span>
                  <span className="text-xs text-stone-400">{s.mode === "stream" ? "Streaming" : "Offline"} · {s.n} images</span>
                </span>
              </button>
            ))}
            {!info &&
              [0, 1].map((i) => (
                <div key={i} className="flex flex-col overflow-hidden rounded-md border border-stone-800">
                  <div className="aspect-[4/3] bg-stone-800/70" />
                  <div className="flex flex-col gap-1.5 p-2">
                    <div className="h-3.5 w-16 rounded bg-stone-800" />
                    <div className="h-3 w-28 rounded bg-stone-800/70" />
                  </div>
                </div>
              ))}
          </div>

          <div
            onDragOver={(e) => {
              e.preventDefault();
              setDrag(true);
            }}
            onDragLeave={() => setDrag(false)}
            onDrop={(e) => {
              e.preventDefault();
              setDrag(false);
              useFiles(e.dataTransfer.files);
            }}
            className={cn(
              "flex flex-col items-center gap-2 rounded-md border border-dashed px-3 py-4 text-center",
              drag ? "border-blue-500 bg-blue-950/30" : "border-stone-700 bg-stone-900/40",
            )}
          >
            <div className="flex gap-2 text-stone-500">
              <Images className="size-5" aria-hidden />
              <FileVideo className="size-5" aria-hidden />
            </div>
            <p className="text-xs text-pretty text-stone-400">
              Drop photos of a scene (JPG or PNG) or a video (MP4, MOV, MKV, WebM) here. Frames are taken from a video
              automatically.
            </p>
            <input ref={fileRef} type="file" multiple accept={ACCEPT} className="hidden" onChange={(e) => useFiles(e.target.files)} />
            <Button size="sm" onClick={() => fileRef.current?.click()} disabled={busy || jobActive}>
              <Upload className="size-3.5" aria-hidden />
              {busy ? "Loading…" : "Browse files"}
            </Button>
            <button
              type="button"
              className="text-xs text-stone-500 underline-offset-2 hover:text-stone-300 hover:underline"
              onClick={() => setShowOpts((v) => !v)}
              aria-expanded={showOpts}
            >
              {showOpts ? "Hide loading options" : "Loading options"}
            </button>
            {showOpts && (
              <div className="flex w-full flex-col gap-2 text-left">
                <div className="grid grid-cols-3 gap-2">
                  <NumberField label="Stride" value={opts.stride} min={1} max={50} onChange={(v) => setOpts({ ...opts, stride: v })} />
                  <NumberField label="Max images" value={opts.maxImages} min={0} max={2000} step={10} hint={opts.maxImages ? undefined : "all"} onChange={(v) => setOpts({ ...opts, maxImages: v })} />
                  <NumberField label="Video frames" value={opts.frames} min={0} max={2000} step={10} hint={opts.frames ? undefined : "auto"} onChange={(v) => setOpts({ ...opts, frames: v })} />
                </div>
                <Switch label="Sharpest video frames" checked={opts.sharpest} onChange={(v) => setOpts({ ...opts, sharpest: v })} />
                <Hint>
                  Stride takes every n-th photo. A video gives frames over its whole length: automatically 3 per second, at
                  least 24 and at most 240. Sharpest frames takes the least blurred frame of each stretch (the video is decoded
                  twice); otherwise the frames are evenly spaced. Frames are kept at up to 3840 px for the splats. You can
                  extract again with another count after loading.
                </Hint>
              </div>
            )}
          </div>
          <Hint>Images are resized to a long side of 504 px (sides multiples of 14), as in training. The first image defines the world frame.</Hint>
        </>
      )}
      <ErrorText>{error}</ErrorText>
      <Confirm
        open={pending !== null}
        onOpenChange={(v) => !v && setPending(null)}
        title="Replace the current reconstruction?"
        description="Loading a new input discards the current solve, its state and its point cloud. Download it first if you want to keep it."
        confirmLabel="Load new input"
        onConfirm={() => {
          pending?.();
          setPending(null);
        }}
      />
    </Section>
  );
}

function sourceLabel(source: string) {
  if (source.startsWith("sample:")) return `Sample: ${source.slice(7)}`;
  if (source.startsWith("video:")) return `Video: ${source.slice(6)}`;
  if (source === "live") return "Webcam";
  return "Uploaded images";
}

/** The upload, then the server filling the pool (probing, frame extraction, resizing). */
function LoadingCard() {
  const uploading = useApp((s) => s.inputProgress);
  const session = useApp((s) => s.session);
  const cancelInput = useApp((s) => s.cancelInput);
  const prep = uploading ? null : session?.preparing;
  const label = uploading?.label ?? prep?.label ?? "Loading";
  const frac = uploading ? uploading.frac : prep?.total ? prep.done / prep.total : null;
  const title = uploading ? "Uploading" : sourceLabel(session?.source ?? "");
  return (
    <div className="flex flex-col gap-2" aria-live="polite">
      <div className="flex items-baseline justify-between gap-2">
        <span className="truncate text-sm font-medium text-stone-100">{title}</span>
        {prep && (
          <Button size="sm" variant="ghost" onClick={cancelInput}>
            Cancel
          </Button>
        )}
      </div>
      <ProgressBar value={frac} label={label} />
      <div className="flex items-baseline justify-between gap-2 text-xs text-stone-400">
        <span className="truncate">{label}</span>
        <span className="shrink-0 tabular-nums">
          {prep?.total ? `${prep.done} / ${prep.total}` : frac !== null ? `${Math.round(frac * 100)}%` : ""}
        </span>
      </div>
    </div>
  );
}

function SessionSummaryCard({ guard }: { guard: (fn: () => void) => void }) {
  const session = useApp((s) => s.session)!;
  const shown = Math.min(session.n, 24);
  const step = session.n / Math.max(shown, 1);
  return (
    <div className="flex flex-col gap-2">
      <div className="flex items-baseline justify-between gap-2">
        <span className="truncate text-sm font-medium text-stone-100">{sourceLabel(session.source)}</span>
        <span className="shrink-0 text-xs text-stone-400 tabular-nums">
          {session.n} {session.video ? "frames" : "images"}
          {session.hw ? ` · ${session.hw[1]}×${session.hw[0]}` : ""}
        </span>
      </div>
      <div className="scroll-thin flex gap-1 overflow-x-auto pb-1">
        {Array.from({ length: shown }, (_, k) => Math.floor(k * step)).map((i) => (
          <img
            key={i}
            src={api.imageUrl(session.id, i, 160)}
            alt={session.names[i]}
            title={session.names[i]}
            loading="lazy"
            className="h-12 w-auto shrink-0 rounded-sm bg-stone-800"
          />
        ))}
      </div>
      {session.video && <VideoFrames key={session.id} video={session.video} guard={guard} />}
      {session.warnings.map((w) => (
        <Hint key={w}>{w}</Hint>
      ))}
    </div>
  );
}

const CODECS: Record<string, string> = { h264: "H.264", hevc: "HEVC", av1: "AV1", vp9: "VP9", vp8: "VP8", mpeg4: "MPEG-4", prores: "ProRes", mjpeg: "MJPEG" };

/** What was taken from a video, and extracting another number of frames from it (no new upload). */
function VideoFrames({ video, guard }: { video: VideoMeta; guard: (fn: () => void) => void }) {
  const reextract = useApp((s) => s.reextract);
  const busy = useApp((s) => s.inputBusy);
  const jobActive = useApp((s) => s.job?.state === "running" || s.job?.state === "queued");
  const max = Math.max(2, Math.min(2000, video.frames || 2000));
  const [frames, setFrames] = React.useState(video.extracted);
  const [sharpest, setSharpest] = React.useState(video.sharpest ?? false);
  const changed = frames !== video.extracted || sharpest !== (video.sharpest ?? false);
  const spacing = video.duration && frames ? video.duration / frames : null;
  return (
    <div className="flex flex-col gap-2 rounded-md border border-stone-800 bg-stone-900/50 p-2.5">
      <p className="text-xs text-stone-400 tabular-nums">
        {fmtDuration(video.duration)} · {video.width}×{video.height} · {+video.fps.toFixed(2)} fps · {CODECS[video.codec] ?? video.codec.toUpperCase()}
      </p>
      <p className="text-xs text-pretty text-stone-300 tabular-nums">
        {video.extracted} {video.sharpest ? "sharpest frames" : "frames"}, one every {(video.spacing ?? video.duration / video.extracted).toFixed(2)} s
        {video.sharpest ? " on average" : ""}
        {video.extracted === video.auto ? " (automatic)" : ""}
      </p>
      <div className="flex items-end gap-2">
        <div className="w-32">
          <NumberField label="Frames" value={frames} min={2} max={max} step={10} onChange={setFrames} disabled={busy || jobActive} />
        </div>
        <Button size="md" className="h-8" onClick={() => guard(() => reextract(frames, sharpest))} disabled={!changed || busy || jobActive}>
          Extract again
        </Button>
        {frames !== video.auto && (
          <Button size="sm" variant="ghost" className="h-8" onClick={() => setFrames(video.auto)} disabled={busy || jobActive}>
            Auto ({video.auto})
          </Button>
        )}
      </div>
      <Switch label="Sharpest frames" checked={sharpest} onChange={setSharpest} disabled={busy || jobActive} />
      {changed && spacing !== null && (
        <Hint className="tabular-nums">
          {frames} frames: one every {spacing.toFixed(2)} s{frames > 120 ? ", streamed by default" : ""}
        </Hint>
      )}
    </div>
  );
}
