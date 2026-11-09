import * as React from "react";
import { FastForward, ImagePlus, Shuffle } from "lucide-react";
import { useApp } from "../lib/store";
import { Button, ErrorText, Hint, NumberField, Section, Select, Slider } from "./ui";

/** Solver properties on the current state: continue, add views, damage part of the state and solve again. */
export function LabSection() {
  const session = useApp((s) => s.session);
  const job = useApp((s) => s.job);
  const runKind = useApp((s) => s.runKind);
  const views = useApp((s) => s.cameraCount);
  const calls = useApp((s) => s.calls.length);
  const lab = useApp((s) => s.params.lab);
  const setParams = useApp((s) => s.setParams);
  const runLab = useApp((s) => s.runLab);
  const addViews = useApp((s) => s.addViews);
  const error = useApp((s) => s.labError);
  const fileRef = React.useRef<HTMLInputElement>(null);

  const active = job?.state === "running" || job?.state === "queued";
  const available = session?.kind === "solve" && runKind === "solve" && calls > 0;
  const disabled = !available || active;

  return (
    <Section title="Solver lab">
      {!available ? (
        <Hint>
          {runKind === "stream"
            ? "These experiments act on a kept solver state; a stream answers frames from a sliding window instead. Run an offline, incremental or any-order solve to use them."
            : "Run an offline, incremental or any-order solve. Its state stays on the GPU, and you can then continue it, add views to it, or damage it and watch the solver repair it."}
        </Hint>
      ) : (
        <Hint>The update has no step index, so the state can be continued, extended or perturbed at any call. The chart marks each action.</Hint>
      )}
      {available && (
      <>
      <NumberField label="Calls per action" value={lab.calls} min={1} max={256} onChange={(v) => setParams((p) => void (p.lab.calls = v))} disabled={disabled} />

      <div className="grid grid-cols-2 gap-2">
        <Button onClick={() => runLab("continue")} disabled={disabled}>
          <FastForward className="size-3.5" aria-hidden />
          Continue +{lab.calls}
        </Button>
        <Button onClick={() => fileRef.current?.click()} disabled={disabled}>
          <ImagePlus className="size-3.5" aria-hidden />
          Add views…
        </Button>
        <input
          ref={fileRef}
          type="file"
          multiple
          accept="image/jpeg,image/png,.jpg,.jpeg,.png"
          className="hidden"
          onChange={(e) => {
            const files = Array.from(e.target.files ?? []);
            e.target.value = "";
            if (files.length) addViews(files);
          }}
        />
      </div>

      <div className="flex flex-col gap-3 rounded-md border border-stone-800 bg-stone-900/50 p-3">
        <div className="flex flex-col gap-1">
          <span className="text-xs font-medium text-stone-200">Correction</span>
          <Hint>Damage the state of some views, see the damaged reconstruction, then let the solver repair it.</Hint>
        </div>
        <Select
          label="Perturbation"
          value={lab.how}
          disabled={disabled}
          onChange={(v) => setParams((p) => void (p.lab.how = v))}
          items={[
            { value: "reset", label: "Reset views to their encoding", description: "Forget everything solved for them" },
            { value: "noise", label: "Add noise to their state", description: "Gaussian, scaled to the state's RMS" },
            { value: "swap", label: "Shuffle states between views", description: "Each view gets another view's solution" },
          ]}
        />
        <div className="grid grid-cols-2 gap-3">
          <NumberField
            label="Views"
            value={Math.min(lab.count, Math.max(1, views - 1))}
            min={lab.how === "swap" ? 2 : 1}
            max={Math.max(2, views - 1)}
            hint={views ? `of ${views}` : undefined}
            onChange={(v) => setParams((p) => void (p.lab.count = v))}
            disabled={disabled}
          />
          {lab.how === "noise" ? (
            <Slider label="Noise σ" value={lab.sigma} min={0.1} max={4} step={0.1} format={(v) => `${v.toFixed(1)}×`} onChange={(v) => setParams((p) => void (p.lab.sigma = v))} disabled={disabled} />
          ) : (
            <div />
          )}
        </div>
        <Button onClick={() => runLab("perturb")} disabled={disabled}>
          <Shuffle className="size-3.5" aria-hidden />
          Perturb and re-solve
        </Button>
        <Hint>The reference view (view 1) is never perturbed, so the world frame stays put.</Hint>
      </div>
      </>
      )}
      <ErrorText>{error}</ErrorText>
    </Section>
  );
}
