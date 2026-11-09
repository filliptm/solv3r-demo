import * as React from "react";
import { ExternalLink } from "lucide-react";
import { BottomPanel } from "./components/BottomPanel";
import { InputSection } from "./components/InputSection";
import { LabSection } from "./components/LabSection";
import { SolveSection } from "./components/SolveSection";
import { SplatSection } from "./components/SplatSection";
import { TooltipProvider } from "./components/ui";
import { ViewerPane } from "./components/ViewerPane";
import { getViewer, useApp } from "./lib/store";

export function App() {
  const loadInfo = useApp((s) => s.loadInfo);
  const restore = useApp((s) => s.restore);

  React.useEffect(() => {
    loadInfo().then(() => restore());
  }, [loadInfo, restore]);

  React.useEffect(() => {
    const onKey = (e: KeyboardEvent) => {
      const t = e.target as HTMLElement;
      if (t.closest("input, textarea, select, [contenteditable], [role=slider], [role=listbox]")) return;
      if (e.key === "Escape") {
        useApp.getState().select(null);
        useApp.getState().setView({ isolate: false });
      } else if (e.key === "f" || e.key === "F") {
        getViewer()?.refit(true);
      }
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, []);

  return (
    <TooltipProvider delay={300}>
      <div className="flex h-dvh w-full overflow-hidden pr-[env(safe-area-inset-right)] pl-[env(safe-area-inset-left)]">
        <aside className="scroll-thin flex w-[22rem] shrink-0 flex-col overflow-y-auto border-r border-stone-800 bg-stone-900/40 pb-[env(safe-area-inset-bottom)]">
          <Header />
          <InputSection />
          <SolveSection />
          <SplatSection />
          <LabSection />
          <Footer />
        </aside>
        <main className="flex min-w-0 flex-1 flex-col pb-[env(safe-area-inset-bottom)]">
          <ViewerPane />
          <BottomPanel />
        </main>
      </div>
    </TooltipProvider>
  );
}

function Header() {
  const info = useApp((s) => s.info);
  const infoError = useApp((s) => s.infoError);
  const status = infoError ? "error" : info?.ready ? "ready" : "loading";
  return (
    <header className="flex flex-col gap-2 border-b border-stone-800 px-4 pt-4 pb-3">
      <div className="flex items-baseline justify-between gap-2">
        <h1 className="text-lg font-semibold text-balance text-stone-50">Solv3R</h1>
        <span className="flex items-center gap-1.5 text-xs text-stone-400">
          <span
            aria-hidden
            className={
              status === "ready" ? "size-2 rounded-full bg-[#0ca30c]" : status === "error" ? "size-2 rounded-full bg-[#d03b3b]" : "size-2 rounded-full bg-stone-500"
            }
          />
          {status === "ready" ? "Model ready" : status === "error" ? "Model error" : "Loading model"}
        </span>
      </div>
      <p className="text-sm text-pretty text-stone-400">3D looped Transformer as a multi-view solver: one block, applied again and again to a state per view.</p>
      <nav className="flex flex-wrap gap-x-3 gap-y-1 text-xs">
        {[
          ["Paper", "https://arxiv.org/abs/2611.02857"],
          ["Project page", "https://ubc-vision.github.io/solv3r/"],
          ["Code", "https://github.com/ubc-vision/solv3r"],
          ["Weights", "https://huggingface.co/jgkwak/solv3r"],
        ].map(([label, href]) => (
          <a key={label} href={href} target="_blank" rel="noreferrer" className="inline-flex items-center gap-1 text-stone-400 underline-offset-2 hover:text-stone-100 hover:underline">
            {label}
            <ExternalLink className="size-3" aria-hidden />
          </a>
        ))}
      </nav>
      {info?.gpu && (
        <p className="text-xs text-stone-500 tabular-nums">
          {info.gpu.name} · {info.gpu.total_gib} GiB · torch {info.torch.split("+")[0]}
        </p>
      )}
    </header>
  );
}

function Footer() {
  return (
    <footer className="mt-auto px-4 py-3 text-xs text-pretty text-stone-600">
      Kwak, Tagliasacchi, Kagami, Ono, Yi. Code Apache-2.0; weights under the NVIDIA license (non-commercial research and evaluation use).
    </footer>
  );
}
