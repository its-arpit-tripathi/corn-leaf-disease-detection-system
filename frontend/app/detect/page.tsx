"use client";
import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import Protected from "@/components/Protected";
import ResultView from "@/components/ResultView";
import { api, type Prediction } from "@/lib/api";

const MAX_MB = 8;
const TYPES = ["image/jpeg", "image/png", "image/webp", "image/bmp"];

function Detect() {
  const [file, setFile] = useState<File | null>(null);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [result, setResult] = useState<Prediction | null>(null);
  const [drag, setDrag] = useState(false);
  const input = useRef<HTMLInputElement>(null);

  const preview = useMemo(() => (file ? URL.createObjectURL(file) : null), [file]);
  useEffect(() => () => { if (preview) URL.revokeObjectURL(preview); }, [preview]);

  const choose = useCallback((f: File | undefined | null) => {
    if (!f) return;
    setResult(null); setError(null);
    if (!TYPES.includes(f.type)) { setFile(null); setError("That file type isn't supported. Choose a JPEG, PNG, WebP or BMP image."); return; }
    if (f.size > MAX_MB * 1024 * 1024) { setFile(null); setError(`That image is larger than ${MAX_MB} MB. Choose a smaller photo.`); return; }
    setFile(f);
  }, []);

  async function analyse() {
    if (!file) return;
    setBusy(true); setError(null);
    const body = new FormData();
    body.append("file", file);
    try { setResult(await api<Prediction>("/api/predict", { method: "POST", body })); }
    catch (e) { setError(e instanceof Error ? e.message : "Analysis failed"); }
    finally { setBusy(false); }
  }

  function reset() { setFile(null); setResult(null); setError(null); if (input.current) input.current.value = ""; }

  if (result) {
    return (
      <div>
        <div className="mb-6 flex flex-wrap items-center justify-between gap-3">
          <h1 className="text-3xl font-semibold text-leaf">Result</h1>
          <button onClick={reset} className="rounded-lg bg-leaf px-4 py-2.5 font-semibold text-white hover:bg-[#173d29]">Check another leaf</button>
        </div>
        <ResultView p={result} />
      </div>
    );
  }

  return (
    <div className="mx-auto max-w-2xl">
      <h1 className="text-3xl font-semibold text-leaf">Check a corn leaf</h1>
      <p className="mt-2 text-muted">Use a sharp, well-lit photo of one leaf. A plain background gives the best result.</p>

      <div
        onDragOver={(e) => { e.preventDefault(); setDrag(true); }} onDragLeave={() => setDrag(false)}
        onDrop={(e) => { e.preventDefault(); setDrag(false); choose(e.dataTransfer.files[0]); }}
        className={`relative mt-6 overflow-hidden rounded-2xl border-2 border-dashed ${drag ? "border-leaf bg-leaf-soft" : "border-line bg-panel"}`}>
        {preview ? (
          <div className="relative">
            {/* eslint-disable-next-line @next/next/no-img-element */}
            <img src={preview} alt="Selected leaf preview" className="mx-auto max-h-[28rem] w-full object-contain" />
            {busy && <div className="scanline pointer-events-none absolute inset-x-0 top-0 h-1 bg-amber/90 shadow-[0_0_18px_6px_rgba(183,121,31,.5)]" aria-hidden="true" />}
          </div>
        ) : (
          <label htmlFor="file" className="flex min-h-64 cursor-pointer flex-col items-center justify-center gap-2 p-8 text-center">
            <span className="font-display text-xl font-semibold">Drop a leaf photo here</span>
            <span className="text-muted">or choose a file. JPEG, PNG, WebP or BMP, up to {MAX_MB} MB.</span>
          </label>
        )}
        <input id="file" ref={input} type="file" accept={TYPES.join(",")} className="sr-only"
          onChange={(e) => choose(e.target.files?.[0])} />
      </div>

      {error && <p role="alert" className="mt-4 rounded-lg border border-danger bg-[#fbe9e5] p-3 text-sm text-danger">{error}</p>}

      <div className="mt-5 flex flex-wrap gap-3">
        {file && (
          <button onClick={analyse} disabled={busy}
            className="rounded-lg bg-leaf px-5 py-3 font-semibold text-white hover:bg-[#173d29] disabled:opacity-60">
            {busy ? "Analysing…" : "Analyse leaf"}
          </button>
        )}
        {file ? (
          <button onClick={reset} disabled={busy} className="rounded-lg border border-line bg-panel px-5 py-3 font-semibold hover:bg-leaf-soft disabled:opacity-60">Choose a different photo</button>
        ) : (
          <button onClick={() => input.current?.click()} className="rounded-lg bg-leaf px-5 py-3 font-semibold text-white hover:bg-[#173d29]">Choose a photo</button>
        )}
      </div>
      <p className="sr-only" role="status">{busy ? "Analysing leaf" : ""}</p>
    </div>
  );
}

export default function Page() { return <Protected><Detect /></Protected>; }
