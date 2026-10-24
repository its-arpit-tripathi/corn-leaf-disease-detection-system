"use client";
import Link from "next/link";
import { useCallback, useEffect, useState } from "react";
import AuthImage from "@/components/AuthImage";
import Protected from "@/components/Protected";
import ResultView from "@/components/ResultView";
import { api, getClassColor, getClassLabel, type Prediction } from "@/lib/api";

function History() {
  const [items, setItems] = useState<Prediction[] | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [open, setOpen] = useState<Prediction | null>(null);

  const load = useCallback(() => {
    api<Prediction[]>("/api/predictions?limit=200")
      .then(setItems)
      .catch((e) => setError(e.message));
  }, []);
  useEffect(load, [load]);

  async function remove(p: Prediction) {
    if (!confirm("Delete this result? This can't be undone.")) return;
    try {
      await api(`/api/predictions/${p.id}`, { method: "DELETE" });
      setOpen(null);
      setItems((l) => l?.filter((x) => x.id !== p.id) ?? null);
    } catch (e) {
      setError(e instanceof Error ? e.message : "Couldn't delete");
    }
  }

  if (open) {
    return (
      <div>
        <div className="mb-6 flex flex-wrap items-center justify-between gap-3">
          <button
            onClick={() => setOpen(null)}
            className="rounded-lg border border-line bg-panel px-4 py-2.5 font-semibold hover:bg-leaf-soft"
          >
            ← Back to history
          </button>
          <button
            onClick={() => remove(open)}
            className="rounded-lg border border-danger px-4 py-2.5 font-semibold text-danger hover:bg-[#fbe9e5]"
          >
            Delete result
          </button>
        </div>
        <p className="mb-4 text-sm text-muted">
          {open.filename} · {new Date(open.created_at).toLocaleString()}
        </p>
        <ResultView p={open} />
      </div>
    );
  }

  return (
    <div>
      <div className="flex flex-wrap items-center justify-between gap-4">
        <div>
          <h1 className="text-3xl font-semibold text-leaf">Inspection History</h1>
          <p className="mt-1 text-sm text-muted">
            All previous maize leaf disease scans and non-maize checks.
          </p>
        </div>
        <Link
          href="/detect"
          className="rounded-lg bg-leaf px-4 py-2 font-semibold text-white hover:bg-[#173d29]"
        >
          + Check new leaf
        </Link>
      </div>

      {error && (
        <p
          role="alert"
          className="mt-4 rounded-lg border border-danger bg-[#fbe9e5] p-3 text-sm text-danger"
        >
          {error}
        </p>
      )}

      {items === null && !error && (
        <p className="mt-6 text-muted" role="status">
          Loading previous scans…
        </p>
      )}

      {items?.length === 0 && (
        <div className="mt-8 rounded-2xl border border-dashed border-line bg-panel p-8 text-center">
          <p className="text-muted">No inspection records found yet.</p>
          <Link
            href="/detect"
            className="mt-3 inline-block font-semibold text-leaf underline"
          >
            Scan your first maize leaf photo
          </Link>
        </div>
      )}

      <ul className="mt-6 grid gap-4 sm:grid-cols-2 lg:grid-cols-3">
        {items?.map((p) => {
          const diseaseTitle = p.disease_name || getClassLabel(p.predicted_class);
          const color = getClassColor(p.predicted_class);
          return (
            <li key={p.id}>
              <button
                onClick={() => setOpen(p)}
                className="group w-full overflow-hidden rounded-xl border border-line bg-panel text-left transition hover:border-leaf hover:shadow-md"
              >
                <div className="relative aspect-[4/3] w-full overflow-hidden bg-zinc-100">
                  <AuthImage
                    path={`/api/predictions/${p.id}/image`}
                    alt=""
                    className="h-full w-full object-cover transition-transform duration-300 group-hover:scale-105"
                  />
                  <div className="absolute top-2 right-2">
                    {p.is_corn_leaf ? (
                      <span className="rounded-md bg-emerald-800/90 px-2 py-0.5 text-[11px] font-semibold text-white backdrop-blur-xs">
                        Maize
                      </span>
                    ) : (
                      <span className="rounded-md bg-amber-800/90 px-2 py-0.5 text-[11px] font-semibold text-white backdrop-blur-xs">
                        Non-Maize
                      </span>
                    )}
                  </div>
                </div>
                <div className="p-4">
                  <p className="flex items-center gap-2 font-display text-base font-semibold text-ink">
                    <span
                      className="inline-block h-3 w-3 shrink-0 rounded-full"
                      style={{ background: color }}
                      aria-hidden="true"
                    />
                    <span className="truncate">{diseaseTitle}</span>
                  </p>
                  <p className="mt-1.5 flex items-center justify-between text-xs text-muted tabular-nums">
                    <span>
                      {p.is_corn_leaf
                        ? `${(p.confidence * 100).toFixed(1)}% confidence`
                        : "Out-of-domain"}
                    </span>
                    <span>{new Date(p.created_at).toLocaleDateString()}</span>
                  </p>
                  <p className="mt-1 truncate text-xs text-muted font-mono">{p.filename}</p>
                </div>
              </button>
            </li>
          );
        })}
      </ul>
    </div>
  );
}

export default function Page() {
  return (
    <Protected>
      <History />
    </Protected>
  );
}
