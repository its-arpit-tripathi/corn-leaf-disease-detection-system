"use client";
import Link from "next/link";
import { useEffect, useState } from "react";
import Protected from "@/components/Protected";
import { api, getClassColor, getClassLabel, type Stats } from "@/lib/api";

function Dashboard() {
  const [s, setS] = useState<Stats | null>(null);
  const [error, setError] = useState<string | null>(null);
  useEffect(() => { api<Stats>("/api/stats").then(setS).catch((e) => setError(e.message)); }, []);

  if (error) return <p role="alert" className="rounded-lg border border-danger bg-[#fbe9e5] p-3 text-danger">{error}</p>;
  if (!s) return <p className="text-muted" role="status">Loading…</p>;
  if (s.total === 0) return (
    <div><h1 className="text-3xl font-semibold text-leaf">Dashboard</h1>
      <p className="mt-6 text-muted">Your charts appear after your first check. <Link href="/detect" className="font-semibold text-leaf underline">Check a leaf</Link>.</p></div>
  );

  const classes = Object.entries(s.by_class) as [string, number][];
  const maxDay = Math.max(1, ...s.last_7_days.map((d) => d.count));
  return (
    <div className="space-y-10">
      <h1 className="text-3xl font-semibold text-leaf">Dashboard</h1>
      <dl className="grid gap-px overflow-hidden rounded-xl border border-line bg-line sm:grid-cols-4">
        {[["Photos checked", s.total], ["Healthy", s.healthy], ["Diseased", s.diseased], ["Average confidence", `${(s.average_confidence * 100).toFixed(0)}%`]].map(([k, v]) => (
          <div key={k} className="bg-panel p-5"><dt className="text-sm text-muted">{k}</dt><dd className="mt-1 font-display text-3xl font-semibold tabular-nums">{v}</dd></div>
        ))}
      </dl>

      <section aria-labelledby="by-class">
        <h2 id="by-class" className="text-xl font-semibold">Results by class</h2>
        <ul className="mt-4 space-y-3">
          {classes.map(([name, n]) => (
            <li key={name} className="grid grid-cols-[9.5rem_1fr_2.5rem] items-center gap-3 text-sm">
              <span>{getClassLabel(name)}</span>
              <span className="h-3 overflow-hidden rounded-full bg-leaf-soft">
                <span className="block h-full rounded-full" style={{ width: `${(n / s.total) * 100}%`, background: getClassColor(name) }} />
              </span>
              <span className="text-right tabular-nums">{n}</span>
            </li>
          ))}
        </ul>
      </section>

      <section aria-labelledby="week">
        <h2 id="week" className="text-xl font-semibold">Checks in the last 7 days</h2>
        <ol className="mt-4 grid h-48 grid-cols-7 items-end gap-3">
          {s.last_7_days.map((d) => (
            <li key={d.date} className="flex h-full flex-col justify-end text-center text-xs text-muted">
              <span className="tabular-nums text-ink">{d.count}</span>
              <span className="mx-auto mt-1 w-full max-w-10 rounded-t bg-leaf" style={{ height: `${(d.count / maxDay) * 100}%`, minHeight: d.count ? 4 : 2, opacity: d.count ? 1 : 0.25 }} />
              <span className="mt-1">{new Date(d.date + "T00:00").toLocaleDateString(undefined, { weekday: "short" })}</span>
            </li>
          ))}
        </ol>
      </section>
    </div>
  );
}

export default function Page() { return <Protected><Dashboard /></Protected>; }
