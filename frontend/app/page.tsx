"use client";
import Link from "next/link";
import LeafArt from "@/components/LeafArt";
import { useAuth } from "@/lib/auth";

const diseases = [
  ["Northern leaf blight", "Long, cigar-shaped gray-green or tan lesions.", "var(--color-c-blight)"],
  ["Common rust", "Small raised reddish-brown pustules on both sides of the leaf.", "var(--color-c-rust)"],
  ["Gray leaf spot", "Narrow rectangular tan lesions that follow the veins.", "var(--color-c-spot)"],
  ["Healthy", "No sign of the three diseases above.", "var(--color-c-healthy)"],
];

export default function Home() {
  const { user } = useAuth();
  return (
    <>
      <section className="grid items-center gap-10 py-4 md:grid-cols-[1.05fr_1fr]">
        <div>
          <h1 className="text-5xl font-semibold leading-[1.05] text-leaf sm:text-6xl">
            Photograph a corn leaf. See what&apos;s wrong, and why.
          </h1>
          <p className="mt-5 max-w-[34rem] text-lg text-muted">
            Upload one photo and get the likely disease, how sure the model is, and a heat map showing which part of the leaf
            led to that answer. You can check the reasoning instead of taking it on trust.
          </p>
          <div className="mt-8 flex flex-wrap gap-3">
            <Link href={user ? "/detect" : "/register"} className="rounded-lg bg-leaf px-5 py-3 font-semibold text-white hover:bg-[#173d29]">
              {user ? "Check a leaf" : "Create a free account"}
            </Link>
            {!user && <Link href="/login" className="rounded-lg border border-leaf px-5 py-3 font-semibold text-leaf hover:bg-leaf-soft">Sign in</Link>}
          </div>
        </div>
        <LeafArt className="w-full rounded-2xl shadow-[0_1px_0_var(--color-line)]" />
      </section>

      <section className="mt-16" aria-labelledby="detects">
        <h2 id="detects" className="text-2xl font-semibold">What it can tell apart</h2>
        <dl className="mt-5 grid gap-px overflow-hidden rounded-xl border border-line bg-line sm:grid-cols-2">
          {diseases.map(([name, text, c]) => (
            <div key={name} className="bg-panel p-5">
              <dt className="flex items-center gap-2 font-display text-lg font-semibold">
                <span className="inline-block h-3 w-3 rounded-full" style={{ background: c }} aria-hidden="true" />{name}
              </dt>
              <dd className="mt-1 text-muted">{text}</dd>
            </div>
          ))}
        </dl>
        <p className="mt-4 max-w-prose text-sm text-muted">
          It is trained on corn leaves only. Check the heat map to see whether the model looked at the leaf or at the background.
        </p>
      </section>
    </>
  );
}
