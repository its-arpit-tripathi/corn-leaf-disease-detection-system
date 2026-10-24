"use client";
import Link from "next/link";
import { useRouter } from "next/navigation";
import { useEffect, useState } from "react";
import { useAuth } from "@/lib/auth";

export default function AuthForm({ mode }: { mode: "login" | "register" }) {
  const { user, login, register } = useAuth();
  const router = useRouter();
  const [name, setName] = useState(""), [email, setEmail] = useState(""), [password, setPassword] = useState("");
  const [error, setError] = useState<string | null>(null), [busy, setBusy] = useState(false);
  const isLogin = mode === "login";

  useEffect(() => { if (user) router.replace("/detect"); }, [user, router]);

  async function submit(e: React.FormEvent) {
    e.preventDefault(); setError(null); setBusy(true);
    try {
      if (isLogin) await login(email, password); else await register(name, email, password);
      router.push("/detect");
    }
    catch (err) { setError(err instanceof Error ? err.message : "Something went wrong"); }
    finally { setBusy(false); }
  }

  const field = "mt-1 w-full rounded-lg border border-line bg-panel px-3 py-2.5 text-base";
  return (
    <div className="mx-auto max-w-sm">
      <h1 className="text-3xl font-semibold text-leaf">{isLogin ? "Sign in" : "Create your account"}</h1>
      <form onSubmit={submit} className="mt-6 space-y-4">
        {!isLogin && (
          <label className="block text-sm font-medium">Name
            <input className={field} value={name} onChange={(e) => setName(e.target.value)} required maxLength={120} autoComplete="name" />
          </label>
        )}
        <label className="block text-sm font-medium">Email
          <input className={field} type="email" value={email} onChange={(e) => setEmail(e.target.value)} required autoComplete="email" />
        </label>
        <label className="block text-sm font-medium">Password
          <input className={field} type="password" value={password} onChange={(e) => setPassword(e.target.value)} required
            minLength={isLogin ? undefined : 8} autoComplete={isLogin ? "current-password" : "new-password"} aria-describedby={isLogin ? undefined : "pw-hint"} />
        </label>
        {!isLogin && <p id="pw-hint" className="-mt-2 text-sm text-muted">Use at least 8 characters.</p>}
        {error && <p role="alert" className="rounded-lg border border-danger bg-[#fbe9e5] p-3 text-sm text-danger">{error}</p>}
        <button disabled={busy} className="w-full rounded-lg bg-leaf px-4 py-3 font-semibold text-white hover:bg-[#173d29] disabled:opacity-60">
          {busy ? (isLogin ? "Signing in…" : "Creating account…") : isLogin ? "Sign in" : "Create account"}
        </button>
      </form>
      <p className="mt-5 text-sm text-muted">
        {isLogin ? "New here? " : "Already have an account? "}
        <Link href={isLogin ? "/register" : "/login"} className="font-semibold text-leaf underline">{isLogin ? "Create an account" : "Sign in"}</Link>
      </p>
    </div>
  );
}
