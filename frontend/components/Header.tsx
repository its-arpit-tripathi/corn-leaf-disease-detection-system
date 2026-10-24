"use client";
import Link from "next/link";
import { usePathname, useRouter } from "next/navigation";
import { useAuth } from "@/lib/auth";

const links = [["/detect", "Check a leaf"], ["/history", "History"], ["/dashboard", "Dashboard"]];

export function Logo() {
  return (
    <span className="flex items-center gap-2 font-display text-lg font-semibold tracking-tight text-leaf">
      <svg width="26" height="26" viewBox="0 0 32 32" aria-hidden="true">
        <path d="M4 28C4 12 14 4 28 4c0 14-8 24-24 24Z" fill="currentColor" />
        <path d="M6 26 22 10" stroke="var(--color-table)" strokeWidth="2" strokeLinecap="round" />
      </svg>
      Leafcheck
    </span>
  );
}

export default function Header() {
  const { user, logout } = useAuth();
  const path = usePathname();
  const router = useRouter();
  return (
    <header className="border-b border-line bg-panel">
      <div className="mx-auto flex max-w-6xl flex-wrap items-center justify-between gap-x-6 gap-y-2 px-4 py-3">
        <Link href={user ? "/detect" : "/"} aria-label="Leafcheck home"><Logo /></Link>
        <nav aria-label="Main" className="flex flex-wrap items-center gap-1 text-sm">
          {user ? (
            <>
              {links.map(([href, label]) => (
                <Link key={href} href={href} aria-current={path === href ? "page" : undefined}
                  className={`rounded-md px-3 py-1.5 font-medium ${path === href ? "bg-leaf text-white" : "text-ink hover:bg-leaf-soft"}`}>
                  {label}
                </Link>
              ))}
              <span className="ml-3 hidden text-muted sm:inline">{user.name}</span>
              <button onClick={() => { logout(); router.push("/"); }}
                className="ml-1 rounded-md px-3 py-1.5 font-medium text-muted hover:bg-leaf-soft hover:text-ink">Sign out</button>
            </>
          ) : (
            <>
              <Link href="/login" className="rounded-md px-3 py-1.5 font-medium hover:bg-leaf-soft">Sign in</Link>
              <Link href="/register" className="rounded-md bg-leaf px-3 py-1.5 font-medium text-white hover:bg-[#173d29]">Create account</Link>
            </>
          )}
        </nav>
      </div>
    </header>
  );
}
