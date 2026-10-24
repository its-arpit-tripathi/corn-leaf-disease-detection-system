"use client";
import { createContext, useCallback, useContext, useEffect, useState } from "react";
import { api, tokenStore, type User } from "./api";

type Ctx = {
  user: User | null; loading: boolean;
  login: (email: string, password: string) => Promise<void>;
  register: (name: string, email: string, password: string) => Promise<void>;
  logout: () => void;
};
const AuthContext = createContext<Ctx | null>(null);
type TokenResponse = { access_token: string; user: User };

export function AuthProvider({ children }: { children: React.ReactNode }) {
  const [user, setUser] = useState<User | null>(null);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    const request = tokenStore.get() ? api<User>("/api/users/me") : Promise.resolve(null);
    request.then(setUser).catch(() => tokenStore.clear()).finally(() => setLoading(false));
  }, []);

  const finish = (r: TokenResponse) => { tokenStore.set(r.access_token); setUser(r.user); };
  const login = useCallback(async (email: string, password: string) => {
    const body = new URLSearchParams({ username: email, password });
    finish(await api<TokenResponse>("/api/auth/login", { method: "POST", body }, false));
  }, []);
  const register = useCallback(async (name: string, email: string, password: string) => {
    finish(await api<TokenResponse>("/api/auth/register", {
      method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ name, email, password }),
    }, false));
  }, []);
  const logout = useCallback(() => { tokenStore.clear(); setUser(null); }, []);

  return <AuthContext.Provider value={{ user, loading, login, register, logout }}>{children}</AuthContext.Provider>;
}

export function useAuth() {
  const c = useContext(AuthContext);
  if (!c) throw new Error("useAuth must be used inside AuthProvider");
  return c;
}
