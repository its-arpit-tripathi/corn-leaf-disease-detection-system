export const API_URL = process.env.NEXT_PUBLIC_API_URL ?? "http://localhost:8000";
const TOKEN_KEY = "corn_leaf_token";

export type User = { id: number; name: string; email: string; created_at: string };

export type DiseaseInfo = {
  title: string;
  disease_name?: string;
  crop_name?: string;
  pathogen?: string;
  severity?: string;
  status?: string;
  summary: string;
  symptoms: string[];
  advice: string[];
};

export type Prediction = {
  id: number;
  filename: string;
  predicted_class: ClassName;
  disease_name?: string;
  crop_name?: string;
  status?: string;
  pathogen?: string;
  severity?: string;
  label: string;
  confidence: number;
  probabilities: Record<string, number>;
  is_corn_leaf: boolean;
  is_maize?: boolean;
  low_confidence: boolean;
  rejection_reason?: string | null;
  suspected_disease?: string | null;
  info: DiseaseInfo;
  created_at: string;
};

export type ClassName =
  | "Blight"
  | "Common_Rust"
  | "Gray_Leaf_Spot"
  | "Healthy"
  | "Not_Leaf"
  | "Not_Corn"
  | string;

export type Stats = {
  total: number;
  healthy: number;
  diseased: number;
  not_leaf: number;
  by_class: Record<string, number>;
  average_confidence: number;
  last_7_days: { date: string; count: number }[];
};

export const CLASS_LABEL: Record<string, string> = {
  Blight: "Northern Leaf Blight",
  Common_Rust: "Common Rust",
  Gray_Leaf_Spot: "Gray Leaf Spot",
  Healthy: "Healthy Maize",
  Not_Leaf: "Not a Maize Leaf",
  Not_Corn: "Not a Maize Leaf",
};

export const CLASS_COLOR: Record<string, string> = {
  Blight: "var(--color-c-blight)",
  Common_Rust: "var(--color-c-rust)",
  Gray_Leaf_Spot: "var(--color-c-spot)",
  Healthy: "var(--color-c-healthy)",
  Not_Leaf: "var(--color-muted)",
  Not_Corn: "var(--color-muted)",
};

export function getClassLabel(name: string): string {
  if (CLASS_LABEL[name]) return CLASS_LABEL[name];
  const cleaned = name.replace(/^Corn_(\(maize\)___)?/i, "").replace(/_/g, " ");
  return cleaned.charAt(0).toUpperCase() + cleaned.slice(1);
}

export function getClassColor(name: string): string {
  return CLASS_COLOR[name] ?? "var(--color-muted)";
}

export const tokenStore = {
  get: () => {
    try {
      return localStorage.getItem(TOKEN_KEY);
    } catch {
      return null;
    }
  },
  set: (t: string) => {
    try {
      localStorage.setItem(TOKEN_KEY, t);
    } catch {}
  },
  clear: () => {
    try {
      localStorage.removeItem(TOKEN_KEY);
    } catch {}
  },
};

export class ApiError extends Error {
  constructor(public status: number, message: string) {
    super(message);
  }
}

function errorMessage(status: number, body: unknown): string {
  const d = (body as { detail?: unknown })?.detail;
  if (typeof d === "string") return d;
  if (Array.isArray(d) && d[0]?.msg) return String(d[0].msg).replace(/^Value error, /, "");
  return `Request failed (${status})`;
}

export async function api<T>(path: string, init: RequestInit = {}, auth = true): Promise<T> {
  const headers = new Headers(init.headers);
  const token = tokenStore.get();
  if (auth && token) headers.set("Authorization", `Bearer ${token}`);
  let res: Response;
  try {
    res = await fetch(`${API_URL}${path}`, { ...init, headers });
  } catch {
    throw new ApiError(0, "Can't reach the server. Check that the backend is running and try again.");
  }
  if (!res.ok) {
    let body: unknown = null;
    try {
      body = await res.json();
    } catch {}
    throw new ApiError(res.status, errorMessage(res.status, body));
  }
  return res.status === 204 ? (undefined as T) : res.json();
}

export async function fetchImageBlob(path: string): Promise<string> {
  const token = tokenStore.get();
  const res = await fetch(`${API_URL}${path}`, {
    headers: token ? { Authorization: `Bearer ${token}` } : {},
  });
  if (!res.ok) throw new ApiError(res.status, "Image unavailable");
  return URL.createObjectURL(await res.blob());
}
