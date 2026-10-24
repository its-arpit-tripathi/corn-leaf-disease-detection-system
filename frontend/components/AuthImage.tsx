"use client";
import { useEffect, useState } from "react";
import { fetchImageBlob } from "@/lib/api";

/** Loads an image that needs the Authorization header, via a blob URL. */
function Loader({ path, alt, className }: { path: string; alt: string; className?: string }) {
  const [src, setSrc] = useState<string | null>(null);
  const [failed, setFailed] = useState(false);
  useEffect(() => {
    let url: string | null = null, cancelled = false;
    fetchImageBlob(path).then((u) => { if (cancelled) URL.revokeObjectURL(u); else { url = u; setSrc(u); } })
      .catch(() => !cancelled && setFailed(true));
    return () => { cancelled = true; if (url) URL.revokeObjectURL(url); };
  }, [path]);
  if (failed) return <div className={`${className} flex items-center justify-center bg-leaf-soft text-sm text-muted`}>Image unavailable</div>;
  if (!src) return <div className={`${className} animate-pulse bg-leaf-soft`} aria-hidden="true" />;
  // eslint-disable-next-line @next/next/no-img-element
  return <img src={src} alt={alt} className={className} />;
}

export default function AuthImage(props: { path: string; alt: string; className?: string }) {
  return <Loader key={props.path} {...props} />;
}
