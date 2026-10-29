"use client";
import { useState } from "react";
import {
  getClassColor,
  getClassLabel,
  type ClassName,
  type Prediction,
} from "@/lib/api";
import AuthImage from "./AuthImage";

export function Probabilities({ p }: { p: Prediction }) {
  const isCorn = Boolean(
    p.is_corn_leaf &&
    p.predicted_class !== "Not_Leaf" &&
    p.predicted_class !== "Not_Corn" &&
    p.predicted_class !== "Non_Maize" &&
    p.predicted_class !== "Other"
  );
  if (!isCorn) return null;

  const rows = (Object.entries(p.probabilities) as [string, number][]).sort(
    (a, b) => b[1] - a[1]
  );
  return (
    <div className="rounded-xl border border-line bg-panel p-4">
      <h3 className="mb-3 text-sm font-semibold uppercase tracking-wider text-muted">
        Class Probability Distribution
      </h3>
      <ul className="space-y-2.5" aria-label="Probability for each class">
        {rows.map(([name, v]) => {
          const isTop = name === p.predicted_class;
          const label = getClassLabel(name);
          const color = getClassColor(name);
          return (
            <li
              key={name}
              className="grid grid-cols-[10rem_1fr_4rem] items-center gap-3 text-sm"
            >
              <span
                className={`truncate ${
                  isTop ? "font-semibold text-ink" : "text-muted"
                }`}
                title={label}
              >
                {label}
              </span>
              <span className="h-2.5 overflow-hidden rounded-full bg-leaf-soft">
                <span
                  className="block h-full rounded-full transition-all duration-500"
                  style={{
                    width: `${Math.max(v * 100, 0.5)}%`,
                    background: color,
                  }}
                />
              </span>
              <span
                className={`text-right tabular-nums ${
                  isTop ? "font-semibold text-ink" : "text-muted"
                }`}
              >
                {(v * 100).toFixed(1)}%
              </span>
            </li>
          );
        })}
      </ul>
    </div>
  );
}

export default function ResultView({
  p,
  originalPreviewUrl,
}: {
  p: Prediction;
  originalPreviewUrl?: string | null;
}) {
  const isCorn = Boolean(
    p.is_corn_leaf &&
    p.predicted_class !== "Not_Leaf" &&
    p.predicted_class !== "Not_Corn" &&
    p.predicted_class !== "Non_Maize" &&
    p.predicted_class !== "Other"
  );

  const [view, setView] = useState<"heatmap" | "original">(isCorn ? "heatmap" : "original");

  const isHealthy = p.predicted_class === "Healthy";
  const displayColor = getClassColor(p.predicted_class);

  const diseaseName =
    p.disease_name ||
    p.info?.disease_name ||
    (isCorn
      ? isHealthy
        ? "Healthy Maize Leaf"
        : getClassLabel(p.predicted_class)
      : "Not a Maize Leaf");

  const cropName = p.crop_name || p.info?.crop_name || (isCorn ? "Maize (Corn)" : "Non-Maize / Unrecognized");
  const pathogen = p.pathogen || p.info?.pathogen;
  const severity = p.severity || p.info?.severity;

  const showConfidenceBar = isCorn || (p.predicted_class === "Not_Leaf" && p.confidence >= 0.5);

  return (
    <div className="grid gap-8 lg:grid-cols-[minmax(0,26rem)_1fr]">
      {/* Left Column: Image Viewer with Grad-CAM and Heatmap Toggle */}
      <div>
        <div className="overflow-hidden rounded-xl border border-line bg-panel shadow-sm">
          {view === "original" && originalPreviewUrl ? (
            // eslint-disable-next-line @next/next/no-img-element
            <img
              src={originalPreviewUrl}
              alt="Uploaded photo"
              className="aspect-square w-full object-cover"
            />
          ) : (
            <AuthImage
              key={`${p.id}-${view}-${p.created_at ? new Date(p.created_at).getTime() : p.id}`}
              path={`/api/predictions/${p.id}/${view === "heatmap" ? "heatmap" : "image"}?v=${p.created_at ? new Date(p.created_at).getTime() : p.id}`}
              alt={
                view === "heatmap"
                  ? "Leaf with Grad-CAM heat map highlighting active regions"
                  : "Uploaded photo"
              }
              className="aspect-square w-full object-cover"
            />
          )}
        </div>

        <div
          role="group"
          aria-label="Image view"
          className="mt-3 inline-flex rounded-lg border border-line bg-panel p-1 text-sm shadow-xs"
        >
          {(["heatmap", "original"] as const).map((v) => (
            <button
              key={v}
              onClick={() => setView(v)}
              aria-pressed={view === v}
              className={`rounded-md px-3 py-1.5 font-medium transition-colors ${
                view === v
                  ? "bg-leaf text-white shadow-xs"
                  : "text-ink hover:bg-leaf-soft"
              }`}
            >
              {v === "heatmap" ? "🔥 Grad-CAM Heatmap" : "📷 Original Photo"}
            </button>
          ))}
        </div>

        <p className="mt-3 max-w-prose text-xs text-muted">
          Grad-CAM highlights regions that influenced the model&apos;s decision.
          Warmer zones (red/yellow) indicate salient diagnostic patterns.
        </p>
      </div>

      {/* Right Column: Detailed Diagnosis & Disease Info */}
      <div className="space-y-6">
        {/* Verification and Status Badges */}
        <div className="flex flex-wrap items-center gap-2">
          {isCorn ? (
            <span className="inline-flex items-center gap-1.5 rounded-full bg-emerald-50 px-3 py-1 text-xs font-semibold text-emerald-800 ring-1 ring-emerald-600/20">
              <span className="h-2 w-2 rounded-full bg-emerald-600" />
              Crop: {cropName}
            </span>
          ) : (
            <span className="inline-flex items-center gap-1.5 rounded-full bg-amber-50 px-3 py-1 text-xs font-semibold text-amber-800 ring-1 ring-amber-600/30">
              <span className="h-2 w-2 rounded-full bg-amber-500" />
              Crop: {cropName}
            </span>
          )}

          {isCorn && !isHealthy && (
            <span className="inline-flex items-center gap-1.5 rounded-full bg-rose-50 px-3 py-1 text-xs font-semibold text-rose-800 ring-1 ring-rose-600/20">
              ⚠️ Foliage Disease Detected
            </span>
          )}

          {isCorn && isHealthy && (
            <span className="inline-flex items-center gap-1.5 rounded-full bg-emerald-50 px-3 py-1 text-xs font-semibold text-emerald-800 ring-1 ring-emerald-600/20">
              ✓ No Disease Detected
            </span>
          )}

          {!isCorn && (
            <span className="inline-flex items-center gap-1.5 rounded-full bg-zinc-100 px-3 py-1 text-xs font-semibold text-zinc-700 ring-1 ring-zinc-500/20">
              Non-Maize Sample
            </span>
          )}

          {isCorn && severity && severity !== "None" && severity !== "N/A" && (
            <span className="inline-flex items-center rounded-full bg-orange-50 px-2.5 py-1 text-xs font-medium text-orange-800 ring-1 ring-orange-500/20">
              Severity: {severity}
            </span>
          )}
        </div>

        {/* Primary Disease Headline */}
        <div>
          <p className="text-xs font-semibold uppercase tracking-wider text-muted">
            {isCorn ? (isHealthy ? "Health Status" : "Disease Diagnosis") : "Classification Status"}
          </p>
          <h2
            className="mt-1 text-3xl font-bold tracking-tight sm:text-4xl"
            style={{ color: isCorn ? displayColor : "var(--color-ink)" }}
          >
            {diseaseName}
          </h2>

          {isCorn && pathogen && pathogen !== "None" && pathogen !== "None detected" && pathogen !== "N/A" && (
            <p className="mt-1 text-sm italic text-muted">
              Causative Pathogen: <span className="font-medium text-ink">{pathogen}</span>
            </p>
          )}

          {showConfidenceBar && (
            <div className="mt-3 flex items-center gap-3">
              <div className="h-2.5 w-32 overflow-hidden rounded-full bg-leaf-soft">
                <div
                  className="h-full rounded-full bg-leaf transition-all duration-500"
                  style={{ width: `${Math.min(100, p.confidence * 100)}%` }}
                />
              </div>
              <span className="text-sm font-semibold text-ink tabular-nums">
                {(p.confidence * 100).toFixed(1)}% Confidence
              </span>
            </div>
          )}
        </div>

        {/* Warnings & Alerts */}
        {!isCorn && (
          <div
            role="alert"
            className="rounded-xl border border-amber/40 bg-amber-50/70 p-4 text-sm text-amber-900"
          >
            <p className="font-semibold">⚠️ Photo Not Identified as Maize Foliage</p>
            <p className="mt-1 text-amber-800">
              This detection model is specialized strictly for maize (corn) leaf diseases.
              Photos of weeds, other crops (sugarcane, tomato, potato, etc.), or non-plant objects are
              flagged to prevent false diagnoses.
            </p>
          </div>
        )}

        {p.low_confidence && isCorn && (
          <div
            role="alert"
            className="rounded-xl border border-amber/40 bg-[#fbf1dc] p-4 text-sm text-ink"
          >
            <strong>The model is uncertain about this photo.</strong> Ensure the leaf is in focus, well-lit by daylight, and photographed close-up without heavy shadows or fingers.
          </div>
        )}

        {/* Probability Breakdown - only shown for verified maize leaves */}
        {isCorn && <Probabilities p={p} />}

        {/* Detailed Agronomic Insights */}
        {p.info && (
          <section
            aria-labelledby="about-result"
            className="rounded-xl border border-line bg-panel p-5 space-y-4 shadow-xs"
          >
            <h3 id="about-result" className="text-lg font-semibold text-ink">
              {isCorn ? "About This Diagnosis" : "Guidance for Maize Photos"}
            </h3>
            <p className="text-sm leading-relaxed text-ink/90">{p.info.summary}</p>

            {isCorn && p.info.symptoms && p.info.symptoms.length > 0 && (
              <div>
                <h4 className="text-sm font-semibold text-ink">Field Symptoms to Verify</h4>
                <ul className="mt-1.5 list-disc space-y-1 pl-5 text-sm text-muted">
                  {p.info.symptoms.map((s) => (
                    <li key={s}>{s}</li>
                  ))}
                </ul>
              </div>
            )}

            {p.info.advice && p.info.advice.length > 0 && (
              <div>
                <h4 className="text-sm font-semibold text-ink">
                  {isCorn ? "Recommended Actions" : "Image Guidelines"}
                </h4>
                <ul className="mt-1.5 list-disc space-y-1 pl-5 text-sm text-muted">
                  {p.info.advice.map((a) => (
                    <li key={a}>{a}</li>
                  ))}
                </ul>
              </div>
            )}
          </section>
        )}

        <p className="text-xs text-muted">
          Automated decision support tool for field scouting. Always confirm severe outbreaks with your local agronomy extension service before applying chemical interventions.
        </p>
      </div>
    </div>
  );
}
