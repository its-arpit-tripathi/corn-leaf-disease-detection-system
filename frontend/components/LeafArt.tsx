/** Original illustration: a corn leaf with lesions and a Grad-CAM style heat blob. */
export default function LeafArt({ className }: { className?: string }) {
  return (
    <svg viewBox="0 0 520 420" className={className} role="img"
      aria-label="Illustration of a corn leaf with a heat map highlighting a lesion">
      <defs>
        <radialGradient id="heat" cx="50%" cy="50%" r="50%">
          <stop offset="0%" stopColor="#e11d2e" stopOpacity=".85" />
          <stop offset="35%" stopColor="#f59e0b" stopOpacity=".6" />
          <stop offset="70%" stopColor="#facc15" stopOpacity=".25" />
          <stop offset="100%" stopColor="#facc15" stopOpacity="0" />
        </radialGradient>
        <linearGradient id="blade" x1="0" y1="1" x2="1" y2="0">
          <stop offset="0" stopColor="#2f7d4f" /><stop offset="1" stopColor="#5fa36f" />
        </linearGradient>
        <clipPath id="bladeClip"><path d="M20 360C70 230 190 90 500 40c-40 150-170 290-480 320Z" /></clipPath>
      </defs>
      <rect width="520" height="420" rx="14" fill="#dfe8e1" />
      <path d="M20 360C70 230 190 90 500 40c-40 150-170 290-480 320Z" fill="url(#blade)" />
      <g clipPath="url(#bladeClip)">
        <path d="M20 360C150 250 300 150 500 40" stroke="#d6e8d0" strokeWidth="5" fill="none" opacity=".8" />
        {[0, 1, 2, 3, 4, 5, 6].map((i) => (
          <path key={i} d={`M${60 + i * 60} ${330 - i * 36}C${120 + i * 60} ${300 - i * 40} ${180 + i * 60} ${260 - i * 40} ${260 + i * 60} ${230 - i * 42}`}
            stroke="#1e4d34" strokeWidth="1.2" fill="none" opacity=".35" />
        ))}
        <g fill="#b98a4a" opacity=".92">
          <ellipse cx="300" cy="190" rx="34" ry="9" transform="rotate(-30 300 190)" />
          <ellipse cx="338" cy="168" rx="22" ry="6" transform="rotate(-30 338 168)" />
          <ellipse cx="262" cy="222" rx="18" ry="5" transform="rotate(-30 262 222)" />
        </g>
        <circle className="heat-in" cx="305" cy="188" r="82" fill="url(#heat)" />
      </g>
      <rect x="0" y="0" width="520" height="3" fill="#fff" opacity=".7" className="scanline" />
    </svg>
  );
}
