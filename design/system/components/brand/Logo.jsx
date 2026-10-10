import React, { useId } from "react";

/**
 * SceneWorks mark — "scene cut". A rounded square split by a diagonal seam: a
 * teal triangle over a solid ground. Colors come from CSS variables
 * (--logo-ground / --logo-seam / --teal) so it tracks the active theme and the
 * user-selectable accent. Verbatim from @sceneworks/ui.
 */
export function Logo({ size = 32, title = "SceneWorks", className }) {
  const clipId = useId();
  return (
    <svg width={size} height={size} viewBox="0 0 100 100" role="img" aria-label={title} className={className}>
      <title>{title}</title>
      <defs>
        <clipPath id={clipId}>
          <rect x="10" y="10" width="80" height="80" rx="14" />
        </clipPath>
      </defs>
      <g clipPath={`url(#${clipId})`}>
        <rect x="10" y="10" width="80" height="80" fill="var(--logo-ground)" />
        <polygon points="10,10 90,10 90,90" fill="var(--teal)" />
        <line x1="10" y1="90" x2="90" y2="10" stroke="var(--logo-seam)" strokeWidth="2.5" strokeLinecap="square" />
      </g>
    </svg>
  );
}

/**
 * The full lockup: scene-cut mark + "SceneWorks" wordmark (Works in --mist).
 * Matches the .brand block in the sidebar.
 */
export function Wordmark({ size = 32 }) {
  return (
    <span className="brand">
      <span className="brand-mark"><Logo size={size} /></span>
      <h1>Scene<span className="light">Works</span></h1>
    </span>
  );
}

export default Logo;
