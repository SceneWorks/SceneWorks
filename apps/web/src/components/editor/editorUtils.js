// Shared helpers for the redesigned Video Editor (design 2a, epic 12798).
// Pure functions + constants used across the editor components — kept out of the
// screen so the timeline/storyboard/rail can share geometry and provenance logic.

// Timeline geometry: the lane content width is `duration * BASE_PX_PER_SEC * zoom`.
// Everything inside is positioned by percentage of duration, so zoom only changes the
// content width (nothing else recomputes). Zoom multiplier ranges ~0.6x–2.6x.
export const BASE_PX_PER_SEC = 24;
export const ZOOM_MIN = 0.6;
export const ZOOM_MAX = 2.6;
export const ZOOM_STEP = 0.2;

// Track ids the backend ships on a default timeline (project_store.rs
// default_timeline_tracks): main video, overlay, and one audio track.
export const MAIN_TRACK_ID = "track_main";
export const OVERLAY_TRACK_ID = "track_overlay";
export const AUDIO_TRACK_ID = "track_audio";

// Camera-motion presets mirrored from Video Studio (the real `motion` control is a
// preset string, not a numeric strength). Kept in sync with VideoStudio's MOTIONS.
export const MOTIONS = [
  "static",
  "slow push-in",
  "pull out",
  "pan left",
  "pan right",
  "tilt up",
  "tilt down",
  "handheld",
];
export const DEFAULT_MOTION = "slow push-in";

// Asset origins stamped by the backend when media is generated in a studio. An asset
// with one of these origins — or (for legacy records with no origin) a generation
// recipe / batch id — is treated as AI-generated for the violet sparkle affordance.
const STUDIO_ORIGINS = new Set(["image_studio", "video_studio", "document_studio", "character_studio"]);

export function isAiAsset(asset) {
  if (!asset) {
    return false;
  }
  if (asset.origin === "upload" || asset.type === "upload") {
    return false;
  }
  if (asset.origin && STUDIO_ORIGINS.has(asset.origin)) {
    return true;
  }
  return Boolean(asset.recipe || asset.generationSetId);
}

// A timeline item counts as an "AI clip" when its backing asset was generated, or when
// its version history records a generative source (extension/bridge/replacement).
export function isAiItem(item, assetsById) {
  if (!item) {
    return false;
  }
  const asset = assetsById?.get?.(item.assetId) ?? null;
  if (isAiAsset(asset)) {
    return true;
  }
  const history = Array.isArray(item.versionHistory) ? item.versionHistory : [];
  return history.some((entry) => ["extension", "bridge", "replacement"].includes(entry?.source));
}

// A small deterministic hash → 0..1 PRNG seeded by a string, so a clip's waveform /
// gradient is stable across renders without storing anything.
function seededRandom(seed) {
  let h = 2166136261 >>> 0;
  const text = String(seed ?? "");
  for (let i = 0; i < text.length; i += 1) {
    h ^= text.charCodeAt(i);
    h = Math.imul(h, 16777619) >>> 0;
  }
  return () => {
    h += 0x6d2b79f5;
    let t = h;
    t = Math.imul(t ^ (t >>> 15), t | 1);
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

// A per-clip hue derived from its id, so each clip block gets a distinct gradient fill
// (design: "gradient fill derived from a per-clip hue"). Kept in a pleasant band.
export function clipHue(seed) {
  const rand = seededRandom(seed);
  return Math.round(rand() * 360);
}

// A mirror-filled waveform silhouette as a single SVG path over a 0 0 100 40 viewBox
// (design: "waveform drawn as a single SVG <path>, preserveAspectRatio=none"). This is
// a SYNTHETIC envelope seeded by the clip id — no real audio analysis exists yet
// (tracked as a backend-gap audit story). `bars` controls the resolution.
export function waveformPath(seed, bars = 48) {
  const rand = seededRandom(seed);
  const step = 100 / bars;
  const top = [];
  const bottom = [];
  for (let i = 0; i <= bars; i += 1) {
    const x = +(i * step).toFixed(2);
    // Envelope: a slow swell times a faster jitter, floored so it never flatlines.
    const swell = 0.35 + 0.5 * Math.abs(Math.sin((i / bars) * Math.PI * 3 + rand() * 2));
    const jitter = 0.55 + 0.45 * rand();
    const amp = Math.min(1, swell * jitter);
    const half = +(amp * 18).toFixed(2);
    top.push(`${x},${(20 - half).toFixed(2)}`);
    bottom.push(`${x},${(20 + half).toFixed(2)}`);
  }
  bottom.reverse();
  return `M ${top.join(" L ")} L ${bottom.join(" L ")} Z`;
}

// Second-granularity ruler ticks across `duration`. Minor tick every second, major
// tick (with an m:ss label) every 5s. Positions are percentages of duration so the
// ruler tracks the same %-geometry as the lanes.
export function buildTicks(duration) {
  const ticks = [];
  const total = Math.max(1, Math.ceil(duration));
  for (let s = 0; s <= total; s += 1) {
    const major = s % 5 === 0;
    ticks.push({
      second: s,
      major,
      leftPct: duration > 0 ? Math.min(100, (s / duration) * 100) : 0,
      label: major ? `${Math.floor(s / 60)}:${String(s % 60).padStart(2, "0")}` : "",
    });
  }
  return ticks;
}

// Percentage geometry for a clip block given the timeline duration.
export function itemGeometry(item, duration) {
  const start = Number(item.timelineStart) || 0;
  const end = Math.max(start, Number(item.timelineEnd) || start);
  const span = Math.max(0.0001, duration);
  return {
    leftPct: Math.max(0, (start / span) * 100),
    widthPct: Math.max(0.5, ((end - start) / span) * 100),
  };
}
