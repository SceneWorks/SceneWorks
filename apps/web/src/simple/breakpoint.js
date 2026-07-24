// Simple UI responsive breakpoints (design handoff "Simple UI for creative studios").
//
// The Simple shell is ONE responsive layout driven by the width of its own mount —
// not `window` — so it stays correct when embedded (the design showcase mounts the
// same app inside desktop / tablet / phone frames). The three bands and the content
// caps below are the design's; every consumer reads them from here so the layout, the
// picker chrome (bottom sheet vs centered modal) and the phone-forces-Simple rule in
// App.jsx can never drift apart.

// Below this the shell is a phone: hamburger + overlay drawer + bottom sheets.
export const PHONE_MAX_WIDTH = 680;
// At or above this the shell gets the roomier desktop content cap.
export const DESKTOP_MIN_WIDTH = 1024;

// Content-column caps. Phone content is full-bleed (no cap).
export const TABLET_CONTENT_MAX = 680;
export const DESKTOP_CONTENT_MAX = 900;

/**
 * The layout band for a measured container width.
 * An unknown (null/0) width resolves to "desktop" — the shell renders with a
 * persistent sidebar until the first ResizeObserver callback lands, which is the
 * non-jarring default for the overwhelmingly common case (a desktop window).
 * @param {number|null|undefined} width
 * @returns {"phone"|"tablet"|"desktop"}
 */
export function breakpointFor(width) {
  if (!Number.isFinite(width) || width <= 0) {
    return "desktop";
  }
  if (width >= DESKTOP_MIN_WIDTH) {
    return "desktop";
  }
  return width >= PHONE_MAX_WIDTH ? "tablet" : "phone";
}

/**
 * True when a measured width is in the phone band. Shared by the shell layout and
 * by App.jsx's "phones are always served the Simple UI" rule, so one number governs
 * both. An unknown width is NOT a phone (never force Simple on a missing signal).
 * @param {number|null|undefined} width
 */
export function isPhoneWidth(width) {
  return Number.isFinite(width) && width > 0 && width < PHONE_MAX_WIDTH;
}

/**
 * The `max-width` for the centered content column at a breakpoint ("none" on phone).
 * @param {"phone"|"tablet"|"desktop"} breakpoint
 */
export function contentMaxWidth(breakpoint) {
  if (breakpoint === "desktop") {
    return `${DESKTOP_CONTENT_MAX}px`;
  }
  if (breakpoint === "tablet") {
    return `${TABLET_CONTENT_MAX}px`;
  }
  return "none";
}

/**
 * Asset-grid column count per band (design: 3 phone / 4 tablet / 6 desktop).
 * @param {"phone"|"tablet"|"desktop"} breakpoint
 */
export function assetGridColumns(breakpoint) {
  if (breakpoint === "desktop") {
    return 6;
  }
  return breakpoint === "tablet" ? 4 : 3;
}

/**
 * Result-grid column count for an image run: 1 column for a single variation,
 * 3 columns for a 6-up run on desktop, otherwise 2 (design spec).
 * @param {number} count
 * @param {"phone"|"tablet"|"desktop"} breakpoint
 */
export function resultGridColumns(count, breakpoint) {
  if (count <= 1) {
    return 1;
  }
  return breakpoint === "desktop" && count >= 6 ? 3 : 2;
}
