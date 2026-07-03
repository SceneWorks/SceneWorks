// Shared URL hardening for manifest-/data-supplied external links (sc-8881).
//
// Model manifests, the bundled-license manifest, and per-model prompt-guide
// metadata all carry free-form URL fields (`licenseUrl`, `homepage`,
// `ui.promptGuide.sources[].url`, …) that we render straight into `<a href>`.
// Today those manifests are first-party and import is disabled (epic 7080), but
// a future user-imported manifest turns a `javascript:` value into a click-time
// script-execution vector. `safeExternalUrl` gates every such render site: it
// returns the URL only when it parses as an absolute `http:`/`https:` URL, and
// otherwise `undefined` so callers render the link inert or omit it entirely.
export function safeExternalUrl(url) {
  if (typeof url !== "string") return undefined;
  const trimmed = url.trim();
  if (!trimmed) return undefined;
  let parsed;
  try {
    parsed = new URL(trimmed);
  } catch {
    // Not an absolute URL (relative, malformed, or scheme-less) — reject rather
    // than guess a base, since manifest links are meant to be absolute externals.
    return undefined;
  }
  if (parsed.protocol !== "http:" && parsed.protocol !== "https:") {
    return undefined;
  }
  return trimmed;
}
