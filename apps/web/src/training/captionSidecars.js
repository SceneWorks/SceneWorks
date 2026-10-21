// Caption sidecars for a dataset import (sc-2025 `.txt`; sc-25685 `.json`). Pure: callers read
// each File's text and hand in `{ name, text }` entries; this pairs them with images by the
// lowercased filename stem.
//
// A `<stem>.json` sidecar is a JSON object of `{ field: text }` — the per-item named captions
// Iris' `caption_field(s)` select from (the item's `captions` map, validated by the Rust store:
// non-empty names without surrounding whitespace, at most 64 characters, string values). The
// `caption` key is the item's ORDINARY caption (iris.rs reads `caption` from `caption.text`), so it
// is not duplicated into the named map: when the stem has no `.txt`, a JSON `caption` (or, failing
// that, `text`) string becomes the caption text instead. A file that is not an object, holds a
// non-string value, or names an invalid field is ignored as a whole and reported.

function baseName(name) {
  return String(name ?? "").replaceAll("\\", "/").split("/").pop() ?? "";
}

export function sidecarExtension(name) {
  const base = baseName(name);
  const dot = base.lastIndexOf(".");
  return dot > 0 ? base.slice(dot + 1).toLowerCase() : "";
}

export function sidecarStem(name) {
  const base = baseName(name);
  const dot = base.lastIndexOf(".");
  return (dot > 0 ? base.slice(0, dot) : base).toLowerCase();
}

export function isJsonCaptionSidecar(file) {
  return sidecarExtension(file?.name) === "json" || file?.type === "application/json";
}

function validFieldName(name) {
  return name.length > 0 && name.trim() === name && name.length <= 64;
}

// `{ ok: true, captions, caption }` or `{ ok: false }`.
export function parseJsonCaptionSidecar(text) {
  let value;
  try {
    value = JSON.parse(String(text ?? ""));
  } catch {
    return { ok: false };
  }
  if (!value || typeof value !== "object" || Array.isArray(value)) {
    return { ok: false };
  }
  const entries = Object.entries(value);
  if (!entries.length || entries.some(([name, field]) => typeof field !== "string" || !validFieldName(name))) {
    return { ok: false };
  }
  const captions = Object.fromEntries(entries.filter(([name]) => name !== "caption"));
  const caption = typeof value.caption === "string" ? value.caption : typeof value.text === "string" ? value.text : null;
  return { ok: true, captions, caption: caption?.trim() ? caption.trim() : null };
}

// Pair `.txt` and `.json` sidecars by stem.
// Returns `{ byStem: Map<stem, { text?, captions? }>, ignored: string[] (file names) }`.
export function collectCaptionSidecars(entries) {
  const byStem = new Map();
  const ignored = [];
  const entry = (stem) => {
    if (!byStem.has(stem)) byStem.set(stem, {});
    return byStem.get(stem);
  };
  const jsonCaptions = new Map();
  for (const { name, text } of entries ?? []) {
    const stem = sidecarStem(name);
    if (sidecarExtension(name) === "json") {
      const parsed = parseJsonCaptionSidecar(text);
      if (!parsed.ok) {
        ignored.push(baseName(name));
        continue;
      }
      if (Object.keys(parsed.captions).length) entry(stem).captions = parsed.captions;
      if (parsed.caption) jsonCaptions.set(stem, parsed.caption);
      if (!Object.keys(parsed.captions).length && !parsed.caption) entry(stem);
      continue;
    }
    const caption = String(text ?? "").trim();
    if (caption) entry(stem).text = caption;
  }
  // A `.txt` caption wins; the JSON caption only fills a stem that has none.
  for (const [stem, caption] of jsonCaptions) {
    if (!entry(stem).text) entry(stem).text = caption;
  }
  return { byStem, ignored };
}
