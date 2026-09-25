// YuE lyrics-to-song controls (epic sc-19373, sc-19385) — the pure half of the Audio Studio's
// YuE surface: the section-labelled lyrics model, genre-tag parsing + suggestions, and the model
// capability predicates. AudioStudio.jsx / audioYueControls.jsx render these; the payload keys
// match the audio job route (apps/rust-api/src/dto.rs AudioJobRequest, sc-19384): the genre tags
// ride `prompt` (space-joined — YuE's `genre.txt`), the sections ride `lyrics`.
//
// Genre-tag suggestions: `data/yue-top-200-tags.json` is a VERBATIM copy of upstream
// `top_200_tags.json` from github.com/multimodal-art-projection/YuE (branch YuE-v1, rev
// 6d4f0b1f8ce6a55fb2392e959394c46e07ee334d), Copyright 2025 Ruibin Yuan and core contributors
// from M-A-P and HKUST, Apache-2.0 — the licence + NOTICE ship at apps/desktop/licenses/yue/.
import YUE_TOP_TAGS from "./data/yue-top-200-tags.json";

// The section labels the editor offers. Upstream splits lyrics on `\[(\w+)\]`, so a label is a
// single word (no hyphen); these are the ones YuE was trained on and the guide documents.
export const YUE_SECTION_LABELS = Object.freeze(["intro", "verse", "chorus", "bridge", "outro"]);

// The tag categories upstream publishes, in its own order, with display names.
export const YUE_TAG_CATEGORIES = Object.freeze([
  { id: "genre", label: "Genre" },
  { id: "instrument", label: "Instrument" },
  { id: "mood", label: "Mood" },
  { id: "gender", label: "Vocal gender" },
  { id: "timbre", label: "Vocal timbre" },
]);

// A model sings segmented lyrics (backend Capabilities.supports_segmented_lyrics — the six YuE
// checkpoints). Capability-driven, never an id match.
export function isSegmentedSongModel(model) {
  return model?.audio?.supportsSegmentedLyrics === true;
}

// An in-context-learning (`_icl`) checkpoint: a segmented-song model that also conditions on
// ReferenceAudio. Mirrors the API's `icl_model` gate in validate_audio_job_for_model.
export function isIclSongModel(model) {
  const conditioning = Array.isArray(model?.audio?.conditioning) ? model.audio.conditioning : [];
  return (
    isSegmentedSongModel(model) &&
    conditioning.some((kind) => String(kind).toLowerCase() === "referenceaudio")
  );
}

// Starter song for a freshly-selected YuE model: a verse and a chorus, the smallest structure
// the default two-segment render uses.
export function defaultYueSections() {
  return [
    { label: "verse", text: "" },
    { label: "chorus", text: "" },
  ];
}

// The `lyrics` string YuE reads: each non-empty section as `[label]\n<text>`, blank line between
// sections (upstream's lyrics.txt shape). Empty sections are dropped.
export function yueLyricsForSubmit(sections) {
  return (Array.isArray(sections) ? sections : [])
    .map((section) => ({
      label: typeof section?.label === "string" && section.label.trim() ? section.label.trim() : "verse",
      text: typeof section?.text === "string" ? section.text.trim() : "",
    }))
    .filter((section) => section.text.length > 0)
    .map((section) => `[${section.label}]\n${section.text}`)
    .join("\n\n");
}

// Split free text into tags: commas / newlines separate tags; whitespace inside a tag is kept
// ("bright vocal" is one upstream tag). Trimmed, empties dropped.
export function parseGenreTags(text) {
  return String(text ?? "")
    .split(/[,\n]+/)
    .map((tag) => tag.trim().replace(/\s+/g, " "))
    .filter(Boolean);
}

// Split a refined tag LINE into tags. A comma / newline list splits as `parseGenreTags`; a bare
// space-separated line (the YuE guide's prompt shape — "inspiring female uplifting pop airy vocal")
// is split greedily: at each word the longest run matching a known multi-word upstream tag
// ("airy vocal") becomes one tag, otherwise the single word does.
export function splitGenreTagLine(text) {
  const raw = String(text ?? "");
  if (/[,\n]/.test(raw)) {
    return parseGenreTags(raw);
  }
  const known = new Set(allYueTagSuggestions().map((tag) => tag.toLowerCase()));
  const longest = Math.max(1, ...[...known].map((tag) => tag.split(" ").length));
  const words = raw.trim().split(/\s+/).filter(Boolean);
  const tags = [];
  for (let index = 0; index < words.length; ) {
    let span = 1;
    for (let size = Math.min(longest, words.length - index); size > 1; size -= 1) {
      if (known.has(words.slice(index, index + size).join(" ").toLowerCase())) {
        span = size;
        break;
      }
    }
    tags.push(words.slice(index, index + span).join(" "));
    index += span;
  }
  return addGenreTags([], tags);
}

// Add tags to a list, skipping case-insensitive duplicates.
export function addGenreTags(current, incoming) {
  const next = Array.isArray(current) ? [...current] : [];
  const seen = new Set(next.map((tag) => tag.toLowerCase()));
  for (const tag of incoming) {
    const key = tag.toLowerCase();
    if (!seen.has(key)) {
      seen.add(key);
      next.push(tag);
    }
  }
  return next;
}

// The `prompt` YuE reads: the tags space-joined (upstream genre.txt is one space-separated line).
export function genreTagsPrompt(tags) {
  return (Array.isArray(tags) ? tags : []).map((tag) => String(tag).trim()).filter(Boolean).join(" ");
}

// Suggested tags per category — upstream's list with its case-variant duplicates ("Pop" / "pop")
// folded to the first spelling, in upstream order.
export function yueTagSuggestions(category) {
  return addGenreTags([], Array.isArray(YUE_TOP_TAGS[category]) ? YUE_TOP_TAGS[category] : []);
}

// Every suggestion across all categories (deduplicated) — the free-text input's datalist.
export function allYueTagSuggestions() {
  return addGenreTags(
    [],
    YUE_TAG_CATEGORIES.flatMap((category) => yueTagSuggestions(category.id)),
  );
}
