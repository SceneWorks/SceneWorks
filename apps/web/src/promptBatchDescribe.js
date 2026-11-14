// Formats per-image describe results into Image Studio batch text (SW-7084). Pure and
// DOM-free: the output is one `---`-separated block per image that splitPromptLines /
// parsePromptResolution (promptBatch.js) read back as one prompt with its `[W:H]` ratio.

// Standard ratios the describer output snaps to: landscape, square, then portrait forms.
const STANDARD_RATIOS = [
  [1, 1], [5, 4], [4, 3], [3, 2], [16, 9], [21, 9],
  [4, 5], [3, 4], [2, 3], [9, 16], [9, 21],
];

// `pose` holds one line per person (the worker writes `<label> — Body line: …; Stance: …; …`),
// so it becomes one `Pose:` line per person; every other field is one line. A result stored
// before `pose` existed has no pose and simply emits no Pose line.
const FIELD_LABELS = [
  ["description", "Description"],
  ["pose", "Pose"],
  ["style", "Style"],
  ["camera", "Camera"],
  ["composition", "Composition"],
  ["lighting", "Lighting"],
  ["environment", "Environment"],
];

// Nearest standard ratio in log-ratio space (so 3:2 and 2:3 are equally far from 1:1).
// Returns "W:H", or null when the dimensions are not positive finite numbers.
export function snapRatio(width, height) {
  const w = Number(width);
  const h = Number(height);
  if (!(w > 0) || !(h > 0) || !Number.isFinite(w) || !Number.isFinite(h)) return null;
  const target = Math.log(w / h);
  let best = STANDARD_RATIOS[0];
  let bestDistance = Infinity;
  for (const ratio of STANDARD_RATIOS) {
    const distance = Math.abs(Math.log(ratio[0] / ratio[1]) - target);
    if (distance < bestDistance) {
      best = ratio;
      bestDistance = distance;
    }
  }
  return `${best[0]}:${best[1]}`;
}

// The describer writes labels as `<lowercase noun>_<digits>` (man_1, woman_2, ...), so any
// such whole token is wrapped; there is no noun whitelist. Tradeoff: legible text that
// happens to read like `item_3` is wrapped too. Existing `{{...}}` spans are left alone,
// which also prevents double-wrapping `{{woman_1}}`.
const LABEL = /(?<![\w])[a-z]+_\d+(?![\w])/g;

export function wrapLabels(text) {
  return String(text)
    .split(/(\{\{[^}]*\}\})/)
    .map((part, i) => (i % 2 === 1 ? part : part.replace(LABEL, "{{$&}}")))
    .join("");
}

const PER_LINE_FIELDS = new Set(["pose"]);

// Returns { text, skipped }. Failed results, ones without usable dimensions, and ones whose
// fields are all empty (a ratio with no prompt text) emit no block and are listed in
// `skipped` (original objects, input order) for the review UI.
export function formatPromptBatch(results) {
  const blocks = [];
  const skipped = [];
  for (const result of Array.isArray(results) ? results : []) {
    const ratio = result && !result.failed ? snapRatio(result.width, result.height) : null;
    if (!ratio) {
      if (result) skipped.push(result);
      continue;
    }
    const lines = [`[${ratio}]`];
    for (const [key, label] of FIELD_LABELS) {
      const value = typeof result.fields?.[key] === "string" ? result.fields[key] : "";
      const parts = PER_LINE_FIELDS.has(key) ? value.split("\n") : [value];
      for (const part of parts) {
        // Newlines collapse so a field can never produce a `---` delimiter line.
        const flat = part.replace(/\s+/g, " ").trim();
        if (flat) lines.push(`${label}: ${wrapLabels(flat)}`);
      }
    }
    if (lines.length === 1) {
      skipped.push(result);
      continue;
    }
    blocks.push(lines.join("\n"));
  }
  return { text: blocks.join("\n---\n"), skipped };
}
