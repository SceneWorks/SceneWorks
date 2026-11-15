// Formats per-image describe results into Image Studio batch text (SW-7084). Pure and
// DOM-free: the output is one `---`-separated block per image that splitPromptLines /
// parsePromptResolution (promptBatch.js) read back as one prompt with its `[W:H]` ratio. With a
// "Fit to" target (SW-7163) an image's block is its condensed prompt instead of its fields.

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

// SW-7163 "Fit to": the smallest prompt limit a target may declare (sceneworks-core
// MIN_FIT_TO_PROMPT_TOKENS; the API refuses a smaller one). Below it the condensing instruction,
// which keeps every limb of every person, cannot be met and most rows would end over the limit.
export const MIN_FIT_TO_PROMPT_TOKENS = 512;

// The image models a describe run can condense for — catalog entries that declare an
// `image.maxPromptTokens` of at least MIN_FIT_TO_PROMPT_TOKENS (installed or not; only their
// limit is needed).
export function fitToTargets(models) {
  return (Array.isArray(models) ? models : [])
    .filter((model) => Number.isInteger(model?.image?.maxPromptTokens) && model.image.maxPromptTokens >= MIN_FIT_TO_PROMPT_TOKENS)
    .map((model) => ({ id: model.id, name: model.name ?? model.id, limit: model.image.maxPromptTokens }));
}

// The row's condensed prompt for the selected target, or null: none is stored for that target,
// condensing failed (the seven fields are used), its text is empty, or the row's fields changed
// since it was condensed (it no longer describes them, so it is never saved).
export function condensedFor(result, fitTo) {
  const condensed = result?.condensed;
  if (!fitTo || !condensed || condensed.target !== fitTo || condensed.failed || result.fieldsEdited) return null;
  return typeof condensed.text === "string" && condensed.text.trim() ? condensed : null;
}

// "612 of 1,024 tokens, leaving room for 8 variable values", with "about" when the count is an
// estimate (counted with the describer's tokenizer, not the target's). Null when no size was
// measured, or the prompt was edited since it was measured.
export function condensedSizeText(condensed) {
  if (condensed?.edited || !Number.isInteger(condensed?.tokens) || !Number.isInteger(condensed?.limitTokens)) return null;
  const about = condensed.tokenCount === "exact" ? "" : "about ";
  const mentions = Number.isInteger(condensed.labelMentions) ? condensed.labelMentions : 0;
  const room = mentions ? `, leaving room for ${mentions} variable ${mentions === 1 ? "value" : "values"}` : "";
  return `${about}${condensed.tokens.toLocaleString("en-US")} of ${condensed.limitTokens.toLocaleString("en-US")} tokens${room}`;
}

// A condense-only request per part of `rows`: each part holds rows of one describer (`model`) at
// one tier (`variant`, null for a describer without tiers), so each row condenses on the model it
// was described with, and stays under `maxBytes` of serialized rows and `maxRows` rows, so a large
// dataset never reaches the API's 10 MiB JSON body limit. `toRequestRow` maps a row to what the
// request carries.
export function condenseChunks(rows, toRequestRow, { maxBytes = 2 * 1024 * 1024, maxRows = 200 } = {}) {
  const chunks = [];
  const open = new Map();
  for (const row of rows ?? []) {
    const request = toRequestRow(row);
    const bytes = new TextEncoder().encode(JSON.stringify(request)).length + 1;
    const model = row.model ?? null;
    const variant = row.variant ?? null;
    const key = JSON.stringify([model, variant]);
    let chunk = open.get(key);
    if (!chunk || chunk.rows.length >= maxRows || chunk.bytes + bytes > maxBytes) {
      chunk = { model, variant, rows: [], bytes: 0 };
      chunks.push(chunk);
      open.set(key, chunk);
    }
    chunk.rows.push(request);
    chunk.bytes += bytes;
  }
  return chunks.map(({ model, variant, rows: chunkRows }) => ({ model, variant, rows: chunkRows }));
}

// Returns { text, skipped }. Failed results, ones without usable dimensions, and ones whose
// fields are all empty (a ratio with no prompt text) emit no block and are listed in
// `skipped` (original objects, input order) for the review UI. With `fitTo`, a result that has
// a condensed prompt for that target emits the `[W:H]` line and that prompt instead of the
// seven `Field:` lines; one without (never condensed, JoyCaption, failed) keeps the fields.
export function formatPromptBatch(results, { fitTo = null } = {}) {
  const blocks = [];
  const skipped = [];
  for (const result of Array.isArray(results) ? results : []) {
    const ratio = result && !result.failed ? snapRatio(result.width, result.height) : null;
    if (!ratio) {
      if (result) skipped.push(result);
      continue;
    }
    const lines = [`[${ratio}]`];
    const condensed = condensedFor(result, fitTo);
    if (condensed) {
      // One line: newlines collapse so the prompt can never produce a `---` delimiter line.
      blocks.push(`${lines[0]}\n${wrapLabels(condensed.text.replace(/\s+/g, " ").trim())}`);
      continue;
    }
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
