// MiniMax-H3-Image prompt expansion (SW-7102): the wording of the Expand prompt control and the
// size it offers for the expander's proposed aspect ratio.
//
// The expander proposes a RATIO, never a size. The size offered here is one of MiniMax-H3-Image's
// own presets (`limits.resolutions`) whose ratio matches, preferring the preset closest in area to
// the size the user already has; with no matching preset, the ratio at the current area on a 16 px
// grid inside the model's `minDimension`/`maxDimension`. Accepting it is the user's separate
// choice — nothing here changes the request on its own.

const RATIO_TOLERANCE = 0.01;
const GRID = 16;

function parseRatio(value) {
  const match = /^(\d+):(\d+)$/.exec(String(value ?? "").trim());
  if (!match) return null;
  const width = Number(match[1]);
  const height = Number(match[2]);
  return width > 0 && height > 0 ? width / height : null;
}

function parseSize(value) {
  const match = /^(\d+)x(\d+)$/.exec(String(value ?? "").trim());
  if (!match) return null;
  const width = Number(match[1]);
  const height = Number(match[2]);
  return width > 0 && height > 0 ? { width, height } : null;
}

// The "WxH" to offer for `whRatio`, or "" when the ratio is unreadable.
export function h3ImageProposedResolution(whRatio, model, currentResolution) {
  const ratio = parseRatio(whRatio);
  if (!ratio) return "";
  const limits = model?.limits ?? {};
  const current = parseSize(currentResolution) ?? parseSize(model?.defaults?.resolution) ?? {
    width: 2048,
    height: 2048,
  };
  const currentArea = current.width * current.height;
  const presets = (Array.isArray(limits.resolutions) ? limits.resolutions : [])
    .map((value) => ({ value, size: parseSize(value) }))
    .filter(({ size }) => size && Math.abs(size.width / size.height / ratio - 1) <= RATIO_TOLERANCE);
  if (presets.length) {
    const areaGap = ({ size }) => Math.abs(Math.log((size.width * size.height) / currentArea));
    return presets.reduce((best, preset) => (areaGap(preset) < areaGap(best) ? preset : best)).value;
  }
  const min = Number(limits.minDimension) || GRID;
  const max = Number(limits.maxDimension) || 4096;
  const snap = (value) => Math.min(max, Math.max(min, Math.round(value / GRID) * GRID));
  let width = Math.sqrt(currentArea * ratio);
  let height = width / ratio;
  const scale = Math.min(1, max / Math.max(width, height));
  width *= scale;
  height *= scale;
  return `${snap(width)}x${snap(height)}`;
}

export const H3_IMAGE_EXPAND_COPY = Object.freeze({
  button: "Expand prompt",
  editButton: "Expand edit instruction",
  hint: "MiniMax-H3-Image’s own prompt expander turns a short prompt into the long caption the model was trained on and proposes a size. You review and accept the expansion — your prompt is not changed on its own, and the size is only offered.",
  editHint: (count) =>
    `MiniMax-H3-Image’s prompt expander reads your ${count} reference${
      count === 1 ? "" : "s"
    } in order as <Picture 1>${count > 1 ? `…<Picture ${count}>` : ""} and expands the instruction. You review and accept the expansion — your prompt is not changed on its own.`,
  busy: "Expanding…",
  missing: (_editing, modelName) =>
    `The prompt expander isn’t installed in a tier this device can run. MiniMax-H3-Image generates normally without it — open the Models screen to download “${modelName}”.`,
  failure: "Prompt expansion failed.",
  draftLabel: "Expanded prompt — edit before applying",
  suggestionLabel: "Proposed size",
  applyResolution: "Use this size",
  again: "Expand again",
  defaultModelName: "MiniMax-H3-Image Prompt Expander",
  // Batch mode: the same expander over every prompt of the batch (BatchExpandPromptsControl).
  batch: Object.freeze({
    tileDescription: "MiniMax-H3-Image’s own expander, run on every prompt in the batch",
    button: (count) => `Expand ${count === 1 ? "the prompt" : `all ${count} prompts`}`,
    hint: "Runs MiniMax-H3-Image’s prompt expander on each prompt in the batch, one after another, before any image renders. You review the expansions and apply them to the batch list — nothing changes until you do, and you can undo it.",
    resolved: (count) =>
      `Variables are filled in first, so each of the ${count.toLocaleString("en-US")} resulting prompts gets its own expansion; applying replaces the templates with one line per prompt (Undo restores them).`,
    confirm: (count) =>
      `That runs the expander ${count.toLocaleString("en-US")} times, one after another, before the batch can render.`,
    tooMany: (count, max) =>
      `These templates fill in to ${count.toLocaleString("en-US")} prompts — more than the ${max.toLocaleString("en-US")} the expander runs in one go. Narrow the variables first.`,
    missing: (keys) => `Add a value for ${keys.map((key) => `{{${key}}}`).join(", ")} before expanding.`,
    groups: "Give every linked group the same number of options before expanding.",
    progress: (current, total) => `Expanding prompt ${current} of ${total}…`,
    summary: (expanded, total) => `Expanded ${expanded} of ${total} prompt${total === 1 ? "" : "s"}.`,
    reviewLabel: "Review the expansions",
    useSizes: "Also start each prompt with its proposed size (prompts with their own [size] keep it)",
    apply: "Apply to batch",
    applied: (count) => `Replaced ${count} prompt${count === 1 ? "" : "s"} with the expansion.`,
    changed: "The batch prompts changed while expanding. Expand again to use the current list.",
    unrepresentable: (items) =>
      `${items.length ? `Prompt${items.length === 1 ? "" : "s"} ${items.slice(0, 8).join(", ")}${items.length > 8 ? ` and ${items.length - 8} more` : ""} contain${items.length === 1 ? "s" : ""} a line of just ---, which` : "The result"} would split the batch list into different prompts. Change that value (or expand again) before applying.`,
    literalPlaceholders: (items) =>
      `Prompt${items.length === 1 ? "" : "s"} ${items.slice(0, 8).join(", ")}${items.length > 8 ? ` and ${items.length - 8} more` : ""} would contain {{…}} text, which the batch list would read as a variable again and render differently. Change that value (or expand again) before applying.`,
  }),
});
