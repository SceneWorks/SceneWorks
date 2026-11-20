// MiniMax-H3-Image prompt expansion (SW-7102): the wording of the Expand prompt control and the
// size it offers for the expander's proposed aspect ratio.
//
// SW-7208: and which model the expander runs (see `h3ExpanderChoices` below).
//
// The expander proposes a RATIO, never a size. The size offered here is one of MiniMax-H3-Image's
// own presets (`limits.resolutions`) whose ratio matches, preferring the preset closest in area to
// the size the user already has; with no matching preset, the ratio at the current area on a 16 px
// grid inside the model's `minDimension`/`maxDimension`. Accepting it is the user's separate
// choice — nothing here changes the request on its own.

import { useCallback, useMemo, useState } from "react";
import { MINIMAX_H3_IMAGE_PE_MODEL_ID } from "./constants.js";
import { H3_EXPANDER_DESCRIBER_MODEL_ID, readH3ExpanderModel, writeH3ExpanderModel } from "./h3ExpanderModelStore.js";

export {
  H3_EXPANDER_DESCRIBER_MODEL_ID,
  H3_EXPANDER_MODEL_IDS,
  normalizeH3ExpanderModel,
  readH3ExpanderModel,
  seedH3ExpanderModelFromServer,
  writeH3ExpanderModel,
} from "./h3ExpanderModelStore.js";

const RATIO_TOLERANCE = 0.01;
// Most prompts a batch run expands in one go; past this the run is refused until the variables
// are narrowed (each prompt is one expander job before any image renders).
export const BATCH_EXPAND_MAX = 2000;
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
  // Batch mode: Run batch expands every resolved prompt first, then queues the images with the
  // expansions (useImageBatchRun). The batch list itself is never rewritten.
  batch: Object.freeze({
    tileDescription: "MiniMax-H3-Image’s own expander, run on every prompt when you run the batch",
    hint: "Run batch first expands each prompt — variables filled in, a leading [size] kept — one after another, then queues the images with the expansions. Your batch list is not changed.",
    useSizes: "Use each prompt’s proposed size (prompts with their own [size] keep it)",
    run: (count) => `Expand & run batch · ${count}`,
    running: "Expanding…",
    confirm: (count) => ` Each prompt is expanded first (${count.toLocaleString("en-US")} expander runs).`,
    tooMany: (count, max) =>
      `This batch fills in to ${count.toLocaleString("en-US")} prompts — more than the ${max.toLocaleString("en-US")} the expander runs in one go. Narrow the variables, or turn Expand prompt off.`,
    progress: (current, total) => `Expanding prompt ${current.toLocaleString("en-US")} of ${total.toLocaleString("en-US")}…`,
    failed: (item, reason) => `Prompt ${item} was not expanded: ${reason}`,
  }),
});

// SW-7208: the expander runs one of two models, both on MiniMax's prompt files and reply contract:
// MiniMax-H3-Image-PE (the default) or the abliterated Qwen3.8-27B the dataset describer uses
// (any installed tier). The saved choice lives in h3ExpanderModelStore.js. Mirrors
// `sceneworks_core::contracts::MINIMAX_H3_IMAGE_EXPAND_PROMPTS_PURPOSE`.
//
// The expander entry's conditional component holding MiniMax's prompt files alone: the abliterated
// choice needs them, but not MiniMax-H3-Image-PE's weights.
export const H3_EXPAND_PROMPTS_PURPOSE = "describer_prompt_expansion";

const EXPANDER_LABELS = Object.freeze({
  [MINIMAX_H3_IMAGE_PE_MODEL_ID]: "MiniMax-H3-Image-PE",
  [H3_EXPANDER_DESCRIBER_MODEL_ID]: "Abliterated Qwen3.8-27B (the describer’s model)",
});

// Each expander model with what it needs on this install. MiniMax-H3-Image-PE needs an installed
// tier (which brings its prompt files). The abliterated choice needs an installed describer tier
// plus MiniMax's prompt files, which an installed expander tier or the prompt-files download
// supplies. `missing` is "weights", "prompts" or null.
export function h3ExpanderChoices(models = []) {
  const find = (id) => models.find((entry) => entry?.id === id) ?? null;
  const installed = (entry) => Boolean(entry) && entry.installState !== "missing";
  const pe = find(MINIMAX_H3_IMAGE_PE_MODEL_ID);
  const describer = find(H3_EXPANDER_DESCRIBER_MODEL_ID);
  const promptsInstalled =
    pe?.conditionalPurposes?.[H3_EXPAND_PROMPTS_PURPOSE]?.installState === "installed";
  return [
    { id: MINIMAX_H3_IMAGE_PE_MODEL_ID, entry: pe, missing: installed(pe) ? null : "weights" },
    {
      id: H3_EXPANDER_DESCRIBER_MODEL_ID,
      entry: describer,
      missing: !installed(describer) ? "weights" : promptsInstalled ? null : "prompts",
    },
  ]
    .filter((choice) => choice.entry)
    .map((choice) => ({ ...choice, label: EXPANDER_LABELS[choice.id], available: choice.missing === null }));
}

// The expander for this install and the saved choice: `selected` is the saved model while it can
// run, else the first model that can. `offered` is whether the Expand tool shows at all: a model
// can run, or the describer is installed and only lacks the prompt files (the panel offers them).
export function resolveH3Expander(models, preferred) {
  const choices = h3ExpanderChoices(models);
  const runnable = choices.filter((choice) => choice.available);
  const selected =
    runnable.find((choice) => choice.id === preferred) ??
    choices.find((choice) => choice.id === preferred && choice.missing === "prompts") ??
    runnable[0] ??
    choices.find((choice) => choice.missing === "prompts") ??
    null;
  return { choices, selected, offered: Boolean(selected) };
}

// The saved expander model as React state, written through on change.
export function useH3ExpanderModel() {
  const [model, setModel] = useState(readH3ExpanderModel);
  const choose = useCallback((value) => setModel(writeH3ExpanderModel(value)), []);
  return [model, choose];
}

// The Expand prompt tool's state for a studio, shared by both Image Studio views: whether the tool
// shows (`available`), the catalog entry of the model that will run (`model`), the expand call
// bound to that model (`prompt`, undefined while it cannot run, which disables the button), and
// the model picker's props.
export function useH3Expander({ models, applies, expandPrompt, downloadConditionalComponents }) {
  const [preferred, choose] = useH3ExpanderModel();
  const { choices, selected, offered } = useMemo(
    () => (applies ? resolveH3Expander(models, preferred) : { choices: [], selected: null, offered: false }),
    [applies, models, preferred],
  );
  const selectedId = selected?.id ?? null;
  const prompt = useCallback(
    (args) => expandPrompt({ ...args, expanderModelId: selectedId }),
    [expandPrompt, selectedId],
  );
  const runnable = Boolean(selected?.available) && typeof expandPrompt === "function";
  const downloadPrompts = useMemo(
    () =>
      typeof downloadConditionalComponents === "function"
        ? () => downloadConditionalComponents(MINIMAX_H3_IMAGE_PE_MODEL_ID, H3_EXPAND_PROMPTS_PURPOSE)
        : undefined,
    [downloadConditionalComponents],
  );
  return {
    available: offered && typeof expandPrompt === "function",
    model: selected?.entry ?? null,
    prompt: runnable ? prompt : undefined,
    picker: { choices, value: selectedId, onChange: choose, onDownloadPrompts: downloadPrompts },
  };
}
