// Iris-3B restoration (sc-25683, epic 25678) — the `iris` engine of the image-upscale surfaces
// (Image Studio post-processing, Image Editor upscale tool, Library single/batch actions).
//
// Unlike Real-ESRGAN / SeedVR2 (a 2×/4× `factor`), Iris is sized by an arbitrary positive `scale`
// (default 4×; 1× = restore only), an input-sizing mode (`budgeted` — the default — fits the input
// into the model's 512/1024 budget first; `original` processes it at its own size) and an optional
// colour correction (default on). The pre-submission dimension preview comes from the API's
// `POST /api/v1/image-restoration/plan`, which runs the engine's own planner — the UI never computes
// "N× of the source" itself, so a budget-downscaled result can never be labelled as N× the original.
//
// Pure + React-free so the payload shapes and the preview wording are unit-tested in isolation.

export const IRIS_RESTORE_ENGINE = "iris";
export const IRIS_RESTORE_MODEL_ID = "iris_3b_restore";
export const IRIS_DEFAULT_SCALE = 4;
export const IRIS_INPUT_SIZINGS = [
  { key: "budgeted", label: "Budgeted input", hint: "Fit the source into 512 px (short) / 1024 px (long) first — the model's trained input size." },
  { key: "original", label: "Original input", hint: "Process the source at its own size — sharper on large sources, but slower and more memory." },
];
// Quick picks; any positive scale can be typed.
export const IRIS_SCALE_PRESETS = [1, 2, 4];

export const IRIS_RESTORE_DEFAULTS = Object.freeze({
  scale: IRIS_DEFAULT_SCALE,
  inputSizing: "budgeted",
  colorFix: true,
});

export function isIrisEngine(engine) {
  return typeof engine === "string" && engine.trim().toLowerCase() === IRIS_RESTORE_ENGINE;
}

// The restoration task's catalog entry (its own install, separate from `iris_3b` generation).
export function irisRestoreModel(models) {
  return (models ?? []).find((model) => model?.id === IRIS_RESTORE_MODEL_ID) ?? null;
}

// Restoration is cache-only at job time: offer the engine only once its `upscaler/` export is
// fully installed (a torn/incomplete install cannot run).
export function irisRestoreInstalled(models) {
  return irisRestoreModel(models)?.installState === "installed";
}

// Whether ANY Iris restoration model can run: the release `iris_3b_restore` export or an installed
// trained restoration model (which runs from its own export, without the release install).
export function irisRestoreRunnable(models) {
  return irisRestoreInstalled(models) || trainedTaskModels(models, "restoration").length > 0;
}

// The restoration model a fresh selection runs: the release export when installed ("" = release),
// else the first installed trained restoration model, else the release (whose missing install the
// plan / job then names).
export function defaultRestoreModel(models) {
  if (irisRestoreInstalled(models)) return "";
  return trainedTaskModels(models, "restoration")[0]?.id ?? "";
}

// A usable scale: any positive finite number, else null (never coerced to another value).
export function normalizeIrisScale(value) {
  const number = typeof value === "string" ? Number(value.trim()) : Number(value);
  return Number.isFinite(number) && number > 0 ? number : null;
}

export function normalizeIrisInputSizing(value) {
  return value === "original" ? "original" : "budgeted";
}

// Installed models a Training Studio depth / restoration run registered (sc-25687 / sc-25688):
// `task` is "depth" | "restoration".
export function trainedTaskModels(models, task) {
  return (models ?? []).filter((model) => model?.trainedCheckpoint?.task === task && model.installState === "installed");
}

// The Iris controls as the job / `upscale` block carries them. No `factor` / `softness` /
// `seed`: the API and worker refuse those for the Iris engine by name. `irisModel` names a trained
// restoration model (sc-25688); absent = the release export.
export function irisRestoreControls({ scale, inputSizing, colorFix, irisModel } = {}) {
  return {
    scale: normalizeIrisScale(scale) ?? IRIS_DEFAULT_SCALE,
    inputSizing: normalizeIrisInputSizing(inputSizing),
    colorFix: colorFix !== false,
    ...(irisModel && irisModel !== IRIS_RESTORE_MODEL_ID ? { irisModel } : {}),
  };
}

// `POST /api/v1/image-restoration/plan` bodies: a not-yet-rendered image (Studio) by size, or
// existing assets (Editor scratch / Library selection) by id — the API measures those upright.
export function restorePlanBodyForSize({ width, height, ...controls }) {
  return { ...irisRestoreControls(controls), width: Number(width), height: Number(height) };
}

// The plan body for a Studio render pass: the Studio size, or — for a prompt batch whose prompts
// carry their own [WxH] directives (`resolutions`, null for a prompt without one) — every DISTINCT
// render size, so the gate covers each render the batch will restore, not only the Studio default.
// Without plan-able dimensions there is nothing to plan (null).
export function restorePlanBodyForRenders({ width, height, resolutions = [] } = {}) {
  const studio = Number(width) > 0 && Number(height) > 0 ? { width: Number(width), height: Number(height) } : null;
  const sizes = new Map();
  for (const resolution of resolutions.length ? resolutions : [null]) {
    const size = resolution && Number(resolution.width) > 0 && Number(resolution.height) > 0
      ? { width: Number(resolution.width), height: Number(resolution.height) }
      : studio;
    if (!size) return null;
    sizes.set(`${size.width}x${size.height}`, size);
  }
  const distinct = [...sizes.values()];
  return distinct.length === 1 ? distinct[0] : { sizes: distinct };
}

export function restorePlanBodyForAssets({ projectId, assetIds, ...controls }) {
  return { ...irisRestoreControls(controls), projectId, assetIds: [...(assetIds ?? [])] };
}

// One preview line. The API's `summary` is the engine planner's honest wording (it names a
// budgeted input as such); a refused item shows the refusal the job would fail with.
export function restorePlanItemText(item) {
  if (!item) return "";
  if (item.error) return `Cannot restore: ${item.error}`;
  return item.plan?.summary ?? "";
}

// Summarize a plan response for a control: every item's line plus whether ANY item is refused
// (which blocks submission — admission may refuse but the UI never silently changes the request).
export function summarizeRestorePlans(response) {
  const items = Array.isArray(response?.items) ? response.items : [];
  const blocked = items.some((item) => Boolean(item?.error));
  return {
    items,
    blocked,
    lines: items.map(restorePlanItemText),
    firstError: items.find((item) => item?.error)?.error ?? null,
  };
}

// Restore saved/reopened settings from an applied `upscale` record (`rawAdapterSettings.upscale`
// or a workflow envelope's `upscale`).
export function irisControlsFromRecord(record) {
  if (!record || !isIrisEngine(record.engine)) return null;
  return irisRestoreControls({
    scale: record.scale,
    inputSizing: record.inputSizing,
    colorFix: record.colorFix,
    irisModel: record.irisModel,
  });
}
