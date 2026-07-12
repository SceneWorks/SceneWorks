import { issue } from "./validation/issues.js";

export const noPresetId = "__no_preset__";

// LoRA-per-job / per-preset caps (sc-8936, F-134). Single source of truth for the web:
// previously the same numbers were hand-copied as bare literals across the Image/Video
// studios, the Character Studio picker, and the Preset Manager, so raising a cap (already
// done once, 3 -> 4/5) meant hunting scattered magic numbers and left stale "/3" badges.
//
// These MUST stay in sync with the worker guards (single source of truth on that side):
//   - MAX_JOB_LORAS_TOTAL == crates/sceneworks-worker/src/image_jobs.rs::MAX_JOB_LORAS (5):
//     the hard per-job total the generation path rejects above (builtin + user combined).
//   - MAX_USER_JOB_LORAS (4): the user-selectable cap the studio pickers enforce, leaving
//     headroom for one auto-applied builtin LoRA within the total (pick 4 -> total 5).
//   - MAX_PRESET_LORAS == the recipe-preset normalizer's cap (5): LoRAs a saved preset holds.
export const MAX_JOB_LORAS_TOTAL = 5;
export const MAX_USER_JOB_LORAS = 4;
export const MAX_PRESET_LORAS = 5;

export function rememberPresetDefault(snapshots, key, currentValue, appliedValue) {
  const previousSnapshot = snapshots.current[key];
  snapshots.current[key] = {
    appliedValue,
    previousValue:
      previousSnapshot && Object.is(currentValue, previousSnapshot.appliedValue)
        ? previousSnapshot.previousValue
        : currentValue,
  };
}

export function clearPresetDefault(setter, snapshots, key) {
  const snapshot = snapshots.current[key];
  if (!snapshot) {
    return;
  }
  setter((current) => (Object.is(current, snapshot.appliedValue) ? snapshot.previousValue : current));
  delete snapshots.current[key];
}

export const defaultModesByWorkflow = {
  text_to_image: ["text_to_image", "character_image"],
  edit_image: ["edit_image"],
  image_to_video: ["image_to_video"],
  text_to_video: ["text_to_video"],
  first_last_frame: ["first_last_frame"],
};

export const modeLabels = {
  text_to_image: "Text",
  edit_image: "Edit",
  character_image: "Character",
  image_to_video: "Image Video",
  text_to_video: "Text Video",
  first_last_frame: "First/Last",
};

export function workflowModelType(workflow) {
  return workflow?.includes("video") || workflow === "first_last_frame" ? "video" : "image";
}

export function workflowModes(workflow) {
  return defaultModesByWorkflow[workflow] ?? [workflow].filter(Boolean);
}

export function compactModeList(workflow) {
  return workflowModes(workflow).map((mode) => modeLabels[mode] ?? mode).join(", ");
}

// Pull the raw families array out of a LoRA/model entry or a lora_import job
// snapshot, trying the various shapes producers use. Pass includeManifest for
// import-job snapshots, whose family metadata lives under payload.manifestEntry.
// Output is raw (unnormalized) — callers that match should normalizeFamilies().
export function extractFamilies(item, { includeManifest = false } = {}) {
  const compatibility = item?.compatibility ?? {};
  const manifest = item?.payload?.manifestEntry ?? {};
  const manifestCompatibility = manifest?.compatibility ?? {};
  const values =
    item?.families ??
    item?.compatibleFamilies ??
    item?.modelFamilies ??
    compatibility.families ??
    (includeManifest
      ? manifest.families ??
        manifest.compatibleFamilies ??
        manifest.modelFamilies ??
        manifestCompatibility.families ??
        item?.payload?.family ??
        manifest.family ??
        item?.family ??
        []
      : item?.family
        ? [item.family]
        : []);
  return Array.isArray(values) ? values : [values].filter(Boolean);
}

export function loraFamilies(lora) {
  return normalizeFamilies(extractFamilies(lora));
}

export function modelLoraFamilies(model) {
  const compatibility = model?.loraCompatibility ?? {};
  const values =
    model?.families ??
    model?.compatibleFamilies ??
    model?.modelFamilies ??
    model?.loraFamilies ??
    compatibility.families ??
    (model?.family ? [model.family] : []);
  return normalizeFamilies(values);
}

export function normalizeLoraFamily(family) {
  const normalized = String(family ?? "").trim().toLowerCase().replaceAll("_", "-");
  // Mirror the backend's canonical resolver: the separator-less `krea2` (ostris
  // ai-toolkit's `ss_base_model_version`) means the same family as `krea-2`/`krea_2`,
  // which the `_`->`-` step alone can't unify. Explicit alias, not a blind strip.
  return normalized === "krea2" ? "krea-2" : normalized;
}

export function normalizeFamilies(values) {
  return (Array.isArray(values) ? values : [values])
    .map(normalizeLoraFamily)
    .filter(Boolean);
}

// True when a LoRA declares at least one resolvable architecture family. A
// family-less LoRA — e.g. an external ComfyUI adapter whose on-disk format the
// detector doesn't recognize (sc-10452 scan, sc-10509) — can never pass the API's
// generate-time compatibility gate (`validate_lora_specs_for_model` 400s on an
// empty family). It must not be selectable in any picker, not even under the
// "Show incompatible" escape hatch, which overrides a known-but-mismatched family,
// not an unknown one. Model Manager still lists it, flagged unusable, so the user
// sees the file was found.
export function loraHasResolvableFamily(lora) {
  return loraFamilies(lora).length > 0;
}

export function loraMatchesModel(lora, model) {
  const modelFamilies = modelLoraFamilies(model);
  const families = loraFamilies(lora);
  // Model side stays permissive: when no model is selected yet, or a model
  // declares no LoRA families, we can't gate — keep showing the LoRA (preset
  // application, the "still importing" warning, and the no-model picker all rely
  // on this). LoRA side fails CLOSED (sc-10509): a LoRA that declares no
  // resolvable family can never pass the API's `validate_lora_specs_for_model`
  // (it 400s on an empty family), so — once there IS a model family to gate
  // against — it is not offered, rather than the old fail-open that surfaced a
  // dead-end selection. External ComfyUI scan rows are the first source to emit
  // family-less LoRAs at scale.
  if (!modelFamilies.length) {
    return true;
  }
  return families.length > 0 && families.some((family) => modelFamilies.includes(family));
}

// Resolve an edit-capable model whose family matches the asset's generating model.
// Prefers the exact generating model when it can edit, then any same-family
// edit-capable model; returns null so Image Studio keeps its default edit model
// when nothing matches.
export function editModelForAsset(asset, imageModels) {
  const sourceModelId = asset?.recipe?.model;
  if (!sourceModelId) {
    return null;
  }
  const models = Array.isArray(imageModels) ? imageModels : [];
  const canEdit = (item) => {
    const caps = item?.capabilities ?? [];
    return caps.includes("edit_image") || caps.includes("image_edit");
  };
  const sourceModel = models.find((item) => item.id === sourceModelId);
  if (sourceModel && canEdit(sourceModel)) {
    return sourceModel.id;
  }
  const families = modelLoraFamilies(sourceModel ?? { family: sourceModelId });
  if (families.length) {
    const sibling = models.find(
      (item) => canEdit(item) && modelLoraFamilies(item).some((family) => families.includes(family)),
    );
    if (sibling) {
      return sibling.id;
    }
  }
  return null;
}

export function loraLooksLikeIcLora(lora) {
  if (lora?.icLora === true || lora?.isIcLora === true) {
    return true;
  }
  if (String(lora?.conditioningRole ?? "").trim().toLowerCase().replaceAll("-", "_") === "ic_lora") {
    return true;
  }
  const source = lora?.source ?? {};
  const files = Array.isArray(source.files) ? source.files : Array.isArray(lora?.files) ? lora.files : [];
  const text = [
    lora?.id,
    lora?.loraId,
    lora?.name,
    lora?.displayName,
    lora?.installedPath,
    lora?.sourcePath,
    lora?.path,
    source.repo,
    source.file,
    source.path,
    ...files,
  ]
    .filter(Boolean)
    .join(" ")
    .toLowerCase()
    .replaceAll("_", "-");
  return text.includes("ic-lora") || text.includes("ltx-2-3-ic-");
}

// A Krea-style image-edit LoRA (epic 10871): declares `conditioningRole: image_edit` (the image
// sibling of `ic_lora`). Unlike IC-LoRAs there is no filename/id convention to fall back on — the
// role is the only signal — so this is a strict role/flag test, matching the worker's
// `lora_declares_image_edit_role`. Selecting one activates the dual-conditioning edit recipe; the
// base can't edit without it.
export function loraLooksLikeImageEditLora(lora) {
  if (lora?.imageEditLora === true) {
    return true;
  }
  return (
    String(lora?.conditioningRole ?? "").trim().toLowerCase().replaceAll("-", "_") === "image_edit"
  );
}

// The image-edit LoRA a model's edit mode requires, resolved from the LoRA catalog (installed or
// not), or null. A model "needs an edit LoRA" exactly when a compatible `image_edit`-role LoRA
// exists — today only Krea 2 (the `krea2_identity_edit` builtin, family `krea_2`); other edit
// models (Qwen-Image-Edit, FLUX.2) have none, so this returns null and their edit is unaffected.
export function findModelEditLora(loras, model) {
  if (!Array.isArray(loras) || !model) {
    return null;
  }
  return loras.find((lora) => loraLooksLikeImageEditLora(lora) && loraMatchesModel(lora, model)) ?? null;
}

// Whether a catalog LoRA's weights are present locally (downloaded). Built-in LoRAs referenced
// from Hugging Face start `installState: "missing"` until fetched; the worker can't load a missing
// file (it does not download at job time), so an auto-applied edit LoRA must clear this first.
export function loraIsInstalled(lora) {
  return Boolean(lora) && lora.installState !== "missing";
}

export function presetMatchesWorkflow(preset, mode) {
  // A preset has one primary workflow for persistence, but modes describe every
  // Studio entry point where the picker should surface it.
  if (preset?.modes?.length) {
    return preset.modes.includes(mode);
  }
  return preset?.workflow === mode;
}

export function presetMatchesModel(preset, model, models = null) {
  if (!preset?.model || !model?.id) {
    return true;
  }
  if (preset.model === model.id) {
    return true;
  }
  // Family-aware fallback: a preset pinned to a sibling model still applies when
  // both models share a LoRA family (e.g. an ltx_2_3 preset under ltx_2_3_eros —
  // both "ltx-video"). Needs the catalog to resolve the preset's pinned model id
  // into its family; without it (e.g. offline fallback) we stay strict.
  if (Array.isArray(models)) {
    const presetModelFamilies = modelLoraFamilies(models.find((item) => item.id === preset.model));
    const modelFamilies = modelLoraFamilies(model);
    return (
      presetModelFamilies.length > 0 &&
      modelFamilies.length > 0 &&
      presetModelFamilies.some((family) => modelFamilies.includes(family))
    );
  }
  return false;
}

export function presetLoras(preset) {
  return preset?.loras ?? preset?.builtInLoras ?? [];
}

export function presetLoraId(presetLora) {
  return typeof presetLora === "string" ? presetLora : presetLora?.id ?? presetLora?.loraId;
}

// Krea 2's distilled, CFG-free Turbo attenuates Raw-trained LoRAs (sc-7579 / sc-7932): the generic
// 0.8 default under-expresses on the few-step student, so a krea-2-family LoRA defaults to a higher
// apply weight (real-weight-validated coherent through scale 4). This is still a DEFAULT — an explicit
// preset weight, a stored `defaultWeight`, or the LoRA's own `weight` still wins. Family token is the
// normalized form (`normalizeLoraFamily`: krea_2 → krea-2).
const KREA_LORA_DEFAULT_WEIGHT = 1.5;

export function loraWeight(lora, presetLora = {}) {
  const fallback = loraFamilies(lora).includes("krea-2") ? KREA_LORA_DEFAULT_WEIGHT : 0.8;
  const value = Number(presetLora.weight ?? lora?.defaultWeight ?? lora?.weight ?? fallback);
  return Number.isFinite(value) ? value : fallback;
}

export function serializePresetLora(lora, presetLora = {}) {
  const id = presetLoraId(presetLora) ?? lora?.id;
  return {
    id,
    name: lora?.name ?? presetLora?.name ?? presetLora?.displayName ?? id,
    scope: lora?.scope ?? presetLora?.scope ?? "global",
    weight: loraWeight(lora, presetLora),
    triggerWords: lora?.triggerWords ?? [],
    compatibility: lora?.compatibility ?? presetLora?.compatibility ?? {},
    icLora: lora?.icLora ?? presetLora?.icLora ?? false,
    conditioningRole: lora?.conditioningRole ?? presetLora?.conditioningRole ?? null,
    installedPath: lora?.installedPath ?? presetLora?.installedPath ?? null,
    source: lora?.source ?? presetLora?.source ?? null,
    presetManaged: true,
  };
}

export function serializeLora(lora, override = {}) {
  return {
    id: lora.id,
    name: lora.name ?? lora.id,
    scope: lora.scope ?? "global",
    weight: Number.isFinite(Number(override.weight)) ? Number(override.weight) : loraWeight(lora),
    triggerWords: lora.triggerWords ?? [],
    notes: lora.notes ?? "",
    compatibility: lora.compatibility ?? {},
    family: lora.family ?? null,
    families: lora.families ?? null,
    compatibleFamilies: lora.compatibleFamilies ?? null,
    modelFamilies: lora.modelFamilies ?? null,
    installedPath: lora.installedPath ?? null,
    sourcePath: lora.sourcePath ?? null,
    source: lora.source ?? null,
    // The conditioning role (`ic_lora` / `image_edit`) is what tells the worker a LoRA
    // is more than a plain style adapter — an LTX IC-LoRA or a Krea image-edit LoRA
    // (epic 10871). The worker reads it straight off the payload `loras` entry (no catalog
    // re-lookup), so it MUST round-trip here: without it, a selected edit LoRA fails the
    // edit lane's role check (R5) even though the user picked it. `icLora` rides along for
    // the flag-based half of the same test.
    conditioningRole: lora.conditioningRole ?? null,
    icLora: lora.icLora ?? false,
    // `installedPath` points at the LoRA directory; for trained LoRAs that
    // directory also holds step checkpoints, so the worker must be told the
    // exact adapter file. Forward the manifest's declared `files`/`file` —
    // without them the worker falls back to the first .safetensors on disk and
    // can load an early checkpoint instead of the final adapter.
    files: lora.files ?? null,
    file: lora.file ?? null,
    presetManaged: Boolean(lora.presetManaged),
  };
}

export function presetLoraDetails(preset, loras) {
  return presetLoras(preset)
    .map((presetLora) => {
      const id = presetLoraId(presetLora);
      const lora = loras.find((item) => item.id === id);
      return lora
        ? { ...serializePresetLora(lora, presetLora), missing: lora.installState === "missing" }
        : { id, name: id, weight: loraWeight(null, presetLora), missing: true };
    })
    .filter((lora) => lora.id);
}

export function presetPromptParts(preset) {
  return [preset?.prompt?.prefix, preset?.prompt?.suffix]
    .map((part) => String(part ?? "").trim())
    .filter(Boolean);
}

export function presetValidation(preset, loras, model) {
  const details = presetLoraDetails(preset, loras);
  const missing = details.filter((lora) => lora.missing).map((lora) => lora.id);
  const incompatible = details
    .filter((detail) => {
      const lora = loras.find((item) => item.id === detail.id);
      return lora && !loraMatchesModel(lora, model);
    })
    .map((lora) => lora.id);
  return {
    missing,
    incompatible,
    ok: missing.length === 0 && incompatible.length === 0,
  };
}

// What gates Save in the Preset Manager, in the app-wide vocabulary (epic 10644,
// sc-10651). Replaces the local saveRequirement/saveError split and the flatten-only
// presetValidationMessage — name and model are silent requirements (the empty fields show
// them); a read-only built-in, broken default values, and the preset's own LoRA problems
// are surfaced errors. The read-only case short-circuits: a locked form's other checks are
// moot.
//
// `valueErrors` are the out-of-range / out-of-menu default values from defaultValueErrors
// (sc-10589) — pre-formatted message strings, one error each.
export function presetSaveValidation({ editable, name, model }, { validation, valueErrors = [] } = {}) {
  const issues = [];
  if (!editable) {
    issues.push(issue.error(null, "Built-in presets are read-only. Duplicate it to make an editable copy."));
    return issues;
  }
  if (!name?.trim()) {
    issues.push(issue.requirement("name", "Name is required."));
  }
  if (!model) {
    issues.push(issue.requirement("model", "Choose a model before saving."));
  }
  for (const message of valueErrors) {
    issues.push(issue.error(null, message));
  }
  const missing = validation?.missing ?? [];
  const incompatible = validation?.incompatible ?? [];
  if (missing.length) {
    issues.push(
      issue.error(
        null,
        `Save blocked: ${missing.join(", ")} ${missing.length === 1 ? "has" : "have"} not finished importing. Wait for imports to finish.`,
      ),
    );
  }
  if (incompatible.length) {
    issues.push(
      issue.error(
        null,
        `Save blocked: ${incompatible.join(", ")} ${incompatible.length === 1 ? "is" : "are"} not compatible with the selected model. Remove incompatible LoRAs or choose a matching model.`,
      ),
    );
  }
  return issues;
}

// ── Studio "Save as Preset" round-trip helpers ───────────────────────────────
// The Image/Video studios snapshot their current working state into a recipe
// preset and restore it on apply. These helpers keep that round-trip in one
// tested place so both studios stay thin.

// Slugify a preset name into a valid recipe-preset id (lowercase letters,
// digits, dash, underscore). Mirrors the backend's slugify_preset_id so the id
// the client previews matches what the server stores.
export function slugifyPresetId(value) {
  return String(value ?? "")
    .trim()
    .toLowerCase()
    .replace(/[^a-z0-9_-]+/g, "_")
    .replace(/^[_-]+|[_-]+$/g, "");
}

// Map an active studio mode to the recipe workflow it persists under. Inverts
// defaultModesByWorkflow, so character_image folds into text_to_image (the only
// workflow whose modes include it).
export function workflowForMode(mode) {
  for (const [workflow, modes] of Object.entries(defaultModesByWorkflow)) {
    if (modes.includes(mode)) {
      return workflow;
    }
  }
  return mode;
}

// True when `name` collides with an existing preset by case-insensitive name or
// by slugged id — the two ways the backend rejects a duplicate. Powers a friendly
// client-side check before POSTing.
export function presetNameTaken(name, presets = []) {
  const trimmed = String(name ?? "").trim().toLowerCase();
  if (!trimmed) {
    return false;
  }
  const id = slugifyPresetId(name);
  return presets.some((preset) => {
    const presetName = String(preset?.name ?? "").trim().toLowerCase();
    return presetName === trimmed || (Boolean(id) && preset?.id === id);
  });
}

// Coerce a possibly-stringy numeric input ("30", 4.5) to a finite number, or
// undefined when blank or non-numeric — so cleanPresetDefaults drops it and the
// stored default stays a real number the backend can range-check.
export function finiteNumberOrUndefined(value) {
  if (value === "" || value === null || value === undefined) {
    return undefined;
  }
  const parsed = Number(value);
  return Number.isFinite(parsed) ? parsed : undefined;
}

// Drop keys whose value is null, undefined, or an empty string (the studios'
// "use the model default" sentinel). Numbers (including 0) and booleans survive
// so 0-valued knobs and `false` toggles round-trip intact.
export function cleanPresetDefaults(defaults = {}) {
  const cleaned = {};
  for (const [key, value] of Object.entries(defaults)) {
    if (value === null || value === undefined || value === "") {
      continue;
    }
    cleaned[key] = value;
  }
  return cleaned;
}

// Build the createPreset payload from a studio's current state. `defaults` is the
// raw snapshot of visible knobs (including the literal `prompt`); callers must
// never include the seed. Top-level model/workflow/loras carry the fields the
// backend validates (model<->workflow capability, lora<->model compatibility).
export function buildStudioPresetPayload({ name, scope = "project", mode, model, loras = [], defaults = {} }) {
  const workflow = workflowForMode(mode);
  return {
    id: slugifyPresetId(name),
    name: String(name ?? "").trim(),
    scope,
    workflow,
    modes: workflowModes(workflow),
    model,
    loras: loras.map((lora) => ({
      id: lora.id,
      weight: Number.isFinite(Number(lora.weight)) ? Number(lora.weight) : loraWeight(lora),
    })),
    defaults: cleanPresetDefaults(defaults),
  };
}

// Apply one preset-default value through the remember/restore snapshot machinery
// so switching back to None (or another preset) restores the user's prior value.
// Generalizes the inline pattern the studios already use for count/resolution.
export function applyPresetDefault(snapshots, key, setter, value) {
  setter((current) => {
    rememberPresetDefault(snapshots, key, current, value);
    return value;
  });
}
