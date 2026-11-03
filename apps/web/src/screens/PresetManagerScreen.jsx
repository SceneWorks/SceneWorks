import { PresetCard } from "./presets/PresetCard.jsx";
import { EditorHeadSection } from "./presets/EditorHeadSection.jsx";
import { KindSection } from "./presets/KindSection.jsx";
import { IdentitySection } from "./presets/IdentitySection.jsx";
import { ModelSection } from "./presets/ModelSection.jsx";
import { WorkflowSection } from "./presets/WorkflowSection.jsx";
import { PromptTemplateSection } from "./presets/PromptTemplateSection.jsx";
import { DefaultsSection } from "./presets/DefaultsSection.jsx";
import { GeneralDefaultsSection } from "./presets/GeneralDefaultsSection.jsx";
import { LorasSection } from "./presets/LorasSection.jsx";
import { PresetAdvancedSection } from "./presets/PresetAdvancedSection.jsx";
import { FooterSection } from "./presets/FooterSection.jsx";
import { downloadSizeText } from "../formatting.js";
import { slugifyPresetId } from "../presetUtils.js";
import React, { useEffect, useMemo, useRef, useState } from "react";

import { AdvancedSection } from "../components/AdvancedSection.jsx";
import { Icon } from "../components/Icons.jsx";
import { WorkPanel } from "../components/WorkPanel.jsx";
import { MAX_PRESET_LORAS, loraMatchesModel, loraWeight, presetLoraId, presetLoras, presetSaveValidation, presetValidation, workflowModes } from "../presetUtils.js";
import { effectiveLimits, samplerOptionsFromModel, schedulerOptionsFromModel, stepsMenuFromModel } from "../samplerOptions.js";
import { useValidation } from "../validation/useValidation.js";

import { useAppStatic } from "../context/AppContext.js";
import { qualityChoices } from "../jobTypes.js";
import { appConfirm } from "../appConfirm.jsx";
import { errorMessage } from "../errorMessage.js";

// The Workflow segment a preset editor offers, and how each choice persists.
//
// `character_image` is NOT a RecipePresetWorkflow — the backend enum is
// text_to_image | edit_image | image_to_video | text_to_video | first_last_frame and
// anything else is a 400. It IS a real `defaults.mode`, which the studios restore on
// preset select (useSavePreset's hydrate effect). So a "Character" preset persists as
// workflow `text_to_image` + `defaults.mode` `character_image`.
//
// A segment only renders when the selected model declares BOTH capabilities, mirroring
// `validate_recipe_preset_model_workflow` (which requires the workflow itself to be in
// the model's capabilities). That is what keeps the control from ever overflowing.
const WORKFLOW_SEGMENTS = [
  { key: "text_to_image", label: "Text", workflow: "text_to_image", mode: "text_to_image", type: "image" },
  { key: "edit_image", label: "Edit", workflow: "edit_image", mode: "edit_image", type: "image" },
  { key: "character_image", label: "Character", workflow: "text_to_image", mode: "character_image", type: "image" },
  { key: "text_to_video", label: "Text → Video", workflow: "text_to_video", mode: "text_to_video", type: "video" },
  { key: "image_to_video", label: "Image → Video", workflow: "image_to_video", mode: "image_to_video", type: "video" },
  { key: "first_last_frame", label: "First–Last Frame", workflow: "first_last_frame", mode: "first_last_frame", type: "video" },
];

// Short mono flags for the model helper line, in the order a reader scans them.
const CAPABILITY_FLAGS = [
  ["text_to_image", "txt2img"],
  ["edit_image", "img2img"],
  ["character_image", "character"],
  ["text_to_video", "txt2vid"],
  ["image_to_video", "img2vid"],
  ["first_last_frame", "first-last"],
];

// Fallbacks for a model that declares no `limits.resolutions` / `.durations` / `.fps`
// (e.g. a user-imported checkpoint). A model that DOES declare them drives the menu
// instead — the studios read the same manifest keys, so the editor must not offer a
// value the launched studio would silently clamp away (sc-10589).
const imageAspectFallback = ["1024x1024", "1536x1024", "1024x1536", "2048x1152"];
const videoResolutionFallback = ["768x512", "1280x720", "720x1280"];
const durationFallback = [3, 4, 6, 8, 10, 12];
const fpsFallback = [24, 25, 30];

// The resolution/aspect menu the selected model honors. Reads the model's effective
// `limits.resolutions` (base `limits`, since a preset is backend-agnostic — it can be
// launched on either worker), falling back to the static list when the model is silent.
function resolutionOptionsForModel(model, isVideo) {
  const values = effectiveLimits(model)?.resolutions;
  if (Array.isArray(values) && values.length) {
    return values.map(String);
  }
  return isVideo ? videoResolutionFallback : imageAspectFallback;
}

// Video-only menus. Durations/fps are numbers in the manifest; the form stores them as
// strings, so keep a numeric list here and stringify at the option boundary.
function durationOptionsForModel(model) {
  const values = effectiveLimits(model)?.durations;
  return Array.isArray(values) && values.length ? values : durationFallback;
}

function fpsOptionsForModel(model) {
  const values = effectiveLimits(model)?.fps;
  return Array.isArray(values) && values.length ? values : fpsFallback;
}

const SORT_CHOICES = [
  ["used", "Recently used"],
  ["updated", "Recently updated"],
  ["name", "Name"],
  ["scope", "Scope"],
];

// Aspect ratios a general (model-agnostic) preset can pin. A general preset can't store a
// concrete WxH — resolution menus differ per model — so it stores a ratio and the Studio
// snaps it to the active model's nearest supported resolution at apply time (epic 11949).
const GENERAL_ASPECT_RATIOS = [
  ["1:1", "1:1 · Square"],
  ["3:2", "3:2 · Landscape"],
  ["2:3", "2:3 · Portrait"],
  ["16:9", "16:9 · Wide"],
  ["9:16", "9:16 · Tall"],
  ["4:3", "4:3"],
  ["3:4", "3:4"],
];

const scopeRank = { builtin: 0, global: 1, project: 2 };

// Desktop-safe discard confirmation for the editor's DESTRUCTIVE in-screen transitions
// (leave to the list, start/switch a preset, or the explicit Discard action) while the
// form carries unsaved edits. Routed through appConfirm — window.confirm silently no-ops
// inside the Tauri desktop WebView, so a raw confirm there would neither confirm nor
// cancel and the guard would be dead on desktop (sc-11968 / sc-11969).
const DISCARD_CONFIRM = {
  title: "Discard changes?",
  message: "This preset has unsaved changes that will be lost.",
  confirmLabel: "Discard changes",
  cancelLabel: "Keep editing",
  tone: "danger",
};

// The `defaults` keys this editor renders. Everything else a preset carries —
// upscale*, ipAdapterScale, controlnetScale, guidanceMethod, trueCfgScale, viewAngle,
// schedulerShift, precision, quantization, motion, the studios' literal `prompt`, … —
// passes through untouched on save. PATCH replaces `defaults` wholesale (manifest.rs
// `merge_object` is a top-level key replace), so rebuilding it from the form alone
// silently destroys everything the studios' Save-as-Preset wrote (sc-10548).
const EDITOR_OWNED_DEFAULTS = [
  "mode",
  "count",
  "steps",
  "guidanceScale",
  "duration",
  "fps",
  "quality",
  "resolution",
  "aspect",
  "negativePrompt",
  "sampler",
  "scheduler",
];

function segmentByKey(key) {
  return WORKFLOW_SEGMENTS.find((segment) => segment.key === key) ?? WORKFLOW_SEGMENTS[0];
}

// A stored default the selected model no longer lists (an in-the-wild preset, or a
// same-type model switch that hasn't cleared yet) still has to be visible and selected
// so the user sees what the preset carries. Flag it rather than blank the select; the
// save stays blocked (defaultValueErrors) until they pick a supported option. Returns
// null when the value is blank or already in the menu.
function outOfMenuOption(value, menu, format) {
  if (value === "" || menu.map(String).includes(String(value))) {
    return null;
  }
  return <option value={value}>{format(value)} — not in this model’s list</option>;
}

// Invert the (workflow, defaults.mode) pair back into the segment the editor shows.
function presetSegmentKey(preset) {
  if (preset?.workflow === "text_to_image" && preset?.defaults?.mode === "character_image") {
    return "character_image";
  }
  return preset?.workflow ?? "text_to_image";
}

function modelCapabilities(model) {
  return Array.isArray(model?.capabilities) ? model.capabilities : [];
}

function segmentAvailable(model, segment) {
  const caps = modelCapabilities(model);
  if (!caps.includes(segment.workflow)) {
    return false;
  }
  return segment.mode === segment.workflow || caps.includes(segment.mode);
}

function availableSegments(model) {
  return WORKFLOW_SEGMENTS.filter((segment) => segmentAvailable(model, segment));
}

function modelHelperLine(model) {
  if (!model) {
    return "";
  }
  const caps = modelCapabilities(model);
  const flags = CAPABILITY_FLAGS.filter(([id]) => caps.includes(id)).map(([, label]) => label);
  const size = downloadSizeText(model);
  return [...flags, size].filter(Boolean).join(" · ");
}

function qualityLabel(value) {
  return qualityChoices.find(([id]) => id === value)?.[1] ?? null;
}


function formLorasFromPreset(preset) {
  return presetLoras(preset)
    .map((lora) => {
      const id = presetLoraId(lora);
      return id ? { id, weight: typeof lora === "object" && lora.weight != null ? String(lora.weight) : "" } : null;
    })
    .filter(Boolean);
}

function numberFieldValue(value) {
  return value === null || value === undefined ? "" : String(value);
}

function formFromPreset(preset, fallbackModel) {
  const defaults = preset?.defaults ?? {};
  return {
    id: preset?.id ?? "",
    name: preset?.name ?? "",
    kind: preset?.kind === "general" ? "general" : "model",
    scope: preset?.scope === "project" ? "project" : "global",
    segment: presetSegmentKey(preset),
    model: preset?.model ?? fallbackModel ?? "",
    order: numberFieldValue(preset?.order),
    count: numberFieldValue(defaults.count),
    aspect: defaults.aspect ?? "",
    duration: numberFieldValue(defaults.duration),
    fps: numberFieldValue(defaults.fps),
    quality: defaults.quality ?? "",
    resolution: defaults.resolution ?? "",
    negativePrompt: defaults.negativePrompt ?? "",
    steps: numberFieldValue(defaults.steps),
    guidanceScale: numberFieldValue(defaults.guidanceScale),
    sampler: defaults.sampler ?? "",
    scheduler: defaults.scheduler ?? "",
    promptPrefix: preset?.prompt?.prefix ?? "",
    promptSuffix: preset?.prompt?.suffix ?? "",
    description: preset?.ui?.description ?? "",
    loras: formLorasFromPreset(preset),
  };
}

function loraLabel(lora) {
  return [lora.scope, lora.family ?? "compatible"].filter(Boolean).join(" | ");
}

// The knobs the backend range-checks (recipe_presets.rs validate_recipe_preset_defaults).
// Catching them here means the CTA explains itself instead of the save round-tripping
// into a 400. Blank always means "no default" and is dropped by buildPayload.
//
// `options` carries the menus the selected model actually honors. A stored value outside
// its menu is a default the model would silently clamp away, so it blocks the save until
// the user picks a supported one — the in-menu flag shows what's there but can't be kept.
// Exported for test (sc-19502): pure, dependency-free, and the only guard stopping a preset from
// storing a default the model's own engine would refuse on every job it drives.
export function defaultValueErrors(form, isVideo, options) {
  const errors = [];
  const checkNumber = (value, label, { min, max, integer = false }) => {
    if (value === "") {
      return;
    }
    const parsed = Number(value);
    if (!Number.isFinite(parsed) || parsed < min || parsed > max || (integer && !Number.isInteger(parsed))) {
      errors.push(`${label} must be ${integer ? "a whole number " : ""}between ${min} and ${max}.`);
    }
  };
  const checkInMenu = (value, label, menu) => {
    if (value === "" || !menu) {
      return;
    }
    if (!menu.map(String).includes(String(value))) {
      errors.push(`${label} ${value} isn't one this model supports — pick a listed option.`);
    }
  };
  checkNumber(form.count, "Variations", { min: 1, max: 8, integer: true });
  checkNumber(form.steps, "Steps", { min: 1, max: 200, integer: true });
  checkNumber(form.guidanceScale, "Guidance", { min: 0, max: 60 });
  // A distilled model renders at exactly one step count (sc-19502). The generic 1..200 range above
  // does not know that, so without this a preset could be SAVED with `steps: 30` for LTX-2.3 and
  // then 400 on every job it drove — a stored default guaranteed to fail, which is the same
  // unsatisfiable-control defect the Video Studio pin removes. Same shape as the fps menu check
  // below: the range bounds sanity, the menu bounds what the model can actually do.
  checkInMenu(form.steps, "Steps", options?.steps);
  checkInMenu(form.resolution, isVideo ? "Resolution" : "Aspect", options?.resolutions);
  if (isVideo) {
    checkNumber(form.duration, "Duration", { min: 1, max: 120 });
    checkNumber(form.fps, "Frames", { min: 1, max: 240 });
    checkInMenu(form.duration, "Duration", options?.durations);
    checkInMenu(form.fps, "Frames", options?.fps);
  }
  return errors;
}

export function PresetManagerScreen() {
  const {
    activeProject,
    createPreset,
    deletePreset,
    duplicatePreset,
    imageModels,
    loras = [],
    models: catalogModels = [],
    presets = [],
    updatePreset,
    videoModels,
    sendPresetToStudio,
    setActiveView,
  } = useAppStatic();
  const onOpenModels = () => setActiveView("Models");
  // `imageModels`/`videoModels` drop anything `installState: "missing"`, so they are the
  // set a user may PICK. A saved preset may legitimately pin a model this install hasn't
  // downloaded — resolve its name and capabilities from the full catalog so the editor
  // still shows the right Workflow options (the backend validates against the catalog too).
  const models = useMemo(() => [...imageModels, ...videoModels], [imageModels, videoModels]);
  const allModels = catalogModels.length ? catalogModels : models;

  const [editing, setEditing] = useState(false);
  const [selectedPresetId, setSelectedPresetId] = useState("");
  const selectedPreset = presets.find((preset) => preset.id === selectedPresetId) ?? null;
  const creating = editing && !selectedPreset;

  const [search, setSearch] = useState("");
  const [scopeFilter, setScopeFilter] = useState("all");
  const [typeFilter, setTypeFilter] = useState("all");
  const [sort, setSort] = useState("updated");

  const [form, setForm] = useState(() => formFromPreset(null, models[0]?.id));
  const [baseline, setBaseline] = useState(() => formFromPreset(null, models[0]?.id));
  const [saving, setSaving] = useState(false);
  const [message, setMessage] = useState({ tone: "neutral", text: "" });
  const [showLoraPicker, setShowLoraPicker] = useState(false);
  const [advancedOpen, setAdvancedOpen] = useState(false);
  const nameInputRef = useRef(null);

  const editable = !selectedPreset || selectedPreset.scope !== "builtin";
  const busy = saving;
  // General (model-agnostic) presets hide the model/workflow/LoRA/model-specific controls
  // and carry only prompt fragments + aspect + variations (epic 11949).
  const isGeneral = form.kind === "general";
  const selectedModel = allModels.find((model) => model.id === form.model) ?? null;
  const modelInstalled = models.some((model) => model.id === form.model);
  const segments = availableSegments(selectedModel);
  const activeSegment = segmentByKey(form.segment);
  const isVideo = activeSegment.type === "video";

  const installedLoras = loras.filter((lora) => lora.installState !== "missing");
  const availableLoras = selectedModel ? installedLoras.filter((lora) => loraMatchesModel(lora, selectedModel)) : [];
  const selectedIds = form.loras.map((lora) => lora.id);
  const addableLoras = availableLoras.filter((lora) => !selectedIds.includes(lora.id));
  const showLoraEmptyState = !availableLoras.length && !form.loras.length;

  const samplerMenu = selectedModel ? samplerOptionsFromModel(selectedModel) : [];
  const schedulerMenu = selectedModel ? schedulerOptionsFromModel(selectedModel) : [];

  // Resolution/duration/fps menus come from the selected model's effective limits, the
  // same source the studios read — so the editor can't offer a default the studio would
  // silently clamp away (sc-10589).
  const resolutionOptions = resolutionOptionsForModel(selectedModel, isVideo);
  const durationOptions = durationOptionsForModel(selectedModel);
  const fpsOptions = fpsOptionsForModel(selectedModel);
  // `steps` has NO fallback menu, unlike the three above: absent means "any count", which is every
  // model but the distilled ones (sc-19502).
  const defaultsOptions = { resolutions: resolutionOptions, durations: durationOptions, fps: fpsOptions, steps: stepsMenuFromModel(selectedModel) };

  const validation = presetValidation({ loras: form.loras }, loras, selectedModel);
  const valueErrors = defaultValueErrors(form, isVideo, defaultsOptions);
  const dirty = editing && !creating && JSON.stringify(form) !== JSON.stringify(baseline);

  // One summary gates Save and carries its reasons, so the disabled state and the messages
  // cannot drift (epic 10644, sc-10651). This generalizes the local sc-10500 split into the
  // shared core: name/model stay silent requirements; a read-only built-in, broken default
  // values (valueErrors, sc-10589), and the preset's own LoRA problems surface as chips.
  const saveDraft = useMemo(
    () => ({ editable, name: form.name, model: form.model, kind: form.kind }),
    [editable, form.name, form.model, form.kind],
  );
  const saveContext = useMemo(() => ({ validation, valueErrors }), [validation, valueErrors]);
  const saveValidity = useValidation(presetSaveValidation, saveDraft, saveContext);
  const canSave = saveValidity.ready;

  const hasPendingCompatibleLoras = Boolean(selectedModel) && loras.some((lora) => lora.installState === "missing" && loraMatchesModel(lora, selectedModel));
  const loraEmptyMessage = !selectedModel
    ? "No model selected"
    : !installedLoras.length
      ? "No uploaded LoRAs yet. Manage LoRAs on the Models page."
      : hasPendingCompatibleLoras
        ? "No installed compatible LoRAs. Imports appear here after the Queue completes."
        : `No installed LoRAs match ${selectedModel.name ?? selectedModel.id}.`;

  useEffect(() => {
    if (selectedPreset && !presets.some((preset) => preset.id === selectedPreset.id)) {
      setSelectedPresetId("");
      setEditing(false);
    }
    // Catalog refreshes replace preset objects; identity alone controls selection validity.
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [presets, selectedPreset?.id]);

  useEffect(() => {
    if (!editing) {
      return;
    }
    const next = formFromPreset(selectedPreset, models[0]?.id);
    setForm(next);
    setBaseline(next);
    setMessage({ tone: "neutral", text: "" });
    // Hydrate only on edit/preset identity changes; model/preset object refreshes must not erase edits.
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [editing, selectedPreset?.id]);

  // Model drives Workflow, not the other way round: when the chosen model can't serve the
  // current segment, fall back to the first one it does serve.
  useEffect(() => {
    if (!editing || !selectedModel) {
      return;
    }
    if (!segmentAvailable(selectedModel, activeSegment) && segments.length) {
      setForm((current) => ({ ...current, segment: segments[0].key }));
    }
    // Model identity and segment keys fully describe this correction without catalog object churn.
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [editing, selectedModel?.id, activeSegment.key, segments.length]);

  function updateField(field, value) {
    setForm((current) => {
      if (field === "name" && !selectedPreset) {
        return { ...current, name: value, id: slugifyPresetId(value) };
      }
      if (field === "model") {
        const nextModel = allModels.find((item) => item.id === value);
        const nextSegments = availableSegments(nextModel);
        const previousSegment = segmentByKey(current.segment);
        const nextSegment = nextSegments.some((segment) => segment.key === current.segment)
          ? previousSegment
          : (nextSegments[0] ?? previousSegment);
        const nextIsVideo = nextSegment.type === "video";
        // Only the value survives a model switch if the new model still lists it; an
        // out-of-menu default (blank untouched) would otherwise persist and be clamped.
        const keepIfListed = (val, menu) => (val !== "" && menu.map(String).includes(String(val)) ? val : "");
        const nextResolutions = resolutionOptionsForModel(nextModel, nextIsVideo);
        return {
          ...current,
          model: value,
          // Model drives Workflow: keep the segment when the new model still serves it,
          // else fall to the first one it does.
          segment: nextSegment.key,
          loras: current.loras.filter((selection) => {
            const lora = loras.find((item) => item.id === selection.id);
            return !lora || loraMatchesModel(lora, nextModel);
          }),
          // Sampler/scheduler menus are per-model; a value the new model doesn't offer
          // would sit invisibly in the form and persist on save.
          sampler: "",
          scheduler: "",
          // Image and video defaults don't transfer: an image resolution isn't in the
          // video resolution menu (and vice versa), and count/duration/fps are exclusive.
          // On a same-type switch keep what the new model still lists and drop the rest,
          // so switching to a model with a narrower menu clears a now-invalid default.
          ...(nextSegment.type === previousSegment.type
            ? nextIsVideo
              ? {
                  resolution: keepIfListed(current.resolution, nextResolutions),
                  duration: keepIfListed(current.duration, durationOptionsForModel(nextModel)),
                  fps: keepIfListed(current.fps, fpsOptionsForModel(nextModel)),
                }
              : { resolution: keepIfListed(current.resolution, nextResolutions) }
            : { resolution: "", count: "", duration: "", fps: "" }),
        };
      }
      return { ...current, [field]: value };
    });
  }

  function addLoraById(id) {
    if (!id) {
      return;
    }
    setForm((current) => {
      const hasLora = current.loras.some((lora) => lora.id === id);
      if (hasLora || current.loras.length >= MAX_PRESET_LORAS) {
        return current;
      }
      const source = loras.find((lora) => lora.id === id);
      // Through the shared resolver, not a hand-rolled copy of its precedence chain: this seeds
      // the SAME starting weight the studio pickers show, and can't drift from the default again.
      const weight = loraWeight(source);
      return { ...current, loras: [...current.loras, { id, weight: String(weight) }] };
    });
  }

  function removeLora(id) {
    setForm((current) => ({ ...current, loras: current.loras.filter((lora) => lora.id !== id) }));
  }

  function updateLoraWeight(id, weight) {
    setForm((current) => ({
      ...current,
      loras: current.loras.map((lora) => (lora.id === id ? { ...lora, weight } : lora)),
    }));
  }

  // A general (model-agnostic) preset carries only prompt fragments + aspect + variations +
  // negative. It never writes model/workflow/modes/loras — the backend rejects those on a
  // general preset (epic 11949). Defaults still spread-then-replace so unrendered keys a
  // studio snapshot wrote survive, and clearing a field really clears it.
  function buildGeneralPayload() {
    const defaults = { ...(selectedPreset?.defaults ?? {}) };
    for (const key of EDITOR_OWNED_DEFAULTS) {
      delete defaults[key];
    }
    if (form.aspect) {
      defaults.aspect = form.aspect;
    }
    if (form.count !== "") {
      defaults.count = Number(form.count);
    }
    if (form.negativePrompt.trim()) {
      defaults.negativePrompt = form.negativePrompt.trim();
    }
    const prompt = {};
    if (form.promptPrefix.trim()) {
      prompt.prefix = form.promptPrefix.trim();
    }
    if (form.promptSuffix.trim()) {
      prompt.suffix = form.promptSuffix.trim();
    }
    const payload = {
      id: slugifyPresetId(form.id || form.name),
      name: form.name.trim(),
      kind: "general",
      scope: form.scope,
      defaults,
      ui: { description: form.description.trim() },
    };
    if (form.order !== "") {
      payload.order = Number(form.order);
    }
    if (Object.keys(prompt).length) {
      payload.prompt = prompt;
    }
    return payload;
  }

  function buildPayload() {
    if (form.kind === "general") {
      return buildGeneralPayload();
    }
    const segment = segmentByKey(form.segment);
    // Start from the preset's stored defaults so unrendered keys survive; drop the ones
    // this form owns so clearing a field really clears it, then re-add from the form.
    const defaults = { ...(selectedPreset?.defaults ?? {}) };
    for (const key of EDITOR_OWNED_DEFAULTS) {
      delete defaults[key];
    }
    defaults.mode = segment.mode;
    const numberInto = (key, value) => {
      if (value !== "") {
        defaults[key] = Number(value);
      }
    };
    numberInto("steps", form.steps);
    numberInto("guidanceScale", form.guidanceScale);
    if (segment.type === "video") {
      numberInto("duration", form.duration);
      numberInto("fps", form.fps);
    } else {
      numberInto("count", form.count);
    }
    for (const key of ["quality", "resolution", "negativePrompt", "sampler", "scheduler"]) {
      if (form[key].trim()) {
        defaults[key] = form[key].trim();
      }
    }
    const prompt = {};
    if (form.promptPrefix.trim()) {
      prompt.prefix = form.promptPrefix.trim();
    }
    if (form.promptSuffix.trim()) {
      prompt.suffix = form.promptSuffix.trim();
    }
    const payload = {
      id: slugifyPresetId(form.id || form.name),
      name: form.name.trim(),
      scope: form.scope,
      workflow: segment.workflow,
      modes: workflowModes(segment.workflow),
      model: form.model,
      defaults,
      loras: form.loras.map((lora) => ({
        id: lora.id,
        weight: Number.isFinite(Number(lora.weight)) ? Number(lora.weight) : 1.0,
      })),
      ui: { description: form.description.trim() },
    };
    if (form.order !== "") {
      payload.order = Number(form.order);
    }
    if (Object.keys(prompt).length) {
      payload.prompt = prompt;
    }
    return payload;
  }

  async function savePreset(event) {
    event.preventDefault();
    if (!canSave) {
      // Submit via Enter can reach here past the disabled button. Show the first real
      // error; when it's only an unfilled field (a silent requirement), main's nudge.
      const [firstError] = saveValidity.surfaced;
      setMessage({ tone: "error", text: firstError?.message ?? "Name this preset before saving." });
      return;
    }
    setSaving(true);
    setMessage({ tone: "neutral", text: "" });
    try {
      const payload = buildPayload();
      if (selectedPreset) {
        await updatePreset(selectedPreset.id, payload, selectedPreset.scope);
        setBaseline(form);
        setMessage({ tone: "success", text: "Preset saved." });
      } else {
        const created = await createPreset(payload);
        setSelectedPresetId(created?.id ?? payload.id);
        setShowLoraPicker(false);
        setMessage({ tone: "success", text: "Preset created." });
      }
    } catch (err) {
      setMessage({ tone: "error", text: errorMessage(err, "Could not save this preset.") });
    } finally {
      setSaving(false);
    }
  }

  async function duplicateOne(preset) {
    setSaving(true);
    setMessage({ tone: "neutral", text: "" });
    try {
      const duplicated = await duplicatePreset(preset.id, preset.scope === "builtin" ? "global" : preset.scope);
      setSelectedPresetId(duplicated.id);
      setMessage({ tone: "success", text: `Duplicated "${preset.name ?? preset.id}".` });
    } catch (err) {
      setMessage({ tone: "error", text: errorMessage(err, "Could not duplicate this preset.") });
    } finally {
      setSaving(false);
    }
  }

  // DELETE archives the preset (`archived: true`) rather than erasing it, so the control
  // stays labelled Archive even though the design draws a trash glyph.
  async function archiveOne(preset) {
    if (preset.scope === "builtin") {
      return;
    }
    setSaving(true);
    setMessage({ tone: "neutral", text: "" });
    try {
      await deletePreset(preset.id, preset.scope);
      if (selectedPresetId === preset.id) {
        setSelectedPresetId("");
      }
      setMessage({ tone: "success", text: `Archived "${preset.name ?? preset.id}".` });
    } catch (err) {
      setMessage({ tone: "error", text: errorMessage(err, "Could not archive this preset.") });
    } finally {
      setSaving(false);
    }
  }

  // Guard a DESTRUCTIVE in-screen transition (leave to the list, start a new preset, or
  // switch to a different preset) that would silently drop an in-progress edit. A clean
  // form proceeds synchronously — the long-standing no-op path, so nothing that isn't
  // genuinely dirty ever prompts — while a dirty edit confirms first via the desktop-safe
  // appConfirm. Plain app navigation is deliberately NOT guarded here: keep-alive
  // (sc-11959) leaves this screen mounted, so an edit survives a nav round trip untouched
  // and a nav prompt would be misleading noise (sc-11969).
  function guardDiscard(proceed) {
    if (!dirty) {
      proceed();
      return;
    }
    appConfirm(DISCARD_CONFIRM).then((ok) => {
      if (ok) {
        proceed();
      }
    });
  }

  function startNewPreset() {
    guardDiscard(() => {
      setSelectedPresetId("");
      setShowLoraPicker(false);
      setAdvancedOpen(false);
      setEditing(true);
    });
  }

  function editPreset(preset) {
    guardDiscard(() => {
      setShowLoraPicker(false);
      setAdvancedOpen(false);
      setSelectedPresetId(preset.id);
      setEditing(true);
    });
  }

  function backToList() {
    guardDiscard(() => {
      setEditing(false);
      setShowLoraPicker(false);
      setMessage({ tone: "neutral", text: "" });
    });
  }

  // Explicit unsaved-state affordance paired with the "Unsaved changes" pill: revert the
  // in-progress edit to the last-saved baseline, confirmed the same desktop-safe way. Only
  // reachable while dirty (the button renders only then), so a clean form is a no-op.
  function discardChanges() {
    if (!dirty) {
      return;
    }
    appConfirm(DISCARD_CONFIRM).then((ok) => {
      if (ok) {
        setForm(baseline);
        setMessage({ tone: "neutral", text: "" });
      }
    });
  }

  const visiblePresets = useMemo(() => {
    const needle = search.trim().toLowerCase();
    const filtered = presets.filter((preset) => {
      if (scopeFilter !== "all" && (preset.scope ?? "global") !== scopeFilter) {
        return false;
      }
      // General presets have no workflow segment, so they only match the "General" and
      // "All" filters — never a specific workflow segment (which would falsely match the
      // "text_to_image" default presetSegmentKey returns for them).
      if (typeFilter !== "all") {
        if (typeFilter === "general") {
          if (preset.kind !== "general") {
            return false;
          }
        } else if (preset.kind === "general" || presetSegmentKey(preset) !== typeFilter) {
          return false;
        }
      }
      if (!needle) {
        return true;
      }
      const haystack = [
        preset.name,
        preset.id,
        preset.model,
        preset.ui?.description,
        preset.prompt?.prefix,
        preset.prompt?.suffix,
      ]
        .filter(Boolean)
        .join(" ")
        .toLowerCase();
      return haystack.includes(needle);
    });
    const byName = (a, b) => String(a.name ?? a.id).localeCompare(String(b.name ?? b.id));
    return filtered.sort((a, b) => {
      if (sort === "name") {
        return byName(a, b);
      }
      if (sort === "scope") {
        return (scopeRank[a.scope] ?? 1) - (scopeRank[b.scope] ?? 1) || byName(a, b);
      }
      if (sort === "used") {
        // Newest-used first; never-used presets (no lastUsedAt) sink to the bottom
        // because "" sorts after any real timestamp under a descending compare.
        return String(b.lastUsedAt ?? "").localeCompare(String(a.lastUsedAt ?? "")) || byName(a, b);
      }
      return String(b.updatedAt ?? "").localeCompare(String(a.updatedAt ?? "")) || byName(a, b);
    });
  }, [presets, search, scopeFilter, typeFilter, sort]);

  const globalCount = visiblePresets.filter((preset) => preset.scope === "global").length;

  if (editing) {
    return (
      <section className="page-frame preset-manager">
        <form className="preset-editor-form" onSubmit={savePreset}>
          <WorkPanel className="preset-editor-panel">
            {renderEditorHead()}
            {renderKind()}
            {renderIdentity()}
            {isGeneral ? (
              <>
                {renderPromptTemplate()}
                {renderGeneralDefaults()}
              </>
            ) : (
              <>
                {renderModel()}
                {renderWorkflow()}
                {renderPromptTemplate()}
                {renderDefaults()}
                {renderLoras()}
                <div className="preset-form-section">
                  <AdvancedSection
                    hint="cleared values → model default"
                    onToggle={() => setAdvancedOpen((value) => !value)}
                    open={advancedOpen}
                  >
                    {renderAdvanced()}
                  </AdvancedSection>
                </div>
              </>
            )}
            {renderFooter()}
          </WorkPanel>
        </form>
      </section>
    );
  }

  return (
    <section className="page-frame preset-manager">
      <WorkPanel className="preset-intro-panel">
        <div className="preset-intro">
          <div className="preset-intro-text">
            <p className="eyebrow">Your presets</p>
            <h2>Reusable generation setups</h2>
            <p>
              A preset bundles a model, a prompt scaffold, defaults, and LoRAs so you can drop it into{" "}
              <button className="link-button" onClick={() => setActiveView("Image")} type="button">
                Image Studio
              </button>{" "}
              in one click. Project presets stay with this workspace; global presets follow you everywhere.
            </p>
          </div>
          <button className="primary-action preset-new" onClick={startNewPreset} type="button">
            <Icon.Plus size={16} /> New preset
          </button>
        </div>

        <div className="work-panel-divider" />

        <div className="preset-toolbar">
          <span className="preset-search">
            <Icon.Search size={15} />
            <input
              aria-label="Search presets"
              onChange={(event) => setSearch(event.target.value)}
              placeholder="Search presets…"
              type="search"
              value={search}
            />
          </span>
          <div aria-label="Scope" className="segmented-control" role="group">
            {[["all", "All"], ["project", "This project"], ["global", "Global"]].map(([value, label]) => (
              <button
                aria-pressed={scopeFilter === value}
                className={scopeFilter === value ? "active" : ""}
                key={value}
                onClick={() => setScopeFilter(value)}
                type="button"
              >
                {label}
              </button>
            ))}
          </div>
          <select aria-label="Type" onChange={(event) => setTypeFilter(event.target.value)} value={typeFilter}>
            <option value="all">All types</option>
            <option value="general">General</option>
            {WORKFLOW_SEGMENTS.map((segment) => (
              <option key={segment.key} value={segment.key}>
                {segment.label}
              </option>
            ))}
          </select>
        </div>
      </WorkPanel>

      {message.text ? (
        <p className={message.tone === "success" ? "inline-success" : "inline-warning"}>{message.text}</p>
      ) : null}

      <section className="preset-results">
        <div className="preset-results-head">
          <div>
            <p className="eyebrow">Results</p>
            <h2>Saved presets</h2>
          </div>
          <span className="preset-results-count">
            {visiblePresets.length} preset{visiblePresets.length === 1 ? "" : "s"} · {globalCount} global
          </span>
          <select aria-label="Sort presets" onChange={(event) => setSort(event.target.value)} value={sort}>
            {SORT_CHOICES.map(([value, label]) => (
              <option key={value} value={value}>
                {label}
              </option>
            ))}
          </select>
        </div>

        {visiblePresets.length ? (
          <div className="preset-grid">{visiblePresets.map(renderPresetCard)}</div>
        ) : (
          <div className="empty-panel compact-panel">
            <span>{presets.length ? "No presets match these filters." : "No presets yet."}</span>
            <button onClick={startNewPreset} type="button">
              New preset
            </button>
          </div>
        )}
      </section>
    </section>
  );

  function renderPresetCard(preset) {

    return (

      <PresetCard

        key={`${preset.scope ?? "global"}-${preset.id}`}

        preset={preset}

        allModels={allModels}

        archiveOne={archiveOne}

        busy={busy}

        duplicateOne={duplicateOne}

        editPreset={editPreset}

        models={models}

        presetSegmentKey={presetSegmentKey}

        qualityLabel={qualityLabel}

        segmentByKey={segmentByKey}

        sendPresetToStudio={sendPresetToStudio}

      />

    );

  }

  function renderEditorHead() {

    return (

      <EditorHeadSection

        backToList={backToList}

        busy={busy}

        canSave={canSave}

        creating={creating}

        dirty={dirty}

        discardChanges={discardChanges}

      />

    );

  }

  function renderSectionHead(title, help, optional) {
    return (
      <div className="preset-form-section-head">
        <h3>
          {title}
          {optional ? <span className="preset-optional">optional</span> : null}
        </h3>
        {help ? <p>{help}</p> : null}
      </div>
    );
  }

  function renderKind() {

    return (

      <KindSection

        creating={creating}

        editable={editable}

        form={form}

        renderSectionHead={renderSectionHead}

        updateField={updateField}

      />

    );

  }

  function renderIdentity() {

    return (

      <IdentitySection

        activeProject={activeProject}

        creating={creating}

        editable={editable}

        form={form}

        nameInputRef={nameInputRef}

        renderSectionHead={renderSectionHead}

        updateField={updateField}

      />

    );

  }

  function renderModel() {

    return (

      <ModelSection

        editable={editable}

        form={form}

        imageModels={imageModels}

        modelHelperLine={modelHelperLine}

        modelInstalled={modelInstalled}

        models={models}

        renderSectionHead={renderSectionHead}

        selectedModel={selectedModel}

        updateField={updateField}

        videoModels={videoModels}

      />

    );

  }

  function renderWorkflow() {

    return (

      <WorkflowSection

        editable={editable}

        form={form}

        renderSectionHead={renderSectionHead}

        segments={segments}

        selectedModel={selectedModel}

        updateField={updateField}

      />

    );

  }

  function renderPromptTemplate() {

    return (

      <PromptTemplateSection

        editable={editable}

        form={form}

        isGeneral={isGeneral}

        renderSectionHead={renderSectionHead}

        updateField={updateField}

      />

    );

  }

  function renderDefaults() {

    return (

      <DefaultsSection

        durationOptions={durationOptions}

        editable={editable}

        form={form}

        fpsOptions={fpsOptions}

        isVideo={isVideo}

        outOfMenuOption={outOfMenuOption}

        renderSectionHead={renderSectionHead}

        resolutionOptions={resolutionOptions}

        updateField={updateField}

      />

    );

  }

  function renderGeneralDefaults() {

    return (

      <GeneralDefaultsSection

        GENERAL_ASPECT_RATIOS={GENERAL_ASPECT_RATIOS}

        editable={editable}

        form={form}

        renderSectionHead={renderSectionHead}

        updateField={updateField}

      />

    );

  }

  function renderLoras() {

    return (

      <LorasSection

        addLoraById={addLoraById}

        addableLoras={addableLoras}

        editable={editable}

        form={form}

        loraEmptyMessage={loraEmptyMessage}

        loraLabel={loraLabel}

        loras={loras}

        onOpenModels={onOpenModels}

        removeLora={removeLora}

        renderSectionHead={renderSectionHead}

        selectedModel={selectedModel}

        setShowLoraPicker={setShowLoraPicker}

        showLoraEmptyState={showLoraEmptyState}

        showLoraPicker={showLoraPicker}

        updateLoraWeight={updateLoraWeight}

      />

    );

  }

  function renderAdvanced() {

    return (

      <PresetAdvancedSection

        activeSegment={activeSegment}

        editable={editable}

        form={form}

        samplerMenu={samplerMenu}

        schedulerMenu={schedulerMenu}

        updateField={updateField}

      />

    );

  }

  function renderFooter() {

    return (

      <FooterSection

        backToList={backToList}

        busy={busy}

        canSave={canSave}

        creating={creating}

        form={form}

        isGeneral={isGeneral}

        isVideo={isVideo}

        message={message}

        saveValidity={saveValidity}

        selectedModel={selectedModel}

      />

    );

  }
}
