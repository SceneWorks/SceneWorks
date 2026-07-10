import React, { useEffect, useMemo, useRef, useState } from "react";
import { LoraKeywordSummary } from "../components/LoraKeywordSummary.jsx";
import { AdvancedSection } from "../components/AdvancedSection.jsx";
import { Icon } from "../components/Icons.jsx";
import { WorkPanel } from "../components/WorkPanel.jsx";
import {
  MAX_PRESET_LORAS,
  compactModeList,
  loraMatchesModel,
  presetLoraId,
  presetLoras,
  presetValidation,
  presetValidationMessage,
  workflowModes,
} from "../presetUtils.js";
import {
  SAMPLER_LABELS,
  SCHEDULER_LABELS,
  effectiveLimits,
  samplerOptionsFromModel,
  schedulerOptionsFromModel,
} from "../samplerOptions.js";
import { useAppStatic } from "../context/AppContext.js";
import { qualityChoices } from "../jobTypes.js";

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
  ["updated", "Recently updated"],
  ["name", "Name"],
  ["scope", "Scope"],
];

const scopeRank = { builtin: 0, global: 1, project: 2 };

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
  const size = model.downloadSizeLabel
    ? `${model.downloadSizeEstimated ? "~" : ""}${model.downloadSizeLabel}`
    : null;
  return [...flags, size].filter(Boolean).join(" · ");
}

function qualityLabel(value) {
  return qualityChoices.find(([id]) => id === value)?.[1] ?? null;
}

function slugify(value) {
  return String(value ?? "")
    .trim()
    .toLowerCase()
    .replace(/[^a-z0-9_-]+/g, "_")
    .replace(/^_+|_+$/g, "");
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
    scope: preset?.scope === "project" ? "project" : "global",
    segment: presetSegmentKey(preset),
    model: preset?.model ?? fallbackModel ?? "",
    order: numberFieldValue(preset?.order),
    count: numberFieldValue(defaults.count),
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
function defaultValueErrors(form, isVideo, options) {
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
  const defaultsOptions = { resolutions: resolutionOptions, durations: durationOptions, fps: fpsOptions };

  const validation = presetValidation({ loras: form.loras }, loras, selectedModel);
  const validationMessage = editable ? presetValidationMessage(validation) : "";
  const valueErrors = defaultValueErrors(form, isVideo, defaultsOptions);
  const dirty = editing && !creating && JSON.stringify(form) !== JSON.stringify(baseline);

  // Two kinds of "can't save" (sc-10500 split). A REQUIREMENT is an unfilled field the
  // form already marks — repeating it as a warning is noise. An ERROR is a value that is
  // present but broken, or a state the user can't see; it must say why the CTA is dead.
  const saveRequirement = !form.name.trim() ? "name" : !form.model ? "model" : "";
  const saveError = !editable
    ? "Built-in presets are read-only. Duplicate it to make an editable copy."
    : valueErrors.length
      ? valueErrors.join(" ")
      : validationMessage;
  const canSave = !saveRequirement && !saveError;

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
  }, [presets, selectedPreset?.id]);

  useEffect(() => {
    if (!editing) {
      return;
    }
    const next = formFromPreset(selectedPreset, models[0]?.id);
    setForm(next);
    setBaseline(next);
    setMessage({ tone: "neutral", text: "" });
  }, [selectedPreset?.id, editing]);

  // Model drives Workflow, not the other way round: when the chosen model can't serve the
  // current segment, fall back to the first one it does serve.
  useEffect(() => {
    if (!editing || !selectedModel) {
      return;
    }
    if (!segmentAvailable(selectedModel, activeSegment) && segments.length) {
      setForm((current) => ({ ...current, segment: segments[0].key }));
    }
  }, [editing, selectedModel?.id, activeSegment.key, segments.length]);

  function updateField(field, value) {
    setForm((current) => {
      if (field === "name" && !selectedPreset) {
        return { ...current, name: value, id: slugify(value) };
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
      const weight = source?.defaultWeight ?? source?.weight ?? 0.8;
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

  function buildPayload() {
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
      id: slugify(form.id || form.name),
      name: form.name.trim(),
      scope: form.scope,
      workflow: segment.workflow,
      modes: workflowModes(segment.workflow),
      model: form.model,
      defaults,
      loras: form.loras.map((lora) => ({
        id: lora.id,
        weight: Number.isFinite(Number(lora.weight)) ? Number(lora.weight) : 0.8,
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
      setMessage({ tone: "error", text: saveError || "Name this preset before saving." });
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
      setMessage({ tone: "error", text: err.message });
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
      setMessage({ tone: "error", text: err.message });
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
      setMessage({ tone: "error", text: err.message });
    } finally {
      setSaving(false);
    }
  }

  function startNewPreset() {
    setSelectedPresetId("");
    setShowLoraPicker(false);
    setAdvancedOpen(false);
    setEditing(true);
  }

  function editPreset(preset) {
    setShowLoraPicker(false);
    setAdvancedOpen(false);
    setSelectedPresetId(preset.id);
    setEditing(true);
  }

  function backToList() {
    setEditing(false);
    setShowLoraPicker(false);
    setMessage({ tone: "neutral", text: "" });
  }

  const visiblePresets = useMemo(() => {
    const needle = search.trim().toLowerCase();
    const filtered = presets.filter((preset) => {
      if (scopeFilter !== "all" && (preset.scope ?? "global") !== scopeFilter) {
        return false;
      }
      if (typeFilter !== "all" && presetSegmentKey(preset) !== typeFilter) {
        return false;
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
            {renderIdentity()}
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
    const segment = segmentByKey(presetSegmentKey(preset));
    const model = allModels.find((item) => item.id === preset.model);
    // The studio snaps its model back into the installed catalog, so launching a preset
    // pinned to an uninstalled model would silently land on a different one.
    const runnable = models.some((item) => item.id === preset.model);
    const scope = preset.scope ?? "global";
    const builtin = scope === "builtin";
    const defaults = preset.defaults ?? {};
    const loraCount = presetLoras(preset).length;
    const prefix = preset.prompt?.prefix ?? "";
    const suffix = preset.prompt?.suffix ?? "";

    const chips = [defaults.resolution ? defaults.resolution.replace("x", " × ") : "Any size"];
    if (segment.type === "video") {
      chips.push(defaults.duration ? `${defaults.duration}s` : "Default length");
      chips.push(defaults.fps ? `${defaults.fps} fps` : "Default fps");
    } else {
      chips.push(defaults.count ? `${defaults.count} variations` : "Default count");
    }
    chips.push(qualityLabel(defaults.quality) ?? "Default quality");

    return (
      <article className="preset-card" key={`${scope}-${preset.id}`}>
        <div className="preset-card-head">
          <div className="preset-card-title">
            <strong>{preset.name ?? preset.id}</strong>
            <div>
              {segment.label} · <span className="preset-card-model">{model?.name ?? preset.model}</span>
            </div>
          </div>
          <span className={scope === "global" ? "preset-scope-chip global" : "preset-scope-chip"}>{scope}</span>
        </div>

        <div className="preset-card-preview">
          {prefix ? <span className="faint">{prefix} </span> : null}
          <span className="token">your prompt</span>
          {suffix ? <span className="faint"> {suffix}</span> : null}
        </div>

        <div className="preset-card-chips">
          {chips.map((chip) => (
            <span className="chip" key={chip}>
              {chip}
            </span>
          ))}
          <span className={loraCount ? "chip accent" : "chip"}>
            {loraCount ? `${loraCount} LoRA${loraCount === 1 ? "" : "s"}` : "No LoRAs"}
          </span>
        </div>

        <div className="preset-card-actions">
          <button
            className="primary-action preset-card-use"
            disabled={busy || !runnable}
            onClick={() => sendPresetToStudio(preset)}
            title={runnable ? undefined : `Install ${model?.name ?? preset.model} to use this preset`}
            type="button"
          >
            <Icon.Play size={13} /> Use in Studio
          </button>
          {builtin ? null : (
            <button className="secondary-action" onClick={() => editPreset(preset)} type="button">
              <Icon.Pencil size={13} /> Edit
            </button>
          )}
          <span className="preset-card-spacer" />
          <button
            aria-label={`Duplicate ${preset.name ?? preset.id}`}
            className="preset-icon-button"
            disabled={busy}
            onClick={() => duplicateOne(preset)}
            title="Duplicate"
            type="button"
          >
            <Icon.Duplicate size={14} />
          </button>
          {builtin ? null : (
            <button
              aria-label={`Archive ${preset.name ?? preset.id}`}
              className="preset-icon-button danger"
              disabled={busy}
              onClick={() => archiveOne(preset)}
              title="Archive"
              type="button"
            >
              <Icon.Trash size={14} />
            </button>
          )}
        </div>
      </article>
    );
  }

  function renderEditorHead() {
    const statusPill = creating ? (
      <span className="preset-status-pill">Draft</span>
    ) : dirty ? (
      <span className="preset-status-pill warn">
        <span className="dot" aria-hidden="true" />
        Unsaved changes
      </span>
    ) : (
      <span className="preset-status-pill">Saved</span>
    );

    return (
      <div className="preset-editor-head">
        <div className="preset-editor-context">
          <button className="preset-back" onClick={backToList} type="button">
            <Icon.ArrowLeft size={14} /> All presets
          </button>
          <p className="eyebrow preset-editor-eyebrow">{creating ? "New preset" : "Edit preset"}</p>
        </div>
        <div className="preset-editor-actions">
          {statusPill}
          <button className="secondary-action" onClick={backToList} type="button">
            Cancel
          </button>
          <button className="primary-action" disabled={!canSave || busy} type="submit">
            <Icon.Save size={15} />
            <span>{busy ? "Saving…" : creating ? "Create preset" : "Save preset"}</span>
          </button>
        </div>
      </div>
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

  function renderIdentity() {
    return (
      <div className="preset-form-section">
        {renderSectionHead("Identity", "Name this preset and choose where it lives.")}
        <div className="preset-identity-row">
          <div className="preset-name-field">
            {creating ? (
              <label className="field field-name">
                <span>Name</span>
                <input
                  onChange={(event) => updateField("name", event.target.value)}
                  placeholder="Name this preset…"
                  ref={nameInputRef}
                  required
                  value={form.name}
                />
              </label>
            ) : (
              <label className="field field-name preset-name-saved">
                <span>Name</span>
                <span className="preset-name-saved-row">
                  <input
                    disabled={!editable}
                    onChange={(event) => updateField("name", event.target.value)}
                    ref={nameInputRef}
                    required
                    value={form.name}
                  />
                  <button
                    className="preset-icon-button"
                    disabled={!editable}
                    onClick={() => nameInputRef.current?.focus()}
                    title="Rename preset"
                    type="button"
                  >
                    <Icon.Pencil size={14} />
                  </button>
                </span>
              </label>
            )}
          </div>
          <label className="field preset-scope-field">
            <span>Scope</span>
            <div aria-label="Scope" className="scope-segment" role="radiogroup">
              <button
                aria-checked={form.scope === "project"}
                className={form.scope === "project" ? "active" : ""}
                disabled={!activeProject || !editable}
                onClick={() => updateField("scope", "project")}
                role="radio"
                type="button"
              >
                <Icon.Folder size={14} /> This project
              </button>
              <button
                aria-checked={form.scope === "global"}
                className={form.scope === "global" ? "active" : ""}
                disabled={!editable}
                onClick={() => updateField("scope", "global")}
                role="radio"
                type="button"
              >
                <Icon.Stars size={14} /> Global
              </button>
            </div>
          </label>
        </div>
        <div className="preset-identity-meta">
          <label className="field">
            <span>ID</span>
            <input
              disabled={!creating || !editable}
              onChange={(event) => updateField("id", event.target.value)}
              placeholder="auto-generated from name"
              required
              value={form.id}
            />
          </label>
          <label className="field">
            <span>Description</span>
            <input
              disabled={!editable}
              onChange={(event) => updateField("description", event.target.value)}
              placeholder="One line — what kind of shot this makes"
              value={form.description}
            />
          </label>
        </div>
      </div>
    );
  }

  function renderModel() {
    return (
      <div className="preset-form-section">
        {renderSectionHead("Base model", "Which checkpoint this preset renders with.")}
        <div className="preset-model-grid">
          <label className="field">
            <span>Model</span>
            <select
              disabled={!editable}
              onChange={(event) => updateField("model", event.target.value)}
              value={form.model}
            >
              {form.model ? null : <option value="">{models.length ? "Select a model…" : "No models installed"}</option>}
              {/* A preset can be pinned to a model this install hasn't downloaded. Say so,
                  rather than rendering an empty select that silently rewrites it on save. */}
              {form.model && !modelInstalled ? (
                <option value={form.model}>{selectedModel?.name ?? form.model} — not installed</option>
              ) : null}
              <optgroup label="Image">
                {imageModels.map((model) => (
                  <option key={model.id} value={model.id}>
                    {model.name ?? model.id}
                  </option>
                ))}
              </optgroup>
              <optgroup label="Video">
                {videoModels.map((model) => (
                  <option key={model.id} value={model.id}>
                    {model.name ?? model.id}
                  </option>
                ))}
              </optgroup>
            </select>
          </label>
          <span className="preset-model-meta-line">{modelHelperLine(selectedModel)}</span>
        </div>
      </div>
    );
  }

  function renderWorkflow() {
    return (
      <div className="preset-form-section">
        {renderSectionHead("Workflow", "Generation modes the selected model supports.")}
        {segments.length ? (
          <div aria-label="Workflow" className="segmented-control preset-workflow" role="radiogroup">
            {segments.map((segment) => (
              <button
                aria-checked={form.segment === segment.key}
                className={form.segment === segment.key ? "active" : ""}
                disabled={!editable}
                key={segment.key}
                onClick={() => updateField("segment", segment.key)}
                role="radio"
                type="button"
              >
                {segment.label}
              </button>
            ))}
          </div>
        ) : (
          <p className="inline-warning">
            {selectedModel?.name ?? "This model"} declares no preset-capable workflow. Pick another model.
          </p>
        )}
      </div>
    );
  }

  function renderPromptTemplate() {
    return (
      <div className="preset-form-section">
        {renderSectionHead(
          "Prompt template",
          "Text wrapped around whatever you type in the Studio. The preview updates as you edit.",
        )}
        <div className="preset-prompt-grid">
          <label className="field">
            <span>Prefix</span>
            <textarea
              disabled={!editable}
              onChange={(event) => updateField("promptPrefix", event.target.value)}
              placeholder="e.g. Cinematic 35mm, warm tungsten,"
              rows={2}
              value={form.promptPrefix}
            />
          </label>
          <label className="field">
            <span>Suffix</span>
            <textarea
              disabled={!editable}
              onChange={(event) => updateField("promptSuffix", event.target.value)}
              placeholder="e.g. shallow depth of field, neutral grade"
              rows={2}
              value={form.promptSuffix}
            />
          </label>
        </div>
        <div className="preset-live-preview">
          <span className="eyebrow">Live preview</span>
          <p>
            {form.promptPrefix.trim() ? <span className="prefix">{form.promptPrefix.trim()} </span> : null}
            <span className="token">your prompt</span>
            {form.promptSuffix.trim() ? <span className="suffix"> {form.promptSuffix.trim()}</span> : null}
          </p>
        </div>
      </div>
    );
  }

  function renderDefaults() {
    return (
      <div className="preset-form-section">
        {renderSectionHead("Defaults", "The settings the Studio starts on. All still editable per run.")}
        <div className="preset-defaults-grid">
          <label className="field">
            <span>{isVideo ? "Resolution" : "Aspect"}</span>
            <select disabled={!editable} onChange={(event) => updateField("resolution", event.target.value)} value={form.resolution}>
              <option value="">No default</option>
              {outOfMenuOption(form.resolution, resolutionOptions, (value) => value.replace("x", " × "))}
              {resolutionOptions.map((value) => (
                <option key={value} value={value}>
                  {value.replace("x", " × ")}
                </option>
              ))}
            </select>
          </label>
          {isVideo ? (
            <>
              <label className="field">
                <span>Duration</span>
                <select disabled={!editable} onChange={(event) => updateField("duration", event.target.value)} value={form.duration}>
                  <option value="">No default</option>
                  {outOfMenuOption(form.duration, durationOptions, (value) => `${value}s`)}
                  {durationOptions.map((d) => (
                    <option key={d} value={String(d)}>
                      {d}s
                    </option>
                  ))}
                </select>
              </label>
              <label className="field">
                <span>Frames</span>
                <select disabled={!editable} onChange={(event) => updateField("fps", event.target.value)} value={form.fps}>
                  <option value="">No default</option>
                  {outOfMenuOption(form.fps, fpsOptions, (value) => `${value} fps`)}
                  {fpsOptions.map((f) => (
                    <option key={f} value={String(f)}>
                      {f} fps
                    </option>
                  ))}
                </select>
              </label>
            </>
          ) : (
            <label className="field">
              <span>Variations</span>
              <select disabled={!editable} onChange={(event) => updateField("count", event.target.value)} value={form.count}>
                <option value="">No default</option>
                {[1, 2, 3, 4, 6, 8].map((n) => (
                  <option key={n} value={String(n)}>
                    {n}
                  </option>
                ))}
              </select>
            </label>
          )}
          <label className="field preset-quality-field">
            <span>Quality</span>
            <div aria-label="Quality" className="quality-pick" role="radiogroup">
              {qualityChoices.map(([value, label]) => (
                <button
                  aria-checked={form.quality === value}
                  className={form.quality === value ? "active" : ""}
                  disabled={!editable}
                  key={value}
                  onClick={() => updateField("quality", form.quality === value ? "" : value)}
                  role="radio"
                  type="button"
                >
                  {label}
                </button>
              ))}
            </div>
          </label>
        </div>
      </div>
    );
  }

  function renderLoras() {
    return (
      <div className="preset-form-section">
        {renderSectionHead(
          "LoRAs",
          `Up to ${MAX_PRESET_LORAS} fine-tunes layered with this preset, compatible with ${selectedModel?.name ?? "the chosen model"}.`,
          true,
        )}
        <section aria-label="Preset LoRAs" className="lora-stack">
          {form.loras.map((selected) => {
            const lora = loras.find((item) => item.id === selected.id);
            const missing = !lora || lora.installState === "missing";
            const incompatible = lora && selectedModel && !loraMatchesModel(lora, selectedModel);
            const name = lora?.name ?? selected.id;
            const weight = Number(selected.weight);
            return (
              <div className={missing || incompatible ? "lora-slot warning" : "lora-slot"} key={selected.id}>
                <div className="lora-slot-head">
                  <span className="lora-slot-meta">
                    <strong>{name}</strong>
                    <small>
                      {missing
                        ? "Missing or still importing"
                        : incompatible
                          ? `${loraLabel(lora)} | incompatible with ${selectedModel?.name ?? selectedModel?.id}`
                          : loraLabel(lora)}
                    </small>
                  </span>
                  <button
                    aria-label={`Remove ${name}`}
                    className="lora-slot-remove"
                    disabled={!editable}
                    onClick={() => removeLora(selected.id)}
                    title="Remove"
                    type="button"
                  >
                    ×
                  </button>
                </div>
                <LoraKeywordSummary lora={lora} />
                <div className="lora-slot-weight">
                  <label>
                    <span>Weight</span>
                    <span className="lora-slot-weight-value">{Number.isFinite(weight) ? weight.toFixed(2) : "—"}</span>
                  </label>
                  {/* -2..2 is the range the preset normalizer accepts, wider than the
                      studios' 0..2 slider — a preset may carry a negative weight. */}
                  <input
                    aria-label={`${name} weight`}
                    disabled={!editable || missing || incompatible}
                    max="2"
                    min="-2"
                    onChange={(event) => updateLoraWeight(selected.id, event.target.value)}
                    step="0.05"
                    type="range"
                    value={Number.isFinite(weight) ? weight : 0}
                  />
                </div>
              </div>
            );
          })}

          {showLoraEmptyState ? (
            <div className="empty-panel compact-panel">
              <span>{loraEmptyMessage}</span>
              <button onClick={onOpenModels} type="button">
                Open Models
              </button>
            </div>
          ) : null}

          {form.loras.length < MAX_PRESET_LORAS ? (
            showLoraPicker ? (
              <div className="lora-picker-panel">
                <strong>Pick a LoRA</strong>
                {addableLoras.length ? (
                  <div className="lora-picker-list">
                    {addableLoras.map((lora) => (
                      <button
                        className="lora-pick-row"
                        key={lora.id}
                        onClick={() => {
                          addLoraById(lora.id);
                          setShowLoraPicker(false);
                        }}
                        type="button"
                      >
                        <span>
                          <strong>{lora.name ?? lora.id}</strong> {lora.family ? <span className="chip">{lora.family}</span> : null}
                        </span>
                        <Icon.Plus size={14} />
                      </button>
                    ))}
                  </div>
                ) : (
                  <p className="lora-pick-empty">{loraEmptyMessage}</p>
                )}
                <div className="lora-picker-actions">
                  <button onClick={() => setShowLoraPicker(false)} type="button">
                    Cancel
                  </button>
                </div>
              </div>
            ) : (
              <button
                className="lora-add"
                data-count={`· ${addableLoras.length} available`}
                disabled={!editable || !addableLoras.length}
                onClick={() => setShowLoraPicker(true)}
                type="button"
              >
                <Icon.Plus size={15} />
                <span>Add LoRA</span>
              </button>
            )
          ) : null}
        </section>
      </div>
    );
  }

  function renderAdvanced() {
    return (
      <div className="advanced-panel">
        <label className="field prompt-field">
          <span>Negative prompt</span>
          <input
            disabled={!editable}
            onChange={(event) => updateField("negativePrompt", event.target.value)}
            placeholder="oversaturated, hands, text, watermark"
            value={form.negativePrompt}
          />
        </label>
        <label className="field">
          <span>Steps</span>
          <input
            disabled={!editable}
            max="200"
            min="1"
            onChange={(event) => updateField("steps", event.target.value)}
            placeholder="model default"
            type="number"
            value={form.steps}
          />
        </label>
        <label className="field">
          <span>Guidance</span>
          <input
            disabled={!editable}
            max="60"
            min="0"
            onChange={(event) => updateField("guidanceScale", event.target.value)}
            placeholder="model default"
            step="0.1"
            type="number"
            value={form.guidanceScale}
          />
        </label>
        {samplerMenu.length > 1 ? (
          <label className="field">
            <span>Sampler</span>
            <select disabled={!editable} onChange={(event) => updateField("sampler", event.target.value)} value={form.sampler}>
              <option value="">No default</option>
              {samplerMenu.map((value) => (
                <option key={value} value={value}>
                  {SAMPLER_LABELS[value] ?? value}
                </option>
              ))}
            </select>
          </label>
        ) : null}
        {schedulerMenu.length > 1 ? (
          <label className="field">
            <span>Scheduler</span>
            <select disabled={!editable} onChange={(event) => updateField("scheduler", event.target.value)} value={form.scheduler}>
              <option value="">No default</option>
              {schedulerMenu.map((value) => (
                <option key={value} value={value}>
                  {SCHEDULER_LABELS[value] ?? value}
                </option>
              ))}
            </select>
          </label>
        ) : null}
        <label className="field">
          <span>Sort order</span>
          <input
            disabled={!editable}
            onChange={(event) => updateField("order", event.target.value)}
            placeholder="0"
            type="number"
            value={form.order}
          />
        </label>
        <label className="field">
          <span>Derived modes</span>
          <input disabled readOnly value={compactModeList(activeSegment.workflow)} />
        </label>
      </div>
    );
  }

  function renderFooter() {
    const loraCount = form.loras.length;
    const chips = [
      selectedModel?.name ?? form.model,
      form.resolution ? form.resolution.replace("x", " × ") : "Any size",
      isVideo
        ? form.duration
          ? `${form.duration}s`
          : "Default length"
        : form.count
          ? `${form.count} variations`
          : "Default count",
      loraCount ? `${loraCount} LoRA${loraCount === 1 ? "" : "s"}` : "No LoRAs",
    ].filter(Boolean);

    return (
      <>
        {saveError ? <p className="inline-warning">{saveError}</p> : null}
        {message.text ? (
          <p className={message.tone === "success" ? "inline-success" : "inline-warning"}>{message.text}</p>
        ) : null}
        <div className="preset-recipe-footer">
          <div className="preset-recipe-chips">
            <span className="eyebrow">Recipe</span>
            {chips.map((chip, index) => (
              <span className={index === 0 ? "chip accent" : "chip"} key={chip}>
                {chip}
              </span>
            ))}
          </div>
          <span className="preset-card-spacer" />
          <button className="secondary-action" onClick={backToList} type="button">
            Cancel
          </button>
          <button className="primary-action" disabled={!canSave || busy} type="submit">
            <Icon.Save size={15} />
            <span>{busy ? "Saving…" : creating ? "Create preset" : "Save preset"}</span>
          </button>
        </div>
      </>
    );
  }
}
