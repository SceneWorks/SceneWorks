import React, { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { LoraKeywordSummary } from "../components/LoraKeywordSummary.jsx";
import { Icon } from "../components/Icons.jsx";
import { terminalStatuses } from "../jobTypes.js";
import {
  LORA_WEIGHT_MAX,
  LORA_WEIGHT_MIN,
  LORA_WEIGHT_STEP,
  MAX_USER_JOB_LORAS,
  applyPresetDefault,
  buildStudioPresetPayload,
  clearPresetDefault,
  loraHasResolvableFamily,
  loraMatchesModel,
  loraWeight,
  noPresetId,
  presetLoraSeedEntries,
  presetLoraDetails as buildPresetLoraDetails,
  presetMatchesModel,
  presetMatchesWorkflow,
  presetNameTaken,
  presetPromptParts as buildPresetPromptParts,
  presetValidation,
  slugifyPresetId,
} from "../presetUtils.js";
import { savePresetDialogValidation } from "../generationValidation.js";
import { useValidation } from "../validation/useValidation.js";
import { ValidationSummary } from "../validation/Validation.jsx";

const completedResultFallbackMs = 30000;

// Cmd/Ctrl+Enter submits the studio form from the prompt textarea.
export function onPromptKeyDown(event) {
  if ((event.metaKey || event.ctrlKey) && event.key === "Enter") {
    event.preventDefault();
    event.currentTarget.form?.requestSubmit();
  }
}

function jobCreatedMs(job) {
  const parsed = Date.parse(job?.createdAt ?? "");
  return Number.isFinite(parsed) ? parsed : 0;
}

function completedAnchorMs(job) {
  return Date.parse(job.completedAt ?? job.updatedAt ?? "");
}

// Pick the runs that belong in a studio's live stack from the tracked set, shared
// by every studio so stacking behaves identically. Runs are kept in the order they
// arrive (callers order oldest-first, so the active run sits on top and queued runs
// follow). Rules:
//   - canceled runs drop immediately (no output to show),
//   - running and queued runs always stack,
//   - a finished run slides out the moment a strictly-newer run starts (leaves the
//     queue), so the next run takes its place,
//   - a finished run with a run still queued behind it stays on top until that run
//     starts.
// `resultRendered(job)` reports whether a lone completed run's output has surfaced
// elsewhere (e.g. the Image studio's latest-batch grid); when it has, the run
// collapses out of the stack. Omit it for studios whose output ships in the job
// result itself (documents), where a lone completed run simply stays as the output.
export function selectStackedJobs(trackedLocalJobs, resultRendered) {
  const successorStarted = (job) =>
    trackedLocalJobs.some(
      (other) =>
        other.id !== job.id &&
        other.status !== "queued" &&
        other.status !== "canceled" &&
        jobCreatedMs(other) > jobCreatedMs(job),
    );
  const hasPendingSuccessor = (job) =>
    trackedLocalJobs.some(
      (other) => other.id !== job.id && other.status !== "canceled" && jobCreatedMs(other) > jobCreatedMs(job),
    );
  return trackedLocalJobs.filter((job) => {
    if (job.status === "canceled") {
      return false;
    }
    if (!terminalStatuses.has(job.status)) {
      return true;
    }
    if (successorStarted(job)) {
      return false;
    }
    if (job.status === "completed") {
      if (hasPendingSuccessor(job)) {
        return true;
      }
      return resultRendered ? !resultRendered(job) : true;
    }
    // Failed/interrupted runs with nothing started behind them stay visible so the
    // outcome is clear until the user moves on.
    return true;
  });
}

// Shared state/derivations for the Image and Video studios: preset selection and
// validation, the catalog-driven model/character resets, and the completed-job
// "keep the progress card until the asset renders" machinery. The studios keep
// their own divergent pieces (preset-default field application, launch-request
// handling, submit payloads) and pass the bits this hook needs as arguments.
export function useGenerationStudio({
  mode,
  presets,
  selectedModel,
  loras,
  models,
  model,
  setModel,
  fallbackModelId,
  characters,
  characterId,
  setCharacterId,
  setCharacterLookId,
  assets,
  latestAssets,
  trackedLocalJobs,
  initialPresetId = null,
  // sc-4196: LoRA selection state + validation, formerly duplicated in both studios.
  // Seeded from the persisted studio snapshot; advancedOpen/setAdvancedOpen are the
  // studio's own advanced-panel toggle (the hook auto-opens it when an incompatible
  // LoRA is selected so the blocking warning is visible).
  advancedOpen = false,
  setAdvancedOpen = () => {},
  initialSelectedLoraIds = [],
  initialLoraWeights = {},
  initialShowIncompatibleLoras = false,
  // The general-preset stack (epic 11949): an ordered set of model-agnostic presets layered
  // on top of whatever model + mode the studio is on. Separate from the base model preset
  // (selectedPresetId) — model presets stay single-select, general presets stack.
  initialGeneralStackIds = [],
}) {
  const [selectedPresetId, setSelectedPresetId] = useState(initialPresetId);
  const [generalStackIds, setGeneralStackIds] = useState(initialGeneralStackIds);
  const [resultFallbackTick, setResultFallbackTick] = useState(0);
  const [selectedLoraIds, setSelectedLoraIds] = useState(initialSelectedLoraIds);
  const [loraWeights, setLoraWeights] = useState(initialLoraWeights);
  const [showIncompatibleLoras, setShowIncompatibleLoras] = useState(initialShowIncompatibleLoras);

  // Snap the model back into range when the catalog changes out from under it.
  useEffect(() => {
    if (!models.some((item) => item.id === model)) {
      setModel(models[0]?.id ?? fallbackModelId);
    }
  }, [models, model, setModel, fallbackModelId]);

  // Drop a character selection that's no longer in the catalog.
  useEffect(() => {
    if (characterId && !characters.some((character) => character.id === characterId)) {
      setCharacterId("");
      setCharacterLookId("");
    }
  }, [characters, characterId, setCharacterId, setCharacterLookId]);

  // The base slot: model presets that match the current workflow + model, single-select
  // exactly as before. General presets are excluded here — they live in their own stack.
  const availablePresets = useMemo(
    () =>
      presets.filter(
        (preset) =>
          preset.kind !== "general" &&
          presetMatchesWorkflow(preset, mode) &&
          presetMatchesModel(preset, selectedModel, models),
      ),
    [mode, presets, selectedModel?.id, models],
  );
  // General (model-agnostic) presets are available on every model in every mode.
  const availableGeneralPresets = useMemo(
    () => presets.filter((preset) => preset.kind === "general"),
    [presets],
  );
  const availableGeneralKey = useMemo(
    () => availableGeneralPresets.map((preset) => preset.id).join("|"),
    [availableGeneralPresets],
  );
  // The active general stack, in selection order. Drops ids no longer in the catalog.
  const generalStack = useMemo(
    () => generalStackIds.map((id) => availableGeneralPresets.find((preset) => preset.id === id)).filter(Boolean),
    [generalStackIds, availableGeneralPresets],
  );
  // sc-5875: presets are opt-in. With no explicit selection (fresh screen / None),
  // resolve to no preset so an unchosen preset's LoRA/resolution/prompt are never
  // silently applied. The dropdown's "None" and the applied config stay in agreement.
  const selectedPreset =
    selectedPresetId && selectedPresetId !== noPresetId
      ? availablePresets.find((preset) => preset.id === selectedPresetId) ?? null
      : null;
  const presetPromptParts = buildPresetPromptParts(selectedPreset);
  const presetLoraDetails = buildPresetLoraDetails(selectedPreset, loras);
  const presetValidationResult = useMemo(
    () => presetValidation(selectedPreset, loras, selectedModel),
    [selectedPreset, loras, selectedModel],
  );

  // An explicitly chosen preset that drops out of the available set falls back to
  // the first available preset (or None) rather than showing stale config.
  useEffect(() => {
    if (!selectedPresetId || selectedPresetId === noPresetId) {
      return;
    }
    if (!selectedPreset) {
      setSelectedPresetId(availablePresets[0]?.id ?? noPresetId);
    }
  }, [availablePresets, selectedPresetId, selectedPreset]);

  // Prune stacked general presets that leave the catalog (archived/deleted). Unlike the
  // base slot this drops silently — a stack has no single "fall back to the first" notion.
  useEffect(() => {
    setGeneralStackIds((ids) => {
      const next = ids.filter((id) => availableGeneralPresets.some((preset) => preset.id === id));
      return next.length === ids.length ? ids : next;
    });
  }, [availableGeneralKey]);

  // Add/remove a general preset from the stack. Toggling never touches the model, mode, or
  // the LoRA picker — general presets carry none of those. Composition into the prompt is
  // the studio's job (epic 11949 Phase 4).
  const toggleGeneralPreset = useCallback((id) => {
    setGeneralStackIds((ids) => (ids.includes(id) ? ids.filter((existing) => existing !== id) : [...ids, id]));
  }, []);

  const resultVisible = useCallback((job) => {
    if (job.result?.generationSetId) {
      return latestAssets.some((asset) => asset.generationSetId === job.result.generationSetId);
    }
    const assetIds = job.result?.assetIds ?? [];
    return assetIds.length > 0 && assetIds.every((id) => assets.some((asset) => asset.id === id));
  }, [assets, latestAssets]);

  const completedWaitExpired = useCallback((job, nowMs = Date.now()) => {
    const anchorMs = completedAnchorMs(job);
    return Number.isFinite(anchorMs) && nowMs - anchorMs > completedResultFallbackMs;
  }, []);

  // A completed job's assets can lag its SSE result by a beat; keep the progress
  // card until the asset renders or the fallback window expires, re-checking on a
  // timer so a card never lingers forever when the asset never arrives.
  useEffect(() => {
    const nowMs = Date.now();
    const pendingCompletedJobs = trackedLocalJobs.filter(
      (job) =>
        job.status === "completed" &&
        Number.isFinite(completedAnchorMs(job)) &&
        !resultVisible(job) &&
        !completedWaitExpired(job, nowMs),
    );
    if (!pendingCompletedJobs.length) {
      return undefined;
    }
    const nextDelay = Math.min(
      ...pendingCompletedJobs.map((job) => Math.max(0, completedResultFallbackMs - (nowMs - completedAnchorMs(job)))),
    );
    const timer = window.setTimeout(() => setResultFallbackTick((value) => value + 1), nextDelay + 50);
    return () => window.clearTimeout(timer);
  }, [trackedLocalJobs, resultVisible, completedWaitExpired, resultFallbackTick]);

  // The visible stack (see selectStackedJobs). A lone completed run collapses once
  // its batch renders in the latest-batch grid or the SSE-lag window expires, so a
  // stale progress card never lingers.
  const localJobs = useMemo(
    () => selectStackedJobs(trackedLocalJobs, (job) => resultVisible(job) || completedWaitExpired(job)),
    [trackedLocalJobs, resultVisible, completedWaitExpired, resultFallbackTick],
  );

  // ---- LoRA selection (sc-4196: shared by Image + Video studios) ----
  const compatibleLoras = useMemo(() => loras.filter((lora) => {
    if (lora.presetManaged) {
      return false;
    }
    if (lora.installState === "missing") {
      return false;
    }
    // "Show incompatible" is an escape hatch for a known-but-mismatched family, not
    // for an unknown one: a family-less LoRA can never generate (the API 400s), so it
    // stays hidden even here — otherwise it's a dead-end selection (sc-10509).
    if (showIncompatibleLoras) {
      return loraHasResolvableFamily(lora);
    }
    return loraMatchesModel(lora, selectedModel);
  }), [loras, selectedModel, showIncompatibleLoras]);
  const compatibleLoraKey = useMemo(() => compatibleLoras.map((lora) => lora.id).join("|"), [compatibleLoras]);
  const selectedLoras = selectedLoraIds.map((id) => compatibleLoras.find((lora) => lora.id === id)).filter(Boolean);
  const userSelectedLoraCount = selectedLoras.filter((lora) => lora.scope !== "builtin").length;
  const selectedLoraValidationResult = useMemo(() => {
    const incompatible = selectedLoras.filter((lora) => !loraMatchesModel(lora, selectedModel)).map((lora) => lora.name ?? lora.id);
    return {
      incompatible,
      ok: incompatible.length === 0,
    };
  }, [selectedLoras, selectedModel]);
  const hasPendingCompatibleLoras = Boolean(selectedModel) && loras.some((lora) => lora.installState === "missing" && loraMatchesModel(lora, selectedModel));
  const loraEmptyMessage = !selectedModel
    ? "No model selected"
    : hasPendingCompatibleLoras
      ? "No installed compatible LoRAs. Imports appear after the Queue completes."
      : showIncompatibleLoras
        ? "No installed LoRAs in the library."
        : `No installed LoRAs match ${selectedModel.name ?? selectedModel.id}.`;

  // Drop selections that fall out of the compatible set (model/filter change).
  useEffect(() => {
    setSelectedLoraIds((ids) => ids.filter((id) => compatibleLoras.some((lora) => lora.id === id)));
  }, [compatibleLoraKey]);
  // Auto-open the advanced panel when an incompatible LoRA is selected so the
  // generate-blocking warning is visible.
  useEffect(() => {
    if (selectedLoraValidationResult.incompatible.length && !advancedOpen) {
      setAdvancedOpen(true);
    }
  }, [advancedOpen, selectedLoraValidationResult.incompatible.length]);

  function toggleLora(lora) {
    setSelectedLoraIds((ids) => {
      if (ids.includes(lora.id)) {
        return ids.filter((id) => id !== lora.id);
      }
      const selected = ids.map((id) => compatibleLoras.find((item) => item.id === id)).filter(Boolean);
      const userCount = selected.filter((item) => item.scope !== "builtin").length;
      if (lora.scope !== "builtin" && userCount >= MAX_USER_JOB_LORAS) {
        return ids;
      }
      return [...ids, lora.id];
    });
  }

  // Per-LoRA strength: the override map falls back to the LoRA's default weight.
  // Order of application is intentionally not exposed — the worker combines
  // adapters additively (set_adapters / dequant-to-bf16 merge), so order has no
  // effect on output.
  function effectiveLoraWeight(lora) {
    const override = loraWeights[lora.id];
    return Number.isFinite(override) ? override : loraWeight(lora);
  }

  function setLoraWeight(id, value) {
    setLoraWeights((current) => ({ ...current, [id]: value }));
  }

  // Preset LoRAs are first-class, visible picker entries — not a hidden server-side merge.
  // Selecting a preset seeds its LoRAs (at the preset's weights) into the selection so they
  // show up and can be retuned, added to, or removed; deselecting it (→ None or another
  // preset) removes the LoRAs it added and restores any weights it overrode. A user's own
  // LoRAs are preserved (the seed is a union computed from the live selection). The server
  // skips its own preset-LoRA merge when the studio sends presetLorasResolvedClientSide, so
  // a removed preset LoRA stays removed instead of being silently re-added at generation.
  const presetLoraSeed = useRef({ presetId: null, addedIds: [], prevWeights: {} });
  // On hydrate, the persisted selection already includes the snapshot preset's LoRAs, so the
  // first pass must adopt (not re-seed) it. Mirrors useSavePreset's skipPresetDefaultsOnHydrate.
  const skipPresetLoraSeedOnHydrate = useRef(initialPresetId != null && initialPresetId !== noPresetId);
  useEffect(() => {
    if (skipPresetLoraSeedOnHydrate.current) {
      skipPresetLoraSeedOnHydrate.current = false;
      if (selectedPreset && selectedPreset.id === initialPresetId) {
        // The restored selection already reflects this preset; adopt it without re-seeding.
        // addedIds is left empty so a post-reload deselect doesn't strip LoRAs we can't prove
        // the preset added (the user removes them by hand instead).
        presetLoraSeed.current = { presetId: selectedPreset.id, addedIds: [], prevWeights: {} };
        return;
      }
    }
    const seed = presetLoraSeed.current;
    const nextPresetId = selectedPreset?.id ?? null;
    if (seed.presetId === nextPresetId) {
      return;
    }
    // Undo the previous preset's contributions, then apply the new preset's LoRAs — both
    // computed from the current selection so the user's own LoRAs survive the switch.
    let ids = selectedLoraIds.filter((id) => !seed.addedIds.includes(id));
    const weights = { ...loraWeights };
    for (const [id, prev] of Object.entries(seed.prevWeights)) {
      if (prev === undefined) {
        delete weights[id];
      } else {
        weights[id] = prev;
      }
    }
    const entries = selectedPreset ? presetLoraSeedEntries(selectedPreset, compatibleLoras) : [];
    const addedIds = [];
    const prevWeights = {};
    for (const { id, weight } of entries) {
      if (!ids.includes(id)) {
        ids = [...ids, id];
        addedIds.push(id);
      }
      prevWeights[id] = Object.prototype.hasOwnProperty.call(weights, id) ? weights[id] : undefined;
      weights[id] = weight;
    }
    setSelectedLoraIds(ids);
    setLoraWeights(weights);
    presetLoraSeed.current = { presetId: nextPresetId, addedIds, prevWeights };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [selectedPreset?.id]);

  return {
    availablePresets,
    selectedPreset,
    selectedPresetId,
    setSelectedPresetId,
    // General-preset stack (epic 11949).
    availableGeneralPresets,
    generalStack,
    generalStackIds,
    toggleGeneralPreset,
    presetPromptParts,
    presetLoraDetails,
    presetValidationResult,
    localJobs,
    // LoRA selection bundle (sc-4196).
    selectedLoraIds,
    setSelectedLoraIds,
    loraWeights,
    setLoraWeights,
    showIncompatibleLoras,
    setShowIncompatibleLoras,
    compatibleLoras,
    selectedLoras,
    userSelectedLoraCount,
    selectedLoraValidationResult,
    loraEmptyMessage,
    toggleLora,
    effectiveLoraWeight,
    setLoraWeight,
  };
}

// Save-as-Preset + preset-default hydrate machinery shared by the Image and Video
// studios (sc-8937). Owns the save panel's state (name/scope/saving/message) and the
// remember/restore snapshot ref, runs the one preset-default hydrate effect, and
// exposes the save handler. The studios differ only in:
//   - `presetDefaultFields`: the [key, setter] pairs the studio hydrates,
//   - `buildDefaults()`: the raw defaults snapshot the save payload carries,
//   - `modeIsPresetable(mode)`: which sub-modes a saved preset may restore ("type"),
//   - `onApplyDefaults(defaults)`: optional per-studio side effect after hydrate
//     (Image marks the prompt box edited so the character-mode default can't clobber
//     the restored prompt),
//   - `extraSaveGuard()`: optional pre-save gate (Video blocks non-video modes),
// so those are passed in. Behavior (validation order, UX, saved/hydrated fields) is
// identical to the per-studio copies this replaced.
export function useSavePreset({
  saved,
  selectedPreset,
  setSelectedPresetId,
  presets,
  mode,
  model,
  selectedLoras,
  effectiveLoraWeight,
  createPreset,
  activeProject,
  presetDefaultFields,
  buildDefaults,
  modeIsPresetable,
  setMode,
  onApplyDefaults = () => {},
  extraSaveGuard = () => null,
}) {
  const [presetName, setPresetName] = useState("");
  const [presetScope, setPresetScope] = useState(activeProject ? "project" : "global");
  const [savingPreset, setSavingPreset] = useState(false);
  const [presetSaveMessage, setPresetSaveMessage] = useState({ tone: "neutral", text: "" });
  const presetDefaultSnapshots = useRef({});

  // When restoring a snapshot, the saved knob values already reflect the user's last
  // state — skip the one preset-default pass that fires as the restored preset resolves
  // so it doesn't overwrite them. "None" applies no defaults, so no guard is needed there.
  const skipPresetDefaultsOnHydrate = useRef(
    Object.keys(saved).length > 0 && saved.selectedPresetId !== noPresetId,
  );

  useEffect(() => {
    // Only the snapshot's OWN preset skips the pass — its knob values are already
    // restored. A different preset resolving first (e.g. Presets → "Use in Studio",
    // sc-10516) must still apply its defaults, or the launch would select the preset
    // and silently ignore everything it carries.
    if (skipPresetDefaultsOnHydrate.current && selectedPreset) {
      skipPresetDefaultsOnHydrate.current = false;
      if (selectedPreset.id === saved.selectedPresetId) {
        return;
      }
    }
    if (!selectedPreset) {
      for (const [key, setter] of presetDefaultFields) {
        clearPresetDefault(setter, presetDefaultSnapshots, key);
      }
      return;
    }
    const defaults = selectedPreset.defaults ?? {};
    for (const [key, setter] of presetDefaultFields) {
      if (Object.prototype.hasOwnProperty.call(defaults, key)) {
        applyPresetDefault(presetDefaultSnapshots, key, setter, defaults[key]);
      }
    }
    onApplyDefaults(defaults);
    // Restore the saved sub-mode ("type") when it's a presetable workflow.
    if (modeIsPresetable(defaults.mode)) {
      setMode(defaults.mode);
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [selectedPreset?.id]);

  // Snapshot the current working config into a named recipe preset in the workspace
  // library. Captures the literal prompt + every visible knob + the selected LoRAs
  // with their weights; the seed is intentionally left out so the preset stays
  // reusable. The backend additionally enforces id uniqueness and model/workflow +
  // LoRA compatibility, surfaced here via err.message.
  async function handleSaveAsPreset() {
    const trimmed = presetName.trim();
    if (!trimmed) {
      setPresetSaveMessage({ tone: "error", text: "Name the preset before saving." });
      return;
    }
    if (!slugifyPresetId(trimmed)) {
      setPresetSaveMessage({ tone: "error", text: "Use letters or numbers in the preset name." });
      return;
    }
    const guardMessage = extraSaveGuard();
    if (guardMessage) {
      setPresetSaveMessage({ tone: "error", text: guardMessage });
      return;
    }
    if (presetScope === "project" && !activeProject) {
      setPresetSaveMessage({ tone: "error", text: "Open a project first, or save to all projects." });
      return;
    }
    if (presetNameTaken(trimmed, presets)) {
      setPresetSaveMessage({ tone: "error", text: `"${trimmed}" already exists — pick a unique name.` });
      return;
    }
    const payload = buildStudioPresetPayload({
      name: trimmed,
      scope: presetScope,
      mode,
      model,
      loras: selectedLoras.map((lora) => ({ id: lora.id, weight: effectiveLoraWeight(lora) })),
      defaults: buildDefaults(),
    });
    setSavingPreset(true);
    setPresetSaveMessage({ tone: "neutral", text: "" });
    try {
      const created = await createPreset(payload);
      setSelectedPresetId(created?.id ?? payload.id);
      setPresetName("");
      setPresetSaveMessage({
        tone: "success",
        text: `Saved "${trimmed}" to ${presetScope === "project" ? "this project" : "all projects"}.`,
      });
    } catch (err) {
      setPresetSaveMessage({ tone: "error", text: err.message });
    } finally {
      setSavingPreset(false);
    }
  }

  return {
    presetName,
    setPresetName,
    presetScope,
    setPresetScope,
    savingPreset,
    presetSaveMessage,
    setPresetSaveMessage,
    handleSaveAsPreset,
  };
}

// The LoRA picker shared by both studios (sc-4196): the compatible-LoRA checklist
// with per-LoRA weight sliders, the "Show incompatible" toggle, and the empty state.
// All state lives in useGenerationStudio; this is a pure presentation of its bundle.
export function LoraPickerSection({
  selectedModel,
  selectedLoras,
  selectedLoraIds,
  compatibleLoras,
  userSelectedLoraCount,
  showIncompatibleLoras,
  setShowIncompatibleLoras,
  toggleLora,
  effectiveLoraWeight,
  setLoraWeight,
  loraEmptyMessage,
}) {
  // Add-on-demand picker (UI-refinement 3b): only the LoRAs you've added render as
  // slots; everything else lives behind the "Add LoRA" dropdown. This replaces the
  // checkbox-per-LoRA wall so a large library no longer floods the panel. "Show
  // incompatible" now filters the dropdown (through compatibleLoras) instead of an
  // always-visible list. Slot styles (.lora-stack/.lora-slot/.lora-add/
  // .lora-picker-panel/.lora-pick-row) already live in styles.css.
  const [pickerOpen, setPickerOpen] = useState(false);
  const availableLoras = compatibleLoras.filter((lora) => !selectedLoraIds.includes(lora.id));
  const atUserLimit = userSelectedLoraCount >= MAX_USER_JOB_LORAS;
  const loraMeta = (lora) => {
    const scope = lora.scope ?? "global";
    return lora.family ? `${scope} · ${lora.family}` : scope;
  };

  return (
    <section className="lora-picker" aria-label="LoRA selection">
      <div>
        <strong>LoRAs</strong>
        <span>
          {selectedLoras.length
            ? `${selectedLoras.length} selected`
            : selectedModel
              ? "Installed and compatible"
              : "Choose a model"}
        </span>
      </div>
      <label className="checkline">
        <input
          checked={showIncompatibleLoras}
          onChange={(event) => setShowIncompatibleLoras(event.target.checked)}
          type="checkbox"
        />
        Show incompatible
      </label>

      {!selectedLoras.length && !availableLoras.length ? (
        <div className="empty-panel compact-panel">{loraEmptyMessage}</div>
      ) : (
        <>
          {selectedLoras.length ? (
            <div className="lora-stack">
              {selectedLoras.map((lora) => {
                const weight = effectiveLoraWeight(lora);
                return (
                  <div className="lora-slot" key={lora.id}>
                    <div className="lora-slot-head">
                      <span className="lora-slot-meta">
                        <strong>{lora.name ?? lora.id}</strong>
                        <small>{loraMeta(lora)}</small>
                      </span>
                      <button
                        aria-label={`Remove ${lora.name ?? lora.id}`}
                        className="lora-slot-remove"
                        onClick={() => toggleLora(lora)}
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
                        <span className="lora-slot-weight-value">{weight.toFixed(2)}</span>
                      </label>
                      <input
                        aria-label={`${lora.name ?? lora.id} weight`}
                        max={LORA_WEIGHT_MAX}
                        min={LORA_WEIGHT_MIN}
                        onChange={(event) => setLoraWeight(lora.id, Number(event.target.value))}
                        step={LORA_WEIGHT_STEP}
                        type="range"
                        value={weight}
                      />
                    </div>
                  </div>
                );
              })}
            </div>
          ) : null}

          <button
            className="lora-add"
            data-count={`· ${availableLoras.length} available`}
            disabled={!availableLoras.length}
            onClick={() => setPickerOpen((open) => !open)}
            type="button"
          >
            <Icon.Plus size={15} />
            <span>Add LoRA</span>
          </button>

          {pickerOpen && availableLoras.length ? (
            <div className="lora-picker-panel">
              <div className="lora-picker-list">
                {availableLoras.map((lora) => {
                  const disabled = lora.scope !== "builtin" && atUserLimit;
                  return (
                    <button
                      className="lora-pick-row"
                      disabled={disabled}
                      key={lora.id}
                      onClick={() => {
                        toggleLora(lora);
                        setPickerOpen(false);
                      }}
                      type="button"
                    >
                      <span className="lora-slot-meta">
                        <strong>{lora.name ?? lora.id}</strong>
                        <small>{loraMeta(lora)}</small>
                      </span>
                      <span className="lora-pick-add">{disabled ? "Limit reached" : "Add"}</span>
                    </button>
                  );
                })}
              </div>
            </div>
          ) : null}
        </>
      )}
    </section>
  );
}

// The "Save as Preset" panel shared by both studios (sc-4196): name field, save
// button, project/global scope segment, and the inline save message. The actual
// save handler differs per studio (different payloads), so it's passed as onSave.
export function SavePresetPanel({
  presetName,
  setPresetName,
  savingPreset,
  presetSaveMessage,
  setPresetSaveMessage,
  onSave,
  presetScope,
  setPresetScope,
  activeProject,
  // Video studio gates saving to a subset of modes; pass an extra disable + a
  // tooltip explaining why. Image studio omits both (always saveable).
  saveDisabled = false,
  saveTitle = undefined,
}) {
  // Button gate and its reason from one summary (epic 10644). A blank name stays silent;
  // an unsaveable mode surfaces the tooltip as an always-visible chip.
  const saveDraft = useMemo(() => ({ presetName, saveDisabled, saveTitle }), [presetName, saveDisabled, saveTitle]);
  const saveValidity = useValidation(savePresetDialogValidation, saveDraft, undefined);
  return (
    <div className="save-preset">
      <div className="save-preset-row">
        <input
          aria-label="Preset name"
          className="save-preset-name"
          disabled={savingPreset}
          onChange={(event) => {
            setPresetName(event.target.value);
            if (presetSaveMessage.text) {
              setPresetSaveMessage({ tone: "neutral", text: "" });
            }
          }}
          onKeyDown={(event) => {
            if (event.key === "Enter") {
              event.preventDefault();
              onSave();
            }
          }}
          placeholder="Name this setup…"
          value={presetName}
        />
        <button
          className="save-preset-btn"
          disabled={savingPreset || !saveValidity.ready}
          onClick={onSave}
          title={saveTitle}
          type="button"
        >
          <Icon.Preset size={14} /> {savingPreset ? "Saving…" : "Save as Preset"}
        </button>
      </div>
      <ValidationSummary issues={saveValidity.surfaced} label="Save-preset errors" />
      <div className="save-preset-scope scope-segment" role="radiogroup" aria-label="Preset scope">
        <button
          aria-checked={presetScope === "project"}
          className={presetScope === "project" ? "active" : ""}
          disabled={!activeProject}
          onClick={() => setPresetScope("project")}
          role="radio"
          type="button"
        >
          <Icon.Folder size={13} /> This project
        </button>
        <button
          aria-checked={presetScope === "global"}
          className={presetScope === "global" ? "active" : ""}
          onClick={() => setPresetScope("global")}
          role="radio"
          type="button"
        >
          <Icon.Stars size={13} /> All projects
        </button>
      </div>
      {presetSaveMessage.text ? (
        <p className={presetSaveMessage.tone === "success" ? "inline-success" : "inline-warning"}>
          {presetSaveMessage.text}
        </p>
      ) : null}
    </div>
  );
}

// The "what this preset adds" strip shown under the preset picker in both studios.
export function PresetGuidanceStrip({ selectedPreset, presetPromptParts, presetLoraDetails }) {
  // Nothing to say when no preset is active — the visible controls already describe the run.
  if (!selectedPreset) {
    return null;
  }
  // A selected preset's installed LoRAs are seeded into the LoRA picker (visible + adjustable),
  // so the strip no longer lists them — it only calls out ones that couldn't be seeded because
  // they aren't installed, so the user knows to import them before the preset fully applies.
  const missingLoras = presetLoraDetails.filter((lora) => lora.missing);
  return (
    <div className="guidance-strip">
      <strong>{selectedPreset.ui?.description ?? "Preset defaults active"}</strong>
      <span>
        {presetPromptParts.length ? `Adds: ${presetPromptParts.join(", ")}` : "No prompt fragments"}
        {missingLoras.length
          ? ` | Preset LoRA not installed: ${missingLoras.map((lora) => lora.name ?? lora.id).join(", ")} — import to apply`
          : ""}
      </span>
    </div>
  );
}
