import React, { useEffect, useMemo, useState } from "react";
import { terminalStatuses } from "../jobTypes.js";
import {
  noPresetId,
  presetLoraDetails as buildPresetLoraDetails,
  presetMatchesModel,
  presetMatchesWorkflow,
  presetPromptParts as buildPresetPromptParts,
  presetValidation,
} from "../presetUtils.js";

const completedResultFallbackMs = 30000;

// Cmd/Ctrl+Enter submits the studio form from the prompt textarea.
export function onPromptKeyDown(event) {
  if ((event.metaKey || event.ctrlKey) && event.key === "Enter") {
    event.preventDefault();
    event.currentTarget.form?.requestSubmit();
  }
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
}) {
  const [selectedPresetId, setSelectedPresetId] = useState(null);
  const [resultFallbackTick, setResultFallbackTick] = useState(0);

  // Snap the model back into range when the catalog changes out from under it.
  useEffect(() => {
    if (!models.some((item) => item.id === model)) {
      setModel(models[0]?.id ?? fallbackModelId);
    }
  }, [models, model]);

  // Drop a character selection that's no longer in the catalog.
  useEffect(() => {
    if (characterId && !characters.some((character) => character.id === characterId)) {
      setCharacterId("");
      setCharacterLookId("");
    }
  }, [characters, characterId]);

  const availablePresets = useMemo(
    () => presets.filter((preset) => presetMatchesWorkflow(preset, mode) && presetMatchesModel(preset, selectedModel, models)),
    [mode, presets, selectedModel?.id, models],
  );
  const selectedPreset =
    selectedPresetId === noPresetId
      ? null
      : selectedPresetId
        ? availablePresets.find((preset) => preset.id === selectedPresetId) ?? null
        : availablePresets[0] ?? null;
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

  function resultVisible(job) {
    if (job.result?.generationSetId) {
      return latestAssets.some((asset) => asset.generationSetId === job.result.generationSetId);
    }
    const assetIds = job.result?.assetIds ?? [];
    return assetIds.length > 0 && assetIds.every((id) => assets.some((asset) => asset.id === id));
  }

  function jobCreatedMs(job) {
    const parsed = Date.parse(job?.createdAt ?? "");
    return Number.isFinite(parsed) ? parsed : 0;
  }

  // A finished run is replaced once a strictly-newer run leaves the queue (i.e.
  // a worker picked it up). Canceled runs never "start", so they don't bump the
  // run above them off the stack.
  function successorStarted(job) {
    return trackedLocalJobs.some(
      (other) =>
        other.id !== job.id &&
        other.status !== "queued" &&
        other.status !== "canceled" &&
        jobCreatedMs(other) > jobCreatedMs(job),
    );
  }

  function hasPendingSuccessor(job) {
    return trackedLocalJobs.some(
      (other) => other.id !== job.id && other.status !== "canceled" && jobCreatedMs(other) > jobCreatedMs(job),
    );
  }

  function completedAnchorMs(job) {
    return Date.parse(job.completedAt ?? job.updatedAt ?? "");
  }

  function completedWaitExpired(job, nowMs = Date.now()) {
    const anchorMs = completedAnchorMs(job);
    return Number.isFinite(anchorMs) && nowMs - anchorMs > completedResultFallbackMs;
  }

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
  }, [assets, latestAssets, trackedLocalJobs, resultFallbackTick]);

  // The visible stack: every running and queued run, plus the most recent finished
  // run, ordered (by trackedLocalJobs) oldest-first so the active run sits on top
  // and queued runs follow. A finished run holds its place — showing its rendered
  // batch above the pending queue — until the next run actually starts, at which
  // point that run slides up to replace it. Canceled runs drop immediately.
  const localJobs = useMemo(
    () =>
      trackedLocalJobs.filter((job) => {
        if (job.status === "canceled") {
          return false;
        }
        // Running and queued runs always stack.
        if (!terminalStatuses.has(job.status)) {
          return true;
        }
        // A finished run slides out the moment its successor starts.
        if (successorStarted(job)) {
          return false;
        }
        if (job.status === "completed") {
          // With a run still queued behind it, keep the completed run (and its
          // rendered batch) on top until that run starts. Standing alone, collapse
          // to the plain latest-batch grid once its assets render (or the wait
          // window expires) so a stale progress card never lingers.
          if (hasPendingSuccessor(job)) {
            return true;
          }
          return !resultVisible(job) && !completedWaitExpired(job);
        }
        // Failed/interrupted runs with nothing started behind them stay visible so
        // the outcome is clear until the user moves on.
        return true;
      }),
    [assets, latestAssets, trackedLocalJobs, resultFallbackTick],
  );

  return {
    availablePresets,
    selectedPreset,
    selectedPresetId,
    setSelectedPresetId,
    presetPromptParts,
    presetLoraDetails,
    presetValidationResult,
    localJobs,
  };
}

// The "what this preset adds" strip shown under the preset picker in both studios.
export function PresetGuidanceStrip({ selectedPreset, presetPromptParts, presetLoraDetails, noPresetHint }) {
  if (!selectedPreset) {
    return (
      <div className="guidance-strip">
        <strong>No preset selected</strong>
        <span>{noPresetHint}</span>
      </div>
    );
  }
  return (
    <div className="guidance-strip">
      <strong>{selectedPreset.ui?.description ?? "Preset defaults active"}</strong>
      <span>
        {presetPromptParts.length ? `Adds: ${presetPromptParts.join(", ")}` : "No prompt fragments"}
        {presetLoraDetails.length
          ? ` | Preset LoRA applied at generation: ${presetLoraDetails.map((lora) => lora.name ?? lora.id).join(", ")}`
          : " | No preset LoRAs"}
        {presetLoraDetails.some((lora) => lora.missing) ? " | Import still pending" : ""}
      </span>
    </div>
  );
}

// The preset "missing"/"incompatible" inline warnings shared by both studios.
export function PresetValidationWarnings({ presetValidationResult, selectedModel }) {
  return (
    <>
      {presetValidationResult.missing.length ? (
        <p className="inline-warning">
          Preset cannot run until LoRA import finishes: {presetValidationResult.missing.join(", ")}. Wait for the Queue or choose another preset.
        </p>
      ) : null}
      {presetValidationResult.incompatible.length ? (
        <p className="inline-warning">
          Preset cannot run with {selectedModel?.name ?? "the selected model"} because these LoRAs are incompatible: {presetValidationResult.incompatible.join(", ")}. Choose another preset or model.
        </p>
      ) : null}
    </>
  );
}
