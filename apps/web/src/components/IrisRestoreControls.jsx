import React, { useEffect, useRef, useState } from "react";
import { apiFetch, isAbortError } from "../api.js";
import { useAppContextOptional } from "../context/AppContext.js";
import {
  IRIS_INPUT_SIZINGS,
  IRIS_SCALE_PRESETS,
  defaultRestoreModel,
  normalizeIrisScale,
  summarizeRestorePlans,
  trainedTaskModels,
} from "../irisRestore.js";

// The trained Iris model picker (sc-25687 depth / sc-25688 restoration): shown only when a Training
// Studio run of `task` is installed (or a reopened setting names one). "" = the release export.
// `models` defaults to the app catalog.
export function TrainedIrisModelSelect({ task, value, onChange, models, releaseLabel, disabled = false }) {
  const catalog = useAppContextOptional()?.models;
  const trained = trainedTaskModels(models ?? catalog, task);
  if (!trained.length && !value) return null;
  return (
    <label className="iris-restore-row">
      Model
      <select aria-label={`Iris ${task} model`} disabled={disabled} onChange={(event) => onChange(event.target.value)} value={value ?? ""}>
        <option value="">{releaseLabel}</option>
        {trained.map((model) => (
          <option key={model.id} value={model.id}>
            {model.name ?? model.id}
          </option>
        ))}
        {value && !trained.some((model) => model.id === value) ? <option value={value}>{value} (not installed)</option> : null}
      </select>
    </label>
  );
}

// The Iris restoration controls (sc-25683) shared by the Image Studio post-processing pass, the
// Image Editor upscale tool and the Library batch panel: scale (presets + any positive value),
// input sizing, colour correction — and the pre-submission preview of the ACTUAL processing and
// output dimensions from the engine's planner (`POST /api/v1/image-restoration/plan`).
//
// `planBody` is the plan request for the source(s) without the controls (`{ width, height }` or
// `{ projectId, assetIds }`), or null when there is nothing to plan yet. `onStatus` reports
// `{ ready, blocked, error }` so the host can gate its Run button: a refused plan (or an invalid
// scale) blocks submission rather than being silently changed.
export function IrisRestoreControls({ value, onChange, planBody = null, token = "", disabled = false, onStatus, idPrefix = "iris", models }) {
  const [scaleText, setScaleText] = useState(String(value?.scale ?? 4));
  const [preview, setPreview] = useState({ state: "idle", lines: [], error: null, blocked: false });
  const statusRef = useRef(onStatus);
  statusRef.current = onStatus;
  const appModels = useAppContextOptional()?.models;
  // With the release export absent, a selection that names no model runs the first installed
  // trained restoration model — set on the value, so the host's job carries it (and re-derived when
  // the catalog loads late or a trained model is installed later).
  const fallbackModel = defaultRestoreModel(models ?? appModels);
  const changeRef = useRef(onChange);
  changeRef.current = onChange;
  useEffect(() => {
    if (!value?.irisModel && fallbackModel) changeRef.current?.({ ...value, irisModel: fallbackModel });
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [fallbackModel, value?.irisModel]);

  // Keep the text box in step with an external change (preset click, restored settings).
  useEffect(() => {
    if (normalizeIrisScale(scaleText) !== value?.scale) setScaleText(String(value?.scale ?? ""));
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [value?.scale]);

  const scale = normalizeIrisScale(scaleText);
  // The selected trained model rides along: it plans with ITS export's tile geometry (S9b).
  const planKey = planBody && scale
    ? JSON.stringify({ ...planBody, scale, inputSizing: value?.inputSizing, colorFix: value?.colorFix, irisModel: value?.irisModel || undefined })
    : null;

  useEffect(() => {
    if (!scale) {
      setPreview({ state: "invalid", lines: [], error: "Enter a scale greater than 0.", blocked: true });
      statusRef.current?.({ ready: false, blocked: true, error: "Enter a scale greater than 0." });
      return undefined;
    }
    if (!planKey) {
      setPreview({ state: "idle", lines: [], error: null, blocked: false });
      statusRef.current?.({ ready: false, blocked: false, error: null });
      return undefined;
    }
    const controller = new AbortController();
    setPreview((current) => ({ ...current, state: "loading" }));
    statusRef.current?.({ ready: false, blocked: false, error: null });
    const timer = setTimeout(() => {
      apiFetch("/api/v1/image-restoration/plan", token, {
        method: "POST",
        body: planKey,
        signal: controller.signal,
      })
        .then((response) => {
          const summary = summarizeRestorePlans(response);
          setPreview({ state: "ready", lines: summary.lines, error: summary.firstError, blocked: summary.blocked });
          statusRef.current?.({ ready: !summary.blocked, blocked: summary.blocked, error: summary.firstError });
        })
        .catch((error) => {
          if (isAbortError(error)) return;
          const message = error?.message || String(error);
          setPreview({ state: "error", lines: [], error: message, blocked: true });
          statusRef.current?.({ ready: false, blocked: true, error: message });
        });
    }, 200);
    return () => {
      clearTimeout(timer);
      controller.abort();
    };
  }, [planKey, scale, token]);

  function update(patch) {
    onChange?.({ ...value, ...patch });
  }

  return (
    <div className="iris-restore-controls">
      <TrainedIrisModelSelect
        disabled={disabled}
        models={models}
        onChange={(irisModel) => update({ irisModel })}
        releaseLabel="Iris 3B Restoration (release)"
        task="restoration"
        value={value?.irisModel}
      />
      <div className="iris-restore-row" role="group" aria-label="Iris scale">
        <span className="iris-restore-label">Scale</span>
        {IRIS_SCALE_PRESETS.map((preset) => (
          <button
            aria-pressed={scale === preset}
            className={scale === preset ? "active" : ""}
            disabled={disabled}
            key={preset}
            onClick={() => {
              setScaleText(String(preset));
              update({ scale: preset });
            }}
            type="button"
          >
            {preset}×
          </button>
        ))}
        <input
          aria-label="Iris scale factor"
          disabled={disabled}
          id={`${idPrefix}-scale`}
          inputMode="decimal"
          min="0.01"
          onChange={(event) => {
            setScaleText(event.target.value);
            const next = normalizeIrisScale(event.target.value);
            if (next) update({ scale: next });
          }}
          step="0.1"
          type="number"
          value={scaleText}
        />
      </div>
      <div className="iris-restore-row" role="radiogroup" aria-label="Iris input sizing">
        {IRIS_INPUT_SIZINGS.map((option) => (
          <label className="checkline" key={option.key} title={option.hint}>
            <input
              checked={(value?.inputSizing ?? "budgeted") === option.key}
              disabled={disabled}
              name={`${idPrefix}-input-sizing`}
              onChange={() => update({ inputSizing: option.key })}
              type="radio"
            />
            {option.label}
          </label>
        ))}
      </div>
      <label className="checkline">
        <input
          checked={value?.colorFix !== false}
          disabled={disabled}
          onChange={(event) => update({ colorFix: event.target.checked })}
          type="checkbox"
        />
        Colour correction
      </label>
      <div className="iris-restore-preview" aria-live="polite" data-state={preview.state}>
        {preview.state === "loading" && preview.lines.length === 0 ? <span>Planning…</span> : null}
        {preview.lines.map((line, index) => (
          <span className={line.startsWith("Cannot restore") ? "iris-restore-refused" : undefined} key={index}>
            {line}
          </span>
        ))}
        {preview.state === "error" || preview.state === "invalid" ? (
          <span className="iris-restore-refused">{preview.error}</span>
        ) : null}
      </div>
    </div>
  );
}
