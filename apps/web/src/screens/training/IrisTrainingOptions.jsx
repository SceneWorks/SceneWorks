import React from "react";

import { invalidProps } from "../../validation/Validation.jsx";
import { irisOptionInPlay, irisRunLabel, irisRunOutcome } from "../../training/irisOptions.js";

// Iris-3B training options (sc-25685 / sc-25686). Rendered inside the Configure panel's Advanced
// disclosure only for the `iris_train` kernel, as its own collapsed disclosure (busy page →
// collapsed by default). Every control maps 1:1 to an `advanced.iris` engine option the trainer
// accepts (crates/sceneworks-core/src/training/iris.rs); a control whose option is not in play for
// this run (muon extras under AdamW, EMA decay with EMA off, init for an adapter, resume knobs
// without a resume, logit-normal knobs on the uniform sampler) is not rendered, matching what the
// snapshot sends. An empty value means "trainer / backbone default" and is not sent.

const choiceLabels = {
  weights: "Released weights",
  random: "Random init (train from scratch)",
  cached: "Cached (frees the text encoder before training)",
  on_the_fly: "On the fly",
  warn: "Warn and truncate",
  error: "Stop the run",
  silent: "Truncate silently",
  raw: "Raw weights",
  ema: "EMA weights",
  fp32: "FP32",
  bf16: "BF16",
  none: "None",
  sqrt: "Square-root scaling",
  linear: "Linear scaling",
  rms_norm: "RMS norm",
  spectral_norm: "Spectral norm",
  exact: "Exact (continue the same data order)",
  new_phase: "New phase (fresh data pass)",
  v: "v-prediction",
  x: "x-prediction",
  sd3: "SD3",
  flux: "Flux",
};

function choiceLabel(value) {
  return choiceLabels[value] ?? value;
}

export function IrisTrainingOptions({
  configDraft,
  updateConfigDraft,
  configValidity,
  previousRuns = [],
  fullRuns = [],
  captionFieldOptions = ["caption"],
}) {
  const iris = configDraft.iris ?? {};
  const setIris = (key, value) => updateConfigDraft("iris", { ...(configDraft.iris ?? {}), [key]: value });
  const inPlay = (key) => irisOptionInPlay(key, configDraft);
  const isFull = String(configDraft.networkType ?? "").trim().toLowerCase() === "full";
  const emaOff = iris.ema_enabled === false;
  const resuming = Boolean(String(configDraft.resumeFromJobId ?? "").trim());

  const numberField = (key, label, { step = "any", min, max, placeholder, title } = {}) => (
    <label key={key} title={title}>
      {label}
      <input
        max={max}
        min={min}
        onChange={(event) => setIris(key, event.target.value)}
        placeholder={placeholder}
        step={step}
        type="number"
        value={iris[key] ?? ""}
        {...invalidProps(configValidity, `iris.${key}`)}
      />
    </label>
  );
  const selectField = (key, label, values, { defaultLabel, title, disabledValue } = {}) => {
    const current = iris[key] ?? "";
    return (
      <label key={key} title={title}>
        {label}
        <select
          onChange={(event) => setIris(key, event.target.value)}
          value={current}
          {...invalidProps(configValidity, `iris.${key}`)}
        >
          {defaultLabel ? <option value="">{defaultLabel}</option> : null}
          {values.map((value) => (
            <option disabled={disabledValue?.(value)} key={value} value={value}>
              {choiceLabel(value)}
            </option>
          ))}
        </select>
      </label>
    );
  };
  const triStateField = (key, label) => (
    <label key={key}>
      {label}
      <select
        onChange={(event) => setIris(key, event.target.value === "" ? "" : event.target.value === "true")}
        value={typeof iris[key] === "boolean" ? String(iris[key]) : ""}
      >
        <option value="">Trainer default</option>
        <option value="true">On</option>
        <option value="false">Off</option>
      </select>
    </label>
  );
  const checkboxField = (key, label) => (
    <label className="training-checkbox-field" key={key}>
      <input
        checked={iris[key] === true}
        onChange={(event) => setIris(key, event.target.checked)}
        type="checkbox"
      />
      {label}
    </label>
  );

  const captionFields = Array.isArray(iris.caption_fields) ? iris.caption_fields : [];
  const captionFieldChoices = Array.from(
    new Set([...captionFieldOptions, ...(iris.caption_field ? [iris.caption_field] : [])]),
  );
  const namedFieldChoices = Array.from(
    new Set([...captionFieldOptions.filter((name) => name !== "caption"), ...captionFields]),
  );
  const toggleCaptionField = (name, checked) =>
    setIris(
      "caption_fields",
      checked ? [...captionFields.filter((field) => field !== name), name] : captionFields.filter((field) => field !== name),
    );
  const emaChoiceBlocked = (value) => value === "ema" && emaOff;

  const selectedResume = previousRuns.find((job) => job.id === configDraft.resumeFromJobId);
  const selectedLoad = fullRuns.find((job) => job.id === configDraft.loadFromJobId);
  const runWarnings = (job) => (Array.isArray(job?.result?.warnings) ? job.result.warnings : []);

  return (
    <details className="training-iris-options" aria-label="Iris training options">
      <summary>Iris training options</summary>

      <div className="training-advanced-grid">
        {inPlay("init")
          ? selectField("init", "Initialization", ["weights", "random"], {
              title: "A full fine-tune can start from the released weights or train from random init.",
            })
          : null}
        {numberField("text_dropout", "Caption dropout", {
          min: 0,
          max: 1,
          step: "0.05",
          title: "Probability of training a step with an empty caption (classifier-free guidance).",
        })}
        {numberField("gradient_clip", "Gradient clip", { min: 0, step: "0.1" })}
        {numberField("num_epochs", "Epochs", {
          min: 1,
          step: "1",
          placeholder: "Steps decide",
          title: "Optional. Stop after this many passes over the dataset; empty = run the configured steps.",
        })}
        {numberField("nan_loss_tolerance", "NaN-loss tolerance", {
          min: 0,
          step: "1",
          title: "Consecutive non-finite losses the run skips before it stops.",
        })}
        {selectField("text_conditioning", "Text conditioning", ["cached", "on_the_fly"])}
      </div>
      <div className="training-advanced-toggles">
        {checkboxField("ema_enabled", "EMA of the trained weights")}
      </div>
      {inPlay("ema_decay") ? (
        <div className="training-advanced-grid">{numberField("ema_decay", "EMA decay", { min: 0, max: 1, step: "0.0001" })}</div>
      ) : null}

      <p className="training-field-hint"><strong>Captions</strong></p>
      <div className="training-advanced-grid">
        {selectField("caption_field", "Caption field", captionFieldChoices, {
          title: "Which caption each item trains on. “caption” is the item's ordinary caption.",
        })}
        {selectField("on_caption_overflow", "Caption overflow", ["warn", "error", "silent"], {
          title: "What happens when a caption is longer than the text encoder's context.",
        })}
      </div>
      {namedFieldChoices.length ? (
        <fieldset className="training-iris-caption-fields">
          <legend>Sample each step from these caption fields (falls back to the caption field)</legend>
          {namedFieldChoices.map((name) => (
            <label className="training-checkbox-field" key={name}>
              <input
                checked={captionFields.includes(name)}
                onChange={(event) => toggleCaptionField(name, event.target.checked)}
                type="checkbox"
              />
              {name}
            </label>
          ))}
        </fieldset>
      ) : null}

      <p className="training-field-hint"><strong>Previews</strong></p>
      <div className="training-advanced-grid">
        {selectField("preview_weights", "Preview weights", ["raw", "ema"], {
          defaultLabel: "Trainer default",
          disabledValue: emaChoiceBlocked,
          title: "Trainer default: EMA for a full run with EMA on, raw for adapters.",
        })}
        <label>
          Preview negative prompt
          <input
            onChange={(event) => setIris("preview_negative_prompt", event.target.value)}
            value={iris.preview_negative_prompt ?? ""}
          />
        </label>
      </div>
      <div className="training-advanced-toggles">
        {checkboxField("upstream_validation_prompts", "Use Iris' seven built-in validation prompts")}
      </div>

      <p className="training-field-hint"><strong>Checkpoints and export</strong></p>
      <div className="training-advanced-grid">
        {numberField("keep_last_checkpoints", "Keep last checkpoints", {
          min: 0,
          step: "1",
          title: "0 keeps every checkpoint. A full-model checkpoint is ~48 GB.",
        })}
        <label title="Optimizer steps (not micro-steps) whose checkpoints are always kept, comma-separated.">
          Milestone optimizer steps
          <input
            onChange={(event) => setIris("milestone_steps", event.target.value)}
            placeholder="e.g. 1000, 5000"
            value={iris.milestone_steps ?? ""}
            {...invalidProps(configValidity, "iris.milestone_steps")}
          />
        </label>
        {selectField("export_weights", "Export weights", ["raw", "ema"], {
          defaultLabel: "Trainer default",
          disabledValue: emaChoiceBlocked,
        })}
        {selectField("export_dtype", "Export precision", ["fp32", "bf16"])}
      </div>

      <p className="training-field-hint"><strong>Optimizer</strong></p>
      <div className="training-advanced-grid">
        <label>
          Beta 1
          <input
            max="1"
            min="0"
            onChange={(event) => setIris("betas", [event.target.value, (iris.betas ?? ["", ""])[1]])}
            step="0.01"
            type="number"
            value={(iris.betas ?? ["", ""])[0]}
            {...invalidProps(configValidity, "iris.betas")}
          />
        </label>
        <label>
          Beta 2
          <input
            max="1"
            min="0"
            onChange={(event) => setIris("betas", [(iris.betas ?? ["", ""])[0], event.target.value])}
            step="0.001"
            type="number"
            value={(iris.betas ?? ["", ""])[1]}
            {...invalidProps(configValidity, "iris.betas")}
          />
        </label>
        {inPlay("muon_momentum") ? numberField("muon_momentum", "Muon momentum", { min: 0, max: 1, step: "0.01", placeholder: "Trainer default" }) : null}
        {inPlay("muon_nesterov") ? triStateField("muon_nesterov", "Muon Nesterov") : null}
        {inPlay("muon_adjust_lr")
          ? selectField("muon_adjust_lr", "Muon LR adjustment", ["rms_norm", "spectral_norm", "none"], { defaultLabel: "Trainer default" })
          : null}
        {selectField("auto_lr", "Auto LR scaling", ["none", "sqrt", "linear"], {
          title: "Scale the learning rate by the effective batch relative to the base batch size.",
        })}
        {inPlay("base_batch_size") ? numberField("base_batch_size", "Base batch size", { min: 1, step: "1" }) : null}
      </div>

      <details className="training-iris-subsection">
        <summary>Flow objective (advanced)</summary>
        <p className="training-field-hint">Empty fields use the backbone's own config.</p>
        <div className="training-advanced-grid">
          {selectField("prediction", "Prediction target", ["v", "x"], { defaultLabel: "Backbone default" })}
          {numberField("flow_shift", "Flow shift", { min: 0, step: "0.1", placeholder: "Backbone default" })}
          {selectField("shift_law", "Shift law", ["none", "sd3", "flux"], { defaultLabel: "Backbone default" })}
          {numberField("shift_base_tokens", "Shift base tokens", { min: 1, step: "1", placeholder: "Backbone default" })}
          {inPlay("logit_mean") ? numberField("logit_mean", "Logit-normal mean", { step: "0.1", placeholder: "Backbone default" }) : null}
          {inPlay("logit_std") ? numberField("logit_std", "Logit-normal std", { step: "0.1", placeholder: "Backbone default" }) : null}
          {numberField("x_pred_sigma_min", "x-prediction sigma floor", { step: "0.01", placeholder: "Backbone default" })}
        </div>
      </details>

      <details className="training-iris-subsection">
        <summary>Resume / continue</summary>
        <div className="training-advanced-grid">
          <label title="Continue a previous Iris run from its newest durable checkpoint.">
            Resume a previous run
            <select
              onChange={(event) => updateConfigDraft("resumeFromJobId", event.target.value)}
              value={configDraft.resumeFromJobId ?? ""}
            >
              <option value="">Start a new run</option>
              {previousRuns.map((job) => (
                <option key={job.id} value={job.id}>
                  {irisRunLabel(job)}
                </option>
              ))}
            </select>
          </label>
          {resuming ? selectField("resume_data_policy", "Data policy", ["exact", "new_phase"], { defaultLabel: "Trainer default" }) : null}
          {isFull ? (
            <label title="Start this full run from a previous full run's trained weights (fresh optimizer state).">
              Start from a previous full run's weights
              <select
                disabled={resuming || iris.init === "random"}
                onChange={(event) => updateConfigDraft("loadFromJobId", event.target.value)}
                value={configDraft.loadFromJobId ?? ""}
                {...invalidProps(configValidity, "loadFromJobId")}
              >
                <option value="">Released weights</option>
                {fullRuns.map((job) => (
                  <option key={job.id} value={job.id}>
                    {irisRunLabel(job)}
                  </option>
                ))}
              </select>
            </label>
          ) : null}
        </div>
        {resuming ? (
          <div className="training-advanced-toggles">
            {checkboxField("override_lr_on_resume", "Use this form's learning rate instead of the checkpoint's")}
          </div>
        ) : null}
        {!previousRuns.length ? (
          <p className="training-field-hint">No finished Iris runs in this project yet.</p>
        ) : null}
        {[selectedResume, selectedLoad].filter(Boolean).map((job) => (
          <p className="training-field-hint" key={job.id}>
            {job.payload?.outputName ?? job.id}: {irisRunOutcome(job)}
            {runWarnings(job).map((warning) => (
              <span className="inline-warning" key={warning}> {warning}</span>
            ))}
          </p>
        ))}
      </details>
    </details>
  );
}
