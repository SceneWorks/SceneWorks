import React, { useState } from "react";

import { errorMessage } from "../../errorMessage.js";
import { invalidProps } from "../../validation/Validation.jsx";
import { irisRunLabel } from "../../training/irisOptions.js";
import { IRIS_TASK_SPECS, POLICY_TEXT, irisCheckSummary, irisTaskOptionInPlay } from "../../training/irisTaskOptions.js";

// Iris-3B depth (sc-25687) / restoration (sc-25688) training options. The run-defining choices (the
// start, the depth training size, the invalid-example policy) and the dataset check sit in the
// panel; every other `advanced.iris` / `advanced.iris_restore` option is in a collapsed disclosure,
// rendered from the objective's spec so each control maps 1:1 to an engine option. A control whose
// option is not in play for this run is not rendered (and not sent).

const GROUPS = {
  depth: [
    ["Paired transforms (RGB, depth, mask move together)", ["resize", "crop", "hflip_prob"]],
    ["Loss and targets", ["gradient_weight", "low_percentile", "high_percentile", "clip_target", "loss_reduction", "min_valid_pixels", "reducer_init"]],
    ["Optimizer", ["betas", "gradient_clip", "muon_momentum", "muon_nesterov", "muon_adjust_lr", "auto_lr", "base_batch_size", "ema_enabled", "ema_decay", "num_epochs", "nan_loss_tolerance"]],
    ["Checkpoints and previews", ["keep_last_checkpoints", "milestone_steps", "preview_items", "preview_weights", "export_weights"]],
  ],
  restoration: [
    ["Loss (pixel + LPIPS + GAN)", ["loss_l2", "loss_lpips", "loss_gan", "gan_real_label"]],
    ["Critic", ["lr_critic", "critic_weight_decay", "critic_betas", "critic_gradient_clip", "critic_dtype"]],
    ["Generator", ["crop", "lora_init", "betas", "gradient_clip", "ema_enabled", "ema_decay", "num_epochs", "nan_loss_tolerance"]],
    ["Checkpoints and export", ["keep_last_checkpoints", "milestone_steps", "preview_weights", "export_weights", "export_dtype"]],
  ],
};

export function IrisTaskOptions({
  objective,
  configDraft,
  updateConfigDraft,
  configValidity,
  previousRuns = [],
  onCheckDataset,
}) {
  const [check, setCheck] = useState(null);
  const specs = IRIS_TASK_SPECS[objective];
  if (!specs) return null;
  const depth = objective === "depth";
  const bag = configDraft.irisTask ?? {};
  const setBag = (key, value) => updateConfigDraft("irisTask", { ...(configDraft.irisTask ?? {}), [key]: value });
  const emaOff = depth ? bag.ema_enabled !== true : bag.ema_enabled === false;
  const resuming = Boolean(String(configDraft.resumeFromJobId ?? "").trim());
  const summary = check?.report ? irisCheckSummary(check.report) : null;

  const field = (key, labelOverride) => {
    const spec = specs[key];
    if (!irisTaskOptionInPlay(objective, key, configDraft)) return null;
    const label = labelOverride ?? spec.label;
    const invalid = invalidProps(configValidity, `irisTask.${key}`);
    if (spec.kind === "bool") {
      return (
        <label className="training-checkbox-field" key={key}>
          <input checked={bag[key] === true} onChange={(event) => setBag(key, event.target.checked)} type="checkbox" />
          {label}
        </label>
      );
    }
    if (spec.kind === "betas" || spec.kind === "size") {
      const pair = Array.isArray(bag[key]) ? bag[key] : ["", ""];
      const names = spec.kind === "size" ? ["height", "width"] : ["1", "2"];
      return names.map((name, index) => (
        <label key={`${key}-${name}`}>
          {label} {name}
          <input
            onChange={(event) => setBag(key, index ? [pair[0], event.target.value] : [event.target.value, pair[1]])}
            step="any"
            type="number"
            value={pair[index]}
            {...invalid}
          />
        </label>
      ));
    }
    if (spec.kind === "choice") {
      return (
        <label key={key}>
          {label}
          <select onChange={(event) => setBag(key, event.target.value)} value={bag[key] ?? ""} {...invalid}>
            <option value="">Trainer default</option>
            {spec.values.map((value) => (
              <option disabled={value === "ema" && emaOff} key={value} value={value}>
                {value}
              </option>
            ))}
          </select>
        </label>
      );
    }
    const text = spec.kind === "ints";
    return (
      <label key={key}>
        {label}
        <input
          onChange={(event) => setBag(key, event.target.value)}
          placeholder={spec.kind === "ints" ? "e.g. 1000, 5000" : undefined}
          step={text ? undefined : "any"}
          type={text ? "text" : "number"}
          value={bag[key] ?? ""}
          {...invalid}
        />
      </label>
    );
  };

  async function runCheck() {
    setCheck({ loading: true });
    try {
      setCheck({ report: await onCheckDataset() });
    } catch (error) {
      setCheck({ error: errorMessage(error, "Could not check the dataset.") });
    }
  }

  return (
    <section className="training-control-note" aria-label={`Iris ${objective} training`}>
      <strong>{depth ? "Iris depth training" : "Iris restoration training"}</strong>
      <div className="training-advanced-grid">
        {depth ? (
          <label>
            Start from
            <select onChange={(event) => updateConfigDraft("depthInit", event.target.value)} value={configDraft.depthInit ?? "export"}>
              <option value="export">Iris 3B Depth export (fine-tune)</option>
              <option value="parent">Generation backbone (paper start)</option>
            </select>
          </label>
        ) : (
          field("init", "Start from")
        )}
        {depth ? field("depth_size") : null}
        {depth ? field("invalid_examples") : null}
      </div>
      <p className="training-field-hint">
        {depth
          ? `Export fine-tunes the installed depth model; the backbone start uses its empty prompt (else the generation text encoder, once). ${POLICY_TEXT.reject} ${POLICY_TEXT.skip}`
          : "Parent needs the Iris 3B install. Use clean images or a degraded image on every item (Data Sets → Iris sidecars). A step = generator + critic window."}
      </p>
      <button className="secondary-action" disabled={!onCheckDataset || check?.loading} onClick={runCheck} type="button">
        {check?.loading ? "Checking…" : "Check dataset"}
      </button>
      {check?.error ? <p className="inline-warning">{check.error}</p> : null}
      {summary ? (
        <div aria-label="Dataset check" className="training-field-hint" role="status">
          <p>
            <strong>{summary.state}.</strong> {summary.counts}. {summary.policy}
          </p>
          {summary.configError ? <p className="inline-warning">{summary.configError}</p> : null}
          {summary.problems.map((item) => (
            <p className="inline-warning" key={item.itemId}>
              {item.displayName ?? item.itemId}: {item.reason}
            </p>
          ))}
        </div>
      ) : null}

      <details className="training-iris-options">
        <summary>Iris {objective} options</summary>
        {GROUPS[objective].map(([title, keys]) => (
          <React.Fragment key={title}>
            <p className="training-field-hint"><strong>{title}</strong></p>
            <div className="training-advanced-grid">{keys.map((key) => field(key))}</div>
          </React.Fragment>
        ))}
        {depth ? null : (
          <label className="training-sample-prompts">
            {specs.degradation.label}
            <textarea
              onChange={(event) => setBag("degradation", event.target.value)}
              placeholder='Optional JSON for clean-image datasets, e.g. {"use_hflip": true}'
              rows={3}
              value={bag.degradation ?? ""}
              {...invalidProps(configValidity, "irisTask.degradation")}
            />
          </label>
        )}
        <p className="training-field-hint"><strong>Resume / continue</strong></p>
        <div className="training-advanced-grid">
          <label>
            Resume a previous run
            <select onChange={(event) => updateConfigDraft("resumeFromJobId", event.target.value)} value={configDraft.resumeFromJobId ?? ""}>
              <option value="">Start a new run</option>
              {previousRuns.map((job) => (
                <option key={job.id} value={job.id}>{irisRunLabel(job)}</option>
              ))}
            </select>
          </label>
          {field("resume_data_policy")}
          {field("override_lr_on_resume", "Use this form's learning rate on resume")}
          {depth ? (
            <label>
              Start from a previous run's weights
              <select
                disabled={resuming || configDraft.depthInit === "parent"}
                onChange={(event) => updateConfigDraft("loadFromJobId", event.target.value)}
                value={configDraft.loadFromJobId ?? ""}
                {...invalidProps(configValidity, "loadFromJobId")}
              >
                <option value="">The selected start</option>
                {previousRuns.map((job) => (
                  <option key={job.id} value={job.id}>{irisRunLabel(job)}</option>
                ))}
              </select>
            </label>
          ) : null}
        </div>
      </details>
    </section>
  );
}
