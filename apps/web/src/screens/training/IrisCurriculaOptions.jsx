import React from "react";

import { invalidProps } from "../../validation/Validation.jsx";
import { IRIS_CURRICULA_SPECS, curriculaInPlay, irisGlobalBatch, validationRows } from "../../training/irisCurricula.js";

// Iris-3B curricula, representation alignment, validation and ranks (sc-25689). Every control is
// one `advanced.iris` option the trainer honors (crates/sceneworks-core/src/training/iris/
// curricula.rs); a control the run's own choices make inert is not rendered (irisCurricula.js).
// Empty = trainer default, not sent.

const GROUPS = [
  ["Curriculum", ["stage", "parent_stage", "run_name", "tracking.project", "tracking.tags"]],
  ["Data shapes", ["shape_policy", "aspect_ratio_bucket", "shape_align", "shape_max_ratio", "rope_aspect"]],
  ["Representation alignment (REPA)", ["repa.weight", "repa.variant", "repa.spatial_norm_gamma", "repa.layer", "repa.proj_hidden_dim", "repa.teacher_image_size", "repa.teacher_match_student"]],
  ["Memory", ["activation_checkpointing", "ac_selective_every"]],
  ["Validation, logging and retention", ["val_every_steps", "val_samples", "log_every", "log_block_grad_norms", "save_every_epochs", "preview_at_step_one", "preview_order", "resume_optimizer"]],
  ["Ranks", ["dist.world.world_size", "dist.world.share_device", "dist.world.timeout_secs", "dist.world.rendezvous_timeout_secs", "dist.world.cancel_grace_secs", "dist.strategy", "dist.sharding", "dist.mesh.dp_shard", "dist.mesh.dp_replicate", "dist.param_dtype", "dist.reduce_dtype", "dist.reshard_after_forward", "dist.checkpoint_io", "expected_global_batch", "scale_warmup_by_world", "perf.lazy_sync"]],
];

export function IrisRunMetrics({ metrics }) {
  const rows = validationRows(metrics);
  if (!rows.length) return null;
  return (
    <table className="training-iris-metrics" aria-label="Holdout validation">
      <thead>
        <tr><th>Step</th><th>Val loss</th><th>t0</th><th>t1</th><th>t2</th><th>t3</th></tr>
      </thead>
      <tbody>
        {rows.slice(-8).map((row) => (
          <tr key={row.step}>
            <td>{row.step}</td>
            {[row.loss, ...row.bands].map((value, index) => (
              <td key={index}>{Number.isFinite(value) ? value.toFixed(4) : "–"}</td>
            ))}
          </tr>
        ))}
      </tbody>
    </table>
  );
}

export function IrisCurriculaOptions({ configDraft, updateConfigDraft, configValidity, presets = [], onApplyPreset, planCheck, resumeMetrics }) {
  const iris = configDraft.iris ?? {};
  const set = (key, value) => updateConfigDraft("iris", { ...iris, [key]: value });
  const field = (key) => {
    if (!curriculaInPlay(key, configDraft)) return null;
    const spec = IRIS_CURRICULA_SPECS[key];
    const value = iris[key] ?? "";
    if (spec.kind === "bool") {
      return (
        <label className="training-checkbox-field" key={key}>
          <input checked={value === true} onChange={(event) => set(key, event.target.checked)} type="checkbox" />
          {spec.label}
        </label>
      );
    }
    return (
      <label key={key}>
        {spec.label}
        {spec.kind === "strings" ? (
          <input
            onChange={(event) => set(key, event.target.value.split(","))}
            placeholder="comma-separated"
            value={Array.isArray(value) ? value.join(",") : ""}
          />
        ) : spec.kind === "choice" ? (
          <select onChange={(event) => set(key, event.target.value)} value={value} {...invalidProps(configValidity, `iris.${key}`)}>
            <option value="">Default</option>
            {spec.values.map((option) => (
              <option key={option} value={option}>{option}</option>
            ))}
          </select>
        ) : (
          <input
            min={spec.min}
            onChange={(event) => set(key, event.target.value)}
            type={spec.kind === "text" ? "text" : "number"}
            value={value}
            {...invalidProps(configValidity, `iris.${key}`)}
          />
        )}
      </label>
    );
  };
  const teacher = planCheck?.teacher;
  const refusal = planCheck?.refusal;
  return (
    <details className="training-iris-options" aria-label="Iris curriculum options">
      <summary>Curriculum, alignment and ranks</summary>
      {presets.length ? (
        <label>
          Official stage preset
          <select onChange={(event) => onApplyPreset?.(presets.find((preset) => preset.id === event.target.value))} value="">
            <option value="">Apply a stage…</option>
            {presets.map((preset) => (
              <option key={preset.id} value={preset.id}>
                {preset.id}
                {preset.parentStage ? ` (continues ${preset.parentStage})` : ""}
              </option>
            ))}
          </select>
        </label>
      ) : null}
      {GROUPS.map(([title, keys]) => (
        <React.Fragment key={title}>
          <p className="training-field-hint"><strong>{title}</strong></p>
          <div className="training-advanced-grid">{keys.map(field)}</div>
        </React.Fragment>
      ))}
      {Number(iris["repa.weight"]) > 0 && teacher && !teacher.installed ? (
        <p className="inline-warning">REPA needs the DINOv2-B/14 teacher (Apache-2.0): install “Iris REPA Teacher — DINOv2-B/14” in the Model Manager.</p>
      ) : null}
      <p className="training-field-hint" role="status">
        Global batch {irisGlobalBatch(configDraft)} = batch × ranks × accumulation
        {planCheck?.resolved?.dataIdentity ? ` · data identity ${planCheck.resolved.dataIdentity}` : ""}
      </p>
      {refusal ? <p className="inline-warning" role="alert">{refusal.message}</p> : null}
      <IrisRunMetrics metrics={resumeMetrics} />
    </details>
  );
}
