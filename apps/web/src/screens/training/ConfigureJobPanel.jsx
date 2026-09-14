import React from "react";

import { AdvancedSection } from "../../components/AdvancedSection.jsx";
import { Icon } from "../../components/Icons.jsx";
import { RequiredModelsNotice } from "../../components/RequiredModelsNotice.jsx";
import { WorkPanel } from "../../components/WorkPanel.jsx";
import { DatasetDoctorReadout } from "./DatasetDoctor.jsx";
import { invalidProps, ReadyPill, ValidationSummary } from "../../validation/Validation.jsx";
import {
  lossTypeOptions,
  ltx25WorkflowPlan,
  networkTypeLabel,
  optimizerLabel,
  optionLabel,
  qualityPresetLabel,
  timestepBiasOptions,
  timestepTypeOptionsForTarget,
  trainingAdapterVersionLabels,
} from "../../training/trainingConfig.js";

// Configure-training-job panel. The Purpose zone of the Training Studio under the
// page-frame standard (sc-10475): one work-panel holding the twelve basic plan
// fields, the Advanced disclosure with every remaining knob, and the actions row.
// All state and handlers are owned by the TrainingStudio screen and passed in.
//
// The basic grid holds the twelve fields the design calls out. Two deviations from
// its literal ordering: the "Base model" slot is our `Target` select (every target
// maps 1:1 to a base model and names it in its own label; the payload carries
// `targetId` alone and the server resolves the base), and `Quality` sits beside
// `Preset` because it is a tier OF the preset, not an independent axis. GPU is a
// runtime routing knob rather than part of the plan, so it lives in Advanced.
//
// The actions row is deliberately just Reset defaults + Start training (sc-10492).
// The preset-value strip, the run-mode select and the dry-run explainer all read as
// noise around the one button that matters. Readiness shows in the head's Ready /
// Needs input pill and in the disabled Start button.
//
// `configValidity` is the whole validation summary (epic 10644): the same object gates
// Start, tones the pill, fills the chip row, and outlines the inputs the chips name.
// Missing-field hints stay suppressed — you can see an empty field — but a cleared
// number leaves nothing on screen to explain the dead button (sc-10501).
export function ConfigureJobPanel({
  setActiveView,
  configValidity,
  trainingTargetsError,
  trainingPresetsError,
  configError,
  configMessage,
  selectedTarget,
  setSelectedTargetId,
  trainingTargets,
  macTargetBlocked,
  updateSelectedPreset,
  updateQualityTier,
  selectedPreset,
  targetPresets,
  openDataset,
  activeDataset,
  datasets,
  updateConfigDraft,
  configDraft,
  outputScopes,
  qualityTiers,
  gpuOptions,
  showAdvancedConfig,
  setShowAdvancedConfig,
  showNetworkType,
  networkTypeOptions,
  macLokrOnWanBlocked,
  isLokrNetwork,
  isFullFinetune,
  visibleOptimizerOptions,
  visibleLrSchedulerOptions,
  showTrainingAdapter,
  visibleTrainingAdapterVersions,
  visibleResolutionOptions,
  submittingJob,
  resetConfigDefaults,
  submitTrainingJob,
  configSnapshot,
  // sc-8942 (F-140): grouped Dataset Doctor readout props (report/loading + the six
  // fix-action callbacks), shared verbatim with DatasetEditorPanel. Spread straight onto
  // DatasetDoctorReadout below. Whether readiness blocks the run is now one of
  // `configValidity`'s issues (sc-10648), so there is no separate readiness prop.
  datasetDoctor,
  // Preprocessor models this run needs but doesn't have (modelEligibility.missingRequiredModels).
  // A pose control branch renders its condition with DWPose, whose resolver is cache-only since
  // epic 17625 — so without it the run reaches the worker and dies there. The same list is folded
  // into `configValidity` by the screen, which is what actually blocks Start training; this only
  // renders the offer. Empty for every LoRA target.
  missingControlModels = [],
  controlModelDownloadJobs = [],
  onDownloadModel,
  onOpenModels,
  onOpenQueue,
  onCancelJob,
}) {
  // ControlNet training (epic 10159) reuses this panel: a `control_branch` target renders the
  // per-image control condition from the selected dataset (the data source) and trains a control
  // branch instead of a LoRA. Surface that so the run reads as ControlNet, not a mislabeled LoRA.
  const isControlTarget = selectedTarget?.outputKind === "control_branch";
  const controlType =
    selectedTarget?.defaults?.advanced?.controlType ?? selectedTarget?.limits?.controlTypes?.[0] ?? "pose";
  const fullFinetuneConfig = isFullFinetune
    ? selectedTarget?.defaults?.advanced?.fullFinetuneConfig
    : null;
  const requiredFullPrecision = fullFinetuneConfig?.mixedPrecision;
  const fullCheckpointingUnsupported = fullFinetuneConfig?.gradientCheckpointing === false;
  const visibleTimestepTypeOptions = timestepTypeOptionsForTarget(selectedTarget);
  const ltxWorkflows = selectedTarget?.baseModel === "ltx_2_5"
    ? (selectedTarget?.limits?.ltxWorkflows ?? [])
    : [];
  const updateLtxWorkflow = (workflow) => {
    const plan = ltx25WorkflowPlan(workflow);
    updateConfigDraft("ltxWorkflow", workflow);
    updateConfigDraft("ltxVideo", plan.video);
    updateConfigDraft("ltxAudio", plan.audio);
  };
  const updateLtxCondition = (field, index, key, value) => {
    const modality = configDraft[field];
    updateConfigDraft(field, {
      ...modality,
      conditions: (modality?.conditions ?? []).map((entry, entryIndex) =>
        entryIndex === index ? { ...entry, [key]: value } : entry,
      ),
    });
  };
  const updateLtxValidation = (key, value) => {
    updateConfigDraft("ltxValidation", { ...(configDraft.ltxValidation ?? {}), [key]: value });
  };
  return (
    <WorkPanel
      className="training-config-panel"
      eyebrow="Configure the run"
      hint="Pick a captioned dataset from the Data Sets library, choose a target and a preset, then queue the plan."
      actions={
        <>
          <button className="secondary-action" onClick={() => setActiveView?.("LibraryDataSets")} type="button">
            <Icon.Library size={14} />
            Data Sets
          </button>
          <ReadyPill ready={configValidity.ready} />
        </>
      }
    >
      {trainingTargetsError ? <p className="inline-warning">{trainingTargetsError}</p> : null}
      {trainingPresetsError ? <p className="inline-warning">{trainingPresetsError}</p> : null}
      {configError ? <p className="inline-warning">{configError}</p> : null}
      {configMessage ? <p className="inline-success">{configMessage}</p> : null}
      {!selectedTarget ? (
        <div className="empty-panel compact-panel">Training target registry unavailable</div>
      ) : (
        <div className="training-config-form" aria-label="Training job configuration">
          <div className="training-config-grid">
            <label>
              Dataset
              <select onChange={(event) => openDataset(event.target.value)} value={activeDataset?.id ?? ""}>
                <option value="">Select a saved dataset</option>
                {datasets.map((dataset) => (
                  <option key={dataset.id} value={dataset.id}>
                    {dataset.name ?? dataset.id}
                  </option>
                ))}
              </select>
            </label>
            <label>
              Target
              <select onChange={(event) => setSelectedTargetId(event.target.value)} value={selectedTarget.id}>
                {trainingTargets.map((target) => {
                  const blocked = macTargetBlocked(target);
                  return (
                    <option key={target.id} value={target.id} disabled={blocked}>
                      {target.ui?.label ?? target.name}
                      {blocked ? " — not on Mac (Cuda only)" : ""}
                    </option>
                  );
                })}
              </select>
            </label>
            <label>
              Preset
              <select onChange={(event) => updateSelectedPreset(event.target.value)} value={selectedPreset?.id ?? ""}>
                {targetPresets.length ? null : <option value="">Target defaults</option>}
                {targetPresets.map((preset) => (
                  <option key={preset.id} value={preset.id}>
                    {preset.name}
                  </option>
                ))}
              </select>
            </label>
            {/* Quality is a tier OF the preset (preset ids are
                <target>.<recipe>.<optimizer>.<quality>), so it sits beside Preset and
                picking a tier swaps in the sibling preset. Groups that ship a single
                tier have nothing to choose, so the picker shows it read-only. */}
            <label>
              Quality
              <select
                disabled={qualityTiers.length < 2}
                onChange={(event) => updateQualityTier(event.target.value)}
                value={selectedPreset?.qualityPreset ?? configDraft.qualityPreset ?? ""}
              >
                {qualityTiers.length ? null : (
                  <option value={configDraft.qualityPreset ?? ""}>
                    {qualityPresetLabel(configDraft.qualityPreset) || "Default"}
                  </option>
                )}
                {qualityTiers.map((preset) => (
                  <option key={preset.id} value={preset.qualityPreset}>
                    {qualityPresetLabel(preset.qualityPreset)}
                  </option>
                ))}
              </select>
            </label>

            <label>
              {/* sc-15036: the same target produces an adapter or a base checkpoint depending on
                  `networkType`, so the RUN decides this label, not the target alone. Kept as an
                  explicit three-way branch (rather than `outputKindLabel`) so a target that
                  declares no `outputKind` still reads "LoRA name" exactly as before. */}
              {isFullFinetune ? "Base checkpoint name" : isControlTarget ? "Control branch name" : "LoRA name"}
              <input onChange={(event) => updateConfigDraft("outputName", event.target.value)} value={configDraft.outputName ?? ""} />
            </label>
            <label>
              Trigger phrase
              <input onChange={(event) => updateConfigDraft("triggerWord", event.target.value)} value={configDraft.triggerWord ?? ""} />
            </label>
            <label>
              Steps
              <input
                onChange={(event) => updateConfigDraft("steps", event.target.value)}
                type="number"
                value={configDraft.steps ?? ""}
                {...invalidProps(configValidity, "steps")}
              />
            </label>
            <label>
              Checkpoint cadence
              <input
                onChange={(event) => updateConfigDraft("saveEvery", event.target.value)}
                type="number"
                value={configDraft.saveEvery ?? ""}
                {...invalidProps(configValidity, "saveEvery")}
              />
            </label>

            {/* sc-15036: a full base fine-tune produces a MODEL, and the model catalog has one
                global user manifest — there is no project-scoped model store to honour a "project"
                scope with. Replace the picker with the explanation rather than leaving a control
                that silently does nothing (the sc-14056 gradient-checkpointing precedent below). */}
            {isFullFinetune ? (
              <label>
                Output scope
                <p className="training-field-hint">
                  A full base fine-tune is registered as a model in your global model library, so it is available in
                  every project. Only adapters are scoped to a single project.
                </p>
              </label>
            ) : (
              <label>
                Output scope
                <select onChange={(event) => updateConfigDraft("outputScope", event.target.value)} value={configDraft.outputScope ?? ""}>
                  {outputScopes.length ? null : <option value={configDraft.outputScope ?? ""}>{configDraft.outputScope || "Default"}</option>}
                  {outputScopes.map((scope) => (
                    <option key={scope} value={scope}>
                      {scope}
                    </option>
                  ))}
                </select>
              </label>
            )}
            <label>
              Sample count
              <input
                min="0"
                onChange={(event) => updateConfigDraft("sampleCount", event.target.value)}
                type="number"
                value={configDraft.sampleCount ?? ""}
              />
            </label>
            <label>
              Sample steps
              <input
                onChange={(event) => updateConfigDraft("sampleSteps", event.target.value)}
                type="number"
                value={configDraft.sampleSteps ?? ""}
              />
            </label>
            <label>
              Sample cadence
              <input
                onChange={(event) => updateConfigDraft("sampleEvery", event.target.value)}
                type="number"
                value={configDraft.sampleEvery ?? ""}
              />
            </label>
          </div>

          {ltxWorkflows.length ? (
            <section className="training-control-note" aria-label="LTX-2.5 workflow">
              <strong>LTX-2.5 prepared workflow</strong>
              <label>
                Workflow
                <select
                  onChange={(event) => updateLtxWorkflow(event.target.value)}
                  value={configDraft.ltxWorkflow ?? ""}
                  {...invalidProps(configValidity, "ltxWorkflow")}
                >
                  {ltxWorkflows.map((workflow) => (
                    <option key={workflow} value={workflow}>{optionLabel(workflow)}</option>
                  ))}
                </select>
              </label>
              {["ltxVideo", "ltxAudio"].map((field) => {
                const modality = configDraft[field];
                if (!modality) return null;
                return (
                  <div key={field} className="training-advanced-grid">
                    <p><strong>{field === "ltxVideo" ? "Video" : "Audio"}</strong> · {modality.isGenerated ? "generated" : "conditioning only"}</p>
                    {(modality.conditions ?? []).map((entry, index) => (
                      <React.Fragment key={`${entry.type}-${index}`}>
                        <label>
                          {optionLabel(entry.type)} probability
                          <input min="0" max="1" step="0.05" type="number" value={entry.probability ?? ""}
                            onChange={(event) => updateLtxCondition(field, index, "probability", event.target.value)} />
                        </label>
                        {["mask", "reference"].includes(entry.type) ? (
                          <label>
                            Tensor key
                            <input value={entry.tensorKey ?? ""}
                              onChange={(event) => updateLtxCondition(field, index, "tensorKey", event.target.value)} />
                          </label>
                        ) : null}
                        {["prefix", "suffix"].includes(entry.type) ? (
                          <label>
                            Temporal boundary
                            <input min="1" step="1" type="number" value={entry.temporalBoundary ?? ""}
                              onChange={(event) => updateLtxCondition(field, index, "temporalBoundary", event.target.value)} />
                          </label>
                        ) : null}
                        {entry.type === "spatialCrop" ? (
                          <label>
                            Spatial region (y1,x1,y2,x2)
                            <input value={Array.isArray(entry.spatialRegion) ? entry.spatialRegion.join(",") : entry.spatialRegion ?? ""}
                              onChange={(event) => updateLtxCondition(field, index, "spatialRegion", event.target.value)} />
                          </label>
                        ) : null}
                        {entry.type === "reference" ? (
                          <>
                            <label>Spatial scale factor<input min="1" step="1" type="number" value={entry.spatialScaleFactor ?? ""}
                              onChange={(event) => updateLtxCondition(field, index, "spatialScaleFactor", event.target.value)} /></label>
                            <label>Temporal scale factor<input min="1" step="1" type="number" value={entry.temporalScaleFactor ?? ""}
                              onChange={(event) => updateLtxCondition(field, index, "temporalScaleFactor", event.target.value)} /></label>
                          </>
                        ) : null}
                      </React.Fragment>
                    ))}
                  </div>
                );
              })}
              <details>
                <summary>Validation recipe</summary>
                <div className="training-advanced-grid">
                  {[
                    ["width", "Width", 32, 4096, 32], ["height", "Height", 32, 4096, 32],
                    ["frames", "Frames", 1, 257, 8], ["fps", "FPS", 1, 120, 1],
                    ["steps", "Steps", 1, 100, 1], ["videoCfgScale", "Video CFG scale", 0, 20, 0.1],
                    ["audioCfgScale", "Audio CFG scale", 0, 20, 0.1], ["videoStgScale", "Video STG scale", 0, 20, 0.1],
                    ["audioStgScale", "Audio STG scale", 0, 20, 0.1], ["guidanceRescale", "Guidance rescale", 0, 1, 0.1],
                    ["videoModalityGuidanceScale", "Video modality guidance", 0, 20, 0.1],
                    ["audioModalityGuidanceScale", "Audio modality guidance", 0, 20, 0.1],
                  ].map(([key, label, min, max, step]) => (
                    <label key={key}>{label}<input min={min} max={max} step={step} type="number"
                      value={configDraft.ltxValidation?.[key] ?? ""}
                      onChange={(event) => updateLtxValidation(key, event.target.value)} /></label>
                  ))}
                  <label>STG block<input min="0" max="47" step="1" type="number" value={Array.isArray(configDraft.ltxValidation?.stgBlocks)
                    ? configDraft.ltxValidation.stgBlocks.join(",") : configDraft.ltxValidation?.stgBlocks ?? ""}
                    onChange={(event) => updateLtxValidation("stgBlocks", event.target.value)} /></label>
                  <label className="training-checkbox-field"><input type="checkbox"
                    checked={Boolean(configDraft.ltxValidation?.generateAudio)}
                    onChange={(event) => updateLtxValidation("generateAudio", event.target.checked)} />Generate validation audio</label>
                </div>
              </details>
            </section>
          ) : null}

          {isControlTarget ? (
            <p className="training-control-note inline-success">
              <strong>ControlNet training.</strong> A {controlType} condition is rendered from each image
              in the selected dataset — your data source — then a control branch is trained for{" "}
              {selectedTarget.ui?.label ?? selectedTarget.name} (applied at generation time). Use a
              captioned dataset; bring-your-own prepared/annotated datasets are coming next.
            </p>
          ) : null}

          {/* The condition above is rendered by a preprocessor, and a pose one needs DWPose. Offer
              it here rather than letting the queued run fail at the worker with an install error
              the Training Studio gives no way to act on. */}
          <RequiredModelsNotice
            detail="Preprocessing runs locally on the native worker, so the run would fail without it."
            downloadJobs={controlModelDownloadJobs}
            feature={`${controlType.charAt(0).toUpperCase()}${controlType.slice(1)} ControlNet training`}
            models={missingControlModels}
            onCancelJob={onCancelJob}
            onDownload={onDownloadModel}
            onOpenModels={onOpenModels}
            onOpenQueue={onOpenQueue}
          />

          <AdvancedSection
            hint="cleared values → preset default"
            onToggle={() => setShowAdvancedConfig(!showAdvancedConfig)}
            open={showAdvancedConfig}
          >
            <div className="training-advanced-grid">
              <label>
                Requested GPU
                <select onChange={(event) => updateConfigDraft("requestedGpu", event.target.value)} value={configDraft.requestedGpu ?? ""}>
                  {gpuOptions.map((gpu) => (
                    <option key={gpu} value={gpu}>
                      {gpu === "auto" ? "Auto" : `GPU ${gpu}`}
                    </option>
                  ))}
                </select>
              </label>
              <label>
                Rank
                <input
                  onChange={(event) => updateConfigDraft("rank", event.target.value)}
                  type="number"
                  value={configDraft.rank ?? ""}
                  {...invalidProps(configValidity, "rank")}
                />
              </label>
              <label>
                Alpha
                <input
                  onChange={(event) => updateConfigDraft("alpha", event.target.value)}
                  type="number"
                  value={configDraft.alpha ?? ""}
                  {...invalidProps(configValidity, "alpha")}
                />
              </label>
              {/* Real hyperparameters the config validates (sc-10689). They live here
                  beside the other numeric knobs so every `> 0` error the rule set can
                  raise names an input the user can reach. The draft always seeds a
                  working value (configDraftFromTarget), so these are never empty. */}
              <label title="Images per optimizer step. Higher batches smooth gradients but cost more VRAM.">
                Batch size
                <input
                  min="1"
                  onChange={(event) => updateConfigDraft("batchSize", event.target.value)}
                  type="number"
                  value={configDraft.batchSize ?? ""}
                  {...invalidProps(configValidity, "batchSize")}
                />
              </label>
              <label title="Optimizer steps accumulated before an update — multiplies the effective batch size without extra VRAM.">
                Gradient accumulation
                <input
                  min="1"
                  onChange={(event) => updateConfigDraft("gradientAccumulation", event.target.value)}
                  type="number"
                  value={configDraft.gradientAccumulation ?? ""}
                  {...invalidProps(configValidity, "gradientAccumulation")}
                />
              </label>
              {showNetworkType ? (
                <label title="What the run trains. LoRA is the standard low-rank adapter; LoKr (LyCORIS Kronecker) trains a much smaller, often more expressive adapter on targets that advertise it; Full base fine-tune updates every base weight and writes a fine-tuned checkpoint instead of an adapter — far more memory, offered only where the engine supports it.">
                  Network type
                  <select
                    onChange={(event) => updateConfigDraft("networkType", event.target.value)}
                    value={configDraft.networkType ?? "lora"}
                  >
                    {networkTypeOptions.map((option) => {
                      const blocked = option === "lokr" && macLokrOnWanBlocked;
                      return (
                        <option key={option} value={option} disabled={blocked}>
                          {networkTypeLabel(option)}
                          {blocked ? " — not on Mac for Wan targets" : ""}
                        </option>
                      );
                    })}
                  </select>
                </label>
              ) : null}
              {showNetworkType && isLokrNetwork ? (
                <label title="LoKr block-decomposition factor. -1 lets LyCORIS pick the largest factor automatically; larger values trade adapter size for capacity.">
                  LoKr factor
                  <input
                    min="-1"
                    onChange={(event) => updateConfigDraft("decomposeFactor", event.target.value)}
                    step="1"
                    type="number"
                    value={configDraft.decomposeFactor ?? ""}
                  />
                </label>
              ) : null}
              <label>
                Optimizer
                <select onChange={(event) => updateConfigDraft("optimizer", event.target.value)} value={configDraft.optimizer ?? ""}>
                  {visibleOptimizerOptions.map((optimizer) => (
                    <option key={optimizer} value={optimizer}>
                      {optimizerLabel(optimizer)}
                    </option>
                  ))}
                </select>
              </label>
              <label>
                Learning rate
                <input
                  onChange={(event) => updateConfigDraft("learningRate", event.target.value)}
                  step="0.00001"
                  type="number"
                  value={configDraft.learningRate ?? ""}
                  {...invalidProps(configValidity, "learningRate")}
                />
              </label>
              <label>
                Weight decay
                <input
                  onChange={(event) => updateConfigDraft("weightDecay", event.target.value)}
                  step="0.00001"
                  type="number"
                  value={configDraft.weightDecay ?? ""}
                />
              </label>
              <label title="Learning-rate scheduler (not the timestep/noise scheduler). Constant holds the LR fixed for the whole run; linear and cosine decay it toward zero over the run.">
                LR scheduler
                <select onChange={(event) => updateConfigDraft("lrScheduler", event.target.value)} value={configDraft.lrScheduler ?? ""}>
                  {visibleLrSchedulerOptions.map((option) => (
                    <option key={option} value={option}>
                      {optionLabel(option)}
                    </option>
                  ))}
                </select>
              </label>
              <label title="Optional linear warmup: number of steps to ramp the LR up from zero before the scheduler body runs. 0 disables warmup.">
                LR warmup steps
                <input
                  min="0"
                  onChange={(event) => updateConfigDraft("lrWarmupSteps", event.target.value)}
                  type="number"
                  value={configDraft.lrWarmupSteps ?? ""}
                />
              </label>
              <label>
                Timestep type
                <select onChange={(event) => updateConfigDraft("timestepType", event.target.value)} value={configDraft.timestepType ?? ""}>
                  {visibleTimestepTypeOptions.map((option) => (
                    <option key={option} value={option}>
                      {optionLabel(option)}
                    </option>
                  ))}
                </select>
              </label>
              <label>
                Timestep bias
                <select onChange={(event) => updateConfigDraft("timestepBias", event.target.value)} value={configDraft.timestepBias ?? ""}>
                  {timestepBiasOptions.map((option) => (
                    <option key={option} value={option}>
                      {optionLabel(option)}
                    </option>
                  ))}
                </select>
              </label>
              <label>
                Loss type
                <select onChange={(event) => updateConfigDraft("lossType", event.target.value)} value={configDraft.lossType ?? ""}>
                  {lossTypeOptions.map((option) => (
                    <option key={option} value={option}>
                      {option === "mse" ? "Mean Squared Error" : optionLabel(option)}
                    </option>
                  ))}
                </select>
              </label>
              {showTrainingAdapter ? (
                <label title="ostris de-distill adapter for the step-distilled Z-Image-Turbo base. Fused in for training, removed at inference. v1 is stable; v2 is a heavier, experimental de-distill.">
                  De-distill adapter
                  <select
                    onChange={(event) => updateConfigDraft("trainingAdapterVersion", event.target.value)}
                    value={configDraft.trainingAdapterVersion ?? ""}
                  >
                    {visibleTrainingAdapterVersions.map((version) => (
                      <option key={version} value={version}>
                        {trainingAdapterVersionLabels[version] ?? version}
                      </option>
                    ))}
                  </select>
                </label>
              ) : null}
              <label>
                Resolution
                <select
                  onChange={(event) => updateConfigDraft("resolution", event.target.value)}
                  value={configDraft.resolution ?? ""}
                  {...invalidProps(configValidity, "resolution")}
                >
                  {visibleResolutionOptions.length ? null : <option value={configDraft.resolution ?? ""}>{configDraft.resolution ?? ""}</option>}
                  {visibleResolutionOptions.map((resolution) => (
                    <option key={resolution} value={resolution}>
                      {resolution}
                    </option>
                  ))}
                </select>
              </label>
              <label>
                Precision
                <input
                  disabled={Boolean(requiredFullPrecision)}
                  onChange={(event) => updateConfigDraft("precision", event.target.value)}
                  value={requiredFullPrecision ?? configDraft.precision ?? ""}
                />
                {requiredFullPrecision ? (
                  <span className="training-field-hint">
                    This backend requires {requiredFullPrecision.toUpperCase()} for full base fine-tuning.
                  </span>
                ) : null}
              </label>
              <label>
                Guidance scale
                <input
                  onChange={(event) => updateConfigDraft("sampleGuidanceScale", event.target.value)}
                  step="0.1"
                  type="number"
                  value={configDraft.sampleGuidanceScale ?? ""}
                />
              </label>
            </div>

            <label className="training-sample-prompts">
              Sample prompts
              <textarea
                onChange={(event) => updateConfigDraft("samplePrompts", event.target.value)}
                placeholder="One prompt per line. Leave blank to use the trigger-phrase defaults."
                rows={4}
                value={configDraft.samplePrompts ?? ""}
              />
              <span className="training-field-hint">
                One prompt per line. Renders one preview per prompt, up to the sample count.
              </span>
            </label>

            <div className="training-advanced-toggles">
              {fullCheckpointingUnsupported ? (
                <p className="training-field-hint">
                  Gradient checkpointing is not available for a full base fine-tune yet, so it is not applied to this
                  run. Lower the training resolution if the run does not fit.
                </p>
              ) : (
                <label className="training-checkbox-field">
                  <input
                    checked={Boolean(configDraft.gradientCheckpointing)}
                    onChange={(event) => updateConfigDraft("gradientCheckpointing", event.target.checked)}
                    type="checkbox"
                  />
                  Gradient checkpointing
                </label>
              )}
            </div>
          </AdvancedSection>

          {/* Dataset Doctor readout before the Train button (sc-6534). Advisory: it
              only hard-blocks training when the gate is Blocked (too few images / a
              fatal flag); warnings stay informational. A Blocked gate now rides in the
              chip row below as one of configValidity's errors (sc-10648), so the
              hand-rolled "isn't ready to train" paragraph is gone. */}
          <DatasetDoctorReadout {...datasetDoctor} compact />

          {/* Only broken values, never the "you haven't picked a dataset yet" hints —
              those are obvious from the form. Sits against the actions row so it reads
              as the reason Start training is dead (sc-10501). */}
          <ValidationSummary issues={configValidity.surfaced} label="Configuration errors" />

          <div className="training-config-actions">
            <button className="secondary-action" onClick={resetConfigDefaults} type="button">
              Reset defaults
            </button>
            <button
              className="primary-action"
              disabled={!configValidity.ready || submittingJob}
              onClick={submitTrainingJob}
              type="button"
            >
              {submittingJob ? "Queuing" : "Start training"}
            </button>
          </div>
          {configSnapshot ? <pre className="training-config-snapshot">{JSON.stringify(configSnapshot, null, 2)}</pre> : null}
        </div>
      )}
    </WorkPanel>
  );
}
