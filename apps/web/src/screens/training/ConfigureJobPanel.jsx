import React from "react";

import { AdvancedSection } from "../../components/AdvancedSection.jsx";
import { Icon } from "../../components/Icons.jsx";
import { RequiredModelsNotice } from "../../components/RequiredModelsNotice.jsx";
import { WorkPanel } from "../../components/WorkPanel.jsx";
import { safeExternalUrl } from "../../urls.js";
import { DatasetDoctorReadout } from "./DatasetDoctor.jsx";
import { IrisTaskOptions } from "./IrisTaskOptions.jsx";
import { IrisTrainingOptions } from "./IrisTrainingOptions.jsx";
import { IrisCurriculaOptions } from "./IrisCurriculaOptions.jsx";
import { invalidProps, ReadyPill, ValidationSummary } from "../../validation/Validation.jsx";
import { numberFromDraft } from "../../training/drafts.js";
import {
  choiceOptionsForTarget,
  gradientCheckpointingLanes,
  gradientCheckpointingUnsupported,
  lossTypeOptions,
  ltx25WorkflowPlan,
  networkTypeLabel,
  optimizerLabel,
  optionLabel,
  qualityPresetLabel,
  rangeOptions,
  resolutionBucketRepeatsMax,
  resolutionBucketsMax,
  resolutionBucketStride,
  seedResolutionBuckets,
  targetSupportsResolutionBuckets,
  subjectMaskCoverage,
  subjectMaskWeightMax,
  targetSupportsSubjectMaskLoss,
  timestepBiasOptions,
  timestepTypeOptionsForTarget,
  targetSupportsWeightNoise,
  trainingAdapterVersionLabels,
  gradientNoiseEtaMax,
  gradientNoiseEtaSuggested,
  gradientNoiseGammaDefault,
  gradientNoiseGammaMax,
  targetSupportsGradientNoise,
  weightNoiseSigmaMax,
  weightNoiseSigmaSuggested,
  depthAnchoringEnabled,
  targetSupportsDepthAnchoring,
  auxModelsInstallNote,
  depthAnchoringEveryDefault,
  depthAnchoringEveryMax,
  depthAnchoringModelLabels,
  depthAnchoringModelOptions,
  depthAnchoringWeightMax,
  depthAnchoringWeightSuggested,
  latentLossEnabled,
  latentPerceptualLosses,
  targetSupportsLatentLoss,
  bodyLosses,
  bodyLossEnabled,
  bodyLossEveryDefault,
  bodyLossEveryMax,
  bodyLossWeightMax,
  bodyLossWeightSuggested,
  bodyShapeMinCosDefault,
  targetSupportsBodyLoss,
  faceLandmarkLossEnabled,
  faceLossEveryDefault,
  faceLossEveryMax,
  faceLossWeightMax,
  faceLossWeightSuggested,
  identityLossEnabled,
  identityLossMinCosDefault,
  identityLossReferenceLabels,
  identityLossReferenceOptions,
  targetSupportsFaceLandmarkLoss,
  targetSupportsIdentityLoss,
} from "../../training/trainingConfig.js";
import { irisFloorSummary, irisMemoryFloors, irisObjective, irisUpstreamPromptsOn } from "../../training/irisOptions.js";
import { irisFloorSize } from "../../training/irisTaskOptions.js";

// A number input bound to the config-draft key `field` (`draft` = { configDraft, configValidity,
// updateConfigDraft }), outlined when a validation chip names it. `input` overrides min/type and adds
// max/step/placeholder.
function DraftNumber({ draft, field, label, title, ...input }) {
  return (
    <label title={title}>
      {label}
      <input
        min="0"
        type="number"
        {...input}
        onChange={(event) => draft.updateConfigDraft(field, event.target.value)}
        value={draft.configDraft[field] ?? ""}
        {...invalidProps(draft.configValidity, field)}
      />
    </label>
  );
}

// One training-technique checkbox (epic 2123); `invalid` is the invalidProps of the key it sets.
function TechniqueToggle({ label, title, invalid, ...input }) {
  return (
    <label className="training-checkbox-field" title={title}>
      <input type="checkbox" {...input} {...invalid} />
      {label}
    </label>
  );
}

// The weight / noise window / alternation knobs of one auxiliary loss (an entry of the panel's
// `auxLosses`), shown while the loss is on, plus its loss-specific `afterWeight` / `after` controls.
function AuxLossKnobs({ loss, draft }) {
  const { prefix, stem, noun, window: bounds } = loss;
  return (
    <>
      <DraftNumber
        draft={draft}
        field={`${prefix}Weight`}
        label={loss.weightLabel ?? `${stem} weight`}
        max={loss.weightMax}
        step={loss.weightStep ?? "0.01"}
        title={loss.weightTitle}
      />
      {loss.afterWeight}
      <DraftNumber
        draft={draft}
        field={`${prefix}MinT`}
        label={`${stem} window min`}
        max="1"
        placeholder={String(bounds[0])}
        step="0.05"
        title={`Lowest noise level (0 = clean image, 1 = pure noise) at which the ${noun} applies. Empty = ${bounds[0]}.`}
      />
      <DraftNumber
        draft={draft}
        field={`${prefix}MaxT`}
        label={`${stem} window max`}
        max="1"
        placeholder={String(bounds[1])}
        step="0.05"
        title={`Highest noise level at which the ${noun} applies. Empty = ${bounds[1]}.`}
      />
      <DraftNumber
        draft={draft}
        field={`${prefix}Every`}
        label={`${stem} every N steps`}
        max={loss.everyMax}
        min="1"
        placeholder={String(loss.everyDefault)}
        step="1"
        title={
          loss.everyTitle ??
          `Every Nth step trains the ${noun} alone; the steps between train the normal loss. 1 adds ${loss.everyIt ?? "it"} to every step instead.`
        }
      />
      {loss.after}
    </>
  );
}

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
  // Why a blocked target is blocked, as the option suffix (sc-24161: off-Mac vs on-Mac gates).
  targetBlockedLabel = () => " — not on Mac (Cuda only)",
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
  // The de-distill training adapter this run's selected version needs but doesn't have (sc-25213);
  // folded into `configValidity` by the screen, rendered here as the download offer.
  missingTrainingAdapterModels = [],
  // The SAM3 Person Segmenter, when this run must generate missing subject masks and it is not
  // installed (sc-2124); folded into `configValidity` by the screen, rendered here as the offer.
  missingSubjectSegmenterModels = [],
  // The licence the selected target's base imposes on what is trained from it
  // (`trainingTargetLicense`, epic 24107 E12 — Qwen Image 2.1's Qwen RESEARCH licence), or null.
  // Rendered as a notice with a required acceptance; whether it blocks Start is one of
  // `configValidity`'s issues, folded in by the screen.
  targetLicense = null,
  licenseAcknowledged = false,
  onLicenseAcknowledgedChange,
  controlModelDownloadJobs = [],
  onDownloadModel,
  onOpenModels,
  onOpenQueue,
  onCancelJob,
  // Iris-3B (sc-25685): the previous runs a new run can resume / start from, the active dataset's
  // caption field names, the active training lane ("mlx" | "candle" | null = unknown) for the
  // declared memory floor, and the data-walk preflight result for the selected dataset/batch.
  irisContext = {},
  // Subject-masked loss (sc-24828): the active dataset's /subject-masks report (null = unknown)
  // and the screen's "Generate subject masks" action (saves the dataset, queues the SAM3 job).
  subjectMaskReport = null,
  onGenerateSubjectMasks,
}) {
  // ControlNet training (epic 10159) reuses this panel: a `control_branch` target renders the
  // per-image control condition from the selected dataset (the data source) and trains a control
  // branch instead of a LoRA. Surface that so the run reads as ControlNet, not a mislabeled LoRA.
  const isControlTarget = selectedTarget?.outputKind === "control_branch";
  const isMossTrainingTarget = selectedTarget?.kernel === "moss_image_lora";
  const controlType =
    selectedTarget?.defaults?.advanced?.controlType ?? selectedTarget?.limits?.controlTypes?.[0] ?? "pose";
  const fullFinetuneConfig = isFullFinetune
    ? selectedTarget?.defaults?.advanced?.fullFinetuneConfig
    : null;
  const requiredFullPrecision = fullFinetuneConfig?.mixedPrecision;
  const fullCheckpointingUnsupported = fullFinetuneConfig?.gradientCheckpointing === false;
  // Adapter noise (epic 2123) is offered only where the target's trainer declares it — the targets
  // endpoint's `limits.supportsWeightNoise` / `limits.supportsGradientNoise` (every LoRA/LoKr target
  // since sc-24827; never the control-branch target) — elsewhere the run would be refused.
  // A toggle is checked for a positive value (a "0" draft is off); the input stays visible while the
  // draft holds any value so an out-of-range entry can be corrected in place (and typing "0.0…"
  // never unmounts the field mid-edit).
  const weightNoiseSupported = targetSupportsWeightNoise(selectedTarget);
  const weightNoiseEnabled = (numberFromDraft(configDraft.weightNoiseSigma) ?? 0) > 0;
  const weightNoiseInputVisible =
    weightNoiseSupported && String(configDraft.weightNoiseSigma ?? "").trim() !== "";
  // Gradient noise: the same on/visible split over eta; gamma is edited alongside it.
  const gradientNoiseSupported = targetSupportsGradientNoise(selectedTarget);
  const gradientNoiseEnabled = (numberFromDraft(configDraft.gradientNoiseEta) ?? 0) > 0;
  const gradientNoiseInputVisible =
    gradientNoiseSupported && String(configDraft.gradientNoiseEta ?? "").trim() !== "";
  // Multi-resolution buckets are on whenever the draft carries a row list (null = off).
  const resolutionBuckets = Array.isArray(configDraft.resolutionBuckets) ? configDraft.resolutionBuckets : null;
  const rectangularBuckets = Boolean(resolutionBuckets?.some((row) => Object.hasOwn(row ?? {}, "width")));
  // Offered only where the target's trainer on this platform declares buckets
  // (`limits.supportsResolutionBuckets`, withheld for LTX-2.5) — elsewhere the run would be refused.
  // A list carried onto an unsupported target keeps its toggle so it can be turned off.
  const resolutionBucketsSupported = targetSupportsResolutionBuckets(selectedTarget);
  // Bucket rows pick from the resolutions the target advertises (the API's menu, E6).
  const bucketResolutionOptions = rangeOptions(selectedTarget?.limits, "resolutions");
  const updateResolutionBucket = (index, key, value) =>
    updateConfigDraft(
      "resolutionBuckets",
      resolutionBuckets.map((row, rowIndex) => (rowIndex === index ? { ...row, [key]: value } : row)),
    );
  // Depth anchoring follows the same target-support mechanism (`limits.supportsDepthAnchoring`);
  // it is on whenever the draft carries a weight (empty = off), and its knobs appear while on.
  // Like the noise toggles, every decoded-x0 / latent loss toggle is gated on target support only:
  // on a combination the engine refuses (a full fine-tune, an LTX-2.5 workflow with no generated
  // video) the toggle stays visible and outlined, and the refusal — attached to its weight key, like
  // the API — blocks Start from the chip row until the user unticks it (epic 2123 review).
  // Body losses (sc-24832): each is offered only where the target's trainer on this platform
  // declares it and its weights are cataloged (`limits.supportsBodyProportionLoss` / …ShapeLoss /
  // …NormalLoss); on whenever the draft carries its weight, knobs shown while on.
  // The frozen models each body loss loads besides the x0 decoder (its help text's install note).
  const bodyLossModels = {
    bodyProportion: ["ViTPose+ Base"],
    bodyShape: ["ViTPose+ Base", "HybrIK ResNet-34"],
    normal: ["ViTPose+ Base", "Sapiens Normal 0.3B"],
  };
  // The face losses (sc-24831) follow the same target-support mechanism
  // (`limits.supportsIdentityLoss` / `limits.supportsFaceLandmarkLoss`); so do the latent-space
  // perceptual losses (sc-24833), one per entry of `latentPerceptualLosses`
  // (`limits.supportsVaeAnchorLoss` / `limits.supportsLatentLpipsLoss`).
  const latentLossHelp = {
    vaeAnchor: `Match the training image at several scales: the model's prediction is decoded and a frozen FLUX.2 VAE encoder compares its features with the training image's. ${auxModelsInstallNote(selectedTarget, ["the FLUX.2 VAE"])} Off by default.`,
    latentLpips:
      "A learned perceptual similarity measured directly on the latent (E-LatentLPIPS) between the model's prediction and the training image. Needs the E-LatentLPIPS weights for this model family installed. Off by default.",
  };
  // Subject-masked loss (sc-24828) is offered only where the target's trainer on this platform
  // declares it (`limits.supportsSubjectMaskLoss`); a carried-over `true` elsewhere blocks Start
  // through configValidation instead of rendering a toggle the run would refuse.
  const subjectMaskLossSupported = targetSupportsSubjectMaskLoss(selectedTarget);
  const subjectMaskLossEnabled = subjectMaskLossSupported && Boolean(configDraft.subjectMaskLoss);
  const maskCoverage = subjectMaskCoverage(subjectMaskReport);
  const [maskJobRequested, setMaskJobRequested] = React.useState(false);
  // A queued-generation note belongs to the dataset it was queued for.
  React.useEffect(() => setMaskJobRequested(false), [activeDataset?.id]);
  // The subject-restricted normal loss (sc-24832) reads the same per-image masks, so it gets the
  // same coverage readout and "Generate subject masks" action (once — under subject-masked loss
  // when that is on too).
  const restrictedNormalsOn =
    selectedTarget?.baseModel !== "ltx_2_5" &&
    Boolean(configDraft.normalRestrictToSubject) &&
    bodyLosses.some((loss) => loss.prefix === "normal" && bodyLossEnabled(configDraft, loss));
  const subjectMaskCoverageAffordance = (
    <>
      <p className="training-field-hint" data-testid="subject-mask-coverage">
        {maskCoverage
          ? `Subject masks: ${maskCoverage.usable} of ${maskCoverage.total} images${
              maskCoverage.empty ? ` (${maskCoverage.empty} found no subject)` : ""
            }.`
          : "Subject mask coverage is unknown until the dataset is saved."}
      </p>
      {maskCoverage?.ungenerated ? (
        <p className="training-field-hint" data-testid="subject-mask-autogenerate">
          {`${maskCoverage.ungenerated} image${maskCoverage.ungenerated === 1 ? "" : "s"} without a subject mask get${
            maskCoverage.ungenerated === 1 ? "s" : ""
          } one generated with SAM3 when training starts.`}
        </p>
      ) : null}
      {maskCoverage && !maskCoverage.complete && typeof onGenerateSubjectMasks === "function" ? (
        maskJobRequested ? (
          <p className="training-field-hint">
            Subject mask generation queued — coverage updates when the job finishes.
          </p>
        ) : (
          <button
            className="secondary-action"
            onClick={() => {
              setMaskJobRequested(true);
              onGenerateSubjectMasks();
            }}
            type="button"
          >
            Generate subject masks
          </button>
        )
      ) : null}
    </>
  );
  // Every decoded-x0 / latent auxiliary loss the target supports, in display order: its toggle
  // (label, help text, suggested weight) and the knobs shown while it is on (label stem, help-text
  // noun, bounds, default window and period, loss-specific extras).
  const draft = { configDraft, configValidity, updateConfigDraft };
  const auxLosses = [
    {
      prefix: "depthAnchoring",
      supported: targetSupportsDepthAnchoring(selectedTarget),
      on: depthAnchoringEnabled(configDraft),
      suggested: depthAnchoringWeightSuggested,
      toggleLabel: "Depth anchoring",
      toggleTitle: `Keep the character's 3D shape consistent: the model's prediction is decoded and a frozen depth model compares its depth to the training image's. ${auxModelsInstallNote(selectedTarget, ["a Depth Anything V2 model"])} Off by default.`,
      weightLabel: "Depth anchoring weight",
      weightTitle:
        "Depth anchoring loss weight. 0.1 is the suggested weight for the Small depth model; use a much smaller weight (around 0.001) with Large.",
      weightMax: depthAnchoringWeightMax,
      stem: "Depth",
      noun: "depth loss",
      window: [0, 1],
      everyMax: depthAnchoringEveryMax,
      everyDefault: depthAnchoringEveryDefault,
      everyIt: "the depth loss",
      afterWeight: (
        <label title={`Which Depth Anything V2 model judges the depth. ${auxModelsInstallNote(selectedTarget, ["the chosen Depth Anything V2 model"])}`}>
          Depth model
          <select
            onChange={(event) => updateConfigDraft("depthAnchoringModel", event.target.value)}
            value={configDraft.depthAnchoringModel ?? ""}
            {...invalidProps(configValidity, "depthAnchoringModel")}
          >
            {depthAnchoringModelOptions.map((option) => (
              <option key={option} value={option}>
                {depthAnchoringModelLabels[option] ?? option}
              </option>
            ))}
          </select>
        </label>
      ),
    },
    ...bodyLosses.map((loss) => ({
      prefix: loss.prefix,
      supported: targetSupportsBodyLoss(selectedTarget, loss),
      on: bodyLossEnabled(configDraft, loss),
      suggested: bodyLossWeightSuggested,
      toggleLabel: `${loss.label} loss`,
      toggleTitle: `${loss.label} loss: the model's prediction is decoded and compared to the training image by a frozen body model; images with no person are skipped. ${auxModelsInstallNote(selectedTarget, bodyLossModels[loss.prefix])} Off by default.`,
      weightTitle: `${loss.label} loss weight. ${bodyLossWeightSuggested} is the suggested starting weight.`,
      weightMax: bodyLossWeightMax,
      stem: loss.label,
      noun: `${loss.label.toLowerCase()} loss`,
      window: loss.window,
      everyMax: bodyLossEveryMax,
      everyDefault: bodyLossEveryDefault,
      after: {
        bodyProportion: (
          <TechniqueToggle
            checked={Boolean(configDraft.bodyProportionIncludeHead)}
            label="Include head proportions"
            onChange={(event) => updateConfigDraft("bodyProportionIncludeHead", event.target.checked)}
            title="Also compare head proportions (nose-to-shoulders height, ear-to-ear width)."
          />
        ),
        bodyShape: (
          <DraftNumber
            draft={draft}
            field="bodyShapeMinCos"
            label="Body shape cosine gate"
            max="1"
            min="-1"
            placeholder={String(bodyShapeMinCosDefault)}
            step="0.05"
            title={`The shape loss only counts while the predicted body shape is already this similar (cosine) to the reference. Empty = ${bodyShapeMinCosDefault}.`}
          />
        ),
        normal: (
          <>
            {selectedTarget?.baseModel !== "ltx_2_5" ? (
              <TechniqueToggle
                checked={Boolean(configDraft.normalRestrictToSubject)}
                invalid={invalidProps(configValidity, "normalRestrictToSubject")}
                label="Normals on the subject only"
                onChange={(event) => updateConfigDraft("normalRestrictToSubject", event.target.checked)}
                title="Average the normal loss over each image's subject mask only (needs a subject mask on every image; missing ones are generated with SAM3 when training starts)."
              />
            ) : null}
            {restrictedNormalsOn && !subjectMaskLossEnabled ? (
              <div className="training-subject-mask-loss" data-testid="restricted-normal-masks">
                {subjectMaskCoverageAffordance}
              </div>
            ) : null}
          </>
        ),
      }[loss.prefix],
    })),
    {
      prefix: "identityLoss",
      supported: targetSupportsIdentityLoss(selectedTarget),
      on: identityLossEnabled(configDraft),
      suggested: faceLossWeightSuggested,
      toggleLabel: "Identity loss",
      toggleTitle: `Keep the character's face: the model's prediction is decoded and a frozen face recognizer (ArcFace) compares the face to the training images'. Images without a detectable face are skipped. ${auxModelsInstallNote(selectedTarget, ["the InstantID face analysis stack"])} Off by default.`,
      weightTitle: "Identity loss weight. Upstream suggests starting between 0.01 and 0.1.",
      weightMax: faceLossWeightMax,
      stem: "Identity loss",
      noun: "identity loss",
      window: [0, 1],
      everyMax: faceLossEveryMax,
      everyDefault: faceLossEveryDefault,
      after: (
        <>
          <DraftNumber
            draft={draft}
            field="identityLossMinCos"
            label="Identity gate (min similarity)"
            max="1"
            min="-1"
            placeholder={String(identityLossMinCosDefault)}
            step="0.05"
            title="A step whose predicted face is no more similar to the reference than this (cosine, -1 to 1) adds no identity loss, so blobs that are not faces are never pushed. Empty = 0.2."
          />
          <label title="What each image's face is pulled toward: the average identity of every training image (steadier), or that image's own face.">
            Identity reference
            <select
              onChange={(event) => updateConfigDraft("identityLossReference", event.target.value)}
              value={configDraft.identityLossReference ?? ""}
              {...invalidProps(configValidity, "identityLossReference")}
            >
              {identityLossReferenceOptions.map((option) => (
                <option key={option} value={option}>
                  {identityLossReferenceLabels[option] ?? option}
                </option>
              ))}
            </select>
          </label>
        </>
      ),
    },
    {
      prefix: "faceLandmarkLoss",
      supported: targetSupportsFaceLandmarkLoss(selectedTarget),
      on: faceLandmarkLossEnabled(configDraft),
      suggested: faceLossWeightSuggested,
      toggleLabel: "Face landmark loss",
      toggleTitle: `Keep the character's face shape: a frozen face-mesh model compares the predicted face's landmarks (jaw, lips, eyes, nose) to the training image's. ${auxModelsInstallNote(selectedTarget, ["the InstantID face analysis stack", "MediaPipe FaceMesh v2"])} Off by default.`,
      weightTitle: "Face landmark loss weight. Upstream suggests starting between 0.01 and 0.1.",
      weightMax: faceLossWeightMax,
      stem: "Face landmark loss",
      noun: "face landmark loss",
      window: [0, 1],
      everyMax: faceLossEveryMax,
      everyDefault: faceLossEveryDefault,
    },
    ...latentPerceptualLosses.map((loss) => ({
      prefix: loss.prefix,
      supported: targetSupportsLatentLoss(selectedTarget, loss),
      on: latentLossEnabled(configDraft, loss),
      suggested: loss.weightSuggested,
      toggleLabel: `${loss.label} loss`,
      toggleTitle: latentLossHelp[loss.prefix],
      weightTitle: `${loss.label} loss weight. ${loss.weightSuggested} is the starting weight.`,
      weightMax: loss.weightMax,
      weightStep: "0.1",
      stem: loss.label,
      noun: `${loss.label} loss`,
      window: [0, loss.maxTDefault],
      everyMax: loss.everyMax,
      everyDefault: loss.everyDefault,
      everyTitle: `1 adds the ${loss.label} loss to every step. N above 1 makes every Nth step train the ${loss.label} loss alone.`,
    })),
  ].filter((loss) => loss.supported);
  const visibleTimestepTypeOptions = timestepTypeOptionsForTarget(selectedTarget);
  // Limit-driven menus: a target that advertises the list gets exactly it; one that advertises a
  // single value (Iris: balanced bias, MSE loss) shows it read-only rather than offering a refusal.
  const visibleTimestepBiasOptions = choiceOptionsForTarget(selectedTarget, "timestepBiases", timestepBiasOptions);
  const visibleLossTypeOptions = choiceOptionsForTarget(selectedTarget, "lossTypes", lossTypeOptions);
  const advertisedPrecisions = choiceOptionsForTarget(selectedTarget, "mixedPrecisions", null);
  const checkpointingUnsupported = gradientCheckpointingUnsupported(selectedTarget, irisContext.trainingLane ?? null);
  const checkpointingLanes = gradientCheckpointingLanes(selectedTarget);
  const sampleCountRange = Array.isArray(selectedTarget?.limits?.sampleCount) ? selectedTarget.limits.sampleCount : null;
  // Iris (sc-25685): generation shows the generation options; depth / restoration (sc-25687 /
  // sc-25688) take no prompt, no sampler and no trigger word, and always produce a model.
  const objective = irisObjective(selectedTarget);
  const iris = objective === "generation";
  const irisTask = objective === "depth" || objective === "restoration";
  const depthTask = objective === "depth";
  const producesModel = isFullFinetune || irisTask;
  const upstreamPrompts = iris && irisUpstreamPromptsOn(configDraft);
  const floorSize = irisFloorSize(selectedTarget, configDraft);
  const pixelFloors = selectedTarget?.memoryFloors?.key === "microBatchPixels";
  const memoryFloors = objective
    ? irisMemoryFloors(
        selectedTarget,
        configDraft.networkType,
        floorSize,
        irisContext.trainingLane ?? null,
        // Generation's policy option; a depth / restoration run under the shared toggle reads the
        // same `full_ac` rows (irisOptions.js picks them only where the lane's table has them).
        iris
          ? configDraft.iris?.activation_checkpointing
          : configDraft.gradientCheckpointing && !checkpointingUnsupported
            ? "full"
            : undefined,
      )
    : [];
  const laneNames = { mlx: "MLX (Apple Silicon unified memory)", candle: "Candle/CUDA (GPU memory)" };
  // Restoration's paper default is a FULL fine-tune with the highest floors; the lighter LoRA arm is
  // offered up front (an explicit switch, never a silent one) with its own floor.
  const restoreLoraFloors =
    objective === "restoration" && isFullFinetune && (networkTypeOptions ?? []).includes("lora")
      ? irisMemoryFloors(selectedTarget, "lora", floorSize, irisContext.trainingLane ?? null)
      : [];
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
  const numericInput = (field, props = {}, validated = false) => (
    <input
      onChange={(event) => updateConfigDraft(field, event.target.value)}
      type="number"
      value={configDraft[field] ?? ""}
      {...props}
      {...(validated ? invalidProps(configValidity, field) : {})}
    />
  );
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
                      {blocked ? targetBlockedLabel(target) : ""}
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
              {producesModel ? "Base checkpoint name" : isControlTarget ? "Control branch name" : "LoRA name"}
              <input onChange={(event) => updateConfigDraft("outputName", event.target.value)} value={configDraft.outputName ?? ""} />
            </label>
            {irisTask ? null : (
              <label>
                Trigger phrase
                <input onChange={(event) => updateConfigDraft("triggerWord", event.target.value)} value={configDraft.triggerWord ?? ""} />
              </label>
            )}
            {[["steps", "Steps"], ["saveEvery", "Checkpoint cadence"]].map(([field, label]) => (
              <label key={field}>
                {label}
                {numericInput(field, {}, true)}
              </label>
            ))}

            {/* sc-15036: a full base fine-tune produces a MODEL, and the model catalog has one
                global user manifest — there is no project-scoped model store to honour a "project"
                scope with. Replace the picker with the explanation rather than leaving a control
                that silently does nothing (the sc-14056 gradient-checkpointing precedent below). */}
            {producesModel ? (
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
            {irisTask ? null : (
              <>
                <label>
                  Sample count
                  {numericInput("sampleCount", { max: sampleCountRange?.[1], min: "0" }, true)}
                </label>
                <label>
                  Sample steps
                  {numericInput("sampleSteps")}
                </label>
              </>
            )}
            <label>
              Sample cadence
              {numericInput("sampleEvery", {}, true)}
            </label>
          </div>

          {memoryFloors.length ? (
            <p className="training-field-hint" aria-label="Memory floor">
              Memory floor: {irisFloorSummary(memoryFloors, laneNames)} for a{" "}
              {pixelFloors ? `${floorSize}-pixel micro-batch` : `${floorSize}px`} {isFullFinetune ? "full fine-tune" : "adapter"} run. On a Mac the floor must fit
              after the macOS/app memory reserve. A machine under its floor is refused before the model loads.
              {irisTask && selectedTarget.memoryFloors?.basis ? ` ${selectedTarget.memoryFloors.basis}` : null}
            </p>
          ) : null}
          {restoreLoraFloors.length ? (
            <p className="training-field-hint" aria-label="Lighter restoration option">
              The paper&apos;s full fine-tune is the default. A LoRA run (merged into the export) needs{" "}
              {irisFloorSummary(restoreLoraFloors, laneNames)}.{" "}
              <button className="secondary-action" onClick={() => updateConfigDraft("networkType", "lora")} type="button">
                Use LoRA instead
              </button>
            </p>
          ) : null}

          {irisTask ? (
            <>
              <IrisTaskOptions
                configDraft={configDraft}
                configValidity={configValidity}
                objective={objective}
                onCheckDataset={activeDataset?.id ? irisContext.checkDataset : undefined}
                previousRuns={irisContext.previousRuns}
                updateConfigDraft={updateConfigDraft}
              />
              <RequiredModelsNotice
                detail="Training-only weights for the restoration loss."
                downloadJobs={controlModelDownloadJobs}
                feature="Iris restoration training"
                models={irisContext.missingResources ?? []}
                onCancelJob={onCancelJob}
                onDownload={onDownloadModel}
                onOpenModels={onOpenModels}
                onOpenQueue={onOpenQueue}
              />
            </>
          ) : null}

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

          {targetLicense ? (
            <div className="model-gated-notice training-license-notice" role="note" aria-label="Base model licence">
              {targetLicense.notice ? <p className="model-license-terms">{targetLicense.notice}</p> : null}
              {safeExternalUrl(targetLicense.url) ? (
                <div className="model-gated-actions">
                  <a href={safeExternalUrl(targetLicense.url)} target="_blank" rel="noreferrer noopener">
                    Review license
                  </a>
                </div>
              ) : null}
              <label className="model-license-ack">
                <input
                  checked={licenseAcknowledged}
                  onChange={(event) => onLicenseAcknowledgedChange?.(event.target.checked)}
                  type="checkbox"
                />
                <span>Accept {targetLicense.name}; adapters inherit it.</span>
              </label>
            </div>
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

          <RequiredModelsNotice
            detail="It is applied to the base only while training, so your LoRA learns the subject instead of fighting the distillation; the run would fail without it."
            downloadJobs={controlModelDownloadJobs}
            feature="Z-Image-Turbo LoRA training"
            models={missingTrainingAdapterModels}
            onCancelJob={onCancelJob}
            onDownload={onDownloadModel}
            onOpenModels={onOpenModels}
            onOpenQueue={onOpenQueue}
          />

          <RequiredModelsNotice
            detail="Images without a subject mask get one generated with it when training starts; the run would be refused without it."
            downloadJobs={controlModelDownloadJobs}
            feature="Subject-masked training"
            models={missingSubjectSegmenterModels}
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
              {(irisTask && isFullFinetune ? [] : [["rank", "Rank"], ["alpha", "Alpha"]]).map(([field, label]) => (
                <label key={field}>
                  {label}
                  {numericInput(field, field === "alpha" && isMossTrainingTarget ? { min: "0", step: "any" } : {}, true)}
                </label>
              ))}
              {/* Real hyperparameters the config validates (sc-10689). They live here
                  beside the other numeric knobs so every `> 0` error the rule set can
                  raise names an input the user can reach. The draft always seeds a
                  working value (configDraftFromTarget), so these are never empty. */}
              <label title="Images per optimizer step. Higher batches smooth gradients but cost more VRAM.">
                Batch size
                {numericInput("batchSize", { min: "1" }, true)}
              </label>
              <label title="Optimizer steps accumulated before an update — multiplies the effective batch size without extra VRAM.">
                Gradient accumulation
                {numericInput("gradientAccumulation", { min: "1" }, true)}
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
                  {numericInput("decomposeFactor", { min: "-1", step: "1" })}
                </label>
              ) : null}
              {isMossTrainingTarget ? (
                <label className="training-sample-prompts" title="One native target path per line. Paths are case-sensitive and sent exactly as written; leave blank to train all supported MOSS projections.">
                  MOSS target modules
                  <textarea
                    aria-label="MOSS target modules"
                    onChange={(event) => updateConfigDraft("loraTargetModules", event.target.value)}
                    placeholder="q\nproj.q\nlayers.0.proj.q"
                    rows={4}
                    value={configDraft.loraTargetModules ?? ""}
                  />
                  <span className="training-field-hint">
                    Exact case-sensitive native paths only (for example q, k, v, o, ffn_up_gate, ffn_down, or an exact dot path). Leave blank for all supported projections.
                  </span>
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
                {numericInput("learningRate", { step: "0.00001" }, true)}
              </label>
              <label>
                Weight decay
                {numericInput("weightDecay", { step: "0.00001" })}
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
                {numericInput("lrWarmupSteps", { min: "0" }, true)}
              </label>
              {weightNoiseInputVisible ? (
                <DraftNumber
                  draft={draft}
                  field="weightNoiseSigma"
                  label="Weight noise sigma"
                  max={weightNoiseSigmaMax}
                  step="0.0025"
                  title="Weight noise strength (sigma): after every optimizer step each adapter weight gets Gaussian noise scaled by sigma times that tensor's RMS. 0.0125 is the suggested strength."
                />
              ) : null}
              {gradientNoiseInputVisible ? (
                <>
                  <DraftNumber
                    draft={draft}
                    field="gradientNoiseEta"
                    label="Gradient noise eta"
                    max={gradientNoiseEtaMax}
                    step="0.001"
                    title="Gradient noise initial scale (eta): every optimizer step adds Gaussian noise of standard deviation eta / (1 + step)^gamma to each adapter gradient, after clipping. 0.01 is the suggested scale."
                  />
                  <DraftNumber
                    draft={draft}
                    field="gradientNoiseGamma"
                    label="Gradient noise gamma"
                    max={gradientNoiseGammaMax}
                    step="0.05"
                    title="Gradient noise annealing exponent (gamma): larger values fade the noise out faster over the run. 0.55 is the default."
                  />
                </>
              ) : null}
              {auxLosses
                .filter((loss) => loss.on)
                .map((loss) => (
                  <AuxLossKnobs draft={draft} key={loss.prefix} loss={loss} />
                ))}
              {irisTask ? null : (
                <>
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
                <select
                  disabled={visibleTimestepBiasOptions.length < 2}
                  onChange={(event) => updateConfigDraft("timestepBias", event.target.value)}
                  value={configDraft.timestepBias ?? ""}
                >
                  {visibleTimestepBiasOptions.map((option) => (
                    <option key={option} value={option}>
                      {optionLabel(option)}
                    </option>
                  ))}
                </select>
              </label>
                </>
              )}
              <label>
                Loss type
                <select
                  disabled={visibleLossTypeOptions.length < 2}
                  onChange={(event) => updateConfigDraft("lossType", event.target.value)}
                  value={configDraft.lossType ?? ""}
                >
                  {visibleLossTypeOptions.map((option) => (
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
              {depthTask ? null : (
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
              )}
              <label>
                Precision
                {advertisedPrecisions && !requiredFullPrecision ? (
                  <select onChange={(event) => updateConfigDraft("precision", event.target.value)} value={configDraft.precision ?? ""}>
                    {advertisedPrecisions.map((option) => (
                      <option key={option} value={option}>
                        {option.toUpperCase()}
                      </option>
                    ))}
                  </select>
                ) : (
                  <input
                    disabled={Boolean(requiredFullPrecision)}
                    onChange={(event) => updateConfigDraft("precision", event.target.value)}
                    value={requiredFullPrecision ?? configDraft.precision ?? ""}
                  />
                )}
                {requiredFullPrecision ? (
                  <span className="training-field-hint">
                    This backend requires {requiredFullPrecision.toUpperCase()} for full base fine-tuning.
                  </span>
                ) : null}
              </label>
              {irisTask ? null : (
                <label>
                  Guidance scale
                  {numericInput("sampleGuidanceScale", { step: "0.1" })}
                </label>
              )}
            </div>

            {irisTask ? null : (
            <label className="training-sample-prompts">
              Sample prompts
              <textarea
                disabled={upstreamPrompts}
                onChange={(event) => updateConfigDraft("samplePrompts", event.target.value)}
                placeholder="One prompt per line. Leave blank to use the trigger-phrase defaults."
                rows={4}
                value={configDraft.samplePrompts ?? ""}
              />
              <span className="training-field-hint">
                {upstreamPrompts
                  ? "Iris' seven built-in validation prompts replace these for this run (Iris training options → Previews)."
                  : "One prompt per line. Renders one preview per prompt, up to the sample count."}
              </span>
            </label>
            )}

            <div className="training-advanced-toggles">
              {isMossTrainingTarget ? (
                <>
                  <label className="training-checkbox-field" title="MOSS keeps the frozen VAE cache enabled; native training refuses it off.">
                    <input checked disabled type="checkbox" />
                    Cache VAE latents (required)
                  </label>
                  <label className="training-checkbox-field" title="MOSS keeps frozen text embeddings cached; native training refuses it off.">
                    <input checked disabled type="checkbox" />
                    Cache text embeddings (required)
                  </label>
                </>
              ) : null}
              {checkpointingUnsupported && checkpointingLanes ? (
                <p className="training-field-hint" aria-label="Gradient checkpointing availability">
                  Gradient checkpointing for this target is implemented on{" "}
                  {checkpointingLanes.map((lane) => laneNames[lane] ?? lane).join(" and ") || "no backend yet"}, not on this
                  machine&apos;s backend, so it is not applied here. Lower the training size if the run does not fit.
                </p>
              ) : checkpointingUnsupported ? (
                <p className="training-field-hint">
                  This target's trainer does not implement gradient checkpointing, so it is not applied. Lower the
                  training resolution if the run does not fit.
                </p>
              ) : fullCheckpointingUnsupported ? (
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
              {weightNoiseSupported ? (
                <TechniqueToggle
                  checked={weightNoiseEnabled}
                  label="Weight noise"
                  onChange={(event) =>
                    updateConfigDraft("weightNoiseSigma", event.target.checked ? String(weightNoiseSigmaSuggested) : "")
                  }
                  title="Perturb the adapter weights with small seeded noise after every optimizer step (relative to each tensor's RMS) — a regularizer against overfitting a small character dataset. Off by default."
                />
              ) : null}
              {gradientNoiseSupported ? (
                <TechniqueToggle
                  checked={gradientNoiseEnabled}
                  label="Gradient noise"
                  onChange={(event) => {
                    updateConfigDraft("gradientNoiseEta", event.target.checked ? String(gradientNoiseEtaSuggested) : "");
                    updateConfigDraft("gradientNoiseGamma", event.target.checked ? String(gradientNoiseGammaDefault) : "");
                  }}
                  title="Add seeded Gaussian noise to the adapter gradients that fades out over the run (eta / (1 + step)^gamma) — helps escape poor early minima. Off by default."
                />
              ) : null}
              {resolutionBucketsSupported || resolutionBuckets ? (
                <TechniqueToggle
                  checked={Boolean(resolutionBuckets)}
                  label="Multi-resolution buckets"
                  onChange={(event) =>
                    updateConfigDraft(
                      "resolutionBuckets",
                      event.target.checked ? seedResolutionBuckets(selectedTarget, configDraft.resolution) : null,
                    )
                  }
                  title="Train every image at several resolutions, each with its own repeat count per epoch (e.g. 512/768/1024 at 16/4/1). Replaces the single Resolution above. Off by default."
                />
              ) : null}
              {auxLosses.map((loss) => (
                <TechniqueToggle
                  checked={loss.on}
                  invalid={invalidProps(configValidity, `${loss.prefix}Weight`)}
                  key={loss.prefix}
                  label={loss.toggleLabel}
                  onChange={(event) =>
                    updateConfigDraft(`${loss.prefix}Weight`, event.target.checked ? String(loss.suggested) : "")
                  }
                  title={loss.toggleTitle}
                />
              ))}
              {subjectMaskLossSupported ? (
                <TechniqueToggle
                  checked={subjectMaskLossEnabled}
                  invalid={invalidProps(configValidity, "subjectMaskLoss")}
                  label="Subject-masked loss"
                  onChange={(event) => updateConfigDraft("subjectMaskLoss", event.target.checked)}
                  title="Weight the training loss by each image's subject mask so the adapter learns the subject, not the background. Needs a subject mask on every image; missing ones are generated with SAM3 when training starts. On in the Z-Image character presets, off otherwise."
                />
              ) : null}
            </div>

            {iris ? (
              <IrisTrainingOptions
                captionFieldOptions={irisContext.captionFieldOptions}
                configDraft={configDraft}
                configValidity={configValidity}
                fullRuns={irisContext.fullRuns}
                previousRuns={irisContext.previousRuns}
                updateConfigDraft={updateConfigDraft}
              />
            ) : null}
            {iris ? (
              <IrisCurriculaOptions
                configDraft={configDraft}
                configValidity={configValidity}
                onApplyPreset={irisContext.applyPreset}
                planCheck={irisContext.planCheck}
                presets={irisContext.presets}
                resumeMetrics={irisContext.resumeMetrics}
                updateConfigDraft={updateConfigDraft}
              />
            ) : null}
            {resolutionBuckets ? (
              <fieldset className="training-resolution-buckets" {...invalidProps(configValidity, "resolutionBuckets")}>
                <legend>Resolution buckets</legend>
                <span className="training-field-hint">
                  Square rows preserve the established every-image-at-each-resolution behavior. Rectangular rows use an explicit width × height geometry; row order is preserved.
                </span>
                <button
                  className="secondary-action"
                  onClick={() => updateConfigDraft("resolutionBuckets", resolutionBuckets.map((row) => {
                    const edge = row.resolution ?? row.width ?? row.height ?? configDraft.resolution ?? "";
                    return rectangularBuckets
                      ? { resolution: edge, repeats: row.repeats }
                      : { width: edge, height: edge, repeats: row.repeats };
                  }))}
                  type="button"
                >
                  {rectangularBuckets ? "Use square buckets" : "Use width × height buckets"}
                </button>
                {resolutionBuckets.map((row, index) => (
                  <div className="training-resolution-bucket-row" key={index}>
                    {rectangularBuckets ? (
                      <label>
                        Width × height
                        <input
                          aria-label={`Bucket ${index + 1} dimensions`}
                          onChange={(event) => {
                            const [width = "", height = ""] = event.target.value.split("x", 2);
                            updateConfigDraft("resolutionBuckets", resolutionBuckets.map((bucket, rowIndex) => (
                              rowIndex === index ? { ...bucket, width, height } : bucket
                            )));
                          }}
                          value={`${row.width ?? ""}x${row.height ?? ""}`}
                        />
                      </label>
                    ) : (
                      <label>
                        Resolution
                        {bucketResolutionOptions.length ? (
                          <select
                            aria-label={`Bucket ${index + 1} resolution`}
                            onChange={(event) => updateResolutionBucket(index, "resolution", event.target.value)}
                            value={row.resolution ?? ""}
                          >
                            {bucketResolutionOptions.map(String).includes(String(row.resolution)) ? null : (
                              <option value={row.resolution ?? ""}>{row.resolution ?? ""}</option>
                            )}
                            {bucketResolutionOptions.map((resolution) => (
                              <option key={resolution} value={resolution}>{resolution}</option>
                            ))}
                          </select>
                        ) : (
                          <input
                            aria-label={`Bucket ${index + 1} resolution`}
                            min={resolutionBucketStride}
                            onChange={(event) => updateResolutionBucket(index, "resolution", event.target.value)}
                            step={resolutionBucketStride}
                            type="number"
                            value={row.resolution ?? ""}
                          />
                        )}
                      </label>
                    )}
                    <label>
                      Repeats
                      <input
                        aria-label={`Bucket ${index + 1} repeats`}
                        max={resolutionBucketRepeatsMax}
                        min="1"
                        onChange={(event) => updateResolutionBucket(index, "repeats", event.target.value)}
                        step="1"
                        type="number"
                        value={row.repeats ?? ""}
                      />
                    </label>
                    <button
                      aria-label={`Remove bucket ${index + 1}`}
                      className="secondary-action"
                      onClick={() =>
                        updateConfigDraft(
                          "resolutionBuckets",
                          resolutionBuckets.filter((_, rowIndex) => rowIndex !== index),
                        )
                      }
                      type="button"
                    >
                      Remove
                    </button>
                  </div>
                ))}
                <button
                  className="secondary-action"
                  disabled={resolutionBuckets.length >= resolutionBucketsMax}
                  onClick={() =>
                    updateConfigDraft("resolutionBuckets", [
                      ...resolutionBuckets,
                      rectangularBuckets
                        ? {
                            width: String(configDraft.resolution ?? ""),
                            height: String(configDraft.resolution ?? ""),
                            repeats: "1",
                          }
                        : { resolution: String(configDraft.resolution ?? ""), repeats: "1" },
                    ])
                  }
                  type="button"
                >
                  Add bucket
                </button>
              </fieldset>
            ) : null}
            {subjectMaskLossEnabled ? (
              <div className="training-subject-mask-loss">
                <DraftNumber
                  draft={draft}
                  field="subjectMaskBackgroundWeight"
                  label="Background weight"
                  max={subjectMaskWeightMax}
                  step="0.05"
                  title="Loss weight of background pixels (outside the subject mask). 0 ignores the background entirely. The loss is still averaged over the whole image, so with a low background weight and a small subject the effective learning rate drops roughly in proportion to the subject's share of the frame."
                />
                <DraftNumber
                  draft={draft}
                  field="subjectMaskSubjectWeight"
                  label="Subject weight"
                  max={subjectMaskWeightMax}
                  step="0.05"
                  title="Loss weight of subject pixels (inside the subject mask). Must be greater than 0."
                />
                {subjectMaskCoverageAffordance}
              </div>
            ) : null}
          </AdvancedSection>

          {/* Dataset Doctor readout before the Train button (sc-6534). Advisory: it
              only hard-blocks training when the gate is Blocked (too few images / a
              fatal flag); warnings stay informational. A Blocked gate now rides in the
              chip row below as one of configValidity's errors (sc-10648), so the
              hand-rolled "isn't ready to train" paragraph is gone. */}
          <DatasetDoctorReadout {...datasetDoctor} compact />

          {/* Iris data-walk preflight (sc-25685): upstream's ranged sampler only visits whole
              640-item chunks, so part of a dataset can go unused. Non-blocking — the run is
              valid, the user should just know before launching. */}
          {iris && irisContext.dataWalk?.warning ? (
            <p className="inline-warning" role="status">{irisContext.dataWalk.warning}</p>
          ) : null}

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
