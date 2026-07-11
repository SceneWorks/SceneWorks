import React, { useState } from "react";
import { PoseLibraryPicker } from "./PoseLibraryPicker.jsx";
import { ImageEditSourcePickerField } from "./AssetPicker.jsx";
import { ControlOverlayPicker } from "./ControlOverlayPicker.jsx";

// Strict-control panel for the text-to-image studios (epic 8236, sc-8245). One picker gated by the
// selected backbone's supported control modes (`ui.controlModes`, mirrored from the manifest /
// STRICT_CONTROL_ENGINES `supported_kinds`), plus the per-mode conditioning input and the control-scale
// slider. The parent owns all state and the request wiring; this component is presentational + controlled:
//
//   * Pose  → the existing PoseLibraryPicker (`ui.poseLibrary`); the parent maps the selected pose ids to
//     `advanced.poses` exactly as the InstantID pose flow does — this never reinvents it.
//   * Canny / depth → a generic control-image upload (the reused ImageEditSourcePickerField, which already
//     supports upload via `importAsset`) plus a preprocess-vs-use-as-is toggle:
//       - preprocess (derive): the worker auto-derives the canny/depth map FROM the uploaded image; the
//         parent sends the asset id as the request `sourceAssetId`.
//       - use-as-is (passthrough): the user-supplied map is fed verbatim; the parent sends the asset id as
//         `advanced.controlImage`.
//     The toggle only changes WHICH request field carries the asset id (see ImageStudio.submit), matching
//     the worker's strict_control.rs distinction (resolve_user_control_map vs resolve_control_source).
//   * Control scale → a slider bound to `controlScale` (range/default from `ui.controlScale`), sent as
//     `advanced.controlScale`.
//
// `supportedModes` is the canonical-ordered, gated list (modelEligibility.supportedControlModes). When it
// is empty the parent hides the panel entirely; this component assumes at least one mode.
const MODE_LABELS = { pose: "Pose", canny: "Canny", depth: "Depth" };

export function ControlPanel({
  supportedModes,
  controlMode,
  onControlModeChange,
  // Pose
  selectedPoseIds,
  onTogglePose,
  onClearPoses,
  loadUserPoses,
  poseBlockText,
  // Trained ControlNet overlay selection (sc-10165 B4). `controlOverlayBaseModel` is the backbone id
  // whose pose control rides a REGISTERED overlay (e.g. `krea_2_turbo`); `null` for the Fun-Union
  // backbones that carry built-in control weights (no overlay picker). The picker self-fetches its list.
  controlOverlayBaseModel,
  selectedOverlayId,
  onOverlayChange,
  // Canny / depth control image
  controlImageAssetId,
  onControlImageChange,
  controlImagePassthrough,
  onControlImagePassthroughChange,
  controlImageAssets,
  importAsset,
  projectId,
  characters,
  // Control scale
  controlScaleConfig,
  controlScale,
  onControlScaleChange,
}) {
  const modes = Array.isArray(supportedModes) ? supportedModes : [];
  // Collapsed by default (this is a large, optional section); the user opts in when they want to
  // lock structure to a reference.
  const [open, setOpen] = useState(false);
  if (!modes.length) {
    return null;
  }
  const activeMode = modes.includes(controlMode) ? controlMode : modes[0];
  const scaleCfg = controlScaleConfig ?? {};

  return (
    <div className={`control-panel${open ? " open" : " collapsed"}`}>
      <button
        aria-expanded={open}
        className="control-panel-head"
        onClick={() => setOpen((prev) => !prev)}
        type="button"
      >
        <span className="control-panel-caret" aria-hidden="true">
          {open ? "▾" : "▸"}
        </span>
        <span className="control-panel-headings">
          <span className="control-panel-label">Structure control</span>
          <span className="muted">
            Lock the output's pose, edges, or depth to a reference.
          </span>
        </span>
      </button>

      {open ? (
        <>
          <div
            className="control-mode-tabs"
            role="tablist"
            aria-label="Control type"
          >
            {modes.map((mode) => (
              <button
                aria-pressed={mode === activeMode}
                className={
                  mode === activeMode
                    ? "control-mode-tab active"
                    : "control-mode-tab"
                }
                key={mode}
                onClick={() => onControlModeChange?.(mode)}
                role="tab"
                type="button"
              >
                {MODE_LABELS[mode] ?? mode}
              </button>
            ))}
          </div>

          {activeMode === "pose" ? (
            poseBlockText ? (
              <p className="mac-gating-note">{poseBlockText}</p>
            ) : (
              <div className="control-pose-section">
                {controlOverlayBaseModel ? (
                  <ControlOverlayPicker
                    baseModel={controlOverlayBaseModel}
                    onOverlayChange={onOverlayChange}
                    selectedOverlayId={selectedOverlayId}
                  />
                ) : null}
                <PoseLibraryPicker
                  loadUserPoses={loadUserPoses}
                  onClear={onClearPoses}
                  onToggle={onTogglePose}
                  selectedIds={selectedPoseIds}
                />
                <p className="muted">
                  Selecting poses generates one image per pose (overrides
                  Variations).
                </p>
              </div>
            )
          ) : (
            <div className="control-image-section">
              <ImageEditSourcePickerField
                assets={controlImageAssets}
                buttonLabel="Select image"
                characters={characters}
                emptyLabel={`No ${activeMode} control image selected`}
                importAsset={importAsset}
                label={
                  activeMode === "canny" ? "Edge/canny source" : "Depth source"
                }
                onChange={onControlImageChange}
                projectId={projectId}
                value={controlImageAssetId}
              />
              <label className="checkline">
                <input
                  checked={Boolean(controlImagePassthrough)}
                  onChange={(event) =>
                    onControlImagePassthroughChange?.(event.target.checked)
                  }
                  type="checkbox"
                />
                Use this image as the control map directly (skip preprocessing)
              </label>
              <p className="muted">
                {controlImagePassthrough
                  ? `Your image is fed in verbatim as the ${activeMode} map — supply an already-prepared ${activeMode} map.`
                  : `The worker auto-derives the ${activeMode} map from your image before generating.`}
              </p>
            </div>
          )}

          <label className="reference-strength control-scale">
            {scaleCfg.label ?? "Control strength"}
            <input
              max={scaleCfg.max ?? 2}
              min={scaleCfg.min ?? 0}
              onChange={(event) =>
                onControlScaleChange?.(Number(event.target.value))
              }
              step={scaleCfg.step ?? 0.05}
              type="range"
              value={controlScale}
            />
            <span>{Number(controlScale).toFixed(2)}</span>
          </label>
        </>
      ) : null}
    </div>
  );
}
