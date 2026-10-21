import React from "react";
import {
  DEFAULT_DEPTH_MAX_SIDE,
  MAX_DEPTH_MAX_SIDE,
  MIN_DEPTH_MAX_SIDE,
  depthProcessingSize,
  depthResolutionError,
  normalizeDepthResolution,
} from "../depthJobs.js";

// The depth resolution choice (sc-25682), shared by the Image Editor depth tool and the Library batch
// panel: an explicit Capped (default 1024 max side, adjustable) vs Native selection. When the source
// size is known the actual model-input size is shown before submission.
export function DepthResolutionFields({ value, onChange, sourceSize = null, idPrefix = "depth" }) {
  const resolution = normalizeDepthResolution(value);
  const capped = resolution.mode === "capped";
  const error = depthResolutionError(resolution);
  const processing =
    sourceSize && !error ? depthProcessingSize(sourceSize.width, sourceSize.height, resolution) : null;
  return (
    <fieldset className="depth-resolution-fields">
      <legend>Processing resolution</legend>
      <label className="checkline">
        <input
          checked={capped}
          name={`${idPrefix}-resolution-mode`}
          onChange={() => onChange?.({ mode: "capped", maxSide: resolution.maxSide ?? DEFAULT_DEPTH_MAX_SIDE })}
          type="radio"
        />
        Capped (faster, bounded memory)
      </label>
      {capped ? (
        <label>
          Max side (px)
          <input
            aria-label="Depth max side"
            max={MAX_DEPTH_MAX_SIDE}
            min={MIN_DEPTH_MAX_SIDE}
            onChange={(event) => onChange?.({ mode: "capped", maxSide: Number(event.target.value) })}
            step={16}
            type="number"
            value={Number.isFinite(resolution.maxSide) ? resolution.maxSide : ""}
          />
        </label>
      ) : null}
      <label className="checkline">
        <input
          checked={!capped}
          name={`${idPrefix}-resolution-mode`}
          onChange={() => onChange?.({ mode: "native" })}
          type="radio"
        />
        Native (source resolution — large photos need much more memory)
      </label>
      {error ? (
        <p className="field-error" role="alert">
          {error}
        </p>
      ) : null}
      {processing ? (
        <p className="muted depth-processing-size">
          Model input {processing.width}×{processing.height}; the depth map is saved at the source size{" "}
          {sourceSize.width}×{sourceSize.height}.
        </p>
      ) : (
        <p className="muted depth-processing-size">
          The depth map is saved at each source image's full (EXIF-corrected) size as raw float32 .npy plus a
          preview.
        </p>
      )}
    </fieldset>
  );
}
