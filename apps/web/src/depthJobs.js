// Iris-3B monocular depth (sc-25682): the shared, React-free contract behind the Image Editor depth
// tool and the Library batch "Depth map" operation. One job per source image, posted to the typed
// `POST /api/v1/image/depth/jobs` route; the result is a `type: "depth"` asset whose file is the raw
// float32 `.npy` and whose `preview` is a separate PNG.

export const DEPTH_MODEL_ID = "iris_3b_depth";
export const DEPTH_ENDPOINT = "/api/v1/image/depth/jobs";
export const DEFAULT_DEPTH_MAX_SIDE = 1024;
export const MIN_DEPTH_MAX_SIDE = 64;
export const MAX_DEPTH_MAX_SIDE = 8192;
export const DEFAULT_DEPTH_RESOLUTION = Object.freeze({ mode: "capped", maxSide: DEFAULT_DEPTH_MAX_SIDE });

export function depthModel(models) {
  return (models ?? []).find((model) => model?.id === DEPTH_MODEL_ID) ?? null;
}

// The installed TRAINED depth models (a Training Studio depth run, sc-25687). Each runs from its own
// export with no release install (`depthInit: parent` training does not need it either).
export function trainedDepthModels(models) {
  return (models ?? []).filter((model) => model?.trainedCheckpoint?.task === "depth" && model.installState === "installed");
}

// The depth model a fresh picker starts on: the release export when it is installed, else the first
// installed trained depth model, else the release (whose missing install then names the fix).
export function defaultDepthModel(models) {
  if (depthModel(models)?.installState === "installed") return "";
  return trainedDepthModels(models)[0]?.id ?? "";
}

// Whether depth can run here, and why not. Both halves are required: a native lane on this platform
// (the platform-intrinsic `imageDepth` capability — false until capabilities load, so the action is
// never offered on a guess) and a COMPLETE install of the model that will run.
//
// `irisModel` is the picker's selection: "" / the release id needs the release `iris_3b_depth` task
// export; a trained depth model id needs only THAT trained model installed (S3 — never the release).
// Omitted (`undefined`) asks whether ANY depth model can run — the entry-point check that offers the
// Library / Editor action at all.
export function depthAvailability(models, macCapabilities, irisModel) {
  if (macCapabilities?.features?.imageDepth?.supported !== true) {
    const detail = macCapabilities?.features?.imageDepth?.reason?.detail;
    return { available: false, reason: detail || "Depth estimation is not available on this platform." };
  }
  const trained = trainedDepthModels(models);
  if (irisModel && irisModel !== DEPTH_MODEL_ID) {
    return trained.some((model) => model.id === irisModel)
      ? { available: true, reason: null }
      : {
          available: false,
          reason: `The trained depth model ${irisModel} is not installed — pick another model or retrain it.`,
        };
  }
  const model = depthModel(models);
  if (model?.installState === "installed") return { available: true, reason: null };
  if (irisModel === undefined && trained.length) return { available: true, reason: null };
  const alternative = trained.length ? " Or choose one of your trained depth models." : "";
  if (!model) {
    return { available: false, reason: `Iris 3B Depth is not in the model catalog.${alternative}` };
  }
  return {
    available: false,
    reason:
      (model.installState === "incomplete"
        ? "The Iris 3B Depth install is incomplete — repair it in Models."
        : "Install Iris 3B Depth in Models to estimate depth.") + alternative,
  };
}

// The request's resolution choice, normalized: `native`, or `capped` with an integer max side.
export function normalizeDepthResolution(resolution) {
  if (resolution?.mode === "native") return { mode: "native" };
  const maxSide = Math.round(Number(resolution?.maxSide ?? DEFAULT_DEPTH_MAX_SIDE));
  return { mode: "capped", maxSide };
}

// Why a resolution choice cannot be submitted, or null. Mirrors the API's bounds exactly.
export function depthResolutionError(resolution) {
  const normalized = normalizeDepthResolution(resolution);
  if (normalized.mode === "native") return null;
  if (!Number.isFinite(normalized.maxSide) || normalized.maxSide < MIN_DEPTH_MAX_SIDE || normalized.maxSide > MAX_DEPTH_MAX_SIDE) {
    return `Max side must be between ${MIN_DEPTH_MAX_SIDE} and ${MAX_DEPTH_MAX_SIDE} pixels (or choose Native).`;
  }
  return null;
}

// The model-input size the worker will actually run for a `width`×`height` upright source: upstream's
// size law (never upscale; each side rounded to the 16-pixel patch grid, half-to-even). Shown before
// submission so the processing size is never a surprise.
export function depthProcessingSize(width, height, resolution, patch = 16) {
  const w = Number(width);
  const h = Number(height);
  if (!(w > 0 && h > 0)) return null;
  const normalized = normalizeDepthResolution(resolution);
  const scale = normalized.mode === "native" ? 1 : Math.min(1, normalized.maxSide / Math.max(w, h));
  const roundHalfEven = (value) => {
    const floor = Math.floor(value);
    const diff = value - floor;
    if (diff > 0.5) return floor + 1;
    if (diff < 0.5) return floor;
    return floor % 2 === 0 ? floor : floor + 1;
  };
  const side = (s) => Math.max(patch, roundHalfEven((s * scale) / patch) * patch);
  return { width: side(w), height: side(h) };
}

// The typed request body. `requestedGpu` rides along like every other job route. `irisModel` names a
// trained depth model (sc-25687); absent = the release export.
export function buildDepthJobBody({ project, requestedGpu, sourceAssetId, resolution, displayName, irisModel }) {
  const body = {
    projectId: project.id,
    projectName: project.name ?? null,
    sourceAssetId,
    resolution: normalizeDepthResolution(resolution ?? DEFAULT_DEPTH_RESOLUTION),
    requestedGpu: requestedGpu ?? "auto",
  };
  if (displayName) body.displayName = displayName;
  if (irisModel && irisModel !== DEPTH_MODEL_ID) body.irisModel = irisModel;
  return body;
}

export function isDepthAsset(asset) {
  return asset?.type === "depth";
}

// The settings a depth asset was made with, for reopening a run (falls back to the default cap).
export function depthSettingsFromAsset(asset) {
  const recorded = asset?.recipe?.rawAdapterSettings?.depth?.resolution ?? asset?.depth?.preprocessing?.resolution;
  return recorded ? normalizeDepthResolution(recorded) : { ...DEFAULT_DEPTH_RESOLUTION };
}

// The trained depth model a depth asset was made with ("" = the release export).
export function depthModelFromAsset(asset) {
  return asset?.recipe?.rawAdapterSettings?.depth?.irisModel ?? "";
}

function resolutionLabel(resolution) {
  const normalized = normalizeDepthResolution(resolution);
  return normalized.mode === "native" ? "Native (source resolution)" : `Capped at ${normalized.maxSide}px max side`;
}

// Label/value rows describing a reloaded depth asset: what it is, where it came from and how it was
// made. Everything shown is read from the persisted sidecar.
export function depthAssetDetails(asset) {
  if (!isDepthAsset(asset)) return [];
  const depth = asset.depth ?? {};
  const source = depth.source ?? {};
  const pre = depth.preprocessing ?? {};
  const model = depth.model ?? {};
  const rows = [
    ["Values", `${depth.valueConvention ?? "unknown"} — relative, not metric; about -1 near … +1 far, not clamped`],
    ["Raw map", `${depth.dtype ?? "float32"} ${Array.isArray(depth.shape) ? depth.shape.join("×") : ""} (${depth.npyDescr ?? "<f4"}, ${depth.layout ?? "row-major"}) .npy`],
    ["Source", `${source.assetId ?? asset.lineage?.sourceAssetId ?? "unknown"} · ${source.width ?? "?"}×${source.height ?? "?"}`],
  ];
  if (source.exifOrientation && source.exifOrientation !== 1) {
    rows.push(["EXIF orientation", `${source.exifOrientation} applied (stored ${source.storedWidth}×${source.storedHeight})`]);
  }
  rows.push(
    ["Resolution", resolutionLabel(pre.resolution ?? depthSettingsFromAsset(asset))],
    ["Model input", `${pre.modelWidth ?? "?"}×${pre.modelHeight ?? "?"} · in ${pre.inputResample ?? "?"} / out ${pre.outputResample ?? "?"}`],
    ["Model", `${model.id ?? DEPTH_MODEL_ID} (${model.backend ?? "?"}, ${model.computeDtype ?? "?"})`],
    // A trained depth model's weights are its Training Studio export, never the release snapshot.
    model.weightsSource === "training"
      ? ["Weights", `trained ${model.trainedModelId ?? "?"} (${model.exportPath ?? "?"})`]
      : ["Weights", `${model.weightsRepo ?? "?"}@${String(model.weightsRevision ?? "?").slice(0, 12)} ${model.weightsSubdir ?? ""}`.trim()],
    ["Config sha256", String(model.configSha256 ?? "?")],
  );
  return rows;
}
