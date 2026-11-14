import { cachedModelFacts, rememberModelFacts } from "./portableModelFacts.js";

const MODEL_FREE_KINDS = new Set(["videoUpscale", "characterTest", "promptRefine", "yue2", "yue2Execution", "personDetect", "personTrack"]);

// A served catalog row or this origin's last served copy can answer pure options.
// Current installation, device choice and memory fit remain server-owned checks.
function validateWithEngine({ kind, payload, modelEntry = null }, validator, version, serverRequired) {
  let entry = modelEntry;
  let metadataSource = "live";
  if (MODEL_FREE_KINDS.has(kind)) {
    entry = {};
    metadataSource = null;
  } else if (kind === "vectorPrompt") {
    if (!entry) {
      const raster = cachedModelFacts(payload?.rasterModel);
      const vector = cachedModelFacts(payload?.vectorModel);
      entry = raster && vector ? { raster, vector } : null;
      metadataSource = "cache";
    }
    if (entry?.raster?.id !== payload?.rasterModel || entry?.vector?.id !== payload?.vectorModel) entry = null;
    if (entry && metadataSource === "live") {
      rememberModelFacts(entry.raster.id, entry.raster);
      rememberModelFacts(entry.vector.id, entry.vector);
    }
  } else {
    const modelId = payload?.model ?? payload?.modelId ?? payload?.targetId ?? payload?.plan?.target?.targetId;
    if (!entry && typeof modelId === "string") {
      entry = cachedModelFacts(modelId);
      metadataSource = "cache";
    }
    if (entry?.id !== modelId) entry = null;
    if (entry && metadataSource === "live") rememberModelFacts(modelId, entry);
  }
  if (!entry) metadataSource = null;
  try {
    const response = JSON.parse(entry
      ? validator.validate_request(kind, JSON.stringify(payload), JSON.stringify(entry))
      : validator.validate_shape(kind, JSON.stringify(payload)));
    if (response.validatorSchemaVersion !== version) {
      return serverRequired("validator-version-mismatch", metadataSource);
    }
    if (response.result.ok && !entry) {
      return { ...serverRequired("model-facts-unavailable"), shapeChecked: true };
    }
    return {
      state: response.result.ok ? "valid" : "invalid",
      ...(response.result.ok ? { resolved: response.result.resolved } : { refusal: response.result.refusal }),
      metadataSource: entry ? metadataSource : null,
      liveChecks: "server",
    };
  } catch {
    return serverRequired("validator-unavailable", metadataSource);
  }
}

export async function load() {
  const validator = await import("./wasm/options-validator/options_validator.js");
  await validator.default();
  return (request, version, serverRequired) => validateWithEngine(request, validator, version, serverRequired);
}
