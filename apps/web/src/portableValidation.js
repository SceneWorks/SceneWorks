import { cachedModelFacts, rememberModelFacts } from "./portableModelFacts.js";

export const VALIDATOR_SCHEMA_VERSION = 1;
let enginePromise;
let loadedEngine;
let serverValidatorVersion = null;

export function setServerValidatorVersion(version) {
  serverValidatorVersion = Number.isInteger(version) && version > 0 ? version : null;
}

function engine() {
  enginePromise ??= import("./wasm/options-validator/options_validator.js")
    .then(async (module) => {
      await module.default();
      loadedEngine = { validate: module.validate_request, shape: module.validate_shape };
      return loadedEngine;
    })
    .catch((error) => {
      enginePromise = undefined;
      throw error;
    });
  return enginePromise;
}

function serverRequired(reason, metadataSource = null) {
  return { state: "server-required", reason, metadataSource, liveChecks: "server" };
}

function versionReason(request) {
  if ((request.validatorSchemaVersion ?? VALIDATOR_SCHEMA_VERSION) !== VALIDATOR_SCHEMA_VERSION) {
    return "validator-version-mismatch";
  }
  if (serverValidatorVersion !== null && serverValidatorVersion !== VALIDATOR_SCHEMA_VERSION) {
    return "server-validator-version-mismatch";
  }
  return null;
}

// The entry must come from a served catalog row or this origin's last served copy.
// Neither source proves current installation, device selection or memory fit; the server
// revalidates every submission. A cache-backed answer is explicitly marked stale.
function validateWithEngine({
  kind,
  payload,
  modelEntry = null,
  validatorSchemaVersion = VALIDATOR_SCHEMA_VERSION,
}, validator) {
  const mismatch = versionReason({ validatorSchemaVersion });
  if (mismatch) return serverRequired(mismatch);
  const modelId = payload?.model ?? payload?.modelId;
  let entry = modelEntry;
  let metadataSource = "live";
  if (!entry && typeof modelId === "string") {
    entry = cachedModelFacts(modelId);
    metadataSource = "cache";
  }
  if (entry?.id !== modelId) entry = null;
  if (!entry) metadataSource = null;
  if (entry && metadataSource === "live") rememberModelFacts(modelId, entry);
  try {
    const response = JSON.parse(entry
      ? validator.validate(kind, JSON.stringify(payload), JSON.stringify(entry))
      : validator.shape(kind, JSON.stringify(payload)));
    if (response.validatorSchemaVersion !== VALIDATOR_SCHEMA_VERSION) {
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

// Submission must not wait for a cold module import: navigation or cancellation may occur
// while it loads. A cold check falls through to the server, which always revalidates.
export function validatePortableRequestNow(request) {
  const mismatch = versionReason(request);
  if (mismatch) return serverRequired(mismatch);
  if (!loadedEngine) {
    void engine().catch(() => {});
    return serverRequired("validator-loading");
  }
  return validateWithEngine(request, loadedEngine);
}

// Forms call this as their draft changes, warming the module before submission and
// showing a typed refusal without waiting for a round trip.
export async function validatePortableRequest(request) {
  const mismatch = versionReason(request);
  if (mismatch) return serverRequired(mismatch);
  try {
    return validateWithEngine(request, await engine());
  } catch {
    return serverRequired("validator-unavailable");
  }
}
