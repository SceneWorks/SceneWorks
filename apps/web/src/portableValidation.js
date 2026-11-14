export const VALIDATOR_SCHEMA_VERSION = 1;
let enginePromise;
let loadedEngine;
let serverValidatorVersion = null;

export function setServerValidatorVersion(version) {
  serverValidatorVersion = Number.isInteger(version) && version > 0 ? version : null;
}

function engine() {
  enginePromise ??= import("./portableValidationEngine.js")
    .then(({ load }) => load())
    .then((validate) => {
      loadedEngine = validate;
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

// Submission must not wait for a cold module import: navigation or cancellation may occur
// while it loads. A cold check falls through to the server, which always revalidates.
export function validatePortableRequestNow(request) {
  const mismatch = versionReason(request);
  if (mismatch) return serverRequired(mismatch);
  if (!loadedEngine) {
    void engine().catch(() => {});
    return serverRequired("validator-loading");
  }
  return loadedEngine(request, VALIDATOR_SCHEMA_VERSION, serverRequired);
}

// Forms call this as their draft changes, warming the module before submission and
// showing a typed refusal without waiting for a round trip.
export async function validatePortableRequest(request) {
  const mismatch = versionReason(request);
  if (mismatch) return serverRequired(mismatch);
  try {
    return (await engine())(request, VALIDATOR_SCHEMA_VERSION, serverRequired);
  } catch {
    return serverRequired("validator-unavailable");
  }
}
