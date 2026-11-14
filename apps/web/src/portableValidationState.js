export const VALIDATOR_SCHEMA_VERSION = 1;
export const MAX_VALIDATION_BATCH = 32;
let serverValidatorVersion = null;

export function setServerValidatorVersion(version) {
  serverValidatorVersion = Number.isInteger(version) && version > 0 ? version : null;
}

export function serverRequired(reason) {
  return { state: "server-required", reason };
}

export function versionReason(request) {
  if ((request.validatorSchemaVersion ?? VALIDATOR_SCHEMA_VERSION) !== VALIDATOR_SCHEMA_VERSION) {
    return "validator-version-mismatch";
  }
  if (serverValidatorVersion !== null && serverValidatorVersion !== VALIDATOR_SCHEMA_VERSION) {
    return "server-validator-version-mismatch";
  }
  return null;
}

// Synchronous submit adapters defer to the final API request, which validates
// the exact body with current authoritative facts before it can create a job.
export function validatePortableRequestNow(request) {
  const mismatch = versionReason(request);
  if (mismatch) return serverRequired(mismatch);
  return serverRequired("server-validation-required");
}
