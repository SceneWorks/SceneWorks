import {
  MAX_VALIDATION_BATCH, VALIDATOR_SCHEMA_VERSION,
  serverRequired, versionReason,
} from "./portableValidationState.js";

export { MAX_VALIDATION_BATCH, VALIDATOR_SCHEMA_VERSION, setServerValidatorVersion, validatePortableRequestNow } from "./portableValidationState.js";

export async function validatePortableBatch({ kind, requests, token = "", signal, validatorSchemaVersion }) {
  const mismatch = versionReason({ validatorSchemaVersion });
  if (mismatch) return requests.map(() => serverRequired(mismatch));
  if (!Array.isArray(requests) || requests.length < 1 || requests.length > MAX_VALIDATION_BATCH) {
    return requests.map(() => serverRequired("invalid-batch-size"));
  }
  const { requestServerValidation } = await import("./portableValidationServer.js");
  return requestServerValidation({ kind, requests, token, signal, version: VALIDATOR_SCHEMA_VERSION });
}

export async function validatePortableRequest(request) {
  const [result] = await validatePortableBatch({
    kind: request.kind, requests: [request.payload], token: request.token,
    signal: request.signal, validatorSchemaVersion: request.validatorSchemaVersion,
  });
  return result;
}

export { portableFeedbackBlocks, portableFeedbackStatus } from "./portableFeedbackState.js";
