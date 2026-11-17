import { apiFetch, isAbortError } from "./api.js";

function serverRequired(reason) {
  return { state: "server-required", reason, metadataSource: null, liveChecks: "server" };
}

export async function requestServerValidation({ kind, requests, token, signal, version }) {
  if (import.meta.env.MODE === "test" && typeof globalThis.__sceneworksOptionsTestServer === "function") {
    return globalThis.__sceneworksOptionsTestServer({ kind, requests, token, signal, version });
  }
  try {
    const response = await apiFetch("/api/v1/options/validate", token, {
      method: "POST", body: JSON.stringify({ validatorSchemaVersion: version, kind, requests }), signal,
    });
    if (response?.validatorSchemaVersion !== version
      || !Array.isArray(response.results) || response.results.length !== requests.length) {
      return requests.map(() => serverRequired("server-validator-version-mismatch"));
    }
    return response.results.map((entry) => entry?.ok === true && entry.resolved && typeof entry.resolved === "object"
      ? { state: "valid", resolved: entry.resolved, metadataSource: "server", liveChecks: "server" }
      : entry?.ok === false && entry.refusal && typeof entry.refusal === "object"
        ? { state: "invalid", refusal: entry.refusal, metadataSource: "server", liveChecks: "server" }
        : serverRequired("invalid-server-response"));
  } catch (error) {
    if (isAbortError(error)) throw error;
    return requests.map(() => serverRequired("server-unavailable"));
  }
}
