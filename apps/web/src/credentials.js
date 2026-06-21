import { apiFetch } from "./api.js";
import { isDesktop, tauriInvoke } from "./runtime.js";

// Service-credential transport shared by the Settings screen (where users manage
// tokens) and the Models screen (which checks token presence for gated models).
// The desktop build stores credentials in the OS keychain via Tauri (sc-1891);
// the server/Docker build manages them over the authed REST API (sc-1893). These
// helpers hide that split so both screens use one transport + presence check.
// `credentialsIsDesktop` is a thin re-export of the unified `isDesktop` (epic 4484
// story 6) kept for existing importers.
export const credentialsIsDesktop = isDesktop;

const invoke = tauriInvoke;

export const SCHEME_LABELS = {
  bearer: "Bearer header",
  query: "Query token",
};

// Server mode authenticates with the same access token the rest of the app uses.
export function serverToken() {
  return (
    (typeof window !== "undefined" && window.localStorage.getItem("sceneworks-token")) || ""
  );
}

export async function loadCredentials() {
  if (credentialsIsDesktop) {
    return (await invoke("list_credentials")) ?? [];
  }
  return (await apiFetch("/api/v1/credentials", serverToken())) ?? [];
}

// Returns the updated, redacted credential list (both transports yield it).
export async function saveCredential({ host, label, scheme, token }) {
  if (credentialsIsDesktop) {
    return (await invoke("set_credential", { host, label, scheme, token })) ?? loadCredentials();
  }
  return apiFetch("/api/v1/credentials", serverToken(), {
    method: "PUT",
    body: JSON.stringify({ host, label, scheme, token }),
  });
}

export async function removeCredentialRequest(host) {
  if (credentialsIsDesktop) {
    return (await invoke("delete_credential", { host })) ?? loadCredentials();
  }
  return apiFetch(`/api/v1/credentials/${encodeURIComponent(host)}`, serverToken(), {
    method: "DELETE",
  });
}

// Normalize a host or URL the way the Rust store does (strip scheme + path,
// lower-case) so a manifest's `credentialHost: "huggingface.co"` matches a stored
// credential that was keyed from e.g. "https://huggingface.co/black-forest-labs".
export function normalizeCredentialHost(value) {
  return String(value ?? "")
    .trim()
    .replace(/^https?:\/\//i, "")
    .split("/")[0]
    .trim()
    .toLowerCase();
}

// Whether a credential for `host` exists with its token actually present (the
// redacted listing flags missing tokens via `present: false`).
export function hasPresentCredential(credentials, host) {
  const target = normalizeCredentialHost(host);
  if (!target) {
    return false;
  }
  return (credentials ?? []).some(
    (credential) =>
      normalizeCredentialHost(credential.host) === target && credential.present !== false,
  );
}
