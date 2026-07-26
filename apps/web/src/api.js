// Resolve the API base URL:
// - explicit non-empty VITE_API_BASE_URL (Docker/server, separate origins) → use it
// - explicit empty string (desktop embedded build, served by the API itself) →
//   same-origin, so fetches hit the API's actual port with no CORS
// - unset (local Vite dev) → the historical localhost:8000 default
const configuredApiBaseUrl = import.meta.env.VITE_API_BASE_URL;
export const API_BASE_URL =
  configuredApiBaseUrl === undefined
    ? "http://localhost:8000"
    : configuredApiBaseUrl === "" && typeof window !== "undefined"
      ? window.location.origin
      : configuredApiBaseUrl;

// True for the DOMException fetch raises when an AbortController aborts a
// request. Callers treat this as "request superseded", not a real error.
export function isAbortError(err) {
  return err?.name === "AbortError";
}

// ---- Failure shape (sc-15105) ---------------------------------------------------
// `apiFetch` used to throw a bare `new Error(message)`, discarding `response.status`.
// Nothing downstream could tell a 401 from any other failure, so a token that went
// stale WHILE a remote tab was open had no seam to re-prompt on: every request 401'd,
// the message piled up in the notice bands, and the media-ticket / SSE-ticket loops
// retried forever behind an app that looked broken rather than locked.
//
// `message` is deliberately byte-identical to what the bare Error carried, so the ~85
// existing `setError(err.message)` call sites are untouched; the status just rides
// along for the few places that want it.
export class ApiError extends Error {
  constructor(message, { status, detail, authRequired, code } = {}) {
    super(message);
    this.name = "ApiError";
    this.status = status;
    this.detail = detail;
    this.code = code;
    // The API tags its auth rejection (`apps/rust-api/src/auth.rs`); other 4xx bodies
    // don't carry the key, so this is a plain boolean rather than a tri-state.
    this.authRequired = authRequired === true;
    // Set on a 401 when the registered handler claims the session (the access gate is
    // re-verifying and will either re-prompt or resume). The retry loops read this to
    // decide between "surrender, the gate has it" and "treat it as an ordinary failure".
    this.reauthenticating = false;
  }
}

// True for the API's "you are not authenticated" rejection. Sibling of `isAbortError`.
export function isUnauthorized(err) {
  return err?.status === 401;
}

// True for the API's per-IP attempt throttle (`apps/rust-api/src/auth.rs`), which answers
// 429 once a peer exceeds its failed-token budget — including on the otherwise public
// /api/v1/auth/verify. A throttled host is refusing, not unreachable, and the two need
// different copy: the lockout clears itself in about a minute.
export function isThrottled(err) {
  return err?.status === 429;
}

// True for a 401 the access gate has claimed: it is re-verifying the token and will
// either re-prompt or resume the session. The retry loops (the media-ticket mint, the
// SSE-ticket connect) check this and surrender rather than backing off forever against
// a token that can only 401. A 401 that arrives with no gate registered — or one the
// gate declined — is false, and stays each caller's ordinary error to handle.
export function isReauthenticating(err) {
  return isUnauthorized(err) && err.reauthenticating === true;
}

// The one place that reacts to a 401, registered by `useAccessGate`. Module-level
// (not React state, same pattern as `setMediaTicket` below) so EVERY `apiFetch` caller
// participates without touching a single call site. Unset outside the gate's lifetime
// and on the desktop/loopback shell, where a 401 is not a thing that happens.
let unauthorizedHandler = null;

// Returns its own unregister function, which clears the slot only if this handler is
// still the registered one — so a late cleanup can never evict a newer registration.
export function setUnauthorizedHandler(handler) {
  const next = typeof handler === "function" ? handler : null;
  unauthorizedHandler = next;
  return () => {
    if (unauthorizedHandler === next) {
      unauthorizedHandler = null;
    }
  };
}

// `options` is forwarded to fetch, so callers can pass an AbortController
// `signal` to cancel a request (e.g. a stale project-scoped load).
export async function apiFetch(path, token, options = {}) {
  const headers = new Headers(options.headers ?? {});
  const isFormData = options.body instanceof FormData;
  if (options.body && !isFormData) {
    headers.set("Content-Type", "application/json");
  }
  if (token) {
    headers.set("X-SceneWorks-Token", token);
  }

  const response = await fetch(`${API_BASE_URL}${path}`, { ...options, headers });
  const payload =
    response.status === 204 ? null : await response.json().catch(() => null);
  if (!response.ok) {
    const detail = payload?.detail;
    const message =
      typeof detail === "string"
        ? detail
        : detail === undefined || detail === null
          ? `Request failed with ${response.status}`
          : JSON.stringify(detail);
    const error = new ApiError(message, {
      status: response.status,
      detail,
      authRequired: payload?.authRequired,
      code: payload?.code,
    });
    // `token` matters: a 401 on a request that deliberately presented NO credential says
    // nothing about the session token, so it must not drag the gate up. The case that
    // motivated this (sc-15105) was the `PUT /api/v1/ui-preferences` writes passing "" on a
    // gated route and swallowing the 401 with `.catch(() => {})`: without this check a remote
    // browser would slam into the full-page blocker on a theme toggle. Those call sites now
    // send the real token (sc-15136), so the guard is no longer load-bearing for THEM — but it
    // stays, both because the reasoning is method-independent and because the remaining
    // credential-less callers (`/api/v1/access`, `/api/v1/health`, the pre-auth ui-preferences
    // GET) are only public by server convention, which a future gating change could revisit.
    if (response.status === 401 && token) {
      // Notify before throwing so the gate reacts even for the callers that swallow
      // their own errors, and record whether it claimed the session. A throwing handler
      // must not mask the real API failure, nor be able to swallow the throw.
      try {
        error.reauthenticating = unauthorizedHandler?.(error) === true;
      } catch {
        // Deliberately ignored: the handler is UI bookkeeping, the ApiError is the truth.
      }
    }
    throw error;
  }
  return payload;
}

export function eventUrl(path, ticket) {
  const url = new URL(`${API_BASE_URL}${path}`);
  if (ticket) {
    url.searchParams.set("ticket", ticket);
  }
  return url.toString();
}

// ---- Media tickets (sc-8810) ---------------------------------------------------
// Element-driven requests (<img src>, <video src>, <a download>) cannot attach the
// X-SceneWorks-Token header, so in remote-auth mode every bare media URL would 401.
// Mirroring the SSE ticket, App.jsx mints a short-lived reusable ticket from
// POST /api/v1/files/ticket (header-authenticated) and stores it here; every media
// URL producer routes through `withMediaTicket` so the ticket rides along as a
// query param that the API honors on the project-files and pose-preview routes.
// Module-level (not React state) because URL producers like assetUrl() are plain
// functions called from render paths and non-component code (browserDownload).
let mediaTicket = "";

export function setMediaTicket(ticket) {
  mediaTicket = typeof ticket === "string" ? ticket : "";
}

export function getMediaTicket() {
  return mediaTicket;
}

// Append the current media ticket to a media URL. No-op while no ticket is set
// (desktop/loopback or auth-off deployments), so local mode is untouched.
export function withMediaTicket(url) {
  if (!url || !mediaTicket) {
    return url;
  }
  return `${url}${url.includes("?") ? "&" : "?"}ticket=${encodeURIComponent(mediaTicket)}`;
}
