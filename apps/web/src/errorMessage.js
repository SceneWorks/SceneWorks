// UI mutations can reject with values from browser APIs, third-party bridges, or
// application code. React can render strings but not arbitrary objects, so keep the
// normalization at the boundary where a rejected value becomes user-facing text.
export function errorMessage(error, fallback) {
  const message = error && typeof error === "object" ? error.message : error;
  return typeof message === "string" && message.trim() ? message : fallback;
}
