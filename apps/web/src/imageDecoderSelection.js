/**
 * Reconcile a recorded decoder selection against the active model/backend.
 *
 * Availability is deliberately not part of compatibility. A recorded alternate
 * decoder may be temporarily unavailable because its donor has not been
 * installed yet; preserving that selection lets the picker explain what is
 * missing instead of silently replaying through the native VAE. Only an option
 * absent from the active model/backend is incompatible and resets to native.
 *
 * @param {unknown} decoder
 * @param {Array<{id?: string, available?: boolean}>} options
 * @returns {string}
 */
export function reconcileDecoderSelection(decoder, options) {
  if (decoder === "native") return "native";
  if (typeof decoder !== "string") return "native";
  return (options ?? []).some((option) => option?.id === decoder) ? decoder : "native";
}
