// Request settings only. Execution reports belong to result history and never enter a new request.
export const defaultVedaSettings = () => restoreVedaSettings();
export function restoreVedaSettings(raw) {
  const { generated = "90%", reference = "90%", fullAttentionLayers = "", fullAttentionSteps = "" } = raw?.veda ?? {};
  return { enabled: raw?.veda?.enabled === true, generated, reference, fullAttentionLayers, fullAttentionSteps };
}
// The shared API/worker validator owns budget/list grammar and schedule bounds. API details
// already reach the Studio's error banner through makeCreateJob; avoid a divergent web parser.
export function vedaControlError(compatible, veda) {
  return veda?.enabled && !compatible ? "Veda requires MiniMax-H3" : null;
}
// Preserve enabled settings even for a stale submit; the API refuses incompatible models.
export const vedaAdvancedSettings = (veda) => veda?.enabled ? { veda } : {};
