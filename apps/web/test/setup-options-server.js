// Legacy UI tests exercise controls and transport without a running API. These are
// finite, source-backed refusal fixtures for the old catalog's authored mutations,
// not browser policy. Mounted API tests cover representative LTX and Iris refusals.
const refused = (field, detail) => ({ state: "invalid", refusal: {
  code: "invalid_generation_option", field, detail: `[shared request validation] ${detail}`,
  context: {},
}, metadataSource: "server", liveChecks: "server" });

function fixtureResult(kind, payload) {
  if (kind === "image" && payload.model === "minimax_h3_image" && payload.mode === "edit_image"
    && new Set(payload.referenceAssetIds ?? []).size === 17) {
    return refused("referenceAssetIds", "minimax_h3_image takes up to 16 reference images, but this request supplies 17. Reduce the ordered conditioning list (source, mask and references together) to 16 or fewer.");
  }
  if (kind === "image" && payload.model === "flux3_klein"
    && payload.mode === "text_to_image" && payload.width === 16) {
    return refused("width", "The unconditioned FLUX 3 width is below its declared floor.");
  }
  if (kind === "video"
    && ["ltx_2_3", "ltx_2_3_eros", "ltx_2_5"].includes(payload.model)
    && ["extend_clip", "video_bridge"].includes(payload.mode)
    && (!Array.isArray(payload.loras) || payload.loras.length === 0)) {
    return refused("mode", "An IC-LoRA is required for this LTX route.");
  }
  if (kind === "training" && ["iris_3b_depth", "iris_3b_restore"].includes(payload.targetId)) {
    const config = payload.config ?? {};
    const mutations = payload.targetId === "iris_3b_depth"
      ? [["steps", 320001], ["saveEvery", 32001], ["advanced.sampleEvery", 32001],
        ["gradientAccumulation", 33], ["advanced.lrWarmupSteps", 1]]
      : [["steps", 10001], ["saveEvery", 1001], ["advanced.sampleEvery", 1001],
        ["advanced.lrWarmupSteps", 1]];
    for (const [field, value] of mutations) {
      const actual = field.split(".").reduce((object, key) => object?.[key], config);
      if (actual === value) return refused(`config.${field}`, "The authored Iris training value exceeds its declared limit.");
    }
  }
  return { state: "valid", resolved: payload, metadataSource: "server", liveChecks: "server" };
}

globalThis.__sceneworksOptionsTestServer = async ({ kind, requests }) => requests.map((payload) => fixtureResult(kind, payload));
