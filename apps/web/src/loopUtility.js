export const LOOP_UTILITY_ID = "vision_sensenova_loop_9b";
export const LOOP_UTILITY_REPO = "sensenova/SenseNova-Loop-9B";
export const LOOP_CAPTIONER = "sensenova_loop";
export const LOOP_UTILITY_NAME = "SenseNova-Loop-9B";
export const LOOP_TIERS = [["q4", "q4"], ["q8", "q8"], ["bf16", "Full precision (BF16)"]];

export function loopTiers(model) {
  return LOOP_TIERS.filter(([tier]) => model?.variants?.some(
    (row) => row.variant === tier && !row.pendingArtifact && row.installState !== "pending",
  ));
}

export function loopUtilityModel(models) {
  const model = models.find((entry) => entry.id === LOOP_UTILITY_ID);
  return loopTiers(model).length ? model : undefined;
}

export function loopTierInstalled(model, tier) {
  return model?.variants?.some((row) => row.variant === tier && row.installState === "installed") ?? false;
}

// An explicit choice stays selected if its artifact is removed. Only an empty selection has a default.
export function loopUtilityTier(model, selected) {
  return selected || loopTiers(model).find(([tier]) => loopTierInstalled(model, tier))?.[0]
    || loopTiers(model)[0]?.[0];
}

export function loopTierAvailable(model, tier) {
  return loopTiers(model).some(([key]) => key === tier);
}

export function loopPromptSource(variant) {
  return variant ? { model: LOOP_UTILITY_REPO, variant } : {};
}
