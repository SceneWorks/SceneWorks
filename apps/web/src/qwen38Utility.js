export const QWEN38_MODEL_ID = "vision_describe_qwen3_8_27b";
export const QWEN38_MODEL_REPO = "huihui-ai/Huihui-Qwen3.8-27B-abliterated";
export const QWEN38_TIERS = [["q4", "q4"], ["q8", "q8"], ["bf16", "Full precision"]];

export function qwen38PromptSource(tier) {
  return tier ? { model: QWEN38_MODEL_REPO, variant: tier } : {};
}

export function installedQwen38Tiers(model) {
  return QWEN38_TIERS.filter(([tier]) => model?.variants?.some(
    (row) => row.variant === tier && row.installState === "installed" && !row.pendingArtifact,
  ));
}
