import { apiFetch } from "./api.js";
import { recipeResolution } from "./recipeFields.js";

// Older assets can outlive their job row. Replay their recorded inputs without opening a form.
// The prompt is already composed; reapplying its preset/style would wrap it a second time.
export function imageRecipeRequest(asset) {
  const shared = asset.generationSet?.recipe ?? {};
  const own = asset.recipe ?? {};
  const recipe = { ...shared, ...own,
    normalizedSettings: { ...shared.normalizedSettings, ...own.normalizedSettings },
    rawAdapterSettings: { ...shared.rawAdapterSettings, ...own.rawAdapterSettings } };
  if (!recipe.model) throw new Error("This image has no recorded generation model.");
  const settings = recipe.normalizedSettings;
  const raw = recipe.rawAdapterSettings;
  // Old sidecars did not record all edit inputs. Never turn a masked/multi-reference edit into
  // an unmasked single-source generation just because the original job has been purged.
  if (recipe.mode === "edit_image" && !["sourceAssetId", "referenceAssetIds", "maskAssetId", "fitMode"].every((key) => Object.hasOwn(raw, key) || Object.hasOwn(settings, key))) {
    throw new Error("The original edit job is no longer stored and this recipe lacks its complete source/mask settings. Use this recipe to review the inputs before generating.");
  }
  const [width, height] = (recipeResolution(recipe) ?? "").split("x").map(Number);
  const advanced = { ...raw };
  // Output receipts are not authored generation controls.
  for (const key of ["promptBatch", "upscale", "hiresFix", "mossImageExecutionReport", "checkpointWeightFacts", "memoryStrategy", "memoryEvidenceRevision", "realModelInference", "promptEnhancement"]) {
    delete advanced[key];
  }
  const prompts = Array.isArray(raw.promptBatch) ? raw.promptBatch : [];
  return {
    mode: recipe.mode ?? "text_to_image",
    model: recipe.model,
    prompt: prompts[0] ?? recipe.prompt ?? "",
    ...(prompts.length ? { prompts } : {}),
    negativePrompt: recipe.negativePrompt ?? "",
    seed: null,
    count: settings.count ?? 1,
    ...(width && height ? { width, height } : {}),
    loras: recipe.loras ?? [],
    characterId: settings.characterId ?? null,
    characterLookId: settings.characterLookId ?? null,
    sourceAssetId: raw.sourceAssetId ?? settings.sourceAssetId ?? asset.lineage?.sourceAssetId ?? null,
    referenceAssetId: raw.referenceAssetId ?? settings.referenceAssetId ?? null,
    referenceAssetIds: raw.referenceAssetIds ?? settings.referenceAssetIds ?? [],
    maskAssetId: raw.maskAssetId ?? settings.maskAssetId ?? null,
    fitMode: raw.fitMode ?? settings.fitMode ?? "crop",
    upscale: raw.upscale ?? settings.upscale,
    hiresFix: raw.hiresFix ?? settings.hiresFix,
    presetPromptResolvedClientSide: true,
    presetLorasResolvedClientSide: true,
    advanced,
  };
}

export async function regenerateImage({ asset, projectId, projectName, token, requestedGpu, beforeCreate, onCreated, isCurrentProject }) {
  if (asset?.type !== "image") return null;
  if (!projectId || asset.projectId !== projectId) throw new Error("Open this image's project before regenerating it.");
  const jobId = asset.generationSet?.jobId ?? asset.lineage?.jobId;
  let original;
  if (jobId) {
    try {
      original = await apiFetch(`/api/v1/jobs/${encodeURIComponent(jobId)}`, token);
    } catch (error) {
      if (error.status !== 404) throw error;
    }
  }
  if (!isCurrentProject()) return null;
  if (!original) {
    const request = imageRecipeRequest(asset);
    if ((await beforeCreate()) === false || !isCurrentProject()) return null;
    const job = await apiFetch("/api/v1/image/jobs", token, {
      method: "POST",
      body: JSON.stringify({ ...request, projectId, projectName, requestedGpu }),
    });
    onCreated(job);
    return job;
  }
  if (original.projectId !== projectId || !["image_generate", "image_edit"].includes(original.type)) {
    throw new Error("This image's original job is not an image-generation job in this project.");
  }
  if ((await beforeCreate()) === false || !isCurrentProject()) return null;
  // Duplicate bypasses the typed image route's random-seed generation. A null seed would reuse
  // the old seeds array (or the worker's deterministic fallback), so supply fresh randomness.
  const seed = crypto.getRandomValues(new Uint32Array(1))[0];
  const job = await apiFetch(`/api/v1/jobs/${encodeURIComponent(jobId)}/duplicate`, token, {
    method: "POST",
    body: JSON.stringify({ requestedGpu, payloadChanges: { seed, seeds: [] } }),
  });
  onCreated(job);
  return job;
}
