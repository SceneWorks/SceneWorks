import { apiFetch } from "./api.js";
import { recipeResolution } from "./recipeFields.js";
import { isMossImage, isMossEdit, mossComponentState, mossEditReferenceIds, mossEditControlsError, mossEditWireOptions } from "./mossImage.js";

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
  if (recipe.mode === "edit_image" && !isMossEdit(recipe.model, recipe.mode) && !["sourceAssetId", "referenceAssetIds", "maskAssetId", "fitMode"].every((key) => Object.hasOwn(raw, key) || Object.hasOwn(settings, key))) {
    throw new Error("The original edit job is no longer stored and this recipe lacks its complete source/mask settings. Use this recipe to review the inputs before generating.");
  }
  const [width, height] = (recipeResolution(recipe) ?? "").split("x").map(Number);
  const advanced = { ...raw };
  // Keep removals explicit for workflow-share coverage: this replays existing controls, it
  // authors no new advanced keys. Processing inputs move to the request fields below.
  delete advanced.promptBatch;
  delete advanced.upscale;
  delete advanced.hiresFix;
  delete advanced.mossImageExecutionReport;
  delete advanced.mossImageEditExecutionReport;
  delete advanced.moss_image_edit_execution_report;
  delete advanced.mossImageSampleIndex;
  delete advanced.checkpointWeightFacts;
  delete advanced.memoryStrategy;
  delete advanced.memoryEvidenceRevision;
  delete advanced.realModelInference;
  delete advanced.promptEnhancement;
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
    sourceAssetId: raw.sourceAssetId ?? settings.sourceAssetId ?? (isMossEdit(recipe.model, recipe.mode) ? null : asset.lineage?.sourceAssetId ?? null),
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

// Background replay uses the same MOSS pre-enqueue checks as the visible Studio.
async function validateMossReplay(request, { models = [], assets = [], projectId }) {
  if (!isMossImage(request?.model)) return;
  const advanced = request.advanced ?? {}, model = models.find((item) => item.id === request.model);
  if (!model) throw new Error("The recorded MOSS model is unavailable. Refresh Models before regenerating.");
  if (!mossComponentState(model, advanced.mossImage, request.mode).ready) throw new Error("Install or repair the exact selected MOSS component before regenerating.");
  if (!isMossEdit(request.model, request.mode)) return;
  if (advanced.mossImage?.acceleration && advanced.mossImage.acceleration !== "off") throw new Error("Disable 8-step acceleration before regenerating a MOSS edit.");
  const references = mossEditReferenceIds(request);
  if (references.some((id) => !assets.some((item) => item.id === id && item.type === "image" && item.projectId === projectId))) throw new Error("A MOSS reference is missing from this project. Reacquire that exact slot before regenerating.");
  mossEditWireOptions(advanced.mossImageEdit);
  const issue = mossEditControlsError({ ...advanced, ...advanced.mossImage });
  if (issue) throw new Error(issue);
  // Keep advanced picker validation out of the eager App bundle until an actual MOSS replay.
  const { mossAdvancedIssue } = await import("./mossImageAdvancedValidation.js");
  const advancedIssue = mossAdvancedIssue(advanced.mossImageAdvanced, { assets: assets.filter((item) => item.projectId === projectId), prompt: request.prompt ?? "", prompts: request.prompts ?? [], negativePrompt: request.negativePrompt ?? "", count: request.count ?? 1, width: request.width, height: request.height, context: advanced.mossImage?.max_sequence_length ?? 4096, upscaleEnabled: request.upscale?.enabled });
  if (advancedIssue) throw new Error(advancedIssue);
}

export async function regenerateImage({ asset, projectId, projectName, token, requestedGpu, models, assets, beforeCreate, onCreated, isCurrentProject }) {
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
    await validateMossReplay(request, { models, assets, projectId });
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
  await validateMossReplay(original.payload, { models, assets, projectId });
  if ((await beforeCreate()) === false || !isCurrentProject()) return null;
  // Duplicate bypasses the typed image route's random-seed generation. A null seed would reuse
  // the old seeds array (or the worker's deterministic fallback), so supply fresh randomness.
  const seed = isMossEdit(original.payload?.model, original.payload?.mode) && original.payload?.advanced?.mossImageAdvanced?.noise ? null : crypto.getRandomValues(new Uint32Array(1))[0];
  const job = await apiFetch(`/api/v1/jobs/${encodeURIComponent(jobId)}/duplicate`, token, {
    method: "POST",
    body: JSON.stringify({ requestedGpu, payloadChanges: { seed, seeds: [] } }),
  });
  onCreated(job);
  return job;
}
