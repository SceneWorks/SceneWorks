import { useAppDomainValue } from "../context/useAppDomainValue.js";
import { useCallback } from "react";
import { assetSeed } from "../components/assetPanels.jsx";
import { editModelForAsset, workflowModelType } from "../presetUtils.js";
import { upsertJobNewest } from "../sorters.js";
import { ADVANCED_MODE, SIMPLE_MODE } from "../simple/uiMode.js";

// F-189: extracted from App along its existing concern boundary.
export function useSendToStudio({
  setSelectedAssetId, setStudioLaunch, setActiveView, imageModels, setEditorLaunch, uiMode, uiModeLocked, setUiModeOverride, activeProjectRef, stableHydrateLaunchDomains, closePreview, token, requestedGpu, models, assets, confirmWorkflowEmbeddingBeforeGeneration, setJobs
}) {
  const sendAssetToImage = useCallback((asset, mode = null) => {
    if (!asset) {
      return;
    }
    setSelectedAssetId(asset.id);
    if (mode) {
      setStudioLaunch({ id: crypto.randomUUID(), view: "Image", assetId: asset.id, mode });
    }
    setActiveView("Image");
  }, [setActiveView, setSelectedAssetId, setStudioLaunch]);

  // Open Image Studio in edit mode with this image as the source, preselecting the
  // family-matched edit model when possible. sc-8730: this is the model-based path,
  // no longer wired to the preview Edit button — it stays on the context so S4
  // (sc-8729) can offer it as an "Edit in > Image Studio" context-menu item.
  const sendAssetToImageEdit = useCallback((asset) => {
    if (!asset) {
      return;
    }
    setSelectedAssetId(asset.id);
    setStudioLaunch({
      id: crypto.randomUUID(),
      view: "Image",
      assetId: asset.id,
      mode: "edit_image",
      model: editModelForAsset(asset, imageModels),
    });
    setActiveView("Image");
  }, [imageModels, setActiveView, setSelectedAssetId, setStudioLaunch]);

  // sc-8730: "Edit" from the fullscreen preview now opens the Image Editor canvas
  // (crop/upscale/refine) with this asset loaded via the editor's openAsset. This is
  // the model-free path; the model-based Image Studio edit_image path lives in
  // sendAssetToImageEdit above (kept for the S4 "Edit in > Image Studio" menu item).
  const sendAssetToImageEditor = useCallback((asset) => {
    if (!asset) {
      return;
    }
    setSelectedAssetId(asset.id);
    setEditorLaunch({ id: crypto.randomUUID(), assetId: asset.id });
    setActiveView("ImageEditor");
  }, [setActiveView, setEditorLaunch, setSelectedAssetId]);

  // The editor consumes editorLaunch and calls this to drop it, so navigating away
  // and back into the Image Editor without a fresh launch doesn't re-open a stale asset.
  const clearEditorLaunch = useCallback(() => setEditorLaunch(null), [setEditorLaunch]);

  function recipeForAsset(asset) {
    return asset?.generationSet?.recipe ?? asset?.recipe ?? null;
  }

  // Shared-workflow prefill (sc-15951) lands in the studio named by the envelope kind. Keep the
  // hydration race guard and launch shape shared so a video envelope reaches Video Studio's
  // existing recipe effect instead of silently loading its controls into Image Studio.
  //
  // `assetId` / `sourceAssetId` are null for a launch with no asset behind it, which is exactly
  // the dropped-file case: the studio's edit branch then leaves its source picker empty rather
  // than pointing at an image this install does not have.
  //
  // Returns `{ ok }`, plus a `reason` when it refused, because "the launch went nowhere" and "the
  // launch cannot go anywhere from here" need different answers on the other side: the first is a
  // race the caller should close its panel over, the second is a control the caller must not
  // pretend worked.
  async function launchImageRecipe({
    recipe,
    view = "Image",
    replaySeed = null,
    assetId = null,
    sourceAssetId = null,
  }) {
    if (!recipe) {
      return { ok: false, reason: "no-recipe" };
    }
    // The Simple shell renders its OWN studios and consumes no `studioLaunch`, so from in there
    // `setActiveView(targetView)` alone points the workspace at a screen nobody is looking at: the
    // panel dismisses, nothing visible happens, and the launch sits queued to fire unprompted the
    // next time the user opens the Advanced studio. Flip the shell the way `SimpleShell`'s own
    // `openInAdvanced` does — and refuse for the same reason it does when the viewport has Simple
    // locked on, because there is no Advanced workspace to land in at that width.
    if (uiMode === SIMPLE_MODE) {
      if (uiModeLocked) {
        return {
          ok: false,
          reason: "locked",
          detail: "That opens in the Advanced workspace — switch on a larger screen.",
        };
      }
      setUiModeOverride(ADVANCED_MODE);
    }
    const projectId = activeProjectRef.current?.id;
    setActiveView(view);
    const hydration = await stableHydrateLaunchDomains(view, projectId);
    if (hydration?.ok !== true || activeProjectRef.current?.id !== projectId) {
      return { ok: false, reason: "raced" };
    }
    setStudioLaunch({
      id: crypto.randomUUID(),
      view,
      assetId,
      sourceAssetId,
      recipe,
      replaySeed,
    });
    return { ok: true };
  }

  async function sendAssetRecipeToImage(asset, options = {}) {
    const recipe = recipeForAsset(asset);
    if (!asset || !recipe) {
      return;
    }
    const launchProjectId = activeProjectRef.current?.id;
    const { mossSeededRecipeReplay } = await import("../mossImageReplay.js");
    if (activeProjectRef.current?.id !== launchProjectId) return;
    // Keep-seed replays THIS image's own seed for a byte-for-byte rerun (e.g. to
    // reproduce and upscale with PiD). assetSeed prefers the per-asset recipe over the
    // set's shared base seed, so reuse honors the exact image the user is viewing.
    // Null → Image Studio leaves the seed random (a close variation), the default.
    const seed = assetSeed(asset);
    const replaySeed = options.keepSeed && seed != null && seed !== "" ? seed : null;
    setSelectedAssetId(asset.id);
    closePreview();
    await launchImageRecipe({
      ...mossSeededRecipeReplay(asset, recipe, replaySeed),
      assetId: asset.id,
      sourceAssetId: asset.lineage?.sourceAssetId ?? null,
    });
  }

  // sc-12324: the video twin of sendAssetRecipeToImage — re-run a generated clip from its own
  // recorded recipe, with or without its seed. Distinct from `sendAssetToVideo` below, which
  // sends an asset as a source clip/frame and replays no settings.
  async function sendAssetRecipeToVideo(asset, options = {}) {
    const recipe = recipeForAsset(asset);
    if (!asset || !recipe) {
      return;
    }
    const seed = assetSeed(asset);
    const replaySeed = options.keepSeed && seed != null && seed !== "" ? seed : null;
    const projectId = activeProjectRef.current?.id;
    setSelectedAssetId(asset.id);
    closePreview();
    setActiveView("Video");
    const hydration = await stableHydrateLaunchDomains("Video", projectId);
    if (hydration?.ok !== true || activeProjectRef.current?.id !== projectId) {
      return;
    }
    setStudioLaunch({
      id: crypto.randomUUID(),
      view: "Video",
      assetId: asset.id,
      recipe,
      replaySeed,
    });
  }

  // Route the viewer's "Use this recipe" to the studio that can actually run the asset's recipe.
  // The two studios own different forms, so a video's recipe must not land in the Image Studio.
  function sendAssetRecipeToStudio(asset, options = {}) {
    if (asset?.type === "video") {
      sendAssetRecipeToVideo(asset, options);
      return;
    }
    if (asset?.type === "vector") {
      setSelectedAssetId(asset.id);
      closePreview();
      setActiveView("VectorStudio");
      setStudioLaunch({ id: crypto.randomUUID(), view: "VectorStudio", assetId: asset.lineage?.sourceAssetId, recipe: asset.recipe });
      return;
    }
    sendAssetRecipeToImage(asset, options);
  }

  // Re-Generate is a background enqueue: never close the viewer, change pages, or touch a draft.
  async function sendAssetRecipeToQueue(asset) {
    const { regenerateImage } = await import("../regenerateImage.js");
    const projectId = activeProjectRef.current?.id;
    return regenerateImage({
      asset,
      projectId,
      projectName: activeProjectRef.current?.name,
      token,
      requestedGpu,
      models,
      assets,
      beforeCreate: confirmWorkflowEmbeddingBeforeGeneration,
      isCurrentProject: () => activeProjectRef.current?.id === projectId,
      onCreated: (job) => setJobs((items) => upsertJobNewest(items, job)),
    });
  }

  // sc-10516: launch a saved preset straight into the studio that can run it.
  //
  // The id alone is not enough: a studio only resolves `selectedPresetId` against its
  // `availablePresets`, which is filtered by the current mode AND model
  // (generationStudio.jsx). So the launch carries the preset's model and sub-mode too,
  // and the studio sets all three together. `presetId` and `recipe` are mutually
  // exclusive — a recipe launch keeps clearing the preset.
  const sendPresetToStudio = useCallback(async (preset) => {
    if (!preset?.id) {
      return;
    }
    // General (model-agnostic) presets don't pin a model/mode — they layer onto whatever the
    // studio is on (epic 11949). Launching one just toggles it into the general stack; default
    // to Image Studio, and it stays active if the user switches to Video. Model presets keep
    // carrying model + sub-mode so they resolve in the target studio (sc-10516).
    if (preset.kind === "general") {
      const projectId = activeProjectRef.current?.id;
      setActiveView("Image");
      const hydration = await stableHydrateLaunchDomains("Image", projectId);
      if (hydration?.ok !== true || activeProjectRef.current?.id !== projectId) return;
      setStudioLaunch({ id: crypto.randomUUID(), view: "Image", presetGeneralId: preset.id });
      return;
    }
    const view = workflowModelType(preset.workflow) === "video" ? "Video" : "Image";
    const projectId = activeProjectRef.current?.id;
    setActiveView(view);
    const hydration = await stableHydrateLaunchDomains(view, projectId);
    if (hydration?.ok !== true || activeProjectRef.current?.id !== projectId) return;
    setStudioLaunch({
      id: crypto.randomUUID(),
      view,
      presetId: preset.id,
      presetModel: preset.model ?? null,
      presetMode: preset.defaults?.mode ?? preset.workflow,
    });
  }, [activeProjectRef, setActiveView, setStudioLaunch, stableHydrateLaunchDomains]);

  const sendAssetToVideo = useCallback((asset, mode = null) => {
    if (!asset) {
      return;
    }
    setSelectedAssetId(asset.id);
    if (mode) {
      setStudioLaunch({ id: crypto.randomUUID(), view: "Video", assetId: asset.id, mode });
    }
    setActiveView("Video");
  }, [setActiveView, setSelectedAssetId, setStudioLaunch]);

  const sendCharacterToImage = useCallback((character, lookId = null, referenceAssetId = null) => {
    if (!character) {
      return;
    }
    setStudioLaunch({
      id: crypto.randomUUID(),
      view: "Image",
      characterId: character.id,
      lookId,
      referenceAssetId,
      mode: "character_image",
    });
    setActiveView("Image");
  }, [setActiveView, setStudioLaunch]);

  const sendCharacterToVideo = useCallback((character, lookId = null) => {
    if (!character) {
      return;
    }
    setStudioLaunch({ id: crypto.randomUUID(), view: "Video", characterId: character.id, lookId, mode: "text_to_video" });
    setActiveView("Video");
  }, [setActiveView, setStudioLaunch]);

  // sc-2022: open a specific dataset in the Dataset editor (Character Studio's
  // "Open" action on an associated dataset). The editor consumes studioLaunch.
  const openDatasetInLibrary = useCallback((datasetId) => {
    if (!datasetId) {
      return;
    }
    setStudioLaunch({ id: crypto.randomUUID(), view: "LibraryDataSets", datasetId });
    setActiveView("LibraryDataSets");
  }, [setActiveView, setStudioLaunch]);
  const context = useAppDomainValue({
    sendAssetToImage,
    sendAssetToImageEdit,
    sendAssetToImageEditor,
    clearEditorLaunch,
    sendPresetToStudio,
    sendAssetToVideo,
    sendCharacterToImage,
    sendCharacterToVideo,
    openDatasetInLibrary
  });
  return { ...context, launchImageRecipe, sendAssetRecipeToStudio, sendAssetRecipeToQueue, context };
}
