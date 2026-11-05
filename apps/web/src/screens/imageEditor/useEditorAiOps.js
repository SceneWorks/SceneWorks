import { editReferenceIds } from "./references.js";
import { editOutputDims } from "./geometry.js";
import { workingToPngFile, blobToEditorImage } from "./export.js";
import { JOB_PATHS, submitJobRequest } from "../../api/jobs.js";
import { mediaResponseWithTicketRetry } from "../../api/media.js";
import { useInsertionEffect, useCallback, useEffect } from "react";
import { nonCanvasJobOutputs } from "../../editorScratch.js";
import { terminalStatuses } from "../../jobTypes.js";
import { assetUrl } from "../../components/assetMedia.jsx";
import { effectiveFitMode } from "../../components/FitModeControl.jsx";
import { activeLayerOf, layerById } from "../../imageLayers.js";
import { buildDetailJobBody, buildEditJobBody, buildUpscaleJobBody, upscaleEngineHasSoftness, upscaleEngineIsRestoration, upscaleFactorsForEngine } from "../../imageJobs.js";
import { irisRestoreControls } from "../../irisRestore.js";
import { maskHasContent } from "./maskShared.js";

// Extracted from ImageEditor along its existing concern boundary.
export function useEditorAiOps({
  working,
  aiOp,
  activeProject,
  confirmFlatten,
  setStatus,
  confirmWorkflowEmbeddingBeforeGeneration,
  mountedRef,
  activeLayerToFile,
  importAsset,
  purgeAsset,
  token,
  trackEditorScratchOp,
  setAiOp,
  setTool,
  runAiOpRef,
  upscaleEngine,
  irisPlanStatus,
  irisRestore,
  requestedGpu,
  upscaleFactor,
  upscaleSoftness,
  detailModel,
  detailStrength,
  detailCnScale,
  mossEdit,
  editPrompt,
  mossEditSettings,
  editModel,
  editLoraRequiredMissing,
  mossEditReady,
  mossEditError,
  editFitMode,
  canMask,
  editAspect,
  maskLines,
  maskBaseImage,
  rasterizeMaskToFile,
  editTransparent,
  selectedEditModel,
  multiRefCapable,
  refAssetIds,
  maxEditReferences,
  editSeed,
  editLoraSelection,
  managedEditLoraId,
  editLora,
  editGuidance,
  mossImageEdit,
  jobs,
  setNativeEditResults,
  checkpoint,
  workingRef,
  replaceLayerImage,
  installWorkingImage,
  setEdits,
  setDirty,
  releaseEditorScratchOp
}) {
  // ── AI ops on the working image (sc-2432 seam) ────────────────────────────
  // Flatten the composited document to a PNG File. `filename` overrides the name
  // (Save/Download use the "-edited" name; the AI-op scratch upload doesn't care).
  const workingImageToFile = useCallback(
    (filename) => workingToPngFile(working, filename),
    [working],
  );

  // Stage the working image as a scratch asset, start a worker job against it, and
  // track it. The watcher below loads the result back and purges scratch + result —
  // intermediates never persist; only Save (sc-2434) lands a Library asset.
  const runAiOp = useCallback(
    async ({
      buildBody,
      label,
      edit,
      endpoint = JOB_PATHS.generic,
      maskFile = null,
      sourceFile = null,
      onComplete = null,
      // The tool-interaction matrix (sc-6119): "activeLayer" ops stage the active
      // layer and write the result back to it (dims unchanged); "composite" ops
      // stage the flattened document and the result becomes a single new base layer.
      layerSource = "activeLayer",
    }) => {
      if (!working || aiOp || !activeProject) return false;
      // A composite-source writeback flattens the stack. Read-only consumers such
      // as Smart Select may stage the composite without changing the document.
      if (layerSource === "composite" && !onComplete && !(await confirmFlatten())) return false;
      setStatus({ loading: false, error: "" });
      const targetLayerId = activeLayerOf(working)?.id ?? null;
      // Stage the source (and, for a masked edit, the mask) as scratch assets. An
      // explicit sourceFile (e.g. the box-baked pass-through, sc-6093) wins; else
      // stage the composite (flatten ops) or just the active layer (active-layer ops).
      let scratch;
      let maskScratch = null;
      try {
        // Share the durable recipe choice while retaining this scratch job's
        // original transport and cleanup (no deferred library-retry callback).
        if (endpoint === JOB_PATHS.image) {
          if ((await confirmWorkflowEmbeddingBeforeGeneration()) === false || !mountedRef.current) return false;
        }
        const staged =
          sourceFile ?? (layerSource === "composite" ? await workingImageToFile() : await activeLayerToFile());
        scratch = await importAsset(staged, { throwOnError: true });
        if (maskFile) maskScratch = await importAsset(maskFile, { throwOnError: true });
      } catch (err) {
        if (scratch) purgeAsset(scratch).catch(() => {});
        setStatus({ loading: false, error: `Could not stage image: ${err.message || err}` });
        return false;
      }
      try {
        const job = await submitJobRequest(endpoint, token, {
          method: "POST",
          body: JSON.stringify(buildBody(scratch, maskScratch)),
        });
        if (!job?.id) throw new Error("The job was not created.");
        // Register the scratch op with App so its scratch/mask (and later result) assets
        // are purged even if the user navigates away mid-job and this editor unmounts
        // before the completion watcher below runs (sc-8850).
        trackEditorScratchOp?.(job.id, [scratch, maskScratch]);
        setAiOp({
          jobId: job.id,
          scratch,
          maskScratch,
          source: working.source,
          label,
          edit,
          onComplete,
          // How the watcher writes the result back: active-layer ops replace the
          // target layer's bitmap; composite ops flatten the stack to one layer.
          writeBack: onComplete ? null : layerSource === "composite" ? "document" : "activeLayer",
          targetLayerId,
        });
        setTool("move");
        return true;
      } catch (err) {
        purgeAsset(scratch).catch(() => {});
        if (maskScratch) purgeAsset(maskScratch).catch(() => {});
        setStatus({ loading: false, error: `Could not start ${label}: ${err.message || err}` });
        return false;
      }
    },
    [working, aiOp, activeProject, confirmFlatten, setStatus, workingImageToFile, activeLayerToFile, importAsset, confirmWorkflowEmbeddingBeforeGeneration, mountedRef, purgeAsset, token, trackEditorScratchOp, setAiOp, setTool],
  );

  // Keep the mask hook's bridge pointed at the committed runAiOp (sc-9752) — runSmartSelect
  // stages a scratch image_segment job through this exact seam.
  useInsertionEffect(() => { runAiOpRef.current = runAiOp; }, [runAiOpRef, runAiOp]);

  function runUpscale() {
    if (upscaleEngineIsRestoration(upscaleEngine)) {
      // Iris restoration (sc-25683): sized by scale; the preview must have planned it (a refused
      // plan is never submitted, and the request is never silently changed).
      if (!irisPlanStatus.ready) return;
      const restore = irisRestoreControls(irisRestore);
      runAiOp({
        label: "restore",
        layerSource: "composite",
        edit: { op: "upscale", engine: upscaleEngine, ...restore },
        buildBody: (scratch) =>
          buildUpscaleJobBody({
            project: activeProject,
            requestedGpu,
            sourceAssetId: scratch.id,
            engine: upscaleEngine,
            displayName: working?.source?.name,
            restore,
          }),
      });
      return;
    }
    const valid = upscaleFactorsForEngine(upscaleEngine);
    const factor = valid.includes(upscaleFactor) ? upscaleFactor : valid[0];
    const softness = upscaleEngineHasSoftness(upscaleEngine) ? upscaleSoftness : undefined;
    runAiOp({
      label: "upscale",
      // Upscale changes dimensions → document-level: flatten the stack, upscale once.
      layerSource: "composite",
      edit: {
        op: "upscale",
        engine: upscaleEngine,
        factor,
        ...(softness !== undefined ? { softness } : {}),
      },
      buildBody: (scratch) =>
        buildUpscaleJobBody({
          project: activeProject,
          requestedGpu,
          sourceAssetId: scratch.id,
          factor,
          engine: upscaleEngine,
          displayName: working?.source?.name,
          softness,
        }),
    });
  }

  function runDetail() {
    if (!detailModel) return;
    runAiOp({
      label: "detail",
      edit: { op: "detail", model: detailModel, strength: detailStrength, cnScale: detailCnScale },
      buildBody: (scratch) =>
        buildDetailJobBody({
          project: activeProject,
          requestedGpu,
          sourceAssetId: scratch.id,
          model: detailModel,
          strength: detailStrength,
          cnScale: detailCnScale,
          displayName: working?.source?.name,
        }),
    });
  }

  async function runEdit() {
    const prompt = mossEdit ? editPrompt : editPrompt.trim();
    if ((!prompt && !(mossEdit && mossEditSettings.mossImageAdvanced?.positive)) || !editModel || !working) return;
    // A required image-edit LoRA that isn't downloaded yet blocks the run (worker R5): the source
    // band renders the actionable Download note (epic 10871, sc-11069). Defensive — the Generate
    // button is already disabled on this condition.
    if (editLoraRequiredMissing) return;
    if (!mossEditReady || mossEditError) return;
    // Canvas-extend / outpaint (sc-2556): resolve the output W×H from the chosen aspect
    // and fit mode (outpaint coerced away when the model can't inpaint). "match" keeps
    // the working size, so the existing same-size edit behavior is unchanged.
    const fitMode = effectiveFitMode(editFitMode, canMask);
    const { width: outWidth, height: outHeight } = mossEdit ? mossEditSettings : editOutputDims(working.width, working.height, editAspect, fitMode);
    // Same-size edit → active layer; a canvas-extend / outpaint (dims change) →
    // document-level flatten (sc-6119 tool matrix).
    const dimsChange = outWidth !== working.width || outHeight !== working.height;
    // A mask is sent only for inpaint-capable models; otherwise it's a whole-image edit (the mask
    // stays as a local guide but isn't uploaded). The mask is brush strokes (sc-2436) and/or a
    // smart-select base (sc-3751), composited together by rasterizeMaskToFile.
    const masked = canMask && (maskHasContent(maskLines) || Boolean(maskBaseImage));
    let maskFile = null;
    if (masked) {
      try {
        maskFile = await rasterizeMaskToFile();
      } catch (err) {
        setStatus({ loading: false, error: `Could not prepare the mask: ${err.message || err}` });
        return;
      }
    }
    runAiOp({
      label: "edit",
      endpoint: JOB_PATHS.image,
      layerSource: dimsChange ? "composite" : "activeLayer",
      edit: { op: "edit", model: editModel, prompt, ...(masked ? { masked: true } : {}) },
      maskFile,
      buildBody: (scratch, maskScratch) =>
        buildEditJobBody({
          project: activeProject,
          requestedGpu,
          sourceAssetId: scratch.id,
          maskAssetId: maskScratch?.id,
          // sc-24114: keep the alpha channel on an alpha-capable model when asked (or when the
          // working image carries alpha and the user has not turned it off).
          transparentBackground: editTransparent,
          modelEntry: selectedEditModel,
          // Multi-reference edit (sc-6107): lead with the working scratch image, then the user's
          // references. Only for a multiReference model with at least one attached reference.
          referenceAssetIds:
            multiRefCapable && refAssetIds.length
              // sc-24113: cap at the MODEL's ceiling, not the module constant. The working scratch
              // image occupies the first slot, so the list is (1 + attached) and both halves are
              // bounded by the same per-model number.
              ? mossEdit ? [scratch.id, ...refAssetIds] : editReferenceIds(scratch.id, refAssetIds, maxEditReferences)
              : null,
          model: editModel,
          prompt,
          seed: editSeed,
          width: outWidth,
          height: outHeight,
          fitMode,
          loras: mossEdit ? [] : editLoraSelection.serializedLoras,
          // Auto-apply the model's managed image-edit LoRA (R5) when installed — deduped inside
          // buildEditJobBody, so a run needs no manual picking (epic 10871, sc-11069).
          editLora: managedEditLoraId ? editLora : null,
          // Identity strength (sc-11798): the user's edit-LoRA weight override, or the default.
          editLoraWeight: managedEditLoraId ? editLoraSelection.weightFor(editLora) : null,
          guidanceScale: editGuidance,
          mossImageEdit,
          mossEditSettings,
        }),
    });
  }

  // When the in-flight op's job terminates, load the result back into the working
  // image (on success) and purge the ephemeral scratch + result assets.
  useEffect(() => {
    if (!aiOp?.jobId) return;
    const job = jobs?.find((item) => item.id === aiOp.jobId);
    if (!job || !terminalStatuses.has(job.status)) return;
    const { jobId, source, edit, onComplete, writeBack, targetLayerId } = aiOp;
    setAiOp(null); // stop tracking immediately so this can't re-enter on the next jobs tick
    const resultAsset = job.status === "completed" ? job.result?.assets?.[0] ?? null : null;
    (async () => {
      try {
        if (!resultAsset) {
          setStatus({ loading: false, error: job.error ?? job.message ?? "The operation failed." });
          return;
        }
        const outputs = nonCanvasJobOutputs(job);
        setNativeEditResults(outputs);
        if (outputs.length) {
          if (outputs.every((asset) => asset.type === "tensor")) {
            // Real tensors remain downloadable Library assets. They never enter the image
            // decoder, canvas writeback, or generic ephemeral-result purge path.
            setStatus({ loading: false, error: "" });
            return;
          }
        }
        // Smart-select (sc-3751): the caller's `onComplete` consumes the result asset itself (loads
        // the mask into the mask layer) — it does NOT replace the working image, so skip the install
        // / history / dirty path entirely.
        if (onComplete) {
          await onComplete(resultAsset);
          return;
        }
        const res = await mediaResponseWithTicketRetry(() => assetUrl(resultAsset));
        if (!res.ok) throw new Error(`Failed to load result (${res.status})`);
        const blob = await res.blob();
        const prepared = await blobToEditorImage(blob);
        if (!mountedRef.current) {
          URL.revokeObjectURL(prepared.objectUrl);
          return;
        }
        checkpoint();
        // Active-layer op (same-size edit / detail) → write the result back into the
        // target layer, preserving the rest of the stack; document op (upscale /
        // outpaint / box edit) → flatten the stack into one new base layer (sc-6119).
        if (writeBack === "activeLayer" && targetLayerId && layerById(workingRef.current, targetLayerId)) {
          replaceLayerImage(
            targetLayerId,
            prepared.image,
            prepared.objectUrl,
            prepared.blob,
          );
        } else {
          const preparedSource = prepared.downscaled
            ? { ...source, editorDownscaled: prepared.downscaled }
            : source;
          installWorkingImage(
            prepared.image,
            prepared.objectUrl,
            prepared.blob,
            preparedSource,
          );
        }
        if (edit) setEdits((prev) => [...prev, edit]);
        setDirty(true);
      } catch (err) {
        if (mountedRef.current) {
          setStatus({ loading: false, error: err.message || "The operation failed." });
        }
      } finally {
        // Hand the purge to App (sc-8850): it owns the scratch registry, so it purges the
        // scratch + mask + result assets through the single survivor path. This also drops
        // the registry entry so the App-level sweep won't double-purge. The result is
        // loaded into the canvas above BEFORE this releases it — intermediates never persist.
        releaseEditorScratchOp?.(jobId, job);
      }
    })();
  }, [aiOp, jobs, installWorkingImage, replaceLayerImage, releaseEditorScratchOp, checkpoint, setAiOp, setNativeEditResults, mountedRef, workingRef, setEdits, setDirty, setStatus]);
  return { workingImageToFile, runAiOp, runUpscale, runDetail, runEdit };
}
