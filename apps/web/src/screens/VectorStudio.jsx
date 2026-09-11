import React, { useEffect, useMemo, useState } from "react";
import { AssetPickerField } from "../components/AssetPicker.jsx";
import { ModelAvailabilityGate } from "../components/ModelAvailabilityGate.jsx";
import { WorkerProgressCard } from "../components/WorkerProgressCard.jsx";
import { assetCanRenderAsImage } from "../components/assetMedia.jsx";
import { useAppContext } from "../context/AppContext.js";
import { downloadOffersFor, vectorModelAvailability, vectorModelServesMode } from "../modelEligibility.js";
import { terminalStatuses } from "../constants.js";

export const VECTOR_DETAIL_PRESETS = Object.freeze({
  draft: { label: "Draft", maxNewTokens: 2048, maxSvgBytes: 131072, maxWallTimeMs: 60000 },
  standard: { label: "Standard", maxNewTokens: 3000, maxSvgBytes: 196608, maxWallTimeMs: 90000 },
  detailed: { label: "Detailed", maxNewTokens: 4000, maxSvgBytes: 262144, maxWallTimeMs: 120000 },
});

// Presentation labels never cross the API's deny_unknown_fields boundary. Catalog limits are
// authoritative; recipe budgets are preserved separately and validated rather than silently clamped.
export function vectorDetailBudget(preset, model) {
  return Object.fromEntries(["maxNewTokens", "maxSvgBytes", "maxWallTimeMs"].map((key) =>
    [key, Math.min(preset[key], model?.vector?.[key] ?? VECTOR_DETAIL_PRESETS.detailed[key])],
  ));
}

function budgetFitsModel(budget, model) {
  return ["maxNewTokens", "maxSvgBytes", "maxWallTimeMs"].every((key) =>
    Number.isSafeInteger(budget?.[key]) && budget[key] > 0 && budget[key] <= (model?.vector?.[key] ?? VECTOR_DETAIL_PRESETS.detailed[key]),
  );
}

const IMMUTABLE_REVISION = /^[0-9a-f]{40}$/;

// A composed workflow may record only one primary artifact identity. Imported/path-backed models,
// mutable refs, and manifests with multiple primary revisions stay ineligible rather than making
// replay guess which bytes generated the retained raster.
export function authoritativeModelRevision(model) {
  const primaryDownloads = (model?.downloads ?? []).filter((download) => download?.coRequisite !== true);
  if (!primaryDownloads.length || primaryDownloads.some((download) => typeof download?.revision !== "string" || !IMMUTABLE_REVISION.test(download.revision))) return null;
  const revisions = new Set(primaryDownloads.map((download) => download.revision));
  return revisions.size === 1 ? [...revisions][0] : null;
}

export function rasterPromptModelAvailability(model, caps) {
  if (model?.type !== "image" || !model?.capabilities?.includes("text_to_image")) {
    return { available: false, reason: "unsupported_mode" };
  }
  const backend = caps?.platform === "macos" || caps?.platform === "darwin" ? "mlx" : "candle";
  const backendSupport = backend === "mlx" ? model?.macSupport : model?.candleSupport;
  if (backendSupport?.supported !== true || model?.usable === false) {
    return { available: false, reason: "backend_unclaimable", backend };
  }
  if (model?.installState !== "installed" || model?.cacheState !== "complete") {
    return { available: false, reason: "model_missing", backend };
  }
  const revision = authoritativeModelRevision(model);
  if (!revision) return { available: false, reason: "artifact_ambiguous", backend };
  return { available: true, reason: null, backend, revision };
}

export function promptVectorModelAvailability(model, caps) {
  const availability = vectorModelAvailability(model, "image_to_svg", caps);
  if (!availability.available) return availability;
  const revision = authoritativeModelRevision(model);
  if (!revision) return { available: false, reason: "artifact_ambiguous", backend: availability.backend };
  return { ...availability, revision };
}

// Strict project-owned raster predicate. This intentionally does not reuse the
// generic picker category because SVG must never become vector conditioning.
export function vectorSourceAssets(assets, projectId) {
  return (assets ?? []).filter((asset) => asset?.projectId === projectId && assetCanRenderAsImage(asset) && !asset.status?.trashed && !asset.status?.rejected);
}

export function VectorStudio() {
  const {
    activeProject,
    assets = [],
    jobs = [],
    models = [],
    macCapabilities,
    macCapabilitiesAuthoritative,
    createVectorJob,
    createVectorPromptWorkflow,
    createModelDownloadJob,
    jobAction,
    setActiveView,
    studioLaunch,
  } = useAppContext();
  const [mode, setMode] = useState("convert_image");
  const [sourceAssetId, setSourceAssetId] = useState("");
  const [detail, setDetail] = useState("standard");
  const [prompt, setPrompt] = useState("");
  const [negativePrompt, setNegativePrompt] = useState("");
  const [rasterModelId, setRasterModelId] = useState("");
  const [vectorModelId, setVectorModelId] = useState("");
  const [conversionModelId, setConversionModelId] = useState("");
  const [replay, setReplay] = useState(null);
  const sources = useMemo(() => vectorSourceAssets(assets, activeProject?.id), [assets, activeProject?.id]);
  const vectorModels = models.filter((model) => model.type === "vector");
  // Select the declaration before provider availability: an installed StarVector
  // whose provider reports pending_terminal_inference_pin must surface that typed
  // state, rather than being misclassified as an unsupported mode.
  const conversionModels = vectorModels.filter((item) => item.capabilities?.includes("image_to_svg"));
  const readyConversionModels = conversionModels.filter((item) => vectorModelAvailability(item, "image_to_svg", macCapabilities).available);
  const conversionModel = conversionModelId
    ? conversionModels.find((item) => item.id === conversionModelId)
    : readyConversionModels[0] ?? conversionModels[0];
  const conversionAvailability = vectorModelAvailability(conversionModel, "image_to_svg", macCapabilities);
  const offers = downloadOffersFor(vectorModels, (item, caps) => vectorModelServesMode(item, "image_to_svg", caps), macCapabilities);
  const rasterWorkflowModels = models.filter((model) => rasterPromptModelAvailability(model, macCapabilities).available);
  const vectorWorkflowModels = vectorModels.filter((model) => promptVectorModelAvailability(model, macCapabilities).available);
  const promptWorkflowAvailable = macCapabilitiesAuthoritative === true && rasterWorkflowModels.length > 0 && vectorWorkflowModels.length > 0;
  const rasterWorkflowModel = (rasterModelId ? rasterWorkflowModels.find((model) => model.id === rasterModelId) : rasterWorkflowModels[0]);
  const vectorWorkflowModel = (vectorModelId ? vectorWorkflowModels.find((model) => model.id === vectorModelId) : vectorWorkflowModels[0]);

  useEffect(() => {
    if (!rasterModelId && rasterWorkflowModels[0]) setRasterModelId(rasterWorkflowModels[0].id);
    if (!vectorModelId && vectorWorkflowModels[0]) setVectorModelId(vectorWorkflowModels[0].id);
  }, [rasterModelId, rasterWorkflowModels, vectorModelId, vectorWorkflowModels]);

  useEffect(() => {
    if (studioLaunch?.view !== "VectorStudio") return;
    const recipe = studioLaunch.recipe;
    const workflow = recipe?.workflow;
    const composed = workflow?.kind === "create_from_prompt";
    const budget = composed ? workflow.vectorStage?.detailBudget : recipe?.detailBudget;
    setReplay(recipe ? {
      mode: recipe.mode ?? "image_to_svg",
      budget,
      sampling: composed ? workflow.vectorStage?.sampling : recipe.sampling,
      seed: composed ? workflow.rasterStage?.seed : undefined,
      width: composed ? workflow.rasterStage?.width : undefined,
      height: composed ? workflow.rasterStage?.height : undefined,
      rasterRevision: composed ? workflow.rasterStage?.revision : undefined,
      vectorRevision: composed ? workflow.vectorStage?.revision : undefined,
    } : null);
    setDetail(budget ? "recipe" : "standard");
    setMode(composed ? "create_from_prompt" : "convert_image");
    setSourceAssetId(studioLaunch.assetId ?? "");
    setPrompt(composed ? workflow.rasterStage?.prompt ?? "" : recipe?.prompt ?? "");
    setNegativePrompt(composed ? workflow.rasterStage?.negativePrompt ?? "" : "");
    if (composed) {
      setRasterModelId(workflow.rasterStage?.model ?? "");
      setVectorModelId(workflow.vectorStage?.model ?? "");
    } else if (recipe?.model) {
      setConversionModelId(recipe.model);
    }
  }, [studioLaunch]);

  const selectedModel = mode === "create_from_prompt" ? vectorWorkflowModel : conversionModel;
  const detailBudget = detail === "recipe" ? replay?.budget : vectorDetailBudget(VECTOR_DETAIL_PRESETS[detail], selectedModel);
  const validBudget = budgetFitsModel(detailBudget, selectedModel);
  const detailOptions = <>{detail === "recipe" ? <option value="recipe">Recorded recipe</option> : null}{Object.entries(VECTOR_DETAIL_PRESETS).map(([key, preset]) => <option key={key} value={key}>{preset.label}</option>)}</>;
  const vectorJobs = jobs.filter((job) => job.type === "vector_generate" && !terminalStatuses.has(job.status));
  const canConvert = Boolean(sources.some((asset) => asset.id === sourceAssetId) && validBudget && (!replay?.mode || replay.mode === "image_to_svg") && conversionAvailability.available && typeof createVectorJob === "function");
  const canCreateFromPrompt = Boolean(prompt.trim() && validBudget && promptWorkflowAvailable && rasterWorkflowModel && vectorWorkflowModel && typeof createVectorPromptWorkflow === "function");
  const submit = async (event) => {
    event.preventDefault();
    if (mode === "create_from_prompt") {
      if (!canCreateFromPrompt) return;
      await createVectorPromptWorkflow({
        prompt: prompt.trim(),
        negativePrompt: negativePrompt.trim() || undefined,
        rasterModel: rasterWorkflowModel.id,
        vectorModel: vectorWorkflowModel.id,
        detailBudget,
        sampling: replay?.sampling,
        seed: replay?.seed,
        width: replay?.width,
        height: replay?.height,
        expectedRasterRevision: replay?.rasterRevision,
        expectedVectorRevision: replay?.vectorRevision,
      });
      return;
    }
    if (!canConvert) return;
    await createVectorJob({
      mode: "image_to_svg",
      sourceAssetId,
      model: conversionModel.id,
      prompt: conversionModel?.vector?.acceptsTextGuidance === true ? prompt.trim() : "",
      detailBudget,
      sampling: replay?.sampling,
    });
  };
  const unavailableCopy = conversionAvailability.reason === "pending_terminal_inference_pin"
    ? "StarVector is installed, but this machine is waiting for the terminal inference pin. Conversion will become available automatically when it is claimable."
    : conversionAvailability.reason === "pending_terminal_candidate"
      ? "StarVector-8B is installed, but dispatch stays disabled until its permanent-pin terminal candidate is accepted."
      : "Install a supported StarVector image-to-SVG checkpoint from Model Manager to convert project raster images.";
  const chooseRasterModel = (event) => {
    setRasterModelId(event.target.value);
    setReplay((current) => current && { ...current, rasterRevision: undefined });
  };
  const chooseVectorModel = (event) => {
    setVectorModelId(event.target.value);
    setReplay((current) => current && { ...current, vectorRevision: undefined });
  };

  return (
    <section className="page-frame vector-studio" aria-labelledby="vector-studio-title">
      <div className="section-heading"><p className="eyebrow">Advanced</p><h2 id="vector-studio-title">Vector Studio</h2><p>Creates canonical SVG plus a safe PNG preview. SVG source is never displayed in the app.</p></div>
      {replay && (!validBudget || (mode === "create_from_prompt" ? !rasterWorkflowModel || !vectorWorkflowModel : !conversionAvailability.available || replay.mode !== "image_to_svg")) ? <p role="alert">This recipe cannot run with the current models and limits. Select an available model and detail preset to change its recorded inputs. <button type="button" onClick={() => {
        setConversionModelId(readyConversionModels[0]?.id ?? "");
        setRasterModelId(rasterWorkflowModels[0]?.id ?? "");
        setVectorModelId(vectorWorkflowModels[0]?.id ?? "");
        setDetail("standard");
        setReplay((current) => current && { ...current, mode: "image_to_svg", rasterRevision: undefined, vectorRevision: undefined });
      }}>Use available models and Standard detail</button></p> : null}
      <div aria-label="Vector workflow" className="mode-tabs" role="tablist">
        <button aria-selected={mode === "convert_image"} onClick={() => setMode("convert_image")} role="tab" type="button">Convert Image</button>
        {promptWorkflowAvailable ? <button aria-selected={mode === "create_from_prompt"} onClick={() => setMode("create_from_prompt")} role="tab" type="button">Create from Prompt</button> : null}
      </div>
      {mode === "create_from_prompt" && rasterWorkflowModel && vectorWorkflowModel ? (
        <form className="work-panel" onSubmit={submit}>
          <p role="note">This is a disclosed two-stage workflow, not direct text-to-SVG: {rasterWorkflowModel.name ?? rasterWorkflowModel.id} at {authoritativeModelRevision(rasterWorkflowModel)} creates a hidden retained raster, then {vectorWorkflowModel.name ?? vectorWorkflowModel.id} at {authoritativeModelRevision(vectorWorkflowModel)} vectorizes it.</p>
          <label>Prompt<textarea aria-label="Vector prompt" onChange={(event) => setPrompt(event.target.value)} required value={prompt} /></label>
          <label>Negative prompt<input aria-label="Negative raster prompt" onChange={(event) => setNegativePrompt(event.target.value)} value={negativePrompt} /></label>
          <label>Raster model<select aria-label="Raster model" onChange={chooseRasterModel} value={rasterWorkflowModel.id}>{rasterWorkflowModels.map((model) => <option key={model.id} value={model.id}>{model.name ?? model.id}</option>)}</select></label>
          <label>Vector model<select aria-label="Vector model" onChange={chooseVectorModel} value={vectorWorkflowModel.id}>{vectorWorkflowModels.map((model) => <option key={model.id} value={model.id}>{model.name ?? model.id}</option>)}</select></label>
          <label>Detail<select aria-label="Vector detail" onChange={(event) => setDetail(event.target.value)} value={detail}>{detailOptions}</select></label>
          <button disabled={!canCreateFromPrompt} type="submit">Create SVG</button>
        </form>
      ) : (
        <ModelAvailabilityGate ready={conversionAvailability.available} title="StarVector is unavailable" description={unavailableCopy} offers={offers} onDownload={createModelDownloadJob} onOpenModels={() => setActiveView("Models")}>
          <form className="work-panel" onSubmit={submit}>
            {readyConversionModels.length > 1 ? <label>Vector model<select aria-label="Conversion model" onChange={(event) => setConversionModelId(event.target.value)} value={conversionModel?.id ?? conversionModelId}>{readyConversionModels.map((model) => <option key={model.id} value={model.id}>{model.name ?? model.id}</option>)}</select></label> : null}
            <AssetPickerField assets={sources} label="Project raster image" onChange={setSourceAssetId} value={sourceAssetId} />
            {conversionModel?.vector?.acceptsTextGuidance === true ? <label>Optional guidance<input aria-label="Optional vector guidance" onChange={(event) => setPrompt(event.target.value)} placeholder="Preserve bold silhouettes" value={prompt} /></label> : null}
            <label>Detail<select aria-label="Vector detail" onChange={(event) => setDetail(event.target.value)} value={detail}>{detailOptions}</select></label>
            <button disabled={!canConvert} type="submit">Convert to SVG</button>
          </form>
        </ModelAvailabilityGate>
      )}
      {vectorJobs.map((job) => <WorkerProgressCard job={job} key={job.id} onCancel={jobAction ? (item) => jobAction(item, "cancel") : undefined} />)}
    </section>
  );
}
