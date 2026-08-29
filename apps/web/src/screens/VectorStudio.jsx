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
  standard: { label: "Standard", maxNewTokens: 4096, maxSvgBytes: 262144, maxWallTimeMs: 120000 },
  detailed: { label: "Detailed", maxNewTokens: 8192, maxSvgBytes: 524288, maxWallTimeMs: 180000 },
});

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
  const [replayRevisions, setReplayRevisions] = useState(null);
  const sources = useMemo(() => vectorSourceAssets(assets, activeProject?.id), [assets, activeProject?.id]);
  const vectorModels = models.filter((model) => model.type === "vector");
  // Select the declaration before provider availability: an installed StarVector
  // whose provider reports pending_terminal_inference_pin must surface that typed
  // state, rather than being misclassified as an unsupported mode.
  const conversionModel = vectorModels.find((item) => item.capabilities?.includes("image_to_svg"));
  const conversionAvailability = vectorModelAvailability(conversionModel, "image_to_svg", macCapabilities);
  const offers = downloadOffersFor(vectorModels, (item, caps) => vectorModelServesMode(item, "image_to_svg", caps), macCapabilities);
  const rasterWorkflowModels = models.filter((model) => rasterPromptModelAvailability(model, macCapabilities).available);
  const vectorWorkflowModels = vectorModels.filter((model) => promptVectorModelAvailability(model, macCapabilities).available);
  const promptWorkflowAvailable = macCapabilitiesAuthoritative === true && rasterWorkflowModels.length > 0 && vectorWorkflowModels.length > 0;
  const rasterWorkflowModel = rasterWorkflowModels.find((model) => model.id === rasterModelId) ?? rasterWorkflowModels[0];
  const vectorWorkflowModel = vectorWorkflowModels.find((model) => model.id === vectorModelId) ?? vectorWorkflowModels[0];

  useEffect(() => {
    if (!rasterModelId && rasterWorkflowModels[0]) setRasterModelId(rasterWorkflowModels[0].id);
    if (!vectorModelId && vectorWorkflowModels[0]) setVectorModelId(vectorWorkflowModels[0].id);
  }, [rasterModelId, rasterWorkflowModels, vectorModelId, vectorWorkflowModels]);

  useEffect(() => {
    if (studioLaunch?.view !== "VectorStudio") return;
    if (sources.some((asset) => asset.id === studioLaunch.assetId)) setSourceAssetId(studioLaunch.assetId);
    const workflow = studioLaunch.recipe?.workflow;
    const budget = workflow?.vectorStage?.detailBudget ?? studioLaunch.recipe?.detailBudget;
    const matching = Object.entries(VECTOR_DETAIL_PRESETS).find(([, value]) => value.maxNewTokens === budget?.maxNewTokens && value.maxSvgBytes === budget?.maxSvgBytes && value.maxWallTimeMs === budget?.maxWallTimeMs);
    if (matching) setDetail(matching[0]);
    if (workflow?.kind === "create_from_prompt") {
      setMode("create_from_prompt");
      setPrompt(workflow.rasterStage?.prompt ?? "");
      setNegativePrompt(workflow.rasterStage?.negativePrompt ?? "");
      setRasterModelId(workflow.rasterStage?.model ?? "");
      setVectorModelId(workflow.vectorStage?.model ?? "");
      setReplayRevisions({ raster: workflow.rasterStage?.revision, vector: workflow.vectorStage?.revision });
    } else if (studioLaunch.recipe?.prompt) {
      setPrompt(studioLaunch.recipe.prompt);
    }
  }, [studioLaunch, sources]);

  const vectorJobs = jobs.filter((job) => job.type === "vector_generate" && !terminalStatuses.has(job.status));
  const canConvert = Boolean(sourceAssetId && conversionAvailability.available && typeof createVectorJob === "function");
  const canCreateFromPrompt = Boolean(prompt.trim() && rasterWorkflowModel && vectorWorkflowModel && typeof createVectorPromptWorkflow === "function");
  const submit = async (event) => {
    event.preventDefault();
    if (mode === "create_from_prompt") {
      if (!canCreateFromPrompt) return;
      await createVectorPromptWorkflow({
        prompt: prompt.trim(),
        negativePrompt: negativePrompt.trim() || undefined,
        rasterModel: rasterWorkflowModel.id,
        vectorModel: vectorWorkflowModel.id,
        detailBudget: VECTOR_DETAIL_PRESETS[detail],
        expectedRasterRevision: replayRevisions?.raster,
        expectedVectorRevision: replayRevisions?.vector,
      });
      return;
    }
    if (!canConvert) return;
    await createVectorJob({ mode: "image_to_svg", sourceAssetId, model: conversionModel.id, prompt: prompt.trim() || undefined, detailBudget: VECTOR_DETAIL_PRESETS[detail] });
  };
  const unavailableCopy = conversionAvailability.reason === "pending_terminal_inference_pin"
    ? "StarVector is installed, but this machine is waiting for the terminal inference pin. Conversion will become available automatically when it is claimable."
    : "Install StarVector-1B from Model Manager to convert project raster images.";
  const chooseRasterModel = (event) => {
    setRasterModelId(event.target.value);
    setReplayRevisions(null);
  };
  const chooseVectorModel = (event) => {
    setVectorModelId(event.target.value);
    setReplayRevisions(null);
  };

  return (
    <section className="page-frame vector-studio" aria-labelledby="vector-studio-title">
      <div className="section-heading"><p className="eyebrow">Advanced</p><h2 id="vector-studio-title">Vector Studio</h2><p>Creates canonical SVG plus a safe PNG preview. SVG source is never displayed in the app.</p></div>
      <div aria-label="Vector workflow" className="mode-tabs" role="tablist">
        <button aria-selected={mode === "convert_image"} onClick={() => setMode("convert_image")} role="tab" type="button">Convert Image</button>
        {promptWorkflowAvailable ? <button aria-selected={mode === "create_from_prompt"} onClick={() => setMode("create_from_prompt")} role="tab" type="button">Create from Prompt</button> : null}
      </div>
      {mode === "create_from_prompt" && promptWorkflowAvailable ? (
        <form className="work-panel" onSubmit={submit}>
          <p role="note">This is a disclosed two-stage workflow, not direct text-to-SVG: {rasterWorkflowModel.name ?? rasterWorkflowModel.id} at {authoritativeModelRevision(rasterWorkflowModel)} creates a hidden retained raster, then {vectorWorkflowModel.name ?? vectorWorkflowModel.id} at {authoritativeModelRevision(vectorWorkflowModel)} vectorizes it.</p>
          <label>Prompt<textarea aria-label="Vector prompt" onChange={(event) => setPrompt(event.target.value)} required value={prompt} /></label>
          <label>Negative prompt<input aria-label="Negative raster prompt" onChange={(event) => setNegativePrompt(event.target.value)} value={negativePrompt} /></label>
          <label>Raster model<select aria-label="Raster model" onChange={chooseRasterModel} value={rasterWorkflowModel.id}>{rasterWorkflowModels.map((model) => <option key={model.id} value={model.id}>{model.name ?? model.id}</option>)}</select></label>
          <label>Vector model<select aria-label="Vector model" onChange={chooseVectorModel} value={vectorWorkflowModel.id}>{vectorWorkflowModels.map((model) => <option key={model.id} value={model.id}>{model.name ?? model.id}</option>)}</select></label>
          <label>Detail<select aria-label="Vector detail" onChange={(event) => setDetail(event.target.value)} value={detail}>{Object.entries(VECTOR_DETAIL_PRESETS).map(([key, preset]) => <option key={key} value={key}>{preset.label}</option>)}</select></label>
          <button disabled={!canCreateFromPrompt} type="submit">Create SVG</button>
        </form>
      ) : (
        <ModelAvailabilityGate ready={conversionAvailability.available} title="StarVector-1B is unavailable" description={unavailableCopy} offers={offers} onDownload={createModelDownloadJob} onOpenModels={() => setActiveView("Models")}>
          <form className="work-panel" onSubmit={submit}>
            <AssetPickerField assets={sources} label="Project raster image" onChange={setSourceAssetId} value={sourceAssetId} />
            <label>Optional guidance<input aria-label="Optional vector guidance" onChange={(event) => setPrompt(event.target.value)} placeholder="Preserve bold silhouettes" value={prompt} /></label>
            <label>Detail<select aria-label="Vector detail" onChange={(event) => setDetail(event.target.value)} value={detail}>{Object.entries(VECTOR_DETAIL_PRESETS).map(([key, preset]) => <option key={key} value={key}>{preset.label}</option>)}</select></label>
            <button disabled={!canConvert} type="submit">Convert to SVG</button>
          </form>
        </ModelAvailabilityGate>
      )}
      {vectorJobs.map((job) => <WorkerProgressCard job={job} key={job.id} onCancel={jobAction ? (item) => jobAction(item, "cancel") : undefined} />)}
    </section>
  );
}
