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

// Strict project-owned raster predicate. This intentionally does not reuse the
// generic picker category because SVG must never become vector conditioning.
export function vectorSourceAssets(assets, projectId) {
  return (assets ?? []).filter((asset) => asset?.projectId === projectId && assetCanRenderAsImage(asset) && !asset.status?.trashed && !asset.status?.rejected);
}

export function VectorStudio() {
  const { activeProject, assets = [], jobs = [], models = [], macCapabilities, createVectorJob, createModelDownloadJob, jobAction, setActiveView, studioLaunch } = useAppContext();
  const [sourceAssetId, setSourceAssetId] = useState("");
  const [detail, setDetail] = useState("standard");
  const [prompt, setPrompt] = useState("");
  const sources = useMemo(() => vectorSourceAssets(assets, activeProject?.id), [assets, activeProject?.id]);
  useEffect(() => {
    if (studioLaunch?.view !== "VectorStudio") return;
    if (sources.some((asset) => asset.id === studioLaunch.assetId)) setSourceAssetId(studioLaunch.assetId);
    const budget = studioLaunch.recipe?.detailBudget;
    const matching = Object.entries(VECTOR_DETAIL_PRESETS).find(([, value]) => value.maxNewTokens === budget?.maxNewTokens && value.maxSvgBytes === budget?.maxSvgBytes && value.maxWallTimeMs === budget?.maxWallTimeMs);
    if (matching) setDetail(matching[0]);
    if (studioLaunch.recipe?.prompt) setPrompt(studioLaunch.recipe.prompt);
  }, [studioLaunch, sources]);
  const vectorModels = models.filter((model) => model.type === "vector");
  // Select the declaration before provider availability: an installed StarVector
  // whose provider reports pending_terminal_inference_pin must surface that typed
  // state, rather than being misclassified as an unsupported mode.
  const model = vectorModels.find((item) => item.capabilities?.includes("image_to_svg"));
  const availability = vectorModelAvailability(model, "image_to_svg", macCapabilities);
  const offers = downloadOffersFor(vectorModels, (item, caps) => vectorModelServesMode(item, "image_to_svg", caps), macCapabilities);
  const vectorJobs = jobs.filter((job) => job.type === "vector_generate" && !terminalStatuses.has(job.status));
  const canSubmit = Boolean(sourceAssetId && availability.available && typeof createVectorJob === "function");
  const submit = async (event) => {
    event.preventDefault();
    if (!canSubmit) return;
    await createVectorJob({ mode: "image_to_svg", sourceAssetId, model: model.id, prompt: prompt.trim() || undefined, detailBudget: VECTOR_DETAIL_PRESETS[detail] });
  };
  const unavailableCopy = availability.reason === "pending_terminal_inference_pin"
    ? "StarVector is installed, but this machine is waiting for the terminal inference pin. Conversion will become available automatically when it is claimable."
    : "Install StarVector-1B from Model Manager to convert project raster images.";

  return (
    <section className="page-frame vector-studio" aria-labelledby="vector-studio-title">
      <div className="section-heading"><p className="eyebrow">Advanced</p><h2 id="vector-studio-title">Convert Image</h2><p>Creates canonical SVG plus a safe PNG preview. SVG source is never displayed in the app.</p></div>
      <ModelAvailabilityGate ready={availability.available} title="StarVector-1B is unavailable" description={unavailableCopy} offers={offers} onDownload={createModelDownloadJob} onOpenModels={() => setActiveView("Models")}>
        <form className="work-panel" onSubmit={submit}>
          <AssetPickerField assets={sources} label="Project raster image" onChange={setSourceAssetId} value={sourceAssetId} />
          <label>Optional guidance<input aria-label="Optional vector guidance" onChange={(event) => setPrompt(event.target.value)} placeholder="Preserve bold silhouettes" value={prompt} /></label>
          <label>Detail<select aria-label="Vector detail" onChange={(event) => setDetail(event.target.value)} value={detail}>{Object.entries(VECTOR_DETAIL_PRESETS).map(([key, preset]) => <option key={key} value={key}>{preset.label}</option>)}</select></label>
          <button disabled={!canSubmit} type="submit">Convert to SVG</button>
        </form>
      </ModelAvailabilityGate>
      {vectorJobs.map((job) => <WorkerProgressCard job={job} key={job.id} onCancel={jobAction ? (item) => jobAction(item, "cancel") : undefined} />)}
    </section>
  );
}
