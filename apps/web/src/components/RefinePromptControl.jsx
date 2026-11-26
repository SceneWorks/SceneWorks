import { LOOP_UTILITY_NAME, loopPromptSource, loopTierInstalled, loopUtilityTier } from "../loopUtility.js";
import { LoopUtilityTier } from "./LoopUtilityTier.jsx";
import { PromptContentOptions } from "./PromptContentOptions.jsx";
import { installedQwen38Tiers, QWEN38_MODEL_ID, qwen38PromptSource } from "../qwen38Utility.js";
import { useAppContextOptional } from "../context/AppContext.js";
import { portableFeedbackBlocks, portableFeedbackStatus } from "../portableFeedbackState.js";
import { formatDecimalGb } from "../formatting.js";
import { fetchPromptGuide } from "../promptGuides.js";
import { isModelMissing } from "../modelMissing.js";
import React, { useEffect, useRef, useState } from "react";
import { Icon } from "./Icons.jsx";
import { usePortableFeedback } from "../hooks/usePortableFeedback.js";
import { ValidationSummary } from "../validation/Validation.jsx";
import { issue } from "../validation/issues.js";


// Detect the "refinement model isn't provisioned" failure (sc-5605) so we can offer a
// one-click download instead of a raw worker error. The native worker resolves an
// already-cached snapshot and fast-fails with "…snapshot is not cached…" when the model
// is absent (the retired Python path auto-downloaded). Prefer the catalog install state
// when the entry is supplied; fall back to the worker message otherwise.

// "Refine my prompt" affordance shared by Image and Video Studio (sc-2041).
// Sends the current prompt + the selected model's guide to the refinement worker
// (via the context `refinePrompt`), then shows the rewrite for review. The
// original prompt is never changed until the user clicks Apply.
//
// When the refinement model isn't provisioned, the worker fails fast; instead of a
// raw error we surface a download affordance (sc-5605). `refineModel` is the catalog
// entry for the refinement LLM (PROMPT_REFINE_MODEL_ID) and `onDownloadRefineModel`
// enqueues its ModelDownload job; both are optional so older callers still work.
export function RefinePromptControl({
  prompt,
  guidePath,
  modelId,
  workflow,
  refinePrompt,
  onApply,
  refineModel,
  onDownloadRefineModel,
  loopModel,
  qwenModel: suppliedQwenModel,
  onDownloadLoopModel,
  // Optional immediate-run mode for callers that need no setup step. Image and Video Studio
  // use the default button so users can choose content exclusions before starting.
  autoStart = false,
}) {
  const app = useAppContextOptional();
  const qwenModel = suppliedQwenModel ?? app?.models?.find((entry) => entry.id === QWEN38_MODEL_ID);
  const installedQwenTiers = installedQwen38Tiers(qwenModel);
  const [sourceChoice, setSourceChoice] = useState("default");
  const useLoop = sourceChoice === "loop";
  const qwenTier = sourceChoice.startsWith("qwen38:") ? sourceChoice.slice(7) : null;
  const useQwen = Boolean(qwenTier);
  const [promptExclusions, setPromptExclusions] = useState([]);
  const exclusions = promptExclusions.length ? { promptExclusions } : {};
  const [loopVariant, setLoopVariant] = useState(null);
  const selectedTier = loopUtilityTier(loopModel, loopVariant);
  const source = useQwen ? qwen38PromptSource(qwenTier) : loopPromptSource(useLoop ? selectedTier : null);
  const sourceReady = useQwen ? installedQwenTiers.some(([key]) => key === qwenTier) : !useLoop || loopTierInstalled(loopModel, selectedTier);

  const [status, setStatus] = useState("idle"); // idle | loading | review | error
  const [refined, setRefined] = useState("");
  const [error, setError] = useState("");
  const [downloadRequested, setDownloadRequested] = useState(false);
  const refineControllerRef = useRef(null);
  const autoStarted = useRef(false);

  const trimmed = (prompt ?? "").trim();
  const portableFeedback = usePortableFeedback(
    "promptRefine", { prompt: trimmed, modelId, workflow, ...source, ...exclusions }, null, Boolean(trimmed),
  );
  const portableRefusal = portableFeedback?.state === "invalid" ? portableFeedback.refusal : null;
  const busy = status === "loading";
  const disabled = !sourceReady || busy || !trimmed || portableFeedbackBlocks(portableFeedback) || typeof refinePrompt !== "function";

  const modelMissing = !useLoop && !useQwen && status === "error" && isModelMissing(refineModel, error);
  const installState = refineModel?.installState;

  // When the refinement model finishes downloading (a catalog refresh flips its
  // installState to "installed"), clear the missing-model error so the user can retry.
  useEffect(() => {
    if (installState === "installed" && downloadRequested) {
      setDownloadRequested(false);
      setStatus((current) => (current === "error" ? "idle" : current));
      setError((current) => (current ? "" : current));
    }
  }, [installState, downloadRequested]);

  useEffect(
    () => () => {
      refineControllerRef.current?.abort();
    },
    [],
  );

  // Start once after the current draft passes server validation.
  useEffect(() => {
    if (autoStart && !loopModel && !installedQwenTiers.length && !useLoop && !useQwen && !autoStarted.current && trimmed && typeof refinePrompt === "function" && !portableFeedbackBlocks(portableFeedback)) {
      autoStarted.current = true;
      handleRefine();
    }
  });

  async function handleRefine() {
    if (disabled) return;
    refineControllerRef.current?.abort();
    const controller = new AbortController();
    refineControllerRef.current = controller;
    const { signal } = controller;
    setStatus("loading");
    setError("");
    try {
      // The guide is first-party context for the rewrite; fetch it best-effort and
      // refine generically if it can't be loaded.
      let guide = "";
      if (guidePath) {
        try {
          guide = await fetchPromptGuide(guidePath, signal);
        } catch (err) {
          if (err?.name === "AbortError") {
            return;
          }
          guide = "";
        }
      }
      const result = await refinePrompt({ prompt: trimmed, modelId, workflow, guide, signal, ...source, ...exclusions });
      if (signal.aborted) {
        return;
      }
      setRefined(result);
      setStatus("review");
    } catch (err) {
      if (err?.name === "AbortError") {
        return;
      }
      setError(err?.message || "Prompt refinement failed.");
      setStatus("error");
    } finally {
      if (refineControllerRef.current === controller) {
        refineControllerRef.current = null;
      }
    }
  }

  async function handleDownloadModel() {
    if (typeof onDownloadRefineModel !== "function") return;
    try {
      // The download enqueuer resolves to the job, or null when it failed (it surfaces
      // its own error). Only switch to the "downloading" note on a real job.
      const job = await onDownloadRefineModel();
      if (job) {
        setDownloadRequested(true);
      }
    } catch (err) {
      // Surface an unexpected enqueue failure inline; the Models screen owns progress.
      setError(err?.message || "Could not start the refinement model download.");
    }
  }

  const sizeLabel = formatDecimalGb(refineModel?.downloadSizeBytes);
  const modelName = refineModel?.name || "Prompt Refiner";

  return (
    <div className="refine-control">
      <PromptContentOptions value={promptExclusions} disabled={busy} onChange={(value) => {
        setPromptExclusions(value);
        setStatus("idle");
        setError("");
      }} />
      {loopModel || useLoop || installedQwenTiers.length || useQwen ? (
        <label>
          Refinement model
          <select disabled={busy} value={sourceChoice} onChange={(event) => {
            setSourceChoice(event.target.value);
            setStatus("idle");
            setError("");
          }}>
            <option value="default">{modelName}</option>
            {loopModel || useLoop ? <option value="loop">{LOOP_UTILITY_NAME}</option> : null}
            {useQwen && !sourceReady ? <option value={sourceChoice}>Qwen3.8-27B {qwenTier} — not installed</option> : null}
            {installedQwenTiers.map(([tier, label]) => <option key={tier} value={`qwen38:${tier}`}>Qwen3.8-27B — {label}</option>)}
          </select>
        </label>
      ) : null}
      {useQwen && !sourceReady ? <p className="inline-warning" role="status">This Qwen3.8-27B tier is no longer installed. Choose an installed tier or install it in the Model Manager.</p> : null}
      {useLoop ? (
        <LoopUtilityTier model={loopModel} tier={selectedTier} disabled={busy}
          onChange={(tier) => { setLoopVariant(tier); setStatus("idle"); setError(""); }}
          onDownload={onDownloadLoopModel} />
      ) : null}
      {/* With autoStart the tile is the trigger, so hide the button while refining/reviewing —
          it reappears (as a re-run) only once the user keeps the original or an error needs a retry. */}
      {autoStart && (status === "loading" || status === "review") ? null : (
        <button className="hero-link refine-button" disabled={disabled} onClick={handleRefine} type="button">
          <Icon.Wand size={14} /> {busy ? "Refining…" : "Refine my prompt"}
        </button>
      )}
      {autoStart && status === "loading" ? (
        <p className="refine-status">
          <Icon.Wand size={14} /> Refining…
        </p>
      ) : null}

      {status === "error" && modelMissing ? (
        <div className="refine-missing-model" role="alert">
          {downloadRequested ? (
            <p className="refine-error">
              Downloading the prompt refinement model… track progress on the Models screen, then try Refine
              again.
            </p>
          ) : (
            <>
              <p className="refine-error">
                The prompt refinement model{sizeLabel ? ` (${sizeLabel})` : ""} isn’t installed yet.
              </p>
              {typeof onDownloadRefineModel === "function" ? (
                <button className="secondary-action" onClick={handleDownloadModel} type="button">
                  Download refinement model
                </button>
              ) : (
                <p className="refine-error">Open the Models screen to download “{modelName}”.</p>
              )}
            </>
          )}
        </div>
      ) : status === "error" ? (
        <p className="refine-error" role="alert">
          {error}
        </p>
      ) : null}
      {portableFeedbackStatus(portableFeedback) ? <p role="status">{portableFeedbackStatus(portableFeedback)}</p> : null}
      <ValidationSummary issues={portableRefusal ? [issue.error(portableRefusal.field, portableRefusal.detail)] : []} label="Prompt refinement errors" />

      {status === "review" ? (
        <div className="refine-review">
          <p className="refine-review-label">Suggested rewrite</p>
          <p className="refine-review-text">{refined}</p>
          <div className="refine-review-actions">
            <button
              className="secondary-action"
              onClick={() => {
                onApply(refined);
                setStatus("idle");
              }}
              type="button"
            >
              Apply
            </button>
            <button className="secondary-action" onClick={() => setStatus("idle")} type="button">
              Keep original
            </button>
          </div>
        </div>
      ) : null}
    </div>
  );
}
