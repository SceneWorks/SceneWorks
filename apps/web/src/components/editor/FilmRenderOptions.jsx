import React, { useEffect, useMemo, useRef, useState } from "react";

import { isAbortError } from "../../api.js";
import { getFilmRenderOptions, previewFilmRenderOptions } from "../../api/films.js";

const REGIMES = [
  { id: "recommended_turbo", label: "Recommended turbo", detail: "Use the installed compatible acceleration adapters selected for this request." },
  { id: "quality", label: "Full quality", detail: "Use the base model without acceleration adapters or a step override." },
  { id: "custom", label: "Custom", detail: "Keep the adapter IDs and step count from Advanced video and budget settings." },
];

const UNAVAILABLE_REASONS = {
  model_unavailable: "The selected video model is unavailable.",
  no_installed_compatible_adapter: "No compatible turbo adapter is installed.",
  incomplete_partition_coverage: "Installed adapters do not cover every request partition.",
  incompatible_resolution: "The selected resolution is incompatible with the installed turbo adapters.",
};

function relevantDraftSignature(draft) {
  const model = draft.productionPlan.model;
  return JSON.stringify({
    revision: draft.revision,
    renderRegime: draft.renderRegime,
    model: {
      id: model.id,
      tier: model.tier,
      resolution: model.resolution,
      loras: model.loras,
      steps: model.advanced?.steps,
    },
    shots: draft.productionPlan.shots.map((shot) => ({
      id: shot.id,
      resolution: shot.resolution,
      mode: shot.conditioning.mode,
      firstFrameRole: shot.conditioning.firstFrameRole,
      lastFrameRole: shot.conditioning.lastFrameRole,
      referenceRoles: shot.conditioning.referenceRoles,
      chainFromShotId: shot.conditioning.chainFromShotId,
    })),
    references: draft.referencePack.references.map((reference) => ({
      role: reference.role,
      kind: reference.kind,
      approved: reference.approved,
      sourceAssetId: reference.sourceAssetId,
      file: reference.file,
    })),
  });
}

function recipeLabel(recipe) {
  if (!recipe) return "Waiting for resolved settings";
  const adapters = recipe.adapterIds?.length ? recipe.adapterIds.join(", ") : "base model";
  const steps = recipe.effectiveSteps == null ? "model-default steps" : `${recipe.effectiveSteps} steps`;
  return `${adapters} · ${steps}`;
}

export function FilmRenderOptions({ disabled, draft, onChange, projectId, token }) {
  const [options, setOptions] = useState(null);
  const [error, setError] = useState("");
  const [loading, setLoading] = useState(true);
  const [loadedDraftKey, setLoadedDraftKey] = useState("");
  const [baselineSignature, setBaselineSignature] = useState("");
  const requestSequence = useRef(0);
  const signature = useMemo(() => relevantDraftSignature(draft), [draft]);
  const currentSignature = useRef(signature);
  currentSignature.current = signature;
  const draftKey = `${projectId}:${draft.id}:${draft.revision}`;

  useEffect(() => {
    const controller = new AbortController();
    const request = ++requestSequence.current;
    const requestedSignature = currentSignature.current;
    setLoading(true);
    setError("");
    setLoadedDraftKey("");
    getFilmRenderOptions(projectId, draft.id, token, { signal: controller.signal })
      .then((value) => {
        if (request !== requestSequence.current || controller.signal.aborted) return;
        setLoadedDraftKey(draftKey);
        setBaselineSignature(requestedSignature);
        if (currentSignature.current === requestedSignature) {
          setOptions(value);
          setLoading(false);
        }
      })
      .catch((cause) => {
        if (request !== requestSequence.current || isAbortError(cause)) return;
        setLoading(false);
        setError(cause.message);
      });
    return () => controller.abort();
  }, [draft.id, draft.revision, draftKey, projectId, token]);

  useEffect(() => {
    if (loadedDraftKey !== draftKey || signature === baselineSignature) return undefined;
    const controller = new AbortController();
    const request = ++requestSequence.current;
    setLoading(true);
    setError("");
    const timer = window.setTimeout(() => {
      previewFilmRenderOptions(projectId, draft.id, draft, token, { signal: controller.signal })
        .then((value) => {
          if (request !== requestSequence.current || controller.signal.aborted) return;
          setOptions(value);
          setBaselineSignature(signature);
          setLoading(false);
        })
        .catch((cause) => {
          if (request !== requestSequence.current || isAbortError(cause)) return;
          setLoading(false);
          setError(cause.message);
        });
    }, 120);
    return () => { window.clearTimeout(timer); controller.abort(); };
  }, [baselineSignature, draft, draftKey, loadedDraftKey, projectId, signature, token]);

  const selected = draft.renderRegime ?? options?.selectedRegime ?? "custom";
  const recommendedUnavailable = options?.recommendedTurbo?.available === false;
  const unavailableReason = options?.recommendedTurbo?.unavailableReason;

  return (
    <section aria-labelledby="film-render-options-heading" className="ve-film-section ve-film-render-options">
      <div className="ve-film-section-heading">
        <div><h3 id="film-render-options-heading">Render choice</h3><p>Choose the saved render intent, then review the exact adapters and steps resolved for the current shots.</p></div>
        {loading ? <span aria-live="polite" className="ve-film-help">Resolving current settings…</span> : null}
      </div>
      <fieldset className="ve-film-render-regimes" disabled={disabled || loading}>
        <legend>Render regime</legend>
        {REGIMES.map((regime) => {
          const unavailable = regime.id === "recommended_turbo" && recommendedUnavailable;
          const recipe = regime.id === "recommended_turbo" ? options?.recommendedTurbo
            : regime.id === "quality" ? options?.quality : null;
          return <label className={selected === regime.id ? "selected" : ""} key={regime.id}>
            <span><input
              checked={selected === regime.id}
              disabled={unavailable}
              name="film-render-regime"
              onChange={() => onChange((next) => { next.renderRegime = regime.id; })}
              type="radio"
            />{regime.label}</span>
            <small>{regime.detail}</small>
            {recipe ? <small className="ve-film-render-recipe">{recipeLabel(recipe)}</small> : null}
            {unavailable ? <small className="ve-film-warning">{UNAVAILABLE_REASONS[unavailableReason] ?? unavailableReason ?? "Recommended turbo is unavailable."}</small> : null}
          </label>;
        })}
      </fieldset>
      {options?.effective ? <p className="ve-film-render-effective"><strong>Effective request:</strong> {recipeLabel(options.effective)}</p> : null}
      {error ? <p role="alert" className="ve-film-finding">Could not resolve render settings: {error}</p> : null}
    </section>
  );
}
