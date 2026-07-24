import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { useAppStatic } from "../context/AppContext.js";
import { PROMPT_REFINE_MODEL_ID } from "../constants.js";

// The Simple studios' prompt refine (design handoff: "Wire to the Anubis-8B refiner").
//
// Same worker job the advanced RefinePromptControl runs — including the best-effort
// fetch of the selected model's prompt guide, which is what gives the rewrite its
// model-specific context — but the review step is dropped: the rewrite replaces the
// prompt and the panel closes, per the design. The one advanced behavior that is NOT
// dropped is the missing-model path: the refiner is a downloadable catalog model, so
// when it isn't installed we surface its download rather than a raw worker error.

function isModelMissing(refineModel, message) {
  if (refineModel?.installState) {
    return refineModel.installState === "missing";
  }
  return /not cached|not installed|snapshot is not/i.test(message ?? "");
}

export function useSimpleRefine(workflow) {
  const { refinePrompt, models = [], createModelDownloadJob } = useAppStatic();
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState("");
  const controllerRef = useRef(null);

  const refineModel = useMemo(
    () => models.find((entry) => entry.id === PROMPT_REFINE_MODEL_ID) ?? null,
    [models],
  );

  useEffect(
    () => () => {
      controllerRef.current?.abort();
    },
    [],
  );

  // Resolves to the rewritten prompt, or null when it failed (the reason is in `error`).
  const run = useCallback(
    async ({ prompt, modelId, guidePath }) => {
      const trimmed = String(prompt ?? "").trim();
      if (!trimmed || typeof refinePrompt !== "function" || busy) {
        return null;
      }
      controllerRef.current?.abort();
      const controller = new AbortController();
      controllerRef.current = controller;
      const { signal } = controller;
      setBusy(true);
      setError("");
      try {
        let guide = "";
        if (guidePath) {
          try {
            const response = await fetch(guidePath, { signal });
            if (response.ok) {
              guide = await response.text();
            }
          } catch (err) {
            if (err?.name === "AbortError") {
              return null;
            }
            guide = "";
          }
        }
        const refined = await refinePrompt({ prompt: trimmed, modelId, workflow, guide, signal });
        return signal.aborted ? null : refined;
      } catch (err) {
        if (err?.name === "AbortError") {
          return null;
        }
        setError(err?.message || "Prompt refinement failed.");
        return null;
      } finally {
        if (controllerRef.current === controller) {
          controllerRef.current = null;
        }
        setBusy(false);
      }
    },
    [busy, refinePrompt, workflow],
  );

  const downloadModel = useCallback(async () => {
    if (!refineModel || typeof createModelDownloadJob !== "function") {
      return;
    }
    const job = await createModelDownloadJob(refineModel);
    if (job) {
      setError("Downloading the refinement model — track it in the Queue, then try again.");
    }
  }, [createModelDownloadJob, refineModel]);

  return {
    busy,
    error,
    run,
    reset: () => setError(""),
    downloadModel: refineModel ? downloadModel : null,
    modelMissing: Boolean(error) && isModelMissing(refineModel, error),
  };
}
