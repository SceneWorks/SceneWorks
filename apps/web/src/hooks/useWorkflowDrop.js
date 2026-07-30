import { useCallback, useRef, useState } from "react";

import { inspectWorkflowFile } from "../api.js";
import {
  looksLikeWorkflowCandidate,
  recipeFromWorkflowShare,
  workflowReplaySeed,
  WORKFLOW_STATUS_WORKFLOW,
} from "../workflowShare.js";

// "Someone sent you an image, you drag it onto SceneWorks, you are in Image Studio with their
// settings loaded" (sc-15951, epic 15945).
//
// Owns the state behind the offer panel: the inspect round trip for a file no in-app dropzone
// claimed, the panel it opens when a workflow is found, and the two things the panel can do with
// it. The drop ROUTING is `useDropNavigationGuard`; the prefill is App's `launchImageRecipe`,
// which is the same seam "Use this recipe" goes through.
//
// # Nothing happens for an image with no workflow
//
// That is the common case — every foreign PNG in the world — and it must stay indistinguishable
// from the swallow that was there before this hook existed. So the panel is opened only AFTER a
// workflow has actually been read: no optimistic open, no spinner, no error band. A
// `no_workflow` response, a file that never passed the client prescreen, and a failure that says
// nothing about the file all leave the app exactly as it was.

// The `code`s worth showing a person. Everything else — a `no_workflow` body, a `not_png` (the
// prescreen already refused those, so one here means the file lied and today's swallow is the
// right answer), a malformed multipart (our bug, not their file), a network failure, an abort —
// resolves to silence.
//
// `too_large` is on the list even though the prescreen bounds the size: the client mirrors the
// server's cap rather than reading it, so an install whose server cap is LOWER than this build's
// constant would otherwise fail silently on a file the user watched disappear.
const SHOWABLE_INSPECT_CODES = new Set([
  "workflow_inspect_unreadable",
  "workflow_inspect_read_failed",
  "workflow_inspect_stage_failed",
  "workflow_inspect_catalog_failed",
  "workflow_inspect_too_large",
]);

// The sentence to show for a failed inspect, or null to stay silent.
export function inspectFailureMessage(error) {
  if (!error || !SHOWABLE_INSPECT_CODES.has(error.code)) {
    return null;
  }
  const message = typeof error.message === "string" ? error.message.trim() : "";
  return message || null;
}

export function useWorkflowDrop({
  enabled = true,
  projectId = null,
  token = "",
  importAsset = null,
  launchRecipe = null,
} = {}) {
  // `null` while there is nothing to offer — which is almost always. When set it is either a read
  // workflow (`share` + `report`) or a failure worth naming (`error`), never both.
  const [offer, setOffer] = useState(null);
  const [importState, setImportState] = useState({ busy: false, done: false, error: "" });
  // Monotonic ticket: a second drop supersedes the first, so a slow inspect cannot open a panel
  // for a file the user has already replaced.
  const ticketRef = useRef(0);

  const dismiss = useCallback(() => {
    ticketRef.current += 1;
    setOffer(null);
    setImportState({ busy: false, done: false, error: "" });
  }, []);

  const handleDroppedFile = useCallback(
    async (file) => {
      if (!enabled) {
        return;
      }
      // The ticket is taken BEFORE the prescreen, not after it. The prescreen is itself
      // asynchronous (a range read off the file), so a ticket taken after it would leave a
      // window in which `dismiss` — or a newer drop — could not supersede this one.
      ticketRef.current += 1;
      const ticket = ticketRef.current;
      if (!(await looksLikeWorkflowCandidate(file))) {
        return;
      }
      if (ticket !== ticketRef.current) {
        return;
      }
      let response = null;
      try {
        response = await inspectWorkflowFile(file, { projectId, token });
      } catch (error) {
        if (ticket !== ticketRef.current) {
          return;
        }
        const detail = inspectFailureMessage(error);
        if (detail) {
          setImportState({ busy: false, done: false, error: "" });
          setOffer({ file, share: null, report: null, error: detail });
        }
        return;
      }
      if (ticket !== ticketRef.current) {
        return;
      }
      // The no-workflow branch, and the only one that has to be invisible.
      if (response?.status !== WORKFLOW_STATUS_WORKFLOW || !response.workflow) {
        return;
      }
      setImportState({ busy: false, done: false, error: "" });
      setOffer({
        file,
        share: response.workflow,
        report: response.resolution ?? null,
        error: "",
      });
    },
    [enabled, projectId, token],
  );

  // "Use this workflow" — build the recipe and hand it to the SAME launch the viewer's
  // "Use this recipe" uses. The panel closes either way: a launch that bailed (a project switch
  // raced the hydration) has already navigated to the studio, and leaving the panel over it would
  // be worse than closing.
  const useWorkflow = useCallback(async () => {
    const current = offer;
    if (!current?.share || typeof launchRecipe !== "function") {
      return;
    }
    const recipe = recipeFromWorkflowShare(current.share, current.report);
    dismiss();
    await launchRecipe({ recipe, replaySeed: workflowReplaySeed(current.share) });
  }, [offer, launchRecipe, dismiss]);

  // "Import the image too" — the ordinary upload path, which is what records
  // `extra.importedWorkflow` on the asset (sc-15949). The panel stays open on success so the
  // user can still take the workflow into the studio; "too" is not "instead".
  const importImage = useCallback(async () => {
    const current = offer;
    if (!current?.file || typeof importAsset !== "function" || importState.busy) {
      return;
    }
    setImportState({ busy: true, done: false, error: "" });
    try {
      await importAsset(current.file, { throwOnError: true });
      setImportState({ busy: false, done: true, error: "" });
    } catch (error) {
      setImportState({ busy: false, done: false, error: error?.message ?? "Import failed." });
    }
  }, [offer, importAsset, importState.busy]);

  return { offer, importState, handleDroppedFile, dismiss, useWorkflow, importImage };
}
