import { useCallback, useEffect, useMemo, useRef, useState } from "react";

import { fetchAssetWorkflow, inspectWorkflowFile, isAbortError } from "../api.js";
import {
  keptSubstituteModelId,
  pickedReferenceAssetIds,
  pickedSourceAssetId,
} from "../workflowReplayValidation.js";
import {
  looksLikeWorkflowCandidate,
  recipeFromWorkflowShare,
  workflowReplaySeed,
  WORKFLOW_STATUS_WORKFLOW,
} from "../workflowShare.js";

// "Someone sent you an image, you drag it onto SceneWorks, you are in Image Studio with their
// settings loaded" (sc-15951, epic 15945) — and its twin, "that image is already in your library
// and you want its recipe back" (sc-15952).
//
// Owns the state behind the offer panel: the round trip that produces a workflow plus a resolution
// report, the panel it opens, and what the panel can do with it. The drop ROUTING is
// `useDropNavigationGuard`; the prefill is App's `launchImageRecipe`, which is the same seam the
// viewer's "Use this recipe" goes through.
//
// # Two ways in, one offer
//
// A dropped FILE goes through `POST /api/v1/workflows/inspect`, which reads a recipe out of bytes
// that are not in the library. An imported ASSET goes through
// `GET /api/v1/projects/:id/assets/:id/workflow`, which reads the envelope sc-15949 already
// recorded at `extra.importedWorkflow` and resolves it against this install's catalogs. Both
// answer the same `{ status, workflow, resolution }` body and land in the same `offer`, because the
// panel's job — say what is in the recipe, say what this machine can do about it, hand it over —
// is identical either way. Two offers that described the same report differently is exactly how
// "the model is missing" becomes "the model is fine" on one of them.
//
// # Nothing happens for an image with no workflow
//
// That is the common case for a DROP — every foreign PNG in the world — and it must stay
// indistinguishable from the swallow that was there before this hook existed. So the panel is
// opened only AFTER a workflow has actually been read: no optimistic open, no spinner, no error
// band. A `no_workflow` response, a file that never passed the client prescreen, and a failure that
// says nothing about the file all leave the app exactly as it was.
//
// The asset route is the opposite case and gets the opposite treatment: the button that starts it
// is rendered only for an asset already known to carry an envelope, so a `no_workflow` answer there
// means the record and the button disagree, and saying nothing would leave a control that visibly
// did not work.

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

// A blank sheet for the panel's own inputs. One constant so opening an offer, dismissing one and
// superseding one cannot each reset a different subset — a substitute model left over from the
// previous image would be the silent substitution this epic exists to prevent, wearing a fresh
// panel's clothes.
const EMPTY_FIXES = { substituteModelId: "", inputPicks: {}, installing: {}, queued: {} };

export function useWorkflowDrop({
  enabled = true,
  projectId = null,
  token = "",
  importAsset = null,
  launchRecipe = null,
  // sc-15952. The installed models the studio would actually offer, so a substitute the user picked
  // is re-checked against the live list on every render rather than trusted once at pick time.
  models = [],
  // A cheap identity for "what this install has on disk", derived by App from the model and LoRA
  // catalogs. It changes when a download completes, because `useJobEvents` refetches both catalogs
  // on a completed `model_download` / `lora_download` — so re-resolution rides the app's existing
  // job stream instead of polling anything.
  catalogRevision = "",
  installModel = null,
  installLora = null,
} = {}) {
  // `null` while there is nothing to offer — which is almost always. When set it is either a read
  // workflow (`share` + `report`) or a failure worth naming (`error`), never both.
  const [offer, setOffer] = useState(null);
  const [importState, setImportState] = useState({ busy: false, done: false, error: "" });
  // What the USER supplied in the panel: a substitute model, the input images, and which installs
  // have been started from here.
  const [fixes, setFixes] = useState(EMPTY_FIXES);
  // Monotonic ticket: a second drop supersedes the first, so a slow inspect cannot open a panel
  // for a file the user has already replaced.
  const ticketRef = useRef(0);
  // The in-flight inspect's controller, so superseding a ticket actually STOPS the upload rather
  // than only ignoring its answer. The endpoint accepts up to 512 MiB and stages the whole body
  // before it reads a byte, so an ignored request is a multi-hundred-megabyte upload still running
  // — on a LAN deployment that is real bandwidth, and on unmount it is a leak.
  const inFlightRef = useRef(null);
  // The RE-RESOLUTION's own controller, kept apart from the one above on purpose: a catalog change
  // must never abort the read that is still opening the panel, and dismissing the panel must abort
  // both. One ref for two independent requests would have them cancelling each other.
  const reresolveRef = useRef(null);
  // The catalog identity the currently-shown report was computed against. Re-armed when an offer
  // opens so the first render never refetches a report it has just received.
  const resolvedRevisionRef = useRef(catalogRevision);
  // The open offer, readable from an effect that must not re-run when it changes.
  const offerRef = useRef(null);
  offerRef.current = offer;

  // Abandon whatever inspect is running and take the next ticket. Every supersede goes through
  // here so there is one place that can forget to abort.
  const supersede = useCallback(() => {
    inFlightRef.current?.abort();
    inFlightRef.current = null;
    ticketRef.current += 1;
    return ticketRef.current;
  }, []);

  const dismiss = useCallback(() => {
    supersede();
    setOffer(null);
    setImportState({ busy: false, done: false, error: "" });
    setFixes(EMPTY_FIXES);
  }, [supersede]);

  // Unmount is a supersede with nobody left to tell. Without this the request outlives the tree
  // that started it.
  useEffect(() => () => {
    inFlightRef.current?.abort();
    inFlightRef.current = null;
    reresolveRef.current?.abort();
    reresolveRef.current = null;
  }, []);

  const handleDroppedFile = useCallback(
    async (file) => {
      if (!enabled) {
        return;
      }
      // The ticket is taken BEFORE the prescreen, not after it. The prescreen is itself
      // asynchronous (a range read off the file), so a ticket taken after it would leave a
      // window in which `dismiss` — or a newer drop — could not supersede this one.
      const ticket = supersede();
      if (!(await looksLikeWorkflowCandidate(file))) {
        return;
      }
      if (ticket !== ticketRef.current) {
        return;
      }
      const controller = typeof AbortController === "function" ? new AbortController() : null;
      inFlightRef.current = controller;
      let response = null;
      try {
        response = await inspectWorkflowFile(file, {
          projectId,
          token,
          signal: controller?.signal,
        });
      } catch (error) {
        if (ticket !== ticketRef.current) {
          return;
        }
        inFlightRef.current = null;
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
      inFlightRef.current = null;
      // The no-workflow branch, and the only one that has to be invisible.
      if (response?.status !== WORKFLOW_STATUS_WORKFLOW || !response.workflow) {
        return;
      }
      setImportState({ busy: false, done: false, error: "" });
      setFixes(EMPTY_FIXES);
      resolvedRevisionRef.current = catalogRevision;
      setOffer({
        file,
        assetId: null,
        share: response.workflow,
        report: response.resolution ?? null,
        error: "",
      });
    },
    [enabled, projectId, token, supersede, catalogRevision],
  );

  // "Use this recipe" on an already-imported asset (sc-15952).
  //
  // The envelope is on the asset and needs no request; the REPORT does, because it is a fact about
  // this machine right now and the catalogs behind it are not in the browser. Same offer, same
  // panel, same launch.
  const offerAssetWorkflow = useCallback(
    async (asset) => {
      if (!enabled || !asset?.id || !projectId) {
        return;
      }
      const ticket = supersede();
      const controller = typeof AbortController === "function" ? new AbortController() : null;
      inFlightRef.current = controller;
      let response = null;
      try {
        response = await fetchAssetWorkflow(projectId, asset.id, {
          token,
          signal: controller?.signal,
        });
      } catch (error) {
        if (ticket !== ticketRef.current) {
          return;
        }
        inFlightRef.current = null;
        if (isAbortError(error)) {
          return;
        }
        // Unlike the drop path there is no silent branch here. The button that ran this is
        // rendered only for an asset whose record carries an envelope, so a failure the user
        // cannot see is a control that visibly did nothing.
        setImportState({ busy: false, done: false, error: "" });
        setFixes(EMPTY_FIXES);
        setOffer({
          file: null,
          assetId: asset.id,
          share: null,
          report: null,
          error: error?.message || "That image's recipe could not be read.",
        });
        return;
      }
      if (ticket !== ticketRef.current) {
        return;
      }
      inFlightRef.current = null;
      setImportState({ busy: false, done: false, error: "" });
      setFixes(EMPTY_FIXES);
      resolvedRevisionRef.current = catalogRevision;
      if (response?.status !== WORKFLOW_STATUS_WORKFLOW || !response.workflow) {
        setOffer({
          file: null,
          assetId: asset.id,
          share: null,
          report: null,
          error:
            response?.detail ||
            "SceneWorks no longer finds a recipe recorded on this image.",
        });
        return;
      }
      setOffer({
        file: null,
        assetId: asset.id,
        share: response.workflow,
        report: response.resolution ?? null,
        error: "",
      });
    },
    [enabled, projectId, token, supersede, catalogRevision],
  );

  // Re-resolve the OPEN offer when this install's catalogs change.
  //
  // The AC is "once installed, re-runs resolution without a reload", and the temptation is a poll.
  // There is no poll here: a download is a job, `useJobEvents` already refetches the model and LoRA
  // catalogs when one completes, and `catalogRevision` is derived from those catalogs — so this
  // fires off the same SSE event that updates the Models page, one hop removed, and fires for an
  // install started ANYWHERE (the Models page in another tab of the app, a queued job finishing
  // while the panel sits open) rather than only for one started from this panel.
  //
  // Only the report is replaced. The envelope is a fact about the file and cannot have changed, the
  // user's own picks must survive, and a failed refresh leaves the last good report standing rather
  // than blanking a panel the user is reading.
  useEffect(() => {
    const current = offerRef.current;
    if (!current?.share) {
      resolvedRevisionRef.current = catalogRevision;
      return undefined;
    }
    if (catalogRevision === resolvedRevisionRef.current) {
      return undefined;
    }
    resolvedRevisionRef.current = catalogRevision;
    let cancelled = false;
    const controller = typeof AbortController === "function" ? new AbortController() : null;
    reresolveRef.current = controller;
    (async () => {
      try {
        const response = current.assetId
          ? await fetchAssetWorkflow(projectId, current.assetId, {
              token,
              signal: controller?.signal,
            })
          : await inspectWorkflowFile(current.file, {
              projectId,
              token,
              signal: controller?.signal,
            });
        if (cancelled || response?.status !== WORKFLOW_STATUS_WORKFLOW || !response.resolution) {
          return;
        }
        setOffer((existing) =>
          existing?.share ? { ...existing, report: response.resolution } : existing,
        );
      } catch {
        // A refresh that failed says nothing new about the recipe.
      }
    })();
    return () => {
      cancelled = true;
      controller?.abort();
      if (reresolveRef.current === controller) {
        reresolveRef.current = null;
      }
    };
  }, [catalogRevision, projectId, token]);

  const chooseSubstituteModel = useCallback((modelId) => {
    setFixes((current) => ({ ...current, substituteModelId: modelId ?? "" }));
  }, []);

  const pickInput = useCallback((slotKey, assetId) => {
    setFixes((current) => ({
      ...current,
      inputPicks: { ...current.inputPicks, [slotKey]: assetId ?? "" },
    }));
  }, []);

  // Start the install the report published for a catalog-known-but-absent requirement.
  //
  // Routed through App's own `createModelDownloadJob` / `createLoraDownloadJob` rather than POSTing
  // `requirement.install.path` directly: those are the Model Manager's flows, and they are what put
  // the job in the queue list, surface its progress and its failure, and — through `useJobEvents` —
  // refetch the catalog that re-resolves this very panel. A bare fetch here would install the model
  // and leave every one of those un-wired.
  const installRequirement = useCallback(
    async (row) => {
      const key = `${row?.kind}:${row?.id}`;
      if (!row?.id || fixes.installing[key] || fixes.queued[key]) {
        return;
      }
      const start = row.kind === "model" ? installModel : installLora;
      if (typeof start !== "function") {
        return;
      }
      setFixes((current) => ({
        ...current,
        installing: { ...current.installing, [key]: true },
      }));
      const job = await start({ id: row.id });
      setFixes((current) => ({
        ...current,
        installing: { ...current.installing, [key]: false },
        queued: { ...current.queued, [key]: Boolean(job) },
      }));
    },
    [fixes.installing, fixes.queued, installModel, installLora],
  );

  // "Use this workflow" — build the recipe and hand it to the SAME launch the viewer's
  // "Use this recipe" uses.
  //
  // The panel then closes for every outcome BUT one. A launch that bailed on a hydration race has
  // already navigated to the studio, and leaving the panel over it would be worse than closing.
  // A launch that REFUSED — today, a viewport locked to the Simple UI, where the Advanced
  // workspace this prefill lands in cannot be opened at all — has navigated nowhere, so closing
  // the panel would leave the button looking like it worked. That one keeps the panel and says
  // why.
  const useWorkflow = useCallback(async () => {
    const current = offer;
    if (!current?.share || typeof launchRecipe !== "function") {
      return;
    }
    const recipe = recipeFromWorkflowShare(current.share, current.report, {
      // Re-checked here as well as in the panel: the catalog can change between the pick and the
      // click, and a substitute the studio's list no longer offers would seed a blank control.
      substituteModelId: keptSubstituteModelId(fixes.substituteModelId, models),
      referenceAssetIds: pickedReferenceAssetIds(current.report, fixes.inputPicks),
    });
    const outcome = await launchRecipe({
      recipe,
      replaySeed: workflowReplaySeed(current.share),
      // `assetId` stays null even when the offer came from an imported asset. That asset is the
      // recipe's OUTPUT, and `ImageStudio` reads `launchRequest.assetId` as the edit SOURCE — so
      // passing it would silently start the edit from the finished image the sender rendered.
      sourceAssetId: pickedSourceAssetId(current.report, fixes.inputPicks) || null,
    });
    if (outcome?.ok === false && outcome.reason === "locked") {
      setOffer((existing) =>
        existing ? { ...existing, launchError: outcome.detail ?? "" } : existing,
      );
      return;
    }
    dismiss();
  }, [offer, launchRecipe, dismiss, fixes.substituteModelId, fixes.inputPicks, models]);

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

  // The panel's props, grouped so App forwards ONE object rather than eight loose props that can
  // drift out of step with each other.
  const missing = useMemo(
    () => ({
      models,
      installing: fixes.installing,
      installed: fixes.queued,
      onInstall: installRequirement,
      substituteModelId: keptSubstituteModelId(fixes.substituteModelId, models),
      onSubstituteModel: chooseSubstituteModel,
      inputPicks: fixes.inputPicks,
      onPickInput: pickInput,
    }),
    [models, fixes, installRequirement, chooseSubstituteModel, pickInput],
  );

  return {
    offer,
    importState,
    handleDroppedFile,
    offerAssetWorkflow,
    dismiss,
    useWorkflow,
    importImage,
    missing,
  };
}
