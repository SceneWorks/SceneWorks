import { editedFilename, buildSaveProvenance } from "./provenance.js";
import { exportEditorFile, editorDownloadPlan, editorDownloadNote } from "./export.js";
import { useCallback, useEffect, useMemo, useState } from "react";
import { inspectWorkflowFile, isAbortError } from "../../api.js";
import { SOURCE_WORKFLOW_PENDING, SOURCE_WORKFLOW_UNKNOWN, workflowStateFromInspect } from "../../editorSourceWorkflow.js";

// Extracted from ImageEditor along its existing concern boundary.
export function useEditorSave({
  working,
  setStatus,
  workingImageToFile,
  importAsset,
  edits,
  setSavedAssetId,
  setDirty,
  activeProject,
  token
}) {
  const [saving, setSaving] = useState(false);

  // ── Save / export (sc-2434) ───────────────────────────────────────────────
  // Persist the working image as a NEW Library asset, never overwriting the
  // source. Lineage links it back to the asset it was opened from (uploads have
  // no source to link); the edit chain rides along as provenance.
  const runSave = useCallback(async () => {
    if (!working || saving) return;
    setSaving(true);
    setStatus({ loading: false, error: "" });
    try {
      const file = await workingImageToFile(editedFilename(working.source));
      const saved = await importAsset(file, {
        throwOnError: true,
        sourceAssetId: working.source.assetId,
        provenance: buildSaveProvenance({
          source: working.source,
          edits,
          width: working.width,
          height: working.height,
          layers: working.layers,
        }),
      });
      setSavedAssetId(saved?.id ?? null);
      setDirty(false);
    } catch (err) {
      setStatus({ loading: false, error: `Could not save: ${err.message || err}` });
    } finally {
      setSaving(false);
    }
  }, [working, saving, setStatus, workingImageToFile, importAsset, edits, setSavedAssetId, setDirty]);

  // ── Does the opened file carry a recipe? Ask the one reader (sc-15954) ─────
  //
  // `POST /api/v1/workflows/inspect` runs `read_workflow_chunk`, the same function the import path
  // and the sc-15951 drop panel use. Nothing in this app parses PNG chunks.
  //
  // The verdict lives HERE and not on `working.source`, for two reasons. It resolves after the
  // document is installed, so writing it back through `setWorking` would either race the user's
  // first edit or bake a stale `pending` into every undo snapshot taken before it landed. And it
  // is keyed by the source blob, which undo restores BY REFERENCE — so undoing back to the opened
  // image reuses the answer instead of asking again.
  const [sourceWorkflow, setSourceWorkflow] = useState({ blob: null, state: SOURCE_WORKFLOW_UNKNOWN });

  const openedExport = working?.source?.originalExport ?? null;

  const openedBlob = openedExport?.blob ?? null;

  const openedStartState = openedExport?.workflow ?? SOURCE_WORKFLOW_UNKNOWN;

  const openedFilename = openedExport?.filename ?? null;

  const activeProjectId = activeProject?.id ?? null;

  useEffect(() => {
    // `describeOpenedImage` already decided whether asking is possible at all: a PNG past the
    // endpoint's own cap starts `unknown` and is never uploaded, because staging half a gigabyte
    // server-side to be handed a 413 is a slower way to learn nothing.
    if (!openedBlob || openedStartState !== SOURCE_WORKFLOW_PENDING) return undefined;
    let live = true;
    const controller = new AbortController();
    setSourceWorkflow({ blob: openedBlob, state: SOURCE_WORKFLOW_PENDING });
    (async () => {
      let state = SOURCE_WORKFLOW_UNKNOWN;
      try {
        const body = await inspectWorkflowFile(
          // Named so the multipart part carries a filename, exactly as the drop path's `File`
          // does. `projectId` only widens the LoRA/preset lookups the resolution report is built
          // against, and this caller reads `status` alone — but it is passed anyway so the server
          // does one kind of work for one kind of request.
          new File([openedBlob], openedFilename ?? "image.png", { type: "image/png" }),
          { projectId: activeProjectId, token, signal: controller.signal },
        );
        state = workflowStateFromInspect(body);
      } catch (err) {
        // Offline, a 413, a 500, a 422 for a file this build refuses to read — none of those is
        // "no recipe", and reporting them as one is the silent loss the AC forbids. `unknown` is
        // rendered, not swallowed.
        if (isAbortError(err)) return;
        state = SOURCE_WORKFLOW_UNKNOWN;
      }
      if (live) setSourceWorkflow({ blob: openedBlob, state });
    })();
    return () => {
      live = false;
      controller.abort();
    };
  }, [openedBlob, openedStartState, openedFilename, activeProjectId, token]);

  // The verdict for THIS document. Before the effect has run — and for any blob the store is not
  // holding an answer for — it falls back to the state the open stamped, which is `pending` for a
  // file on its way to the reader and `unknown` for one that will never go. Never `absent`.
  const sourceWorkflowState =
    sourceWorkflow.blob && sourceWorkflow.blob === openedBlob
      ? sourceWorkflow.state
      : openedStartState;

  // The one decision the pill, the banner and the export branch all read.
  const downloadPlan = useMemo(
    () => (working ? editorDownloadPlan(working, sourceWorkflowState) : null),
    [working, sourceWorkflowState],
  );

  const downloadNote = useMemo(
    () =>
      downloadPlan
        ? editorDownloadNote(downloadPlan, { downscaled: working?.source?.editorDownscaled ?? null })
        : null,
    [downloadPlan, working],
  );

  // Export straight to disk (no project involvement). An untouched PNG goes out as the file it
  // came in as — same bytes, same recipe chunk, same colour profile — and anything else is
  // flattened to a fresh PNG with no recipe in it (sc-15954). `downloadNote`, rendered beside the
  // button, says which of the two the click will produce — and the click reads the SAME plan
  // object the pill was rendered from, so the two cannot disagree about the file that lands.
  const runDownload = useCallback(async () => {
    if (!working || !downloadPlan) return;
    try {
      const file =
        downloadPlan.mode === "original"
          ? // The File wraps the SAME blob — no re-encode, so this is the one export path in the
            // editor whose bytes are the source's bytes.
            new File([working.source.originalExport.blob], downloadPlan.filename, {
              type: "image/png",
            })
          : await workingImageToFile(downloadPlan.filename);
      await exportEditorFile(file);
    } catch (err) {
      setStatus({ loading: false, error: `Could not export: ${err.message || err}` });
    }
  }, [working, downloadPlan, workingImageToFile, setStatus]);
  return { saving, runSave, downloadPlan, downloadNote, runDownload };
}
