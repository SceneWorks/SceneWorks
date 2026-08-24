import React, { useCallback, useEffect, useMemo, useState } from "react";
import { WorkerProgressCard } from "./WorkerProgressCard.jsx";
import { appConfirm } from "../appConfirm.jsx";
import { formatBytes } from "../formatting.js";
import { hasPresentCredential } from "../credentials.js";
import { isDesktop, tauriInvoke } from "../runtime.js";
import { macModelBlock } from "../macGating.js";
import { modelCapabilityChips } from "../modelCapabilities.js";
import { normalizeLoraFamily } from "../presetUtils.js";
import {
  MANAGED_SOURCES,
  MANAGED_SOURCE_CREDENTIAL_HOST,
  OWNERSHIP_CHOICES,
  OWNERSHIP_LINKED,
  approveLibraryRoot,
  describeRefusal,
  duplicateWarningText,
  fetchLibraryRoots,
  isLocalOnlyRefusal,
  linkedCorrection,
  linkedImportBody,
  managedImportBody,
  managedSourceProblem,
  modelCheckpointId,
  removeLibraryRoot,
  rescanLibraryCheckpoint,
  rootRemovalCopy,
  scanLibraryRoot,
  updateLibraryRoot,
} from "../checkpointLibrary.js";

// The one "add a model" experience, for both ownerships and both shells (epic 20398, sc-20650).
//
// It is ONE component on purpose. The two ownerships share the validation surface, the queued-job
// progress cards, cancel, retry and the refusal rendering; the only thing that differs is what the
// user names — a checkpoint inside a library they already keep, or a transfer SceneWorks performs.
// Two panels would have meant two places for a refusal to be swallowed, and the epic's whole point
// is that a checkpoint's ownership is a choice rather than a fork in the product.
//
// Everything the panel needs from the outside is injected, which is what lets the tests drive the
// full flow — including the desktop bridge — without a transport:
//   `library`     — the six library-root routes.
//   `pickFolder`  — the native folder chooser. Defaults to the desktop bridge's `choose_folder` and
//                   is absent in a remote browser, where the path is typed instead.
//   `onImportModel` — the app's existing `createModelImportJob`. BOTH ownerships end here.
//
// The panel opens COLLAPSED. The Model Manager is a busy page and this is a disclosure on it; it is
// a button + labelled region rather than a `<details>` element so the open state is assertable by
// role and so the screen's "no collapsible cards" invariant is untouched.

const DEFAULT_LIBRARY = {
  fetchRoots: fetchLibraryRoots,
  approve: approveLibraryRoot,
  update: updateLibraryRoot,
  remove: removeLibraryRoot,
  scan: scanLibraryRoot,
  rescan: rescanLibraryCheckpoint,
};

const NEUTRAL = Object.freeze({ tone: "neutral", text: "", detail: "", reason: null });

function toneClass(tone) {
  if (tone === "success") return "inline-success";
  if (tone === "error") return "inline-warning";
  return "inline-note";
}

// Every failure that reaches the user goes through here, so no call site can quietly turn a typed
// refusal into a default. `detail` carries the store's `[checkpoint-plan:<code>] …` sentence, which
// is where the actionable evidence lives (the inspector's diagnostics, the two drift digests).
function failureMessage(error) {
  const described = describeRefusal(error);
  return {
    tone: "error",
    text: isLocalOnlyRefusal(error)
      ? "A model library can only be added or relinked from SceneWorks running on this machine."
      : described.message,
    detail: isLocalOnlyRefusal(error) ? described.message : "",
    reason: described.reason,
  };
}

export function CheckpointImportPanel({
  token,
  families = [],
  models = [],
  credentials = [],
  macCapabilities,
  pendingJobs = [],
  completedJobs = [],
  onImportModel,
  onCancelJob,
  onRetryJob,
  onOpenQueue,
  onRefreshCatalog,
  compact = false,
  defaultOpen = false,
  library = DEFAULT_LIBRARY,
  pickFolder = isDesktop ? () => tauriInvoke("choose_folder") : null,
  headingId = "checkpoint-import-heading",
}) {
  const [open, setOpen] = useState(defaultOpen);
  const [ownership, setOwnership] = useState(OWNERSHIP_LINKED);
  const [message, setMessage] = useState(NEUTRAL);
  const [busy, setBusy] = useState("");

  // Linked state.
  const [roots, setRoots] = useState([]);
  const [rootsLoaded, setRootsLoaded] = useState(false);
  const [activeRootId, setActiveRootId] = useState("");
  const [scan, setScan] = useState(null);
  const [pathDraft, setPathDraft] = useState("");
  const [labelDraft, setLabelDraft] = useState("");
  const [renaming, setRenaming] = useState("");

  // Managed state. `type` is image-only for the same reason the pre-epic form fixed it: the import
  // route writes this type verbatim into the user manifest and never reconciles it against the
  // detected family, so offering the others would let an image checkpoint be mis-typed (sc-14020).
  const [managed, setManaged] = useState({
    kind: "upload",
    file: null,
    path: "",
    url: "",
    repo: "",
    revision: "",
    expectedSha256: "",
    modelVersionId: "",
    fileId: "",
    name: "",
    family: "",
    type: "image",
  });
  const [fileInputKey, setFileInputKey] = useState(0);
  // The last submission, kept so "Try again" re-sends exactly what failed rather than whatever the
  // form happens to hold by then.
  const [lastAttempt, setLastAttempt] = useState(null);

  const modelsByCheckpoint = useMemo(() => {
    const index = new Map();
    for (const model of models) {
      const id = modelCheckpointId(model);
      if (id) index.set(id, model);
    }
    return index;
  }, [models]);

  const loadRoots = useCallback(async () => {
    setBusy("roots");
    try {
      const response = await library.fetchRoots(token);
      const next = Array.isArray(response?.roots) ? response.roots : [];
      setRoots(next);
      setRootsLoaded(true);
      setMessage((current) => (current.tone === "error" ? NEUTRAL : current));
      return next;
    } catch (error) {
      setMessage(failureMessage(error));
      setRootsLoaded(true);
      return null;
    } finally {
      setBusy("");
    }
  }, [library, token]);

  useEffect(() => {
    if (!open || ownership !== OWNERSHIP_LINKED || rootsLoaded) return;
    loadRoots();
  }, [open, ownership, rootsLoaded, loadRoots]);

  const runScan = useCallback(
    async (rootId) => {
      setActiveRootId(rootId);
      setBusy(`scan:${rootId}`);
      try {
        const result = await library.scan(token, rootId);
        setScan(result);
        setMessage(NEUTRAL);
        return result;
      } catch (error) {
        setScan(null);
        setMessage(failureMessage(error));
        return null;
      } finally {
        setBusy("");
      }
    },
    [library, token],
  );

  // A library the user has to press a button to look inside is a library whose Needs Relink /
  // Needs Rescan state they never see. The first root scans as soon as the roots load, so the
  // states that require an action are on screen at the moment the panel opens.
  useEffect(() => {
    if (!open || ownership !== OWNERSHIP_LINKED || !rootsLoaded) return;
    if (activeRootId || roots.length === 0) return;
    runScan(roots[0].rootId);
  }, [open, ownership, rootsLoaded, activeRootId, roots, runScan]);

  async function choosePath(setter) {
    if (!pickFolder) return;
    const picked = await pickFolder().catch(() => null);
    if (picked) setter(String(picked));
  }

  async function addLibrary() {
    const path = pathDraft.trim();
    if (!path) {
      setMessage({ ...NEUTRAL, tone: "error", text: "Choose the folder your checkpoints live in." });
      return;
    }
    setBusy("approve");
    try {
      const root = await library.approve(token, { path, label: labelDraft.trim() || undefined });
      setPathDraft("");
      setLabelDraft("");
      const next = await loadRoots();
      setMessage({ ...NEUTRAL, tone: "success", text: `Added ${root?.displayLabel ?? path}.` });
      if (root?.rootId) await runScan(root.rootId);
      return next;
    } catch (error) {
      setMessage(failureMessage(error));
      return null;
    } finally {
      setBusy("");
    }
  }

  async function relinkRoot(rootId) {
    let path = pathDraft.trim();
    if (pickFolder) {
      const picked = await pickFolder().catch(() => null);
      if (picked) path = String(picked);
    }
    if (!path) {
      setMessage({ ...NEUTRAL, tone: "error", text: "Choose where this library lives now." });
      return;
    }
    setBusy(`relink:${rootId}`);
    try {
      await library.update(token, rootId, { path });
      setPathDraft("");
      await loadRoots();
      await runScan(rootId);
      setMessage({ ...NEUTRAL, tone: "success", text: "Library relinked. Its checkpoints are selectable again." });
      onRefreshCatalog?.();
    } catch (error) {
      setMessage(failureMessage(error));
    } finally {
      setBusy("");
    }
  }

  async function renameRoot(rootId) {
    const label = labelDraft.trim();
    if (!label) {
      setMessage({ ...NEUTRAL, tone: "error", text: "Enter a name for this library." });
      return;
    }
    setBusy(`rename:${rootId}`);
    try {
      await library.update(token, rootId, { label });
      setRenaming("");
      setLabelDraft("");
      await loadRoots();
      setMessage({ ...NEUTRAL, tone: "success", text: "Library renamed." });
    } catch (error) {
      setMessage(failureMessage(error));
    } finally {
      setBusy("");
    }
  }

  async function forgetRoot(root) {
    const copy = rootRemovalCopy(root, activeRootId === root.rootId ? scan : null);
    if (!(await appConfirm({ ...copy, cancelLabel: "Cancel", tone: "danger" }))) return;
    setBusy(`forget:${root.rootId}`);
    try {
      const result = await library.remove(token, root.rootId);
      if (activeRootId === root.rootId) {
        setActiveRootId("");
        setScan(null);
      }
      await loadRoots();
      const dropped = result?.removedCheckpoints?.length ?? 0;
      setMessage({
        ...NEUTRAL,
        tone: "success",
        text: `SceneWorks forgot this library${dropped ? ` and ${dropped} checkpoint record${dropped === 1 ? "" : "s"}` : ""}. Your files were not touched.`,
      });
      onRefreshCatalog?.();
    } catch (error) {
      setMessage(failureMessage(error));
    } finally {
      setBusy("");
    }
  }

  async function rescanCheckpoint(rootId, relativePath) {
    setBusy(`rescan:${relativePath}`);
    try {
      const status = await library.rescan(token, rootId, relativePath);
      await runScan(rootId);
      setMessage(
        status?.state === "ready"
          ? { ...NEUTRAL, tone: "success", text: `${relativePath} is selectable again.` }
          : { ...NEUTRAL, tone: "error", text: status?.detail || `${relativePath} still cannot be used.`, reason: status?.state ?? null },
      );
      onRefreshCatalog?.();
    } catch (error) {
      setMessage(failureMessage(error));
    } finally {
      setBusy("");
    }
  }

  // BOTH ownerships submit here. One queue, one progress surface, one retry.
  async function submit(attempt) {
    if (!onImportModel) return;
    setLastAttempt(attempt);
    setBusy("import");
    setMessage({ ...NEUTRAL, tone: "neutral", text: attempt.file ? "Uploading the checkpoint before queueing." : "Validating and queueing." });
    try {
      const job = await onImportModel(attempt.file ? { ...attempt.body, file: attempt.file } : attempt.body);
      const modelId = job?.payload?.modelId;
      // The detector's verdict, when the user left family on Auto-detect (sc-14020). Reported
      // because "which family did this turn out to be" is the one thing an auto-detected import
      // leaves the user unable to predict.
      const resolvedFamily = job?.payload?.manifestEntry?.family;
      const detected = !attempt.body.family && resolvedFamily ? ` Detected family: ${normalizeLoraFamily(resolvedFamily)}.` : "";
      setMessage({
        ...NEUTRAL,
        tone: "success",
        text: `${modelId ? `Import queued for ${modelId}.` : "Import queued."}${detected}`,
      });
      if (attempt.file) setFileInputKey((current) => current + 1);
      setManaged((current) => ({ ...current, file: null, path: "", url: "", repo: "", name: "" }));
    } catch (error) {
      setMessage(failureMessage(error));
    } finally {
      setBusy("");
    }
  }

  function submitLinked(candidate) {
    return submit({
      body: linkedImportBody({
        rootId: activeRootId,
        relativePath: candidate.candidate.relativePath,
        name: candidate.candidate.relativePath.split("/").pop(),
        family: managed.family || undefined,
      }),
    });
  }

  function submitManaged(event) {
    event?.preventDefault?.();
    const problem = managedSourceProblem(managed);
    if (problem) {
      setMessage({ ...NEUTRAL, tone: "error", text: problem });
      return;
    }
    return submit({ body: managedImportBody(managed), file: managed.kind === "upload" ? managed.file : null });
  }

  const managedSource = MANAGED_SOURCES.find((source) => source.kind === managed.kind) ?? MANAGED_SOURCES[0];
  const credentialHost = MANAGED_SOURCE_CREDENTIAL_HOST[managed.kind] ?? null;
  const credentialMissing = credentialHost ? !hasPresentCredential(credentials, credentialHost) : false;

  return (
    <section aria-labelledby={headingId} className={compact ? "checkpoint-import compact" : "checkpoint-import"}>
      <h3 className="checkpoint-import-heading" id={headingId}>
        Add a model
      </h3>
      <button
        aria-controls="checkpoint-import-body"
        aria-expanded={open}
        className="checkpoint-import-toggle"
        onClick={() => setOpen((current) => !current)}
        type="button"
      >
        {open ? "Hide options" : "Add a model"}
      </button>

      {open ? (
        <div className="checkpoint-import-body" id="checkpoint-import-body">
          <div aria-label="Model ownership" className="checkpoint-ownership" role="radiogroup">
            {OWNERSHIP_CHOICES.map((choice) => (
              <button
                aria-checked={ownership === choice.id}
                className={ownership === choice.id ? "checkpoint-ownership-choice active" : "checkpoint-ownership-choice"}
                key={choice.id}
                onClick={() => setOwnership(choice.id)}
                role="radio"
                type="button"
              >
                <span className="checkpoint-ownership-label">{choice.label}</span>
                <span className="checkpoint-ownership-summary">{choice.summary}</span>
              </button>
            ))}
          </div>

          {ownership === OWNERSHIP_LINKED
            ? renderLinked()
            : renderManaged()}

          <p aria-live="polite" className={toneClass(message.tone)} role="status">
            {message.text}
          </p>
          {message.detail ? <p className="checkpoint-import-detail">{message.detail}</p> : null}
          {message.tone === "error" && lastAttempt ? (
            <button disabled={busy === "import"} onClick={() => submit(lastAttempt)} type="button">
              Try again
            </button>
          ) : null}
        </div>
      ) : null}

      {/* Progress and duplicate warnings sit OUTSIDE the disclosure. An import already running is
          not an option the user is choosing between — collapsing it away would hide a job that is
          consuming the machine, and hide the cancel button that stops it. */}
      {pendingJobs.length ? (
        <div className="checkpoint-import-progress">
          <strong>Imports in progress</strong>
          <div className="local-job-stack">
            {pendingJobs.map((job) => (
              <WorkerProgressCard
                job={job}
                key={job.id}
                onCancel={onCancelJob}
                onOpenQueue={onOpenQueue}
                onRetry={onRetryJob}
              />
            ))}
          </div>
        </div>
      ) : null}

      {completedJobs
        .map((job) => ({ job, warning: duplicateWarningText(job) }))
        .filter((entry) => entry.warning)
        .map((entry) => (
          <p className="inline-note checkpoint-duplicate-warning" key={entry.job.id} role="note">
            {entry.warning}
          </p>
        ))}
    </section>
  );

  function renderLinked() {
    return (
      <div className="checkpoint-linked">
        <div className="checkpoint-linked-add">
          <label>
            Library folder
            <input
              disabled={busy === "approve"}
              onChange={(event) => setPathDraft(event.target.value)}
              placeholder={pickFolder ? "Choose a folder" : "/Volumes/Models/checkpoints"}
              value={pathDraft}
            />
          </label>
          {pickFolder ? (
            <button onClick={() => choosePath(setPathDraft)} type="button">
              Choose folder
            </button>
          ) : null}
          <label>
            Name
            <input
              disabled={busy === "approve"}
              onChange={(event) => setLabelDraft(event.target.value)}
              placeholder="Optional"
              value={labelDraft}
            />
          </label>
          <button disabled={busy === "approve"} onClick={addLibrary} type="button">
            Add library
          </button>
        </div>

        {roots.length === 0 && rootsLoaded ? (
          <p className="checkpoint-empty">
            No linked libraries yet. Point SceneWorks at a folder of checkpoints — the files stay where they are.
          </p>
        ) : null}

        <ul aria-label="Linked libraries" className="checkpoint-root-list">
          {roots.map((root) => (
            <li className="checkpoint-root" key={root.rootId}>
              <div className="checkpoint-root-head">
                <span className="checkpoint-root-label">{root.displayLabel ?? root.label ?? root.path}</span>
                <span className="checkpoint-root-path">{root.path}</span>
              </div>
              <div className="checkpoint-root-actions">
                <button disabled={busy === `scan:${root.rootId}`} onClick={() => runScan(root.rootId)} type="button">
                  {busy === `scan:${root.rootId}` ? "Scanning…" : "Rescan library"}
                </button>
                <button disabled={busy === `relink:${root.rootId}`} onClick={() => relinkRoot(root.rootId)} type="button">
                  Relink library
                </button>
                <button onClick={() => { setRenaming(root.rootId); setLabelDraft(root.label ?? ""); }} type="button">
                  Rename
                </button>
                <button disabled={busy === `forget:${root.rootId}`} onClick={() => forgetRoot(root)} type="button">
                  Remove library
                </button>
              </div>
              {renaming === root.rootId ? (
                <div className="checkpoint-root-rename">
                  <label>
                    New name
                    <input onChange={(event) => setLabelDraft(event.target.value)} value={labelDraft} />
                  </label>
                  <button disabled={busy === `rename:${root.rootId}`} onClick={() => renameRoot(root.rootId)} type="button">
                    Save name
                  </button>
                  <button onClick={() => setRenaming("")} type="button">
                    Cancel rename
                  </button>
                </div>
              ) : null}
            </li>
          ))}
        </ul>

        {scan ? renderScan() : null}
      </div>
    );
  }

  function renderScan() {
    // The root directory is not there. Every checkpoint under it is Needs Relink — a lifecycle
    // state with a button, NOT a missing model. Saying "not installed" here would invite the user
    // to re-download bytes that are sitting on a drive they only have to plug back in.
    if (scan.available === false) {
      return (
        <div className="checkpoint-scan checkpoint-scan-unavailable" role="group" aria-label="Library needs relink">
          <p className="checkpoint-state-headline">Needs relink</p>
          <p>
            This library’s folder is not available right now — it was moved, renamed, or its drive is not attached.
            Nothing was lost, and its models are all still installed.
          </p>
          <button disabled={busy === `relink:${scan.root.rootId}`} onClick={() => relinkRoot(scan.root.rootId)} type="button">
            Relink library
          </button>
        </div>
      );
    }
    const candidates = scan.candidates ?? [];
    return (
      <div className="checkpoint-scan">
        <ul aria-label="Checkpoints in this library" className="checkpoint-candidate-list">
          {candidates.map((entry) => renderCandidate(entry))}
        </ul>
        {candidates.length === 0 ? <p className="checkpoint-empty">No checkpoints found in this library.</p> : null}
        {(scan.unmatched ?? []).length ? (
          <div className="checkpoint-unmatched" role="group" aria-label="Checkpoints no longer found">
            <p className="checkpoint-state-headline">Needs rescan</p>
            <p>
              SceneWorks has records for these checkpoints but did not find them in the library this time. They were
              renamed, moved, or replaced, and their SceneWorks records are intact.
            </p>
            <ul>
              {scan.unmatched.map((status) => (
                <li key={status.checkpointId}>
                  <span>{status.relativePath}</span>
                  <button
                    disabled={busy === `rescan:${status.relativePath}`}
                    onClick={() => rescanCheckpoint(status.rootId, status.relativePath)}
                    type="button"
                  >
                    Rescan checkpoint
                  </button>
                  {status.detail ? <span className="checkpoint-import-detail">{status.detail}</span> : null}
                </li>
              ))}
            </ul>
          </div>
        ) : null}
        {(scan.diagnostics ?? []).length ? (
          <ul aria-label="Library scan notes" className="checkpoint-diagnostics">
            {scan.diagnostics.map((diagnostic, index) => (
              <li key={`${diagnostic.code ?? "note"}-${index}`}>{diagnostic.message ?? diagnostic.code}</li>
            ))}
          </ul>
        ) : null}
      </div>
    );
  }

  function renderCandidate(entry) {
    const candidate = entry.candidate ?? {};
    const correction = linkedCorrection(entry.status);
    const model = entry.status ? modelsByCheckpoint.get(entry.status.checkpointId) : null;
    const chips = model ? modelCapabilityChips(model) : [];
    const block = model ? macModelBlock(model, macCapabilities) : null;
    return (
      <li className="checkpoint-candidate" key={entry.checkpointId ?? candidate.relativePath}>
        <div className="checkpoint-candidate-head">
          <span className="checkpoint-candidate-path">{candidate.relativePath}</span>
          <span className="checkpoint-candidate-meta">
            {[
              candidate.container,
              candidate.sizeBytes ? formatBytes(candidate.sizeBytes) : null,
              candidate.headerFamily,
              candidate.headerRole,
              candidate.quantization,
            ]
              .filter(Boolean)
              .join(" · ")}
          </span>
        </div>
        {chips.length ? (
          <ul aria-label={`${candidate.relativePath} capabilities`} className="model-capabilities">
            {chips.map((chip) => (
              <li className="chip" key={chip}>
                {chip}
              </li>
            ))}
          </ul>
        ) : null}
        {block?.blocked ? <p className="inline-warning checkpoint-eligibility">{block.text}</p> : null}
        {correction ? (
          <div className="checkpoint-candidate-correction" role="group" aria-label={`${candidate.relativePath} ${correction.headline}`}>
            <p className="checkpoint-state-headline">{correction.headline}</p>
            <p>{correction.summary}</p>
            {correction.detail ? <p className="checkpoint-import-detail">{correction.detail}</p> : null}
            <button
              disabled={busy === `rescan:${candidate.relativePath}` || busy === `relink:${activeRootId}`}
              onClick={() =>
                correction.action === "relink"
                  ? relinkRoot(activeRootId)
                  : rescanCheckpoint(activeRootId, candidate.relativePath)
              }
              type="button"
            >
              {correction.label}
            </button>
          </div>
        ) : null}
        {entry.selectable ? (
          <button disabled={busy === "import"} onClick={() => submitLinked(entry)} type="button">
            Use this checkpoint
          </button>
        ) : correction ? null : (
          <p className="checkpoint-candidate-unselectable">
            SceneWorks has only read this file’s header. Selecting it reads the whole file first, which is what makes it
            safe to run.
            <button disabled={busy === "import"} onClick={() => submitLinked(entry)} type="button">
              Validate and use
            </button>
          </p>
        )}
      </li>
    );
  }

  function renderManaged() {
    return (
      <form aria-label="Add to SceneWorks" className="checkpoint-managed" onSubmit={submitManaged}>
        <div aria-label="Where the checkpoint comes from" className="segmented-control compact-segment" role="radiogroup">
          {MANAGED_SOURCES.map((source) => (
            <button
              aria-checked={managed.kind === source.kind}
              className={managed.kind === source.kind ? "active" : ""}
              key={source.kind}
              onClick={() => setManaged((current) => ({ ...current, kind: source.kind }))}
              role="radio"
              type="button"
            >
              {source.label}
            </button>
          ))}
        </div>
        <p className="checkpoint-source-hint">{managedSource.hint}</p>

        <div className="models-import-grid">
          {managed.kind === "upload" ? (
            <label>
              Model File
              <span className="file-picker-row">
                <span className="file-upload-button">
                  Choose
                  <input
                    accept=".safetensors,.ckpt,.pt,.bin"
                    key={fileInputKey}
                    onChange={(event) => setManaged((current) => ({ ...current, file: event.target.files?.[0] ?? null }))}
                    type="file"
                  />
                </span>
                <span className="selected-file-name">{managed.file?.name ?? "No file selected"}</span>
              </span>
            </label>
          ) : null}
          {managed.kind === "localPath" ? (
            <label>
              Source path
              <input
                onChange={(event) => setManaged((current) => ({ ...current, path: event.target.value }))}
                placeholder="/Users/you/Downloads/model.safetensors"
                value={managed.path}
              />
            </label>
          ) : null}
          {managed.kind === "localPath" && pickFolder ? (
            <button
              onClick={() => choosePath((value) => setManaged((current) => ({ ...current, path: value })))}
              type="button"
            >
              Choose source folder
            </button>
          ) : null}
          {managed.kind === "url" || managed.kind === "civitai" ? (
            <label>
              {managed.kind === "civitai" ? "Civitai URL" : "Source URL"}
              <input
                onChange={(event) => setManaged((current) => ({ ...current, url: event.target.value }))}
                placeholder="https://..."
                value={managed.url}
              />
            </label>
          ) : null}
          {managed.kind === "civitai" ? (
            <>
              <label>
                Model version id
                <input
                  onChange={(event) => setManaged((current) => ({ ...current, modelVersionId: event.target.value }))}
                  placeholder="Optional"
                  value={managed.modelVersionId}
                />
              </label>
              <label>
                File id
                <input
                  onChange={(event) => setManaged((current) => ({ ...current, fileId: event.target.value }))}
                  placeholder="Optional"
                  value={managed.fileId}
                />
              </label>
            </>
          ) : null}
          {managed.kind === "huggingFace" ? (
            <>
              <label>
                Hugging Face repo
                <input
                  onChange={(event) => setManaged((current) => ({ ...current, repo: event.target.value }))}
                  placeholder="owner/name"
                  value={managed.repo}
                />
              </label>
              <label>
                Revision
                <input
                  onChange={(event) => setManaged((current) => ({ ...current, revision: event.target.value }))}
                  placeholder="Optional"
                  value={managed.revision}
                />
              </label>
            </>
          ) : null}
          {managed.kind === "url" || managed.kind === "civitai" ? (
            <label>
              Expected SHA-256
              <input
                onChange={(event) => setManaged((current) => ({ ...current, expectedSha256: event.target.value }))}
                placeholder="Optional"
                value={managed.expectedSha256}
              />
            </label>
          ) : null}
          <label>
            Type
            {/* Image-only, exactly as the pre-epic form was: the import route writes this value
                verbatim into the user manifest and never reconciles it against the detected family
                (sc-14020). */}
            <select aria-readonly="true" disabled value="image">
              <option value="image">Image</option>
            </select>
          </label>
          <label>
            Family
            <select
              disabled={!families.length}
              onChange={(event) => setManaged((current) => ({ ...current, family: event.target.value }))}
              value={managed.family}
            >
              {families.length ? (
                <>
                  <option value="">Auto-detect</option>
                  {families.map((family) => (
                    <option key={family} value={family}>
                      {family}
                    </option>
                  ))}
                </>
              ) : (
                <option value="">No known families</option>
              )}
            </select>
          </label>
          <label>
            Name
            <input
              onChange={(event) => setManaged((current) => ({ ...current, name: event.target.value }))}
              placeholder="Optional"
              value={managed.name}
            />
          </label>
          <button disabled={busy === "import"} type="submit">
            {busy === "import" ? "Queueing…" : "Queue Import"}
          </button>
        </div>
        {credentialHost && credentialMissing ? (
          <p className="inline-note checkpoint-credential-notice">
            No {credentialHost} credential is stored. A gated or paid download will be refused until you add one in
            Settings.
          </p>
        ) : null}
      </form>
    );
  }
}
