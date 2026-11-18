import React, { useEffect } from "react";
import { Modal } from "./Modal.jsx";

// The blocked prompt for a model that is installed on a library the app cannot reach right now
// (sc-19709). Everything it renders comes from the seam's typed context — the model's name and the
// expected library location — so the user never sees a raw filesystem error, and the app never
// guesses at availability.
//
// Three exits, and only three: reconnect and retry, name a different library, or abandon the
// action. The gate owns which of them may fire; this component only renders its state, which is
// why a double click or a reconnect landing mid-retry cannot produce a second submission.
//
// One more step can sit in front of a relocation: the chosen library lacks models SceneWorks
// recorded (the drive was reformatted or replaced and only some models were downloaded again).
// Adopting it anyway is the user's call, so the gate stops in `confirmingReplacement` and this
// component lists exactly which models will need re-downloading before anything is written.
//
// Reuses `Modal`, so focus is trapped and restored, Escape closes, and the dialog is portaled —
// no bespoke overlay, no `window.confirm` (inert in the Tauri WebView).
export function ModelLibraryDialog({
  state,
  onRetry,
  onRelocate,
  onUseCurrentLibrary,
  onConfirmReplacement,
  onCancelReplacement,
  onCancel,
  canRelocate,
  autoProbeMs = 5000,
  autoProbePaused = false,
}) {
  const blocked = state?.status === "blocked";
  const confirming = state?.status === "confirmingReplacement";
  const busy = state?.status === "retrying" || state?.status === "relocating";
  const open = blocked || busy || confirming;

  // While the prompt is open, keep re-probing: plugging the drive back in should resolve the
  // prompt without demanding a click. Every tick funnels through the same guarded retry the button
  // uses, so a tick that lands during an in-flight attempt is a no-op rather than a second submit.
  //
  // Paused while the NATIVE folder picker is up: that dialog is modal and opaque to this window, so
  // a drive returning mid-selection would otherwise resume the submission behind something the user
  // cannot see past — a job appearing out of an interaction they had not finished.
  useEffect(() => {
    if (!blocked || autoProbePaused || !autoProbeMs || !onRetry) return undefined;
    const id = setInterval(() => onRetry({ auto: true }), autoProbeMs);
    return () => clearInterval(id);
  }, [blocked, autoProbePaused, autoProbeMs, onRetry]);

  if (!open) return null;

  if (confirming) {
    return <ReplacementConfirmation state={state} onBack={onCancelReplacement} onConfirm={onConfirmReplacement} />;
  }

  const context = state.context ?? {};
  const modelLabel = context.modelName || context.modelId || "This model";
  const libraryPath = context.expectedLibraryPath || context.configuredLibraryPath;
  // The configured library is right there and readable, but it is not the one SceneWorks recorded
  // — a different disk mounted where the old one was, or a location that moved. "Reconnect the
  // drive" is the wrong instruction for that: there is nothing to reconnect, and without saying so
  // the state is a dead end the user has no reason to associate with the relocate control.
  const mismatched = context.libraryPresent === true;
  // The library at the configured location may be the user's own drive, reformatted or replaced.
  // Offering it directly saves a folder picker for the commonest recovery; the server still
  // judges it, and anything it lacks goes through the same confirmation as any other folder.
  const canUseCurrent =
    mismatched && canRelocate && Boolean(context.configuredLibraryPath) && Boolean(onUseCurrentLibrary);

  return (
    <Modal
      className="discard-confirm-modal model-library-modal"
      describedBy="model-library-body"
      labelledBy="model-library-title"
      onClose={onCancel}
    >
      <h2 className="discard-confirm-title" id="model-library-title">
        {mismatched
          ? `${modelLabel} is on a different model library`
          : `${modelLabel} needs its model library`}
      </h2>
      <p className="discard-confirm-body" id="model-library-body">
        {mismatched
          ? `${modelLabel} is installed, but the folder SceneWorks is configured to read is not the library it was installed on. Nothing was lost — point SceneWorks at the library holding your models, or, if this drive was reformatted or replaced, use it as your library and download again what it no longer holds.`
          : `${modelLabel} is installed, but the library holding its files is not available right now. Nothing was lost — reconnect the drive and SceneWorks will pick it up again.`}
      </p>
      {libraryPath ? (
        <p className="model-library-path">
          <span className="model-library-path-label">Expected location</span>
          <code>{libraryPath}</code>
        </p>
      ) : null}
      {/* One live region for every transient answer: "still not connected", a rejected folder, or
          the in-flight state. Screen readers hear the outcome of a retry without a focus jump. */}
      <p aria-live="polite" className="model-library-status" role="status">
        {state.status === "retrying"
          ? "Checking for the library…"
          : state.status === "relocating"
            ? "Checking that library…"
            : state.error || state.hint || ""}
      </p>
      {/* The same exits either way — only which one leads changes. A mismatched library is not
          fixed by retrying, so relocation takes the primary action; a disconnected one is, so
          retry keeps it. */}
      <div className="discard-confirm-actions">
        <button disabled={busy} onClick={onCancel} type="button">
          Cancel
        </button>
        {canRelocate ? (
          <button
            className={mismatched && !canUseCurrent ? "primary-action" : undefined}
            disabled={busy}
            onClick={onRelocate}
            type="button"
          >
            Choose a different library location
          </button>
        ) : null}
        {canUseCurrent ? (
          <button
            className="primary-action"
            disabled={busy}
            onClick={() => onUseCurrentLibrary(context.configuredLibraryPath)}
            type="button"
          >
            Use the library at this location
          </button>
        ) : null}
        <button
          className={mismatched ? undefined : "primary-action"}
          disabled={busy}
          onClick={() => onRetry?.()}
          type="button"
        >
          Connect drive and retry
        </button>
      </div>
      {canRelocate ? null : (
        <p className="model-library-hint">
          {mismatched
            ? "Changing the model library location is a desktop setting. On a shared server, set the library location on the machine running SceneWorks."
            : "Relocating the model library is a desktop setting. On a shared server, reconnect the library on the machine running SceneWorks, then retry."}
        </p>
      )}
    </Modal>
  );
}

// The confirmation in front of a library replacement. It names every model that will need
// re-downloading, says what stays installed, and states that nothing is deleted — so adopting a
// decoy is never silent: the list is the warning.
function ReplacementConfirmation({ state, onBack, onConfirm }) {
  const replacement = state.replacement ?? {};
  const demoted = replacement.demotedRepositories ?? [];
  const retained = replacement.retainedRepositories ?? [];
  const location = replacement.libraryRoot || replacement.path;
  const count = demoted.length;
  const plural = count === 1 ? "model" : "models";
  return (
    <Modal
      className="discard-confirm-modal model-library-modal"
      describedBy="model-library-body"
      labelledBy="model-library-title"
      onClose={onBack}
    >
      <h2 className="discard-confirm-title" id="model-library-title">
        Use this library and re-download {count} {plural}?
      </h2>
      <p className="discard-confirm-body" id="model-library-body">
        {`This library does not contain ${count} ${plural} SceneWorks recorded as installed — the drive may have been reformatted or replaced. If you continue, SceneWorks uses this library from now on${retained.length ? `, keeps the ${retained.length} ${retained.length === 1 ? "model" : "models"} it does contain,` : ""} and marks these as not installed so you can download them again. Nothing is deleted.`}
      </p>
      {location ? (
        <p className="model-library-path">
          <span className="model-library-path-label">Library</span>
          <code>{location}</code>
        </p>
      ) : null}
      <p className="model-library-path-label">Will need re-downloading</p>
      <ul aria-label="Models that will need re-downloading" className="model-library-demoted">
        {demoted.map((repository) => (
          <li key={repository}>
            <code>{repository}</code>
          </li>
        ))}
      </ul>
      {replacement.restartRequired ? (
        <p className="model-library-hint">
          This is a different location, so SceneWorks applies it after a restart.
        </p>
      ) : null}
      <p aria-live="polite" className="model-library-status" role="status">
        {state.hint || ""}
      </p>
      <div className="discard-confirm-actions">
        <button onClick={onBack} type="button">
          Back
        </button>
        <button className="primary-action" onClick={onConfirm} type="button">
          Use this library
        </button>
      </div>
    </Modal>
  );
}
