import React, { useRef, useState } from "react";
import { Modal } from "./Modal.jsx";

// The disclosure shown after a model library has been successfully relocated (sc-19709).
//
// The relocated path reaches the API and GPU worker as spawn environment (`HF_HOME`), so it cannot
// apply to the sidecars already running. That is a real, product-approved property of the app —
// not an error — so it is stated plainly, with a restart offered rather than demanded.
//
// "Restart now" is gated on there being no running job. The teardown behind it is the same
// SIGTERM-then-grace path as quitting, but interrupting a render still costs the user the work in
// flight, so while a job is running the affordance says so and only "Later" is available.
export function ModelLibraryRestartDialog({
  relocation,
  canRestart,
  runningJobCount = 0,
  onRestart,
  onLater,
}) {
  // A ref, not just the disabled attribute: two clicks dispatched in the same tick would both see
  // the pre-render `disabled` value, and restarting twice is exactly the kind of double-fire this
  // epic's gate exists to prevent.
  const requested = useRef(false);
  const [restarting, setRestarting] = useState(false);
  if (!relocation) return null;

  const jobRunning = runningJobCount > 0;
  const restartNow = () => {
    if (requested.current) return;
    requested.current = true;
    setRestarting(true);
    onRestart?.();
  };

  return (
    <Modal
      className="discard-confirm-modal model-library-modal"
      describedBy="model-library-restart-body"
      labelledBy="model-library-restart-title"
      onClose={onLater}
    >
      <h2 className="discard-confirm-title" id="model-library-restart-title">
        Model library updated
      </h2>
      <p className="discard-confirm-body" id="model-library-restart-body">
        SceneWorks will read models from this library after it restarts. Until then, models stored
        there stay unavailable — nothing was moved, copied, or downloaded.
      </p>
      {/* Say what happened to the work they were doing. From the user's seat they pressed Generate,
          answered a dialog, and got a settings message: without this line, the missing job reads as
          a bug rather than the deliberate consequence of relocating. */}
      <p className="discard-confirm-body">
        The generation you started was not queued. Submit it again after SceneWorks restarts.
      </p>
      <p className="model-library-path">
        <span className="model-library-path-label">New location</span>
        <code>{relocation.hfHome ?? relocation.libraryRoot}</code>
      </p>
      <p aria-live="polite" className="model-library-status" role="status">
        {restarting
          ? "Restarting SceneWorks…"
          : !canRestart
            ? "Restart the SceneWorks server to apply it."
            : jobRunning
              ? `${runningJobCount === 1 ? "A job is" : `${runningJobCount} jobs are`} still running. Restarting would interrupt ${runningJobCount === 1 ? "it" : "them"} — restart once ${runningJobCount === 1 ? "it is" : "they are"} finished.`
              : ""}
      </p>
      <div className="discard-confirm-actions">
        <button disabled={restarting} onClick={onLater} type="button">
          Later
        </button>
        {canRestart ? (
          <button
            className="primary-action"
            disabled={restarting || jobRunning}
            onClick={restartNow}
            type="button"
          >
            Restart now
          </button>
        ) : null}
      </div>
    </Modal>
  );
}
