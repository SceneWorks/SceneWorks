import React from "react";

import { Modal } from "./Modal.jsx";
import { ValidationSummary } from "../validation/Validation.jsx";
import { useValidation } from "../validation/useValidation.js";
import { workflowReplayValidation } from "../workflowReplayValidation.js";
import { WorkflowMissingInputs } from "./WorkflowMissingInputs.jsx";
import { WorkflowResolutionReport } from "./WorkflowResolutionReport.jsx";
import {
  workflowModeLabel,
  workflowResolutionLabel,
  workflowSettingRows,
} from "../workflowShare.js";

// "Workflow found" — the offer opened by an unclaimed image drop whose file turned out to carry a
// SceneWorks recipe (sc-15951), and by "Use this recipe" on an already-imported asset that is
// carrying one (sc-15952).
//
// Shows what is actually in the recipe, then what this install can do with it
// (`WorkflowResolutionReport`), then what it needs from the user (`WorkflowMissingInputs`), then
// the actions. It is opened only after a workflow has been read, so there is no loading state here
// by design: an image with no workflow — every foreign PNG — never reaches this component at all.
//
// # The CTA and its reason come from one call
//
// `workflowReplayValidation` through `useValidation`, which is the same mechanism Image Studio's
// own Generate uses (epic 10644) and is here for the same reason: a "Use this recipe" that is dead
// because the model could not be resolved must SAY that, and a `disabled` expression maintained
// separately from the message is how a screen ends up dead and silent. It is not a parallel gate —
// the studio's rules still apply once the recipe lands there; this is the same vocabulary applied
// at the earlier point where the hand-over happens.
//
// # Two surfaces, one panel
//
// `offer.assetId` is set when the offer came from an asset already in the library. The only things
// that differ are the title and the import action — "Import the image too" is meaningless for an
// image that is already imported, so it is not rendered rather than rendered disabled. Everything
// else is deliberately identical: two panels describing one report is exactly how "the model is
// missing" becomes "the model is fine" on one of them.

function Field({ label, children }) {
  return (
    <div className="workflow-drop-field">
      <span className="workflow-drop-field-label">{label}</span>
      <span className="workflow-drop-field-value">{children}</span>
    </div>
  );
}

export function WorkflowDropPanel({
  offer,
  importState,
  canImport,
  onUse,
  onImport,
  onDismiss,
  // The missing-inputs surface's props, bundled by `useWorkflowDrop`. Defaulted so a caller that
  // renders the panel without them (the sc-15951 tests) still gets the report and the actions.
  missing = {},
  // The project's images, offered by the input pickers.
  assets = [],
}) {
  // Hooks run before the early returns below can skip them — a `return null` above a `useMemo` is
  // the hook-order violation React refuses at runtime.
  const draft = React.useMemo(
    () => ({
      report: offer?.share ? (offer.report ?? null) : null,
      substituteModelId: missing.substituteModelId ?? "",
      inputPicks: missing.inputPicks ?? {},
    }),
    [offer?.share, offer?.report, missing.substituteModelId, missing.inputPicks],
  );
  const validity = useValidation(workflowReplayValidation, draft, undefined);
  if (!offer) {
    return null;
  }
  const { share, report, error, launchError } = offer;
  if (error) {
    return (
      <Modal
        className="workflow-drop-modal"
        labelledBy="workflow-drop-title"
        onClose={onDismiss}
      >
        <h2 className="workflow-drop-title" id="workflow-drop-title">
          That image could not be read
        </h2>
        <p className="workflow-drop-error">{error}</p>
        <div className="workflow-drop-actions">
          <button className="btn" onClick={onDismiss} type="button">
            Close
          </button>
        </div>
      </Modal>
    );
  }

  const resolutionLabel = workflowResolutionLabel(share);
  const settings = workflowSettingRows(share, report);
  const batchCount = Number(share.count);
  const producer = share.producer ?? {};
  // A model this install cannot resolve is NOT prefilled — the studio keeps the model it is
  // already on unless the user picked one here. Saying so is the whole difference between "we could
  // not reproduce this" and a prompt quietly rendered by somebody else's model.
  const modelUnresolved =
    (report?.model ? report.model.state !== "resolved" : true) && !missing.substituteModelId;
  // An offer opened from the library rather than from a drop.
  const fromLibrary = Boolean(offer.assetId);

  return (
    <Modal className="workflow-drop-modal" labelledBy="workflow-drop-title" onClose={onDismiss}>
      <h2 className="workflow-drop-title" id="workflow-drop-title">
        {fromLibrary ? "The recipe in this image" : "Workflow found"}
      </h2>
      <p className="workflow-drop-lede">
        This image carries the recipe that made it
        {producer.name ? ` (written by ${producer.name}${producer.version ? ` ${producer.version}` : ""})` : ""}.
      </p>

      <div className="workflow-drop-summary">
        <Field label="Mode">{workflowModeLabel(share)}</Field>
        <Field label="Model">{report?.model?.name ?? share.model ?? "Not named"}</Field>
        {resolutionLabel ? <Field label="Resolution">{resolutionLabel}</Field> : null}
        {share.seed === null || share.seed === undefined ? null : (
          <Field label="Seed">{String(share.seed)}</Field>
        )}
      </div>

      <div className="workflow-drop-prompt">
        <span className="workflow-drop-field-label">Prompt</span>
        <p>{share.prompt || <em>No prompt recorded.</em>}</p>
      </div>
      {share.negativePrompt ? (
        <div className="workflow-drop-prompt">
          <span className="workflow-drop-field-label">Negative prompt</span>
          <p>{share.negativePrompt}</p>
        </div>
      ) : null}

      {settings.length ? (
        <div className="workflow-drop-settings">
          <span className="workflow-drop-field-label">Settings</span>
          <ul>
            {/* Each row says whether "Use this workflow" will actually apply it. A grid where
                "PiD decoder — On" and "Steps — 28" look identical, when only one of them reaches a
                control, tells the user a recipe replayed faithfully when it did not — so the mark
                is a state on the row, not a footnote under the list. */}
            {settings.map((row) => (
              <li className={row.restored ? undefined : "not-restored"} key={row.key}>
                <span>{row.label}</span>
                <strong>{row.value}</strong>
                {row.restored ? null : (
                  <em className="workflow-drop-setting-mark" title={row.detail}>
                    not restored
                  </em>
                )}
              </li>
            ))}
          </ul>
        </div>
      ) : null}

      <WorkflowResolutionReport report={report} share={share} />

      {/* The repairs, when there are any. Renders NOTHING for a fully-resolvable recipe with no
          input images — no empty container, no zero-state line — so a working case looks exactly
          as it did before this section existed. */}
      <WorkflowMissingInputs
        assets={assets}
        installed={missing.installed}
        installing={missing.installing}
        inputPicks={missing.inputPicks}
        models={missing.models}
        onInstall={missing.onInstall}
        onPickInput={missing.onPickInput}
        onSubstituteModel={missing.onSubstituteModel}
        report={report}
        substituteModelId={missing.substituteModelId}
      />

      {/* The count note. The envelope's seed is THIS image's; `count` is what the run asked for,
          so a naive replay would make the shared image the first of an N-image batch. Prefill
          asks for one and says so rather than quietly reproducing the batch. */}
      {Number.isFinite(batchCount) && batchCount > 1 ? (
        <p className="workflow-drop-note">
          This image was one of a batch of {batchCount}. Only its own seed travelled, so the studio
          is set to make a single image.
        </p>
      ) : null}
      {modelUnresolved ? (
        <p className="workflow-drop-note warn">
          The model is not prefilled, so Image Studio keeps the model it is already on. Nothing
          here substitutes one for the model this image names.
        </p>
      ) : null}

      {importState.done ? (
        <p className="workflow-drop-note">The image was imported into this project.</p>
      ) : null}
      {importState.error ? <p className="workflow-drop-error">{importState.error}</p> : null}
      {/* The chips sit against the actions so they read as the reason the CTA is dead — the
          epic-10644 rule, and the reason `disabled` below is derived from the same summary rather
          than from its own expression. */}
      <ValidationSummary issues={validity.surfaced} label="Use this recipe messages" />
      {/* "Use this workflow" refused, and the panel stays open saying why. The one case today is a
          viewport locked to the Simple UI: Image Studio's prefill lands in the Advanced workspace,
          which cannot be opened at that width, and closing the panel over a launch that went
          nowhere is the inert-control failure this whole path exists to avoid. */}
      {launchError ? <p className="workflow-drop-error">{launchError}</p> : null}

      <div className="workflow-drop-actions">
        <button className="btn" onClick={onDismiss} type="button">
          Cancel
        </button>
        {/* Not rendered for an image already in the library. A disabled "Import the image too"
            beside an imported image is a control that describes a state, and this panel has a
            report for that. */}
        {fromLibrary ? null : (
          <button
            className="btn"
            disabled={!canImport || importState.busy || importState.done}
            onClick={onImport}
            title={canImport ? undefined : "Open a project to import this image."}
            type="button"
          >
            {importState.busy ? "Importing…" : "Import the image too"}
          </button>
        )}
        <button className="btn primary" disabled={!validity.ready} onClick={onUse} type="button">
          {fromLibrary ? "Use this recipe" : "Use this workflow"}
        </button>
      </div>
    </Modal>
  );
}
