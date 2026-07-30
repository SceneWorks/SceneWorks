import React from "react";

import { Modal } from "./Modal.jsx";
import { WorkflowResolutionReport } from "./WorkflowResolutionReport.jsx";
import {
  workflowModeLabel,
  workflowResolutionLabel,
  workflowSettingRows,
} from "../workflowShare.js";

// "Workflow found" — the offer an unclaimed image drop opens when the file turned out to carry a
// SceneWorks recipe (sc-15951, epic 15945).
//
// Shows what is actually in the file, then what this install can do with it
// (`WorkflowResolutionReport`, which sc-15952 reuses), then three actions. It is opened only
// after a workflow has been read, so there is no loading state here by design: an image with no
// workflow — every foreign PNG — never reaches this component at all.

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
}) {
  if (!offer) {
    return null;
  }
  const { share, report, error } = offer;
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
  const settings = workflowSettingRows(share);
  const batchCount = Number(share.count);
  const producer = share.producer ?? {};
  // A model this install cannot resolve is NOT prefilled — the studio keeps the model it is
  // already on. Saying so is the whole difference between "we could not reproduce this" and a
  // prompt quietly rendered by somebody else's model.
  const modelUnresolved = report?.model ? report.model.state !== "resolved" : true;

  return (
    <Modal className="workflow-drop-modal" labelledBy="workflow-drop-title" onClose={onDismiss}>
      <h2 className="workflow-drop-title" id="workflow-drop-title">
        Workflow found
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
            {settings.map((row) => (
              <li key={row.key}>
                <span>{row.label}</span>
                <strong>{row.value}</strong>
              </li>
            ))}
          </ul>
        </div>
      ) : null}

      <WorkflowResolutionReport report={report} />

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

      <div className="workflow-drop-actions">
        <button className="btn" onClick={onDismiss} type="button">
          Cancel
        </button>
        <button
          className="btn"
          disabled={!canImport || importState.busy || importState.done}
          onClick={onImport}
          title={canImport ? undefined : "Open a project to import this image."}
          type="button"
        >
          {importState.busy ? "Importing…" : "Import the image too"}
        </button>
        <button className="btn primary" onClick={onUse} type="button">
          Use this workflow
        </button>
      </div>
    </Modal>
  );
}
