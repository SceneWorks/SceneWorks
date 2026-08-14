import React from "react";
import { Modal } from "./Modal.jsx";
import {
  GALLERY_READABLE_COPY,
  SAVE_WITHOUT_WORKFLOW_LABEL,
  WORKFLOW_SHARE_DOC_URL,
  inFileItems,
  notInFileItems,
  proseFieldSentence,
} from "../workflowEmbed.js";

// The user-facing copy for the embedded-workflow feature (sc-15953, epic 15945): the block of
// detail under the Settings toggle, and the one-time first-run disclosure.
//
// Both live here so there is ONE set of claims about what leaves with an image. Every sentence
// below is checked against `docs/workflow-share-envelope.md`, which is itself pinned in both
// directions against the shipped sanitizer by `crates/sceneworks-core/tests/workflow_share_doc.rs`.
//
// The three lists are not written out as prose at all. They are rendered from
// `EMBEDDED_PROSE_FIELDS`, `WORKFLOW_FIELDS_IN_FILE` and `WORKFLOW_FIELDS_NOT_IN_FILE`, each of
// which that same Rust test pins against a table in the document — so a seventh prose field, a new
// allow-listed advanced setting, or a withheld key that quietly became shared all fail this copy
// rather than silently going unmentioned in it. That gap was real: `advanced.poses` (the
// `keypoints` / `hands` / `face` coordinate arrays) travelled while the copy named none of it.
//
// The standing failure mode this epic keeps hitting is an artifact claiming a stronger guarantee
// than the code delivers. So the wording is deliberately narrow: it says what IS in the file rather
// than promising the file is safe, it names the two-segment-path hole rather than claiming every
// path is caught, and it says the switch applies to the NEXT generation rather than implying
// anything about images already on disk.

// The detailed paragraphs under the Settings toggle. The first-run modal deliberately uses a
// concise summary so the decision is readable on phones.
export function WorkflowEmbedDetails() {
  return (
    <>
      <p className="settings-note">
        <strong>Your prompt travels as you wrote it.</strong> {proseFieldSentence()} are recorded
        exactly as authored, so a file path or a client&apos;s name typed into any of them leaves
        with every copy of the image. Every other value is dropped if it looks like a file path —
        except a two-segment name like <code>Clients/Acme</code>, which is the shape of a Hugging
        Face repo id and is not treated as a location.
      </p>
      <p className="settings-note">
        <strong>Also in the file:</strong>
      </p>
      <ul className="settings-note settings-note-list">
        {inFileItems().map((item) => (
          <li key={item}>{item}</li>
        ))}
      </ul>
      <p className="settings-note">
        <strong>Readable by image galleries.</strong> {GALLERY_READABLE_COPY}
      </p>
      <p className="settings-note">
        <strong>Not in the file:</strong>
      </p>
      <ul className="settings-note settings-note-list">
        {notInFileItems().map((item) => (
          <li key={item}>{item}</li>
        ))}
      </ul>
      <p className="settings-note">
        Turning this off applies from the next generation. Images already on disk keep the block
        they were written with — to share one of those without it, use{" "}
        <strong>{SAVE_WITHOUT_WORKFLOW_LABEL}</strong>.
      </p>
      <p className="settings-note">
        <a href={WORKFLOW_SHARE_DOC_URL} rel="noreferrer noopener" target="_blank">
          What travels, exactly
        </a>
      </p>
    </>
  );
}

// The one-time decision surface. The first undecided generation waits behind it, and only an
// explicit yes/no choice records the durable decision. The detailed sharing envelope remains in
// Settings; this modal intentionally keeps only the context needed to choose.
export function WorkflowEmbedDecisionModal({ error, onChoose, onOpenSettings, saving }) {
  return (
    <Modal
      className="workflow-embed-decision-modal"
      describedBy="workflow-embed-decision-description"
      labelledBy="workflow-embed-decision-title"
      onClose={() => {}}
    >
      <div className="workflow-embed-decision-copy">
        <h2 id="workflow-embed-decision-title">Include the recipe in generated images?</h2>
        <p id="workflow-embed-decision-description">
          SceneWorks can embed the prompt, settings, model, and LoRA details used to make an image.
          That makes recipes reusable and helps galleries such as Civitai identify the resources.
        </p>
        <p>You can change this later from Settings.</p>
      </div>
      {error ? (
        <p className="workflow-embed-decision-error" role="alert">
          {error}
        </p>
      ) : null}
      <div className="workflow-embed-decision-actions">
        <button className="settings-btn" disabled={saving} onClick={onOpenSettings} type="button">
          Go to settings
        </button>
        <button
          className="settings-btn"
          disabled={saving}
          onClick={() => onChoose(false)}
          type="button"
        >
          No thanks!
        </button>
        <button
          className="settings-btn settings-btn--primary"
          disabled={saving}
          onClick={() => onChoose(true)}
          type="button"
        >
          {saving ? "Saving..." : "Ok, got it"}
        </button>
      </div>
    </Modal>
  );
}
