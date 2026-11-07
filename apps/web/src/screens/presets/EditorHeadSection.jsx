import React from "react";
import { Icon } from "../../components/Icons.jsx";

export function EditorHeadSection({
  backToList,
  busy,
  canSave,
  creating,
  dirty,
  discardChanges,
}) {

    const statusPill = creating ? (
      <span className="preset-status-pill">Draft</span>
    ) : dirty ? (
      <span className="preset-status-pill warn">
        <span className="dot" aria-hidden="true" />
        Unsaved changes
      </span>
    ) : (
      <span className="preset-status-pill">Saved</span>
    );

    return (
      <div className="preset-editor-head">
        <div className="preset-editor-context">
          <button className="preset-back" onClick={backToList} type="button">
            <Icon.ArrowLeft size={14} /> All presets
          </button>
          <p className="eyebrow preset-editor-eyebrow">{creating ? "New preset" : "Edit preset"}</p>
        </div>
        <div className="preset-editor-actions">
          {statusPill}
          {dirty ? (
            <button className="secondary-action preset-discard" onClick={discardChanges} type="button">
              Discard
            </button>
          ) : null}
          <button className="secondary-action" onClick={backToList} type="button">
            Cancel
          </button>
          <button className="primary-action" disabled={!canSave || busy} type="submit">
            <Icon.Save size={15} />
            <span>{busy ? "Saving…" : creating ? "Create preset" : "Save preset"}</span>
          </button>
        </div>
      </div>
    );

}
