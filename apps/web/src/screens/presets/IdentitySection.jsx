import React from "react";
import { Icon } from "../../components/Icons.jsx";

export function IdentitySection({
  activeProject,
  creating,
  editable,
  form,
  nameInputRef,
  renderSectionHead,
  updateField,
}) {

    return (
      <div className="preset-form-section">
        {renderSectionHead("Identity", "Name this preset and choose where it lives.")}
        <div className="preset-identity-row">
          <div className="preset-name-field">
            {creating ? (
              <label className="field field-name">
                <span>Name</span>
                <input
                  onChange={(event) => updateField("name", event.target.value)}
                  placeholder="Name this preset…"
                  ref={nameInputRef}
                  required
                  value={form.name}
                />
              </label>
            ) : (
              <label className="field field-name preset-name-saved">
                <span>Name</span>
                <span className="preset-name-saved-row">
                  <input
                    disabled={!editable}
                    onChange={(event) => updateField("name", event.target.value)}
                    ref={nameInputRef}
                    required
                    value={form.name}
                  />
                  <button
                    className="preset-icon-button"
                    disabled={!editable}
                    onClick={() => nameInputRef.current?.focus()}
                    title="Rename preset"
                    type="button"
                  >
                    <Icon.Pencil size={14} />
                  </button>
                </span>
              </label>
            )}
          </div>
          <label className="field preset-scope-field">
            <span>Scope</span>
            <div aria-label="Scope" className="scope-segment" role="radiogroup">
              <button
                aria-checked={form.scope === "project"}
                className={form.scope === "project" ? "active" : ""}
                disabled={!activeProject || !editable}
                onClick={() => updateField("scope", "project")}
                role="radio"
                type="button"
              >
                <Icon.Folder size={14} /> This project
              </button>
              <button
                aria-checked={form.scope === "global"}
                className={form.scope === "global" ? "active" : ""}
                disabled={!editable}
                onClick={() => updateField("scope", "global")}
                role="radio"
                type="button"
              >
                <Icon.Stars size={14} /> Global
              </button>
            </div>
          </label>
        </div>
        <div className="preset-identity-meta">
          <label className="field">
            <span>ID</span>
            <input
              disabled={!creating || !editable}
              onChange={(event) => updateField("id", event.target.value)}
              placeholder="auto-generated from name"
              required
              value={form.id}
            />
          </label>
          <label className="field">
            <span>Description</span>
            <input
              disabled={!editable}
              onChange={(event) => updateField("description", event.target.value)}
              placeholder="One line — what kind of shot this makes"
              value={form.description}
            />
          </label>
        </div>
      </div>
    );

}
