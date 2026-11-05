import React, { useState } from "react";

export function InlineLayerName({ layer, onRename }) {
  const [editing, setEditing] = useState(false);
  const [draft, setDraft] = useState(layer.name);

  if (!editing) {
    return (
      <button
        className="ie-layer-name"
        onClick={(event) => {
          event.stopPropagation();
          setDraft(layer.name);
          setEditing(true);
        }}
        title="Rename layer"
        type="button"
      >
        {layer.name}
      </button>
    );
  }

  return (
    <input
      aria-label={`Rename ${layer.name}`}
      autoFocus
      className="ie-input ie-layer-name-input"
      onBlur={() => {
        const name = draft.trim();
        if (name) onRename(layer.id, name);
        setEditing(false);
      }}
      onChange={(event) => setDraft(event.target.value)}
      onClick={(event) => event.stopPropagation()}
      onKeyDown={(event) => {
        if (event.key === "Enter") event.currentTarget.blur();
        if (event.key === "Escape") setEditing(false);
      }}
      value={draft}
    />
  );
}
