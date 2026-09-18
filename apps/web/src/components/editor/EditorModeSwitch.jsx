import React from "react";
import { Icon } from "../Icons.jsx";

const MODES = [
  { id: "film", label: "Film", icon: Icon.Video },
  { id: "timeline", label: "Timeline", icon: Icon.Editor },
];

// Film plans and renders shots from a script; Timeline is the manual cut. Both act on the
// same project timelines, so the switch lives in the editor toolbar in either mode.
export function EditorModeSwitch({ mode, onChange }) {
  return (
    <div aria-label="Editor mode" className="ve-mode-switch" role="group">
      {MODES.map(({ id, label, icon: Glyph }) => (
        <button aria-pressed={mode === id} className={mode === id ? "selected" : ""} key={id} onClick={() => onChange?.(id)} type="button">
          <Glyph size={14} />
          {label}
        </button>
      ))}
    </div>
  );
}
