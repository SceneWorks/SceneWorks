import React from "react";

// A More-options dropdown plus a "Make my default" pin. Changing the dropdown is
// session-only; the button persists the current value as the user's default.
// Shared by Make a picture and Make a video.
export function AdvField({ label, children, isDefault, onMakeDefault }) {
  return (
    <div className="sw-advf">
      <label>
        {label}
        {children}
      </label>
      <button type="button" className="sw-mkdefault" onClick={onMakeDefault} disabled={isDefault}>
        {isDefault ? "Default ✓" : "Make my default"}
      </button>
    </div>
  );
}
