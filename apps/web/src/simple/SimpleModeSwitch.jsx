import React from "react";
import { ADVANCED_MODE, SIMPLE_MODE } from "./uiMode.js";
import "./simple.css";

// The Simple ⇄ Advanced segmented switch.
//
// ONE component renders in BOTH shells — the Simple sidebar footer and the full
// workspace's `.sidebar-footer` — so the two can never present the control differently.
// This is the only addition the Simple UI makes to the existing workspace; everything
// else about that shell is untouched.
//
// On a phone the switch is LOCKED to Simple (the full workspace is unusable at that
// width), so it renders disabled with a one-line reason rather than silently ignoring
// the click.
export function SimpleModeSwitch({ mode, onChange, locked = false, className = "" }) {
  return (
    <div className={className}>
      <div aria-label="Interface mode" className="su-switch" role="radiogroup">
        <button
          aria-checked={mode === SIMPLE_MODE}
          className={mode === SIMPLE_MODE ? "active" : ""}
          disabled={locked}
          onClick={() => onChange(SIMPLE_MODE)}
          role="radio"
          type="button"
        >
          Simple
        </button>
        <button
          aria-checked={mode === ADVANCED_MODE}
          className={mode === ADVANCED_MODE ? "active" : ""}
          disabled={locked}
          onClick={() => onChange(ADVANCED_MODE)}
          role="radio"
          type="button"
        >
          Advanced
        </button>
      </div>
      {locked ? <p className="su-switch-note">Phones always use the Simple UI.</p> : null}
    </div>
  );
}
