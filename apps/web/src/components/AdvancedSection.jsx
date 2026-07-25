import React from "react";
import { Icon } from "./Icons.jsx";

// AdvancedSection — the one canonical "Advanced" disclosure (Page-Frame standard,
// direction 1b, epic sc-10433). A bordered, header-labelled block whose controls
// open contiguously beneath its own header, inside the same block: the header row
// IS the toggle, so collapsed it is header-only and reserves no space no matter
// how many optional knobs a screen has (sc-10474).
//
// Deliberately NOT a floating full-width button that spawns a separate container
// below it — the button and the panel it revealed read as two disconnected things.
//
// Controlled via `open` / `onToggle`; presentational only. `actions` (e.g. "Reset
// to model defaults") sit to the left of the caret and keep their own handlers.
//
// `leading` is optional non-interactive content pinned to the START of the header row,
// pushing the "Advanced" trigger and its caret to the right (epic 14361: the Audio Studio
// settings bar shares this row with the selected model's capability hint). It is outside the
// toggle button on purpose — the row's click target stays the trigger itself.
export function AdvancedSection({
  open,
  onToggle,
  label = "Advanced",
  hint,
  actions,
  leading,
  className,
  children,
}) {
  const rootClass = ["advanced-section", open ? "open" : "", leading ? "advanced-section--leading" : "", className]
    .filter(Boolean)
    .join(" ");
  return (
    <section className={rootClass}>
      <div className="advanced-section-head">
        {leading ? <div className="advanced-section-leading">{leading}</div> : null}
        <button
          aria-expanded={Boolean(open)}
          className="advanced-section-toggle"
          onClick={onToggle}
          type="button"
        >
          <span className="eyebrow advanced-section-label">{label}</span>
          {hint ? <span className="advanced-section-hint">{hint}</span> : null}
        </button>
        {actions ? <div className="advanced-section-actions">{actions}</div> : null}
        <button
          aria-expanded={Boolean(open)}
          aria-label={open ? "Collapse advanced" : "Expand advanced"}
          className="advanced-section-caret-btn"
          onClick={onToggle}
          type="button"
        >
          <span className="advanced-section-caret-label">{open ? "Hide" : "Show"}</span>
          <Icon.ChevDown className={open ? "advanced-section-caret open" : "advanced-section-caret"} />
        </button>
      </div>
      {open ? <div className="advanced-section-body">{children}</div> : null}
    </section>
  );
}
