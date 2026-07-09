import React from "react";

// WorkPanel — the one elevated card per page (Page-Frame standard, direction 1b,
// epic sc-10433). It holds a screen's Purpose zone: its primary controls + tabs.
// Presentational only — no state. A 3px accent top-rule sits across the top.
// When `eyebrow` / `hint` / `actions` are omitted the children render directly
// (so screens that lead with mode-tabs can skip the head).
export function WorkPanel({ eyebrow, hint, actions, className, children, ...rest }) {
  const hasHead = eyebrow || hint || actions;
  return (
    <div className={className ? `work-panel ${className}` : "work-panel"} {...rest}>
      <span className="work-panel-rule" aria-hidden="true" />
      {hasHead ? (
        <div className="work-panel-head">
          <div className="work-panel-head-text">
            {eyebrow ? <p className="eyebrow work-panel-eyebrow">{eyebrow}</p> : null}
            {hint ? <p className="work-panel-hint">{hint}</p> : null}
          </div>
          {actions ? <div className="work-panel-actions">{actions}</div> : null}
        </div>
      ) : null}
      {children}
    </div>
  );
}
