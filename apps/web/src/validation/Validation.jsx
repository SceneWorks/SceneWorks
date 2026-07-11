// The rendering layer for the app-wide validation core (epic 10644, sc-10646).
//
// Three visuals converge here: the danger chips of `.training-config-warnings`, the
// bordered `.inline-warning` paragraphs the studios use for preset/LoRA problems, and
// the unstyled red text of `.structured-error` / `.refine-error` / `.batch-warning`.
//
// There is one message renderer and it is `<ValidationSummary>`. A field never speaks
// for itself — `invalidProps` only outlines it. Two components both rendering an
// issue's text would be two views that can disagree about who owns it, which is the
// defect the core exists to prevent (see validation/issues.js).

import React from "react";

// The chip row. Sits against a form's actions so the chips read as the reason the CTA
// is dead (sc-10501), and renders nothing at all when there is nothing to say — an
// empty bordered box is worse than silence.
//
// Feed it `summary.surfaced`. Requirements never reach here: the core has already
// dropped them, because a chip reading "Name the output" beside an empty Name box is
// the noise sc-10492 removed.
export function ValidationSummary({ issues, label = "Validation messages" }) {
  if (!issues?.length) {
    return null;
  }
  return (
    <div className="validation-chips" role="status" aria-label={label}>
      {issues.map((item) => (
        <span className={`validation-chip tone-${item.kind}`} key={`${item.field ?? ""}:${item.message}`}>
          {item.message}
        </span>
      ))}
    </div>
  );
}

// The compact always-on readiness signal, generalized from `.training-status-pill`.
//
// That pill was accent-toned in BOTH states, so "Needs input" was styled as success
// and only its text said otherwise. The two states carry different classes here, and
// styles.css tones them apart.
export function ReadyPill({ ready }) {
  return (
    <span className={`ready-pill ${ready ? "is-ready" : "is-pending"}`}>
      {ready ? "Ready" : "Needs input"}
    </span>
  );
}

// Props for an input whose value the user broke. Returns an attribute, never a node:
// anything added inside a CSS-grid form body becomes a grid item and reflows the row
// (the sc-10481 sweep lesson), and `.training-config-grid` is exactly that shape.
//
// Only errors reach `invalidFields`. An untouched required field is not outlined, or a
// fresh form paints itself red before the user has typed.
export function invalidProps(summary, field) {
  return summary?.invalidFields?.has(field) ? { "aria-invalid": "true" } : {};
}
