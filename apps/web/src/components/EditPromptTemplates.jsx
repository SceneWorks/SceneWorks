import React from "react";
import { Icon } from "./Icons.jsx";
import { EDIT_PROMPT_TEMPLATES } from "../data/editPromptTemplates.js";

// One pill row, three surfaces: the advanced Image Studio's Edit tab, the Image Editor's
// Edit tool panel, and Simple's Edit tab. Each surface has its own chip vocabulary, so the
// variant only picks class names — the templates and the click behaviour are shared, which
// is what keeps the three Edit surfaces from drifting apart.
const VARIANTS = {
  studio: { row: "suggestion-row", label: "suggestion-row-label", button: "suggestion", icon: true },
  editor: { row: "ie-chip-row", label: "ie-field-label", button: "ie-chip", icon: false },
  // `su-pill-row` is a non-wrapping inline row built for one or two pills; five of them
  // need their own wrapping row on a phone-width screen.
  simple: { row: "su-edit-templates", label: "su-section-label", button: "su-pill-btn", icon: false },
};

export function EditPromptTemplates({ label = "Quick edits:", onApply, variant = "studio" }) {
  const classes = VARIANTS[variant] ?? VARIANTS.studio;
  return (
    <div aria-label="Quick edit instructions" className={classes.row}>
      {label ? <span className={classes.label}>{label}</span> : null}
      {EDIT_PROMPT_TEMPLATES.map((template) => (
        <button
          className={classes.button}
          key={template.id}
          onClick={() => onApply(template.prompt)}
          title={`${template.hint} — “${template.prompt}”`}
          type="button"
        >
          {classes.icon ? <Icon.Sparkle size={11} /> : null}
          {template.label}
        </button>
      ))}
    </div>
  );
}
