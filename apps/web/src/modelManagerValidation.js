// The two import-form gates in the Model Manager, in the app-wide vocabulary
// (epic 10644, sc-10651). Both produce only silent requirements today — a project for a
// project-scoped import, and a file or a URL to import from — so nothing surfaces; the
// empty inputs speak for themselves. Expressed as rules so the button's readiness is a
// named thing rather than an inline boolean, and so a future error (a malformed URL, say)
// slots straight in.
//
// The `importing…` busy flags and the `onImport…` capability checks are not validation;
// they stay in the button's `disabled` expression.

import { issue } from "./validation/issues.js";

export function loraImportValidation({ scope, activeProject, isFileImport, file, sourceUrl } = {}) {
  const issues = [];
  if (scope === "project" && !activeProject) {
    issues.push(issue.requirement("project", "Choose a project for a project-scoped import"));
  }
  if (isFileImport ? !file : !sourceUrl?.trim()) {
    issues.push(issue.requirement("source", isFileImport ? "Choose a file to import" : "Paste a source URL"));
  }
  return issues;
}

export function modelImportValidation({ isFileImport, file, sourceUrl } = {}) {
  const issues = [];
  if (isFileImport ? !file : !sourceUrl?.trim()) {
    issues.push(issue.requirement("source", isFileImport ? "Choose a file to import" : "Paste a source URL"));
  }
  return issues;
}
