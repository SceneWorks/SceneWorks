import { describe, expect, it } from "vitest";

import { batchOperationValidation } from "./components/BatchOperationsPanel.jsx";
import { batchSaveValidation } from "./components/BatchPromptPanel.jsx";
import { documentComposeValidation } from "./screens/DocumentStudio.jsx";
import { credentialValidation, remotePasswordValidation } from "./screens/SettingsScreen.jsx";
import { downloadSelectionValidation, firstProjectValidation } from "./screens/SetupWizard.jsx";
import { summarize } from "./validation/issues.js";

// The remaining small screen gates (epic 10652). One surfaces errors — the batch
// operations Run — and the rest are requirement-only, so their contract is just `ready`
// with nothing shown.

describe("batchOperationValidation (the one that surfaces)", () => {
  const whole = { count: 3, op: "edit", missingModel: false, missingPrompt: false };

  it("passes a ready operation", () => {
    expect(summarize(batchOperationValidation(whole)).ready).toBe(true);
    expect(summarize(batchOperationValidation(whole)).surfaced).toEqual([]);
  });

  it("treats an empty selection as a silent requirement", () => {
    const issues = batchOperationValidation({ ...whole, count: 0 });
    expect(issues.find((i) => i.field === "selection").kind).toBe("requirement");
    expect(summarize(issues).surfaced).toEqual([]);
    expect(summarize(issues).ready).toBe(false);
  });

  it("surfaces a missing model as an error, worded per op", () => {
    expect(summarize(batchOperationValidation({ ...whole, op: "detail", missingModel: true })).surfaced[0].message).toContain(
      "No compatible model",
    );
    expect(summarize(batchOperationValidation({ ...whole, op: "upscale", missingModel: true })).surfaced[0].message).toContain(
      "No upscale engine",
    );
  });

  it("surfaces a missing edit prompt as an error", () => {
    const summary = summarize(batchOperationValidation({ ...whole, missingPrompt: true }));
    expect(summary.surfaced).toHaveLength(1);
    expect(summary.surfaced[0].kind).toBe("error");
    expect(summary.surfaced[0].message).toContain("Enter a prompt");
  });

  // Bidirectional: the broken model shows; the empty selection beside it stays silent.
  it("shows the model error without the empty-selection hint", () => {
    const summary = summarize(batchOperationValidation({ count: 0, op: "edit", missingModel: true, missingPrompt: false }));
    expect(summary.surfaced).toHaveLength(1);
    expect(summary.surfaced[0].message).toContain("No compatible model");
    expect(summary.ready).toBe(false);
  });
});

describe("requirement-only screen gates block silently", () => {
  const blocksSilently = (issues) => {
    const summary = summarize(issues);
    expect(summary.ready).toBe(false);
    expect(summary.surfaced).toEqual([]);
  };
  const passes = (issues) => expect(summarize(issues).ready).toBe(true);

  it("documentComposeValidation", () => {
    passes(documentComposeValidation({ activeProject: { id: "p" }, hasModel: true, prompt: "hi" }));
    blocksSilently(documentComposeValidation({ activeProject: null, hasModel: true, prompt: "hi" }));
    blocksSilently(documentComposeValidation({ activeProject: { id: "p" }, hasModel: true, prompt: "  " }));
    // The no-model case (unreachable past ModelAvailabilityGate) still blocks, silently.
    blocksSilently(documentComposeValidation({ activeProject: { id: "p" }, hasModel: false, prompt: "hi" }));
  });

  it("batchSaveValidation", () => {
    passes(batchSaveValidation({ name: "My batch", promptCount: 3 }));
    blocksSilently(batchSaveValidation({ name: "", promptCount: 3 }));
    blocksSilently(batchSaveValidation({ name: "My batch", promptCount: 0 }));
  });

  it("downloadSelectionValidation", () => {
    passes(downloadSelectionValidation({ selectionCount: 2 }));
    blocksSilently(downloadSelectionValidation({ selectionCount: 0 }));
  });

  it("firstProjectValidation", () => {
    passes(firstProjectValidation({ projectName: "My project" }));
    blocksSilently(firstProjectValidation({ projectName: "   " }));
  });

  it("credentialValidation", () => {
    passes(credentialValidation({ host: "api.x", token: "abc" }));
    blocksSilently(credentialValidation({ host: "", token: "abc" }));
    blocksSilently(credentialValidation({ host: "api.x", token: "" }));
  });

  it("remotePasswordValidation", () => {
    passes(remotePasswordValidation({ password: "hunter2" }));
    blocksSilently(remotePasswordValidation({ password: "" }));
  });
});
