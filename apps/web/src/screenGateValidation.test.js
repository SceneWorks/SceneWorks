import { describe, expect, it } from "vitest";

import { batchOperationValidation } from "./components/BatchOperationsPanel.jsx";
import { summarize } from "./validation/issues.js";

// The batch operations Run gate (epic 10652) — the one small screen gate that surfaces
// messages, so the one worth the validation core. missingModel / missingPrompt used to
// dim the button with no stated reason (the sc-10492 defect); an empty selection is a
// silent requirement the header count already shows.
//
// The other small gates on these screens (Document compose, batch save, Setup Wizard,
// Settings) are requirement-only: they surface nothing, so they stay plain boolean
// `disabled` expressions rather than joining the core (epic 10644 boundary — see the epic
// description). There is nothing to unit-test about `disabled={!name.trim()}`.

describe("batchOperationValidation", () => {
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
