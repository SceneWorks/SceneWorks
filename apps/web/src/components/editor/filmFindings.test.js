import { describe, expect, it } from "vitest";

import { planRefusalMessages, unroutedPackFindings } from "./filmFindings.jsx";

// sc-24028. A refused film-document write arrives as `PlanDiagnostic`'s `Display` output, which
// carries the internal scope and field path. The operator is shown the message half only.
describe("planRefusalMessages", () => {
  it("strips the scope and field path from one diagnostic", () => {
    expect(planRefusalMessages(
      "[plan] referencePack.references[2].description: reference \"silent\" has no `file` and no `description`",
    )).toEqual(["reference \"silent\" has no `file` and no `description`"]);
  });

  it("splits several joined diagnostics and strips each", () => {
    expect(planRefusalMessages(
      "[plan] referencePack.references[0].role: duplicate reference role \"courier\"; "
      + "[plan] referencePack.references[1].kind: unknown reference kind \"costume\"",
    )).toEqual([
      "duplicate reference role \"courier\"",
      "unknown reference kind \"costume\"",
    ]);
  });

  // The shared-file refusal quotes role names, a path and an example sentence, so it contains both
  // "; " and ": " itself. Splitting on those blindly would cut one message into fragments.
  it("keeps a message that itself contains '; ' and ': ' whole", () => {
    const message = "roles \"courier\", \"guard\" share the file \"references/a.png\", so each needs "
      + "a `locator`: a phrase completing \"The courier is …\"; article included; \"guard\" declares none";
    expect(planRefusalMessages(`[plan] referencePack.references.locator: ${message}`))
      .toEqual([message]);
  });

  it("strips a shot-scoped diagnostic the same way", () => {
    expect(planRefusalMessages("[SH010] audio: audio is required: say what this shot sounds like"))
      .toEqual(["audio is required: say what this shot sounds like"]);
  });

  // An ordinary route or transport error is already operator-facing and must survive untouched —
  // including one that happens to contain a clause shaped like a diagnostic, which must NOT be
  // taken apart on the strength of its second half.
  it("returns a non-diagnostic error unchanged", () => {
    expect(planRefusalMessages("Film draft not found")).toEqual(["Film draft not found"]);
    const contains = "Saving failed; [plan] referencePack.version: retry the save";
    expect(planRefusalMessages(contains)).toEqual([contains]);
    expect(planRefusalMessages("")).toEqual([]);
    expect(planRefusalMessages(undefined)).toEqual([]);
  });
});

describe("unroutedPackFindings", () => {
  it("returns pack findings that no field rendered and the panel owns", () => {
    const findings = [
      { field: "referencePack.references[0].role", message: "duplicate role" },
      { field: "referencePack.references[0].description", message: "already beside its row" },
      { field: "referencePack.sound[0].role", message: "another panel owns this" },
      { shotId: "SH010", field: "audio", message: "a shot finding" },
    ];
    expect(unroutedPackFindings(
      findings,
      new Set(["referencePack.references[0].description"]),
      (field) => !field.startsWith("referencePack.sound"),
    )).toEqual([{ field: "referencePack.references[0].role", message: "duplicate role" }]);
  });
});
