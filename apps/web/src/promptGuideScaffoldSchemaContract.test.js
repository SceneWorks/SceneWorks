import { readFileSync } from "node:fs";
import { fileURLToPath } from "node:url";
import { dirname, resolve } from "node:path";
import JSON5 from "json5";
import { describe, expect, it } from "vitest";
import { PROMPT_GUIDE_EXEMPT_TYPES, promptGuideRequiredForModel } from "./promptGuideContract.js";

// Contract guard for sc-13783 (epic 13678).
//
// TWO authorities decide whether a catalog entry must ship `ui.promptGuide`:
//   1. scripts/check-scaffold.mjs — the web/scaffold CI gate (assertBuiltinPromptGuides).
//   2. packages/schemas/model-manifest.schema.json — the authoring JSON Schema.
// They used to disagree: the scaffold REQUIRED a guide on every entry while the schema treated it
// as OPTIONAL, so a schema-valid entry could still RED the scaffold lane. This test reads the rule
// out of BOTH real source files and asserts they encode the *same* exemption, so the two can never
// silently diverge again. If either authority is edited without the other, one of these assertions
// fails.
//
// The scaffold side reads apps/web/src/promptGuideContract.js (the shared predicate the scaffold
// imports). The schema side is derived by evaluating the schema's own if/then conditional against a
// candidate entry — i.e. it reads the exemption VALUE out of the committed schema, not a copy of it.

const HERE = dirname(fileURLToPath(import.meta.url));
const SCHEMA_PATH = resolve(HERE, "../../../packages/schemas/model-manifest.schema.json");
const MANIFEST_PATH = resolve(HERE, "../../../config/manifests/builtin.models.jsonc");

const schema = JSON.parse(readFileSync(SCHEMA_PATH, "utf8"));
const manifestModels = JSON5.parse(readFileSync(MANIFEST_PATH, "utf8")).models;

const modelItemSchema = schema.properties.models.items;

// Does `then` impose the ui.promptGuide requirement (ui present AND ui contains promptGuide)?
function thenRequiresPromptGuide(then) {
  return Boolean(
    then?.required?.includes("ui") && then?.properties?.ui?.required?.includes("promptGuide"),
  );
}

// Minimal, faithful evaluation of a property constraint used inside the schema's `if`. Handles the
// exact keywords the promptGuide conditional uses (`const`, `not.const`). Anything else THROWS so a
// future schema refactor forces this test to be updated instead of silently false-passing.
function propertyConstraintMatches(constraint, value) {
  if (constraint && typeof constraint === "object") {
    if ("const" in constraint) return value === constraint.const;
    if (constraint.not && "const" in constraint.not) return value !== constraint.not.const;
  }
  throw new Error(
    `promptGuide contract test cannot evaluate schema property constraint ${JSON.stringify(
      constraint,
    )}; update this test to match the schema's new conditional shape.`,
  );
}

// Does the schema's `if` subschema match a candidate entry?
function ifMatches(ifSchema, entry) {
  for (const key of ifSchema.required ?? []) {
    if (!(key in entry)) return false;
  }
  for (const [prop, constraint] of Object.entries(ifSchema.properties ?? {})) {
    if (!propertyConstraintMatches(constraint, entry[prop])) return false;
  }
  return true;
}

// The schema side of the contract: does the committed schema REQUIRE ui.promptGuide for `entry`?
// Derived by scanning the model-item `allOf` for the conditional whose `then` imposes the
// promptGuide requirement and evaluating its `if`. Returns false when no such satisfied conditional
// exists (e.g. someone deleted the schema requirement) — which would then mismatch the scaffold
// predicate for picker entries and fail the agreement assertions below.
function schemaRequiresPromptGuide(entry) {
  for (const branch of modelItemSchema.allOf ?? []) {
    if (thenRequiresPromptGuide(branch.then) && ifMatches(branch.if, entry)) {
      return true;
    }
  }
  return false;
}

const modelTypes = modelItemSchema.properties.type.enum;

describe("promptGuide scaffold ↔ schema contract (sc-13783)", () => {
  it("exposes a non-empty exemption set that includes the utility type", () => {
    // The whole point of the reconciliation: utility entries are exempt. If this regresses, the
    // scaffold would demand a meaningless prompt guide on validation dependencies again.
    expect(PROMPT_GUIDE_EXEMPT_TYPES.length).toBeGreaterThan(0);
    expect(PROMPT_GUIDE_EXEMPT_TYPES).toContain("utility");
  });

  it("schema and scaffold agree on the requirement for EVERY declared model type", () => {
    // The core anti-divergence assertion. For each type in the schema's own enum, the scaffold
    // predicate and the schema-derived requirement must return the same boolean. Deleting the schema
    // conditional (schema→false for pickers) or dropping utility from the exempt set (scaffold→true
    // for utility) breaks this.
    expect(modelTypes.length).toBeGreaterThan(0);
    for (const type of modelTypes) {
      const entry = { id: `probe_${type}`, name: `Probe ${type}`, type };
      expect(
        schemaRequiresPromptGuide(entry),
        `schema requiredness for type=${type}`,
      ).toBe(promptGuideRequiredForModel(entry));
    }
  });

  it("requires a promptGuide for a picker (image) entry in BOTH authorities", () => {
    const picker = { id: "probe_image", name: "Probe image", type: "image" };
    expect(promptGuideRequiredForModel(picker)).toBe(true);
    expect(schemaRequiresPromptGuide(picker)).toBe(true);
  });

  it("exempts a utility entry from the promptGuide requirement in BOTH authorities", () => {
    const utility = { id: "probe_utility", name: "Probe utility", type: "utility" };
    expect(promptGuideRequiredForModel(utility)).toBe(false);
    expect(schemaRequiresPromptGuide(utility)).toBe(false);
  });

  it("treats the sc-13684 utility entries (whisper-base, clap-htsat-unfused) as exempt", () => {
    // These real entries carry a promptGuide today (the sc-13684 workaround), but the reconciled
    // rule must NOT REQUIRE one — a later cleanup that drops their guide must stay green.
    for (const id of ["whisper_base", "clap_htsat_unfused"]) {
      const entry = manifestModels.find((model) => model.id === id);
      expect(entry, `manifest must contain ${id}`).toBeTruthy();
      expect(entry.type).toBe("utility");
      expect(promptGuideRequiredForModel(entry)).toBe(false);
      expect(schemaRequiresPromptGuide(entry)).toBe(false);
    }
  });

  it("still requires a promptGuide for real picker (non-utility) manifest entries", () => {
    const picker = manifestModels.find((model) => model.type !== "utility");
    expect(picker, "manifest must contain a non-utility entry").toBeTruthy();
    expect(promptGuideRequiredForModel(picker)).toBe(true);
    expect(schemaRequiresPromptGuide(picker)).toBe(true);
  });
});
