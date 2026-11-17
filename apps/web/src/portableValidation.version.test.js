import { afterEach, expect, it } from "vitest";
import { setServerValidatorVersion, validatePortableRequestNow } from "./portableValidation.js";

afterEach(() => setServerValidatorVersion(null));

it("does not use local rules when the served API advertises a newer validator", () => {
  setServerValidatorVersion(2);
  expect(validatePortableRequestNow({ kind: "image", payload: { model: "m" } })).toMatchObject({
    state: "server-required", reason: "server-validator-version-mismatch",
  });
});

it("treats an unknown request-validator interface as a server decision", () => {
  expect(validatePortableRequestNow({
    kind: "image", payload: { model: "m" }, validatorSchemaVersion: 2,
  })).toMatchObject({ state: "server-required", reason: "validator-version-mismatch" });
});
