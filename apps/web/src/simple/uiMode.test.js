import { beforeEach, describe, expect, it } from "vitest";
import {
  ADVANCED_MODE,
  SIMPLE_MODE,
  readStoredSimpleDefault,
  resolveUiMode,
  uiModeLockedToSimple,
  writeStoredSimpleDefault,
} from "./uiMode.js";
import { PHONE_MAX_WIDTH } from "./breakpoint.js";

describe("uiMode — which shell renders", () => {
  beforeEach(() => {
    window.localStorage.clear();
  });

  it("keeps the full workspace for an untouched install", () => {
    // The whole point of the switch: an existing user's UI does not change under them.
    expect(readStoredSimpleDefault()).toBe(false);
    expect(resolveUiMode({ simpleDefault: false, override: null, width: 1440 })).toBe(ADVANCED_MODE);
  });

  it("honors the stored default once the user opts in", () => {
    writeStoredSimpleDefault(true);
    expect(readStoredSimpleDefault()).toBe(true);
    expect(resolveUiMode({ simpleDefault: true, override: null, width: 1440 })).toBe(SIMPLE_MODE);
  });

  it("lets a session override win over the stored default in BOTH directions", () => {
    expect(resolveUiMode({ simpleDefault: false, override: SIMPLE_MODE, width: 1440 })).toBe(SIMPLE_MODE);
    expect(resolveUiMode({ simpleDefault: true, override: ADVANCED_MODE, width: 1440 })).toBe(ADVANCED_MODE);
  });

  it("forces Simple on a phone even against an explicit Advanced override", () => {
    const phone = PHONE_MAX_WIDTH - 1;
    expect(resolveUiMode({ simpleDefault: false, override: ADVANCED_MODE, width: phone })).toBe(SIMPLE_MODE);
    expect(uiModeLockedToSimple(phone)).toBe(true);
  });

  it("does not force Simple at the phone boundary or on an unknown width", () => {
    // The boundary is exclusive — PHONE_MAX_WIDTH itself is the tablet band.
    expect(uiModeLockedToSimple(PHONE_MAX_WIDTH)).toBe(false);
    expect(resolveUiMode({ simpleDefault: false, override: null, width: PHONE_MAX_WIDTH })).toBe(ADVANCED_MODE);
    // An unmeasured viewport must never silently swap the user's shell.
    expect(uiModeLockedToSimple(null)).toBe(false);
    expect(resolveUiMode({ simpleDefault: false, override: null, width: null })).toBe(ADVANCED_MODE);
  });

  it("ignores an unrecognized override rather than treating it as Simple", () => {
    expect(resolveUiMode({ simpleDefault: false, override: "sImPlE", width: 1440 })).toBe(ADVANCED_MODE);
    expect(resolveUiMode({ simpleDefault: true, override: "nonsense", width: 1440 })).toBe(SIMPLE_MODE);
  });

  it("round-trips the stored default through localStorage", () => {
    writeStoredSimpleDefault(true);
    expect(window.localStorage.getItem("sceneworks-simple-ui-default")).toBe("true");
    writeStoredSimpleDefault(false);
    expect(readStoredSimpleDefault()).toBe(false);
  });
});
