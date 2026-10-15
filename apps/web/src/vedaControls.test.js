import { expect, it } from "vitest";
import { defaultVedaSettings, restoreVedaSettings, vedaAdvancedSettings, vedaControlError } from "./vedaControls.js";
it("legacy recipes restore frozen off defaults and never restore execution", () => {
  expect(restoreVedaSettings({ vedaExecution: { effective: "sparse" } })).toEqual(defaultVedaSettings());
  expect(vedaAdvancedSettings(defaultVedaSettings())).toEqual({});
});
it("enabled incompatible model changes are refused, never silently dropped", () => {
  const veda = { ...defaultVedaSettings(), enabled: true };
  expect(vedaControlError(false, veda)).toContain("requires MiniMax-H3");
  expect(vedaAdvancedSettings(veda)).toEqual({ veda });
});
