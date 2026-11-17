import React, { useEffect } from "react";
import { afterEach, describe, expect, it, vi } from "vitest";
import { act } from "react";
import { mountRoot, unmountRoot, settle } from "../testUtils/dom.js";
import { ErrorBoundary } from "./ErrorBoundary.jsx";
import { lazyScreen } from "./LazyScreen.jsx";
import { isSettingsRefusal } from "./SettingsRefusal.jsx";
let mounted;
afterEach(async () => { if (mounted) await unmountRoot(mounted.root, mounted.container); vi.restoreAllMocks(); localStorage.clear(); });

for (const code of ["UnsupportedSchemaVersion", "ConflictingAliases"]) {
  describe(code, () => {
    it("shows the typed refusal and prevents writes in a failed studio", async () => {
      vi.spyOn(console,"error").mockImplementation(() => {});
      const raw = '{"optionsSchemaVersion":99,"prompt":"keep this"}';
      localStorage.setItem("original",raw);
      const error=Object.assign(new Error("Stored options cannot be read"),{code,field:"optionsSchemaVersion"});
      function Studio() {
        useEffect(() => { localStorage.setItem("original","overwritten"); },[]);
        throw error;
      }
      const Screen=lazyScreen(async () => ({Studio}),"Studio","Image Studio");
      mounted=mountRoot();
      await act(async () => { mounted.root.render(<ErrorBoundary><Screen /></ErrorBoundary>); });
      await settle();
      expect(mounted.container.textContent).toContain("Saved settings could not be opened");
      expect(mounted.container.textContent).toContain("optionsSchemaVersion");
      expect(mounted.container.querySelector("[data-refusal-code]").dataset.refusalCode).toBe(code);
      expect(localStorage.getItem("original")).toBe(raw);
      expect(mounted.container.textContent).not.toContain("screen bundle");
    });
  });
}
it("does not classify ordinary bundle failures as stored option refusals",()=> {
  expect(isSettingsRefusal(new Error("offline"))).toBe(false);
});
