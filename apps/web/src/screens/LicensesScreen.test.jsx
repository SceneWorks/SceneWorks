import React, { act } from "react";
import { createRoot } from "react-dom/client";
import { afterEach, beforeEach, describe, expect, it } from "vitest";

import { LicensesScreen } from "./LicensesScreen.jsx";
import { bundledLicenses } from "../data/bundledLicenses.js";

// The corpus is imported from apps/desktop/licenses/ at build time, so these tests
// assert against the real bundled notices rather than a mock.
describe("LicensesScreen", () => {
  let container;
  let root;

  beforeEach(() => {
    container = document.createElement("div");
    document.body.appendChild(container);
    root = createRoot(container);
  });

  afterEach(async () => {
    await act(async () => root.unmount());
    container.remove();
  });

  async function render() {
    await act(async () => {
      root.render(<LicensesScreen />);
    });
  }

  it("lists every bundled component", async () => {
    await render();
    const items = container.querySelectorAll(".licenses-item");
    expect(items.length).toBe(bundledLicenses.length);
    expect(container.textContent).toContain("FFmpeg");
    expect(container.textContent).toContain("ONNX Runtime");
  });

  it("shows the first component's license text by default", async () => {
    await render();
    // ffmpeg is first: its written-offer notice mentions GPLv3 §6.
    expect(container.querySelector(".licenses-text").textContent).toContain(
      "Written offer for corresponding source",
    );
  });

  it("switches the displayed component on selection", async () => {
    await render();
    const onnx = [...container.querySelectorAll(".licenses-item")].find((b) =>
      b.textContent.includes("ONNX Runtime"),
    );
    await act(async () => onnx.click());
    expect(container.textContent).toContain("Microsoft Corporation");
    expect(container.querySelector(".licenses-text").textContent).toContain("MIT License");
  });

  it("switches between a component's license documents", async () => {
    await render();
    // ffmpeg has two docs (notice + GPL text); pick the GPL tab.
    const gplTab = [...container.querySelectorAll(".segmented-control button")].find((b) =>
      b.textContent.includes("General Public License"),
    );
    await act(async () => gplTab.click());
    expect(container.querySelector(".licenses-text").textContent).toContain(
      "GNU GENERAL PUBLIC LICENSE",
    );
  });
});
