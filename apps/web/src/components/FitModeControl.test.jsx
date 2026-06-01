import React, { act } from "react";
import { createRoot } from "react-dom/client";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";

import { FitModeControl, fitModeOptions, effectiveFitMode } from "./FitModeControl.jsx";

describe("fit mode options (epic 2551)", () => {
  it("offers crop + pad always, outpaint only for inpaint-capable models — never stretch", () => {
    expect(fitModeOptions(false).map((m) => m.id)).toEqual(["crop", "pad"]);
    expect(fitModeOptions(true).map((m) => m.id)).toEqual(["crop", "pad", "outpaint"]);
    expect(fitModeOptions(true).some((m) => m.id === "stretch")).toBe(false);
  });

  it("coerces a stale/invalid value to a selectable one (outpaint sticks only when capable)", () => {
    expect(effectiveFitMode("outpaint", false)).toBe("crop"); // not inpaint-capable
    expect(effectiveFitMode("outpaint", true)).toBe("outpaint");
    expect(effectiveFitMode("pad", false)).toBe("pad");
    expect(effectiveFitMode("bogus", true)).toBe("crop");
    expect(effectiveFitMode(undefined, true)).toBe("crop");
  });
});

describe("FitModeControl render", () => {
  let container;
  let root;

  beforeEach(() => {
    global.IS_REACT_ACT_ENVIRONMENT = true;
    container = document.createElement("div");
    document.body.appendChild(container);
    root = createRoot(container);
  });

  afterEach(async () => {
    await act(async () => root.unmount());
    container.remove();
    vi.clearAllMocks();
  });

  const labels = () => [...container.querySelectorAll(".segmented-control button")].map((b) => b.textContent.trim());

  it("hides Outpaint when the model can't inpaint, shows it when it can", async () => {
    await act(async () => root.render(<FitModeControl value="crop" onChange={() => {}} inpaintCapable={false} />));
    expect(labels()).toEqual(["Crop", "Pad"]);

    await act(async () => root.render(<FitModeControl value="crop" onChange={() => {}} inpaintCapable={true} />));
    expect(labels()).toEqual(["Crop", "Pad", "Outpaint"]);
  });

  it("marks the active mode and fires onChange on click", async () => {
    const onChange = vi.fn();
    await act(async () => root.render(<FitModeControl value="pad" onChange={onChange} inpaintCapable={true} />));
    const buttons = [...container.querySelectorAll(".segmented-control button")];
    const pad = buttons.find((b) => b.textContent.trim() === "Pad");
    const outpaint = buttons.find((b) => b.textContent.trim() === "Outpaint");
    expect(pad.classList.contains("active")).toBe(true);
    expect(pad.getAttribute("aria-pressed")).toBe("true");
    await act(async () => outpaint.dispatchEvent(new MouseEvent("click", { bubbles: true })));
    expect(onChange).toHaveBeenCalledWith("outpaint");
  });
});
