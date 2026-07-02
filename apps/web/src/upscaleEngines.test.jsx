import React, { act } from "react";
import { createRoot } from "react-dom/client";
import { afterEach, describe, it, expect, vi } from "vitest";
import {
  UPSCALE_ENGINES,
  upscaleFactorsForEngine,
  upscaleEngineHasSoftness,
  availableUpscaleEngines,
  useUpscaleEngineFallback,
  DEFAULT_UPSCALE_ENGINE,
} from "./upscaleEngines.js";

// sc-8853: the upscale table + gating hook are now single-sourced. These lock in
// the canonical `key` shape and the stale-selection fallback both studios share.

// Minimal hook harness (this project has no @testing-library/react — existing
// tests drive react-dom/client directly, mirrored here).
let container = null;
let root = null;
afterEach(() => {
  if (root) act(() => root.unmount());
  root = null;
  container = null;
});
function renderHook(props) {
  container = document.createElement("div");
  function Harness() {
    useUpscaleEngineFallback(props);
    return null;
  }
  act(() => {
    root = createRoot(container);
    root.render(<Harness />);
  });
}

describe("UPSCALE_ENGINES table (canonical key shape)", () => {
  it("keys every engine on `key` (not the old ImageStudio `id`)", () => {
    for (const engine of UPSCALE_ENGINES) {
      expect(engine.key).toBeTruthy();
      expect(engine.id).toBeUndefined();
      expect(engine.label).toBeTruthy();
      expect(Array.isArray(engine.factors)).toBe(true);
    }
  });

  it("exposes the three known engines", () => {
    expect(UPSCALE_ENGINES.map((e) => e.key)).toEqual(["real-esrgan", "seedvr2", "aura-sr"]);
  });

  it("resolves factors per engine key, defaulting to [2,4]", () => {
    expect(upscaleFactorsForEngine("real-esrgan")).toEqual([2, 4]);
    expect(upscaleFactorsForEngine("aura-sr")).toEqual([4]);
    expect(upscaleFactorsForEngine("nonexistent")).toEqual([2, 4]);
  });

  it("reports softness only for seedvr2", () => {
    expect(upscaleEngineHasSoftness("seedvr2")).toBe(true);
    expect(upscaleEngineHasSoftness("real-esrgan")).toBe(false);
    expect(upscaleEngineHasSoftness("aura-sr")).toBe(false);
  });
});

describe("availableUpscaleEngines", () => {
  it("drops aura-sr and seedvr2 when their capabilities are absent", () => {
    const caps = { features: {} };
    expect(availableUpscaleEngines(caps).map((e) => e.key)).toEqual(["real-esrgan"]);
  });

  it("keeps seedvr2 when the platform supports it", () => {
    const caps = { features: { imageUpscaleSeedvr2: { supported: true } } };
    expect(availableUpscaleEngines(caps).map((e) => e.key)).toEqual(["real-esrgan", "seedvr2"]);
  });
});

describe("useUpscaleEngineFallback", () => {
  it("snaps a gated-out engine back to the default and clamps the factor", () => {
    const setUpscaleEngine = vi.fn();
    const setUpscaleFactor = vi.fn();
    renderHook({
        macCapabilities: { features: {} }, // aura-sr gated out
        upscaleEngine: "aura-sr",
        setUpscaleEngine,
        upscaleFactor: 4, // aura-sr's factor, not supported by real-esrgan? it is (2,4)
        setUpscaleFactor,
      });
    expect(setUpscaleEngine).toHaveBeenCalledWith(DEFAULT_UPSCALE_ENGINE);
    // factor 4 IS valid for real-esrgan, so it is not re-clamped.
    expect(setUpscaleFactor).not.toHaveBeenCalled();
  });

  it("clamps the factor when the current one is invalid for the default engine", () => {
    const setUpscaleEngine = vi.fn();
    const setUpscaleFactor = vi.fn();
    renderHook({
        macCapabilities: { features: {} },
        upscaleEngine: "aura-sr",
        setUpscaleEngine,
        upscaleFactor: 3, // not in real-esrgan factors [2,4]
        setUpscaleFactor,
      });
    expect(setUpscaleEngine).toHaveBeenCalledWith(DEFAULT_UPSCALE_ENGINE);
    expect(setUpscaleFactor).toHaveBeenCalledWith(2);
  });

  it("does nothing when the selected engine is allowed", () => {
    const setUpscaleEngine = vi.fn();
    const setUpscaleFactor = vi.fn();
    renderHook({
        macCapabilities: { features: { imageUpscaleSeedvr2: { supported: true } } },
        upscaleEngine: "seedvr2",
        setUpscaleEngine,
        upscaleFactor: 2,
        setUpscaleFactor,
      });
    expect(setUpscaleEngine).not.toHaveBeenCalled();
    expect(setUpscaleFactor).not.toHaveBeenCalled();
  });
});
