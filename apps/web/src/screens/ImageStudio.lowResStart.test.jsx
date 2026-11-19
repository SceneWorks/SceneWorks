import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
vi.mock("../api.js", async (original) => ({ ...await original(), apiFetch: vi.fn(async () => ({})) }));
import { currentCandidate } from "../testUtils/catalogBaseline/candidate.js";
import { change, click, mount, submissionButton } from "../testUtils/catalogBaseline/harness.jsx";
import {
  lowResStartFromRecipe,
  resolveLowResStart,
  resolveLowResStartDimensions,
  resolveLowResStartExecutedSteps,
  supportsLowResStart,
} from "./imageStudio/samplingControls.js";

beforeEach(() => {
  globalThis.IS_REACT_ACT_ENVIRONMENT = true;
  vi.spyOn(Date, "now").mockReturnValue(1794000000000);
  vi.spyOn(Math, "random").mockReturnValue(0.25);
  vi.stubGlobal("fetch", vi.fn(() => { throw new Error("Studio surface tests must not fetch or execute a model"); }));
});
afterEach(() => {
  vi.restoreAllMocks();
  vi.unstubAllGlobals();
});

const checkboxByLabel = (container, text) =>
  [...container.querySelectorAll('input[type="checkbox"]')].find((node) =>
    node.closest("label")?.textContent.includes(text),
  );

describe("SW-6942 · Low-res start controls & M6 public catalog population preservation", () => {
  it("projects features.low_res_start onto migrated base FLUX.1 routes in the public catalog and preserves non-participating routes (M6 / §10.3 / HOOK-4)", async () => {
    const catalog = currentCandidate();
    const imageModels = catalog.models.filter((model) => model.type === "image");
    const lowResRoutes = imageModels.filter((model) => model.features?.low_res_start === true).map((model) => model.id);
    expect(lowResRoutes.sort()).toEqual(["flux_dev", "flux_schnell"]);

    // Verify non-participating public image routes never expose the Low-res start control or emit lowResStart (PAT-8):
    for (const modelId of ["sdxl", "krea_2_raw", "flux2_dev", "qwen_image_2_1"]) {
      const model = imageModels.find((entry) => entry.id === modelId);
      expect(model).toBeTruthy();
      expect(supportsLowResStart(model, { mode: "text_to_image" })).toBe(false);

      const mounted = await mount(catalog, {
        platform: "linux",
        studio: "image",
        model: modelId,
        mode: "text_to_image",
      });
      try {
        expect(checkboxByLabel(mounted.container, "Low-res start")).toBeUndefined();
        const submit = submissionButton(mounted.container, "image");
        if (submit && !submit.disabled) {
          await click(submit);
          expect(mounted.calls[0]?.payload?.advanced).not.toHaveProperty("lowResStart");
        }
      } finally {
        await mounted.close();
      }
    }
  });

  it("preserves exact disabled behavior on participating FLUX.1 routes and submits advanced.lowResStart when opted in (FEAT-1, E5)", async () => {
    const catalog = currentCandidate();
    const mounted = await mount(catalog, {
      platform: "linux",
      studio: "image",
      model: "flux_dev",
      mode: "text_to_image",
    });
    try {
      const toggle = checkboxByLabel(mounted.container, "Low-res start");
      expect(toggle).toBeTruthy();
      expect(toggle.checked).toBe(false);

      // 1. Disabled by default: submitting omits advanced.lowResStart completely.
      const submit = submissionButton(mounted.container, "image");
      expect(submit?.disabled).toBe(false);
      await click(submit);
      expect(mounted.calls).toHaveLength(1);
      expect(mounted.calls[0].payload.advanced).not.toHaveProperty("lowResStart");

      // 2. Enable Low-res start: renders K number input, linked K/M slider, scale input, and quantized dimensions readout.
      await click(toggle);
      expect(toggle.checked).toBe(true);

      const kInput = mounted.container.querySelector('input[aria-label="Low-res steps (K)"]');
      const kmSlider = mounted.container.querySelector('input[aria-label="Low-res K / M slider"]');
      const scaleInput = mounted.container.querySelector('input[aria-label="Low-res scale"]');
      const dimsReadout = mounted.container.querySelector('[data-testid="low-res-start-dimensions"]');

      expect(kInput).toBeTruthy();
      expect(kmSlider).toBeTruthy();
      expect(scaleInput).toBeTruthy();
      expect(dimsReadout).toBeTruthy();

      // Default for flux_dev (M = 28 steps): K = floor(28 / 2) = 14, scale = 0.5, 512×512 → 1024×1024.
      expect(kInput.value).toBe("14");
      expect(kmSlider.value).toBe("14");
      expect(kmSlider.max).toBe("27");
      expect(scaleInput.value).toBe("0.5");
      expect(dimsReadout.textContent).toContain("512×512 → 1024×1024 (K=14 / M=28)");

      // 3. Linked K number input and K/M slider stay synchronized and scale quantizes to multiples of 16:
      await change(kmSlider, "8");
      expect(kInput.value).toBe("8");
      await change(scaleInput, "0.6");
      // floor((1024 * 0.6) / 16) * 16 = floor(614.4 / 16) * 16 = 38 * 16 = 608
      expect(dimsReadout.textContent).toContain("608×608 → 1024×1024 (K=8 / M=28)");

      await click(submit);
      expect(mounted.calls).toHaveLength(2);
      expect(mounted.calls[1].payload.advanced.lowResStart).toEqual({
        version: 1,
        k: 8,
        scale: 0.6,
      });
    } finally {
      await mounted.close();
    }
  });

  it("computes executed step count M after img2img strength truncation and refuses invalid K or scale without clamping (§6.13, FEAT-1)", () => {
    const fluxDev = { id: "flux_dev", defaults: { steps: 25 }, features: { low_res_start: true } };

    // T2I: M = 25
    expect(
      resolveLowResStartExecutedSteps({
        model: fluxDev,
        stepsOverride: "",
        img2imgActive: false,
        img2imgStrength: 0.5,
      }),
    ).toBe(25);

    // img2img with strength 0.5 on 25 steps: init_step = floor(25 * 0.5) = 12 -> M = 25 - 12 = 13
    const truncatedM = resolveLowResStartExecutedSteps({
      model: fluxDev,
      stepsOverride: "",
      img2imgActive: true,
      img2imgStrength: 0.5,
    });
    expect(truncatedM).toBe(13);

    // Valid K = 6 < M = 13 resolves cleanly:
    const valid = resolveLowResStart({
      supported: true,
      options: { enabled: true, k: "6", scale: "0.5" },
      executedSteps: truncatedM,
      width: 1024,
      height: 1024,
    });
    expect(valid.error).toBeNull();
    expect(valid.block).toEqual({ version: 1, k: 6, scale: 0.5 });

    // K = 13 >= M = 13 after img2img strength truncation is refused without clamping:
    const invalidK = resolveLowResStart({
      supported: true,
      options: { enabled: true, k: "13", scale: "0.5" },
      executedSteps: truncatedM,
      width: 1024,
      height: 1024,
    });
    expect(invalidK.block).toBeNull();
    expect(invalidK.error).toContain("1 ≤ K < 13");

    // Scale outside [0.25, 0.9] is refused without clamping:
    const invalidScale = resolveLowResStart({
      supported: true,
      options: { enabled: true, k: "6", scale: "0.95" },
      executedSteps: truncatedM,
      width: 1024,
      height: 1024,
    });
    expect(invalidScale.block).toBeNull();
    expect(invalidScale.error).toContain("0.25 and 0.9");

    // Dimensions quantize to multiples of 16 and enforce min_dim = 256:
    expect(resolveLowResStartDimensions({ width: 1024, height: 1024, scale: 0.5 })).toMatchObject({
      valid: true,
      lowWidth: 512,
      lowHeight: 512,
    });
    expect(resolveLowResStartDimensions({ width: 512, height: 512, scale: 0.25 })).toMatchObject({
      valid: false,
      lowWidth: 128,
      lowHeight: 128,
    });

    // Recipe round-trip hydration:
    expect(lowResStartFromRecipe({ version: 1, k: 8, scale: 0.5 })).toEqual({
      version: 1,
      enabled: true,
      k: 8,
      scale: 0.5,
    });
  });
});
