import { describe, expect, it } from "vitest";
import {
  GUIDANCE_METHOD_LABELS,
  SAMPLER_LABELS,
  SCHEDULER_LABELS,
  guidanceDefaultFromModel,
  guidanceMethodDefaultFromModel,
  guidanceMethodOptionsFromModel,
  samplerDefaultFromModel,
  samplerOptionsFromModel,
  schedulerDefaultFromModel,
  schedulerOptionsFromModel,
  schedulerShiftDefaultFromModel,
  stepsDefaultFromModel,
  stepsMenuFromModel,
} from "./samplerOptions.js";
import { fallbackModels } from "./constants.js";

describe("samplerOptions", () => {
  it("falls back to default-only when limits are missing", () => {
    expect(samplerOptionsFromModel(undefined)).toEqual(["default"]);
    expect(samplerOptionsFromModel({})).toEqual(["default"]);
    expect(schedulerOptionsFromModel({})).toEqual(["default"]);
  });

  it("returns options in canonical order regardless of manifest ordering", () => {
    const model = { limits: { samplers: ["uni_pc", "default", "euler"] } };
    expect(samplerOptionsFromModel(model)).toEqual(["default", "euler", "uni_pc"]);
  });

  it("preserves unknown sampler keys (forward-compat)", () => {
    const model = { limits: { samplers: ["default", "euler", "future"] } };
    expect(samplerOptionsFromModel(model)).toEqual(["default", "euler", "future"]);
  });

  it("scheduler options are ordered canonically", () => {
    const model = {
      limits: {
        schedulers: ["beta", "default", "karras", "sgm_uniform"],
      },
    };
    expect(schedulerOptionsFromModel(model)).toEqual([
      "default",
      "karras",
      "sgm_uniform",
      "beta",
    ]);
  });

  it("resolves the per-backend limits override (epic 7114)", () => {
    // SDXL-shaped asymmetry: full curated on MLX (base), narrow on candle.
    const model = {
      limits: { samplers: ["default", "euler", "dpmpp_2m"] },
      candle: { limits: { samplers: ["default", "ddim"] } },
    };
    expect(samplerOptionsFromModel(model, "mlx")).toEqual(["default", "euler", "dpmpp_2m"]);
    expect(samplerOptionsFromModel(model, "candle")).toEqual(["default", "ddim"]);
    // No / unknown backend falls back to the base menu.
    expect(samplerOptionsFromModel(model)).toEqual(["default", "euler", "dpmpp_2m"]);
    // Lens-shaped asymmetry: default-only on MLX (base), curated on candle.
    const lens = {
      limits: { samplers: ["default"] },
      candle: { limits: { samplers: ["default", "euler", "heun"] } },
    };
    expect(samplerOptionsFromModel(lens, "mlx")).toEqual(["default"]);
    expect(samplerOptionsFromModel(lens, "candle")).toEqual(["default", "euler", "heun"]);
  });

  it("default helpers read defaults block with sensible fallbacks", () => {
    const model = {
      defaults: {
        sampler: "dpmpp_2m",
        scheduler: "karras",
        schedulerShift: 4.5,
        steps: 20,
        guidanceScale: 3.5,
      },
    };
    expect(samplerDefaultFromModel(model)).toBe("dpmpp_2m");
    expect(schedulerDefaultFromModel(model)).toBe("karras");
    expect(schedulerShiftDefaultFromModel(model)).toBe(4.5);
    expect(stepsDefaultFromModel(model)).toBe(20);
    expect(guidanceDefaultFromModel(model)).toBe(3.5);
  });

  it("invalid default values fall back gracefully", () => {
    expect(samplerDefaultFromModel({ defaults: { sampler: "" } })).toBe("default");
    expect(schedulerShiftDefaultFromModel({ defaults: { schedulerShift: -1 } })).toBe(3.0);
    expect(stepsDefaultFromModel({ defaults: { steps: 0 } })).toBeNull();
    expect(guidanceDefaultFromModel({ defaults: { guidanceScale: "n/a" } })).toBeNull();
  });

  // sc-19502 — `limits.steps` is the EXACT set of renderable step counts. The studio pins the Steps
  // control when there is exactly one, so this reader must agree with the backend gate
  // (`allowed_steps` in crates/sceneworks-core/src/video_request.rs) about what counts as a menu.
  it("reads limits.steps as an exact menu and treats an unusable one as no menu", () => {
    expect(stepsMenuFromModel({ limits: { steps: [8] } })).toEqual([8]);
    expect(stepsMenuFromModel({ limits: { steps: [4, 8] } })).toEqual([4, 8]);

    // No menu declared — the overwhelmingly common case, and it must stay unpinned.
    expect(stepsMenuFromModel(undefined)).toBeNull();
    expect(stepsMenuFromModel({})).toBeNull();
    expect(stepsMenuFromModel({ limits: {} })).toBeNull();

    // Unusable declarations fall back to NO menu rather than pinning the control to nothing — the
    // same tolerance the Rust reader has, because a stricter UI would block legal requests.
    expect(stepsMenuFromModel({ limits: { steps: [] } })).toBeNull();
    expect(stepsMenuFromModel({ limits: { steps: [0] } })).toBeNull();
    expect(stepsMenuFromModel({ limits: { steps: [8.5] } })).toBeNull();
    expect(stepsMenuFromModel({ limits: { steps: ["8"] } })).toBeNull();
    expect(stepsMenuFromModel({ limits: { steps: [null] } })).toBeNull();
    expect(stepsMenuFromModel({ limits: { steps: 8 } })).toBeNull();

    // A partially-unusable menu keeps its usable entries, matching the Rust reader.
    expect(stepsMenuFromModel({ limits: { steps: [8, 0, "12"] } })).toEqual([8]);
  });

  // sc-19502 — the fallback catalog is a MIRROR of builtin.models.jsonc, and the Steps control reads
  // it when the live catalog is unavailable. If the mirror drifted, the offline studio would offer a
  // free-text Steps box for a model whose backend refuses every value but 8.
  it("the fallback catalog pins LTX to its single baked step count", () => {
    for (const id of ["ltx_2_3", "ltx_2_3_eros"]) {
      const model = fallbackModels.find((entry) => entry.id === id);
      expect(model, `${id} present in the fallback catalog`).toBeTruthy();
      expect(stepsMenuFromModel(model), `${id} step menu`).toEqual([8]);
      // …and the studio's own default is that one legal value, so the common path never trips the
      // enqueue gate.
      expect(stepsDefaultFromModel(model), `${id} default steps`).toBe(8);
    }

    // The other video models must stay UNPINNED — a blanket pin would remove a working control.
    const unpinned = fallbackModels.filter(
      (model) => model.type === "video" && !String(model.id).startsWith("ltx_2_3"),
    );
    expect(unpinned.length).toBeGreaterThan(0);
    for (const model of unpinned) {
      expect(stepsMenuFromModel(model), `${model.id} must keep a free Steps control`).toBeNull();
    }
  });

  it("guidance methods fall back to cfg-only and order canonically (epic 7434)", () => {
    // No advertisement → cfg-only, so the studio hides the picker (length 1).
    expect(guidanceMethodOptionsFromModel(undefined)).toEqual(["cfg"]);
    expect(guidanceMethodOptionsFromModel({})).toEqual(["cfg"]);
    // Manifest ordering is normalized to the canonical guidance order.
    const model = { limits: { guidanceMethods: ["cfg_pp", "cfg"] } };
    expect(guidanceMethodOptionsFromModel(model)).toEqual(["cfg", "cfg_pp"]);
  });

  it("resolves guidance methods per-backend — CFG++ is MLX-only (sc-7447)", () => {
    // SDXL-shaped: MLX advertises cfg_pp via the mlx override; candle (base) is cfg-only.
    const sdxl = {
      limits: { guidanceMethods: ["cfg"] },
      mlx: { limits: { guidanceMethods: ["cfg", "cfg_pp"] } },
    };
    expect(guidanceMethodOptionsFromModel(sdxl, "mlx")).toEqual(["cfg", "cfg_pp"]);
    expect(guidanceMethodOptionsFromModel(sdxl, "candle")).toEqual(["cfg"]);
    // No / unknown backend falls back to the base (cfg-only) menu.
    expect(guidanceMethodOptionsFromModel(sdxl)).toEqual(["cfg"]);
  });

  it("guidance default is cfg unless the model pins one", () => {
    expect(guidanceMethodDefaultFromModel(undefined)).toBe("cfg");
    expect(guidanceMethodDefaultFromModel({})).toBe("cfg");
    expect(guidanceMethodDefaultFromModel({ defaults: { guidanceMethod: "" } })).toBe("cfg");
    expect(guidanceMethodDefaultFromModel({ defaults: { guidanceMethod: "cfg_pp" } })).toBe(
      "cfg_pp",
    );
  });

  it("guidance labels cover the full vocabulary (epic 7434)", () => {
    for (const key of ["cfg", "cfg_rescale", "apg", "cfg_pp"]) {
      expect(GUIDANCE_METHOD_LABELS[key]).toBeTruthy();
    }
  });

  it("labels cover the full curated menu (epic 7114)", () => {
    for (const key of [
      "default",
      "euler",
      "euler_ancestral",
      "heun",
      "dpmpp_2m",
      "dpmpp_sde",
      "uni_pc",
      "lcm",
      "ddim",
    ]) {
      expect(SAMPLER_LABELS[key]).toBeTruthy();
    }
    for (const key of [
      "default",
      "normal",
      "simple",
      "karras",
      "exponential",
      "sgm_uniform",
      "beta",
      "ddim_uniform",
    ]) {
      expect(SCHEDULER_LABELS[key]).toBeTruthy();
    }
  });
});
