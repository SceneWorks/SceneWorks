import { readFileSync } from "node:fs";
import { dirname, resolve } from "node:path";
import { fileURLToPath } from "node:url";
import JSON5 from "json5";
import { describe, expect, it } from "vitest";
import { fallbackModels } from "./constants.js";

const HERE = dirname(fileURLToPath(import.meta.url));
const manifest = JSON5.parse(
  readFileSync(resolve(HERE, "../../../config/manifests/builtin.models.jsonc"), "utf8"),
);
const manifestModels = Array.isArray(manifest) ? manifest : manifest.models;

describe("model catalog truthful no-op removal surface (sc-18481)", () => {
  const containsRetiredStyleMode = (value) => {
    if (value === "style_variations") return true;
    if (Array.isArray(value)) return value.some(containsRetiredStyleMode);
    if (value && typeof value === "object") return Object.values(value).some(containsRetiredStyleMode);
    return false;
  };

  it("removes the retired style mode recursively from every builtin and fallback claim", () => {
    for (const model of [...manifestModels, ...fallbackModels]) {
      expect(containsRetiredStyleMode(model), `${model.id} contains a retired style claim`).toBe(false);
    }
  });

  it.each(["lens", "lens_turbo", "chroma1_hd", "chroma1_base", "chroma1_flash", "qwen_image"])(
    "%s does not restore the removed hidden style-variation no-op",
    (id) => {
      const model = fallbackModels.find((entry) => entry.id === id);
      expect(model, `${id} must be present in fallbackModels`).toBeTruthy();
      expect(model.capabilities).not.toContain("style_variations");
    },
  );

  it.each([
    "lens",
    "lens_turbo",
    "chroma1_hd",
    "chroma1_base",
    "chroma1_flash",
    "qwen_image",
    "realvisxl_lightning",
  ])("builtin %s exposes no style-variation chip or recommendation", (id) => {
    const model = manifestModels.find((entry) => entry.id === id);
    expect(model, `${id} must be present in builtin.models.jsonc`).toBeTruthy();
    expect(model.capabilities).not.toContain("style_variations");
    expect(model.ui?.recommendedFor ?? []).not.toContain("style_variations");
  });

  it("Wan2.2 I2V 14B exposes no unsupported first/last-frame chip", () => {
    const id = "wan_2_2_i2v_14b";
    const builtin = manifestModels.find((entry) => entry.id === id);
    const fallback = fallbackModels.find((entry) => entry.id === id);
    expect(builtin, `${id} must be present in builtin.models.jsonc`).toBeTruthy();
    expect(fallback, `${id} must be present in fallbackModels`).toBeTruthy();
    for (const model of [builtin, fallback]) {
      expect(model.capabilities).not.toContain("first_last_frame");
      expect(model.ui?.recommendedFor ?? []).not.toContain("first_last_frame");
    }
  });

  it("retains AuraSR only as a non-downloadable stale-selection identity", () => {
    const aura = manifestModels.find((entry) => entry.id === "aura_sr_v2");
    expect(aura).toBeTruthy();
    expect(aura.capabilities).toEqual([]);
    expect(aura.ui?.recommendedFor).toEqual([]);
    expect(aura.downloads).toEqual([]);
  });

  it.each([
    "flux2_klein_9b",
    "flux2_klein_9b_true_v2",
    "flux2_dev",
    "sd3_5_large",
    "sd3_5_large_turbo",
    "sd3_5_medium",
    "anima_base",
    "anima_aesthetic",
    "anima_turbo",
    "boogu_image",
    "boogu_image_turbo",
    "boogu_image_edit",
  ])("builtin %s describes both native backends", (id) => {
    const model = manifestModels.find((entry) => entry.id === id);
    expect(model, `${id} must be present in builtin.models.jsonc`).toBeTruthy();
    expect(model.ui?.description).toMatch(/MLX/i);
    expect(model.ui?.description).toMatch(/Candle/i);
    expect(model.ui?.description).not.toMatch(/MLX[- ]only|Apple Silicon only/i);
  });

  it.each(["ideogram-4.md", "vision-caption.md", "pulid-flux.md"])(
    "%s documents both native backends",
    (filename) => {
      const guide = readFileSync(resolve(HERE, `../public/prompt-guides/${filename}`), "utf8");
      expect(guide).toMatch(/MLX/i);
      expect(guide).toMatch(/Candle/i);
      expect(guide).not.toMatch(/Apple Silicon only|macOS only|Mac required/i);
    },
  );
});

// SD3.5 Image Studio surfacing (epic 7841 / sc-7873). The fallbackModels list seeds the picker
// before the live catalog loads; the three SD3.5 variants must be present as text-to-image image
// models with the SD3.5 prompt guide and per-variant defaults (Turbo fast, Large high-fidelity,
// Medium smaller-RAM). The real macOnly/gated/minMemoryGb gating comes from the manifest/macSupport
// at runtime — these fallback entries only carry the picker shape + per-variant defaults.
describe("fallbackModels SD3.5 surfacing (sc-7873)", () => {
  const byId = (id) => fallbackModels.find((model) => model.id === id);

  it("includes all three SD3.5 variants as text-to-image image models", () => {
    for (const id of ["sd3_5_large", "sd3_5_large_turbo", "sd3_5_medium"]) {
      const model = byId(id);
      expect(model, `${id} must be present in fallbackModels`).toBeTruthy();
      expect(model.type).toBe("image");
      expect(model.capabilities).toContain("text_to_image");
      expect(model.ui?.promptGuide?.path).toBe("/prompt-guides/sd3-5.md");
    }
  });

  it("Turbo is the fast default: few-step, CFG-free", () => {
    const turbo = byId("sd3_5_large_turbo");
    expect(turbo.defaults.steps).toBe(4);
    expect(turbo.defaults.guidanceScale).toBe(1.0);
  });

  it("Large is high-fidelity: more steps + true CFG", () => {
    const large = byId("sd3_5_large");
    expect(large.defaults.steps).toBe(28);
    expect(large.defaults.guidanceScale).toBe(3.5);
  });

  it("Medium is the smaller-RAM mid-tier: its own recipe", () => {
    const medium = byId("sd3_5_medium");
    expect(medium.defaults.steps).toBe(40);
    expect(medium.defaults.guidanceScale).toBe(4.5);
  });
});

describe("fallbackModels audio prompt-guide parity (sc-14353)", () => {
  it("gives every selectable built-in audio fallback its model-specific guide", () => {
    const expected = {
      kokoro_82m: "/prompt-guides/kokoro-82m.md",
      moss_tts_realtime: "/prompt-guides/moss-tts-realtime.md",
      moss_ttsd_v05: "/prompt-guides/moss-ttsd-v05.md",
      moss_sfx_v2: "/prompt-guides/moss-soundeffect-v2.md",
      acestep_v15_turbo: "/prompt-guides/acestep-v15-turbo.md",
      openvoice_v2: "/prompt-guides/openvoice-v2.md",
      chatterbox_tts: "/prompt-guides/chatterbox-tts.md",
      chatterbox_ve: "/prompt-guides/chatterbox-ve.md",
    };

    for (const [id, path] of Object.entries(expected)) {
      const model = fallbackModels.find((entry) => entry.id === id);
      expect(model, `${id} must be present in fallbackModels`).toBeTruthy();
      expect(model.ui?.promptGuide?.path).toBe(path);
    }
  });
});
