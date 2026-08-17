import { readFileSync } from "node:fs";
import { dirname, resolve } from "node:path";
import { fileURLToPath } from "node:url";

import { describe, expect, it } from "vitest";
import { fallbackModels } from "./constants.js";
import {
  EXTRA_COMPATIBLE_LORA_FAMILIES,
  acceptedLoraFamilies,
  applyPresetDefault,
  buildStudioPresetPayload,
  cleanPresetDefaults,
  clearPresetDefault,
  editModelForAsset,
  findModelEditLora,
  finiteNumberOrUndefined,
  loraHasResolvableFamily,
  loraIsInstalled,
  loraLooksLikeImageEditLora,
  loraMatchesModel,
  loraWeight,
  normalizeLoraFamily,
  presetMatchesModel,
  presetNameTaken,
  serializeLora,
  slugifyPresetId,
  workflowForMode,
} from "./presetUtils.js";

const ltx = { id: "ltx_2_3", family: "ltx-video" };
const ltxEros = { id: "ltx_2_3_eros", family: "ltx-video" };
const sdxl = { id: "sdxl", family: "sdxl" };
const catalog = [ltx, ltxEros, sdxl];

describe("normalizeLoraFamily", () => {
  it("collapses every Krea 2 spelling variant to the UI's krea-2 token", () => {
    for (const variant of ["krea_2", "krea-2", "krea2", "KREA2", " Krea_2 "]) {
      expect(normalizeLoraFamily(variant)).toBe("krea-2");
    }
  });

  it("lower-cases and hyphenates other families, leaving unrelated tokens intact", () => {
    expect(normalizeLoraFamily("Z_Image")).toBe("z-image");
    expect(normalizeLoraFamily("wan-video")).toBe("wan-video");
    expect(normalizeLoraFamily("")).toBe("");
  });
});

describe("presetMatchesModel", () => {
  it("matches when the preset pins no model", () => {
    expect(presetMatchesModel({ id: "p" }, ltxEros, catalog)).toBe(true);
  });

  it("matches when the selected model has no id", () => {
    expect(presetMatchesModel({ model: "ltx_2_3" }, {}, catalog)).toBe(true);
  });

  it("matches on exact model id", () => {
    expect(presetMatchesModel({ model: "ltx_2_3" }, ltx, catalog)).toBe(true);
  });

  it("matches a sibling model in the same family (ltx_2_3 preset under ltx_2_3_eros)", () => {
    expect(presetMatchesModel({ model: "ltx_2_3" }, ltxEros, catalog)).toBe(true);
  });

  it("does not match across families", () => {
    expect(presetMatchesModel({ model: "ltx_2_3" }, sdxl, catalog)).toBe(false);
  });

  it("stays strict (no family fallback) when the catalog is unavailable", () => {
    expect(presetMatchesModel({ model: "ltx_2_3" }, ltxEros)).toBe(false);
  });
});

describe("loraMatchesModel", () => {
  const sdxlModel = { id: "sdxl", family: "sdxl", loraCompatibility: { families: ["sdxl"] } };
  const noFamilyModel = { id: "some_model", loraCompatibility: { families: [] } };
  const sdxlLora = { id: "l", family: "sdxl" };
  const fluxLora = { id: "l", family: "flux" };
  const familylessLora = { id: "l", name: "mystery.safetensors" };

  it("matches when the LoRA family is one the model can load", () => {
    expect(loraMatchesModel(sdxlLora, sdxlModel)).toBe(true);
  });

  it("does not match a known-but-different family", () => {
    expect(loraMatchesModel(fluxLora, sdxlModel)).toBe(false);
  });

  it("fails closed on a family-less LoRA (was offered-then-400d before sc-10509)", () => {
    expect(loraMatchesModel(familylessLora, sdxlModel)).toBe(false);
  });

  it("stays permissive when the model declares no LoRA families (can't gate)", () => {
    // Model side is fail-open: a model that declares no families — or no model
    // selected yet — can't be gated against, so the LoRA is still shown (preset
    // application + the "still importing" warning rely on this). The LoRA-family
    // gate only applies once the model actually declares families.
    expect(loraMatchesModel(sdxlLora, noFamilyModel)).toBe(true);
    expect(loraMatchesModel(familylessLora, noFamilyModel)).toBe(true);
  });

  it("fails CLOSED when the API withdrew the model's LoRA advertisement", () => {
    // The API emits this shape for an imported/fine-tuned model no backend lane on this deployment
    // can serve adapters for — a Mage-Flow full fine-tune (whose native single-file loaders reject
    // inference adapters), or a model whose active registered provider serves the base assembly but
    // advertises no adapter format (ComfyUI Qwen-Image). Imported Krea routes adapters on both
    // native backends. Submitting the withdrawn shape 400s, so offering it is a dead-end selection.
    //
    // The empty family list ALONE is not enough: it lands in the permissive "cannot gate" branch
    // asserted above (`noFamilyModel`), which is why the explicit `supported: false` signal exists.
    // These two cases differ only by that flag, so a regression that drops it is caught here.
    const withdrawnFineTune = {
      id: "user_mage_flow_finetune",
      family: "mage-flow",
      loraCompatibility: { families: [], supported: false },
    };
    const withdrawn = {
      id: "external_base_qwen_image",
      family: "qwen-image",
      loraCompatibility: { families: [], supported: false },
    };
    expect(loraMatchesModel({ id: "l", family: "mage-flow" }, withdrawnFineTune)).toBe(false);
    expect(loraMatchesModel(familylessLora, withdrawnFineTune)).toBe(false);
    expect(loraMatchesModel({ id: "l", family: "qwen-image" }, withdrawn)).toBe(false);
    expect(loraMatchesModel(familylessLora, withdrawn)).toBe(false);
    // ...while a model that genuinely serves adapters is untouched.
    expect(loraMatchesModel(sdxlLora, { ...sdxlModel, loraCompatibility: { families: ["sdxl"], supported: true } })).toBe(true);
  });

  it("offers no LoRA for SenseNova when its schema-valid catalog rows omit the advertisement", () => {
    const seededSenseNova = fallbackModels.filter((model) => model.id.startsWith("sensenova_u1_"));
    expect(seededSenseNova.map((model) => model.id)).toEqual([
      "sensenova_u1_8b",
      "sensenova_u1_8b_fast",
    ]);
    for (const model of seededSenseNova) {
      expect(model.loraCompatibility).toBeUndefined();
      expect(loraMatchesModel(sdxlLora, model)).toBe(false);
      expect(loraMatchesModel(familylessLora, model)).toBe(false);
    }

    // Live catalog rows include the family even when the pre-catalog seed does not. Keep that
    // representation equally fail-closed, including future variants that share the same engine.
    expect(
      loraMatchesModel(sdxlLora, {
        id: "sensenova_u1_8b_infographic_v3",
        family: "sensenova-u1",
      }),
    ).toBe(false);
  });

  it("normalizes separators/underscores on both sides before comparing", () => {
    expect(loraMatchesModel({ id: "l", family: "Z_Image" }, { id: "z", loraCompatibility: { families: ["z-image"] } })).toBe(true);
  });

  it("surfaces the Krea 2 turbo accelerator LoRA under Krea 2 Raw (sc-13882)", () => {
    // The manifest-shaped accelerator (family krea_2, role accelerator) must be picker-compatible
    // with the Raw model, whose loraCompatibility.families is ["krea_2"]. The `role` marker is inert
    // to this gate (family is the only signal) — it exists for the S3 sampling-regime routing.
    const turboAccel = {
      id: "krea2_turbo_accel",
      family: "krea_2",
      role: "accelerator",
      compatibility: { families: ["krea_2"] },
    };
    const kreaRaw = { id: "krea_2_raw", loraCompatibility: { families: ["krea_2"], types: ["character", "style", "acceleration"] } };
    expect(loraMatchesModel(turboAccel, kreaRaw)).toBe(true);
  });

  // sc-15017 — the extra-compatible registry, mirrored from `lora_family.rs`. Each case asserts
  // BOTH directions: the model that declares the extra accepts the base family's LoRA, and the base
  // model does NOT thereby accept the variant's. A blind two-way table would pass the first
  // assertion of every pair and fail the second.
  describe("extra-compatible families are honored one-directionally", () => {
    // Transcribed from config/manifests/builtin.models.jsonc: Krea Realtime declares its OWN
    // family, precisely so `wan-video` membership is not inherited by every other family-keyed gate.
    const kreaRealtime = {
      id: "krea_realtime_14b",
      family: "krea-realtime",
      loraCompatibility: { families: ["krea-realtime"], types: ["style", "motion", "character", "enhance"] },
    };
    const wanModel = { id: "wan_2_2_t2v_14b", family: "wan-video", loraCompatibility: { families: ["wan-video"] } };
    const wanLora = { id: "origami", name: "Origami_WanLora", family: "wan-video" };
    const kreaRealtimeLora = { id: "kr", family: "krea-realtime" };

    it("offers a Wan-family LoRA on Krea Realtime 14B (its DiT IS Wan 2.1 T2V 14B)", () => {
      expect(loraMatchesModel(wanLora, kreaRealtime)).toBe(true);
      // …and its own family still matches, so the extra is additive, not a replacement.
      expect(loraMatchesModel(kreaRealtimeLora, kreaRealtime)).toBe(true);
    });

    it("does NOT offer a Krea-Realtime LoRA on a Wan model (the relation is one-directional)", () => {
      expect(loraMatchesModel(kreaRealtimeLora, wanModel)).toBe(false);
      // Control: the Wan model does accept its OWN family, so the `false` above is the direction
      // and not a broken fixture.
      expect(loraMatchesModel(wanLora, wanModel)).toBe(true);
    });

    it("does not leak the relation to an unrelated video family", () => {
      const ltx = { id: "ltx_2_3", family: "ltx-video", loraCompatibility: { families: ["ltx-video"] } };
      expect(loraMatchesModel(wanLora, ltx)).toBe(false);
      expect(loraMatchesModel(kreaRealtimeLora, ltx)).toBe(false);
    });

    it("keeps the pre-existing image-side entries working in the same one direction", () => {
      const chroma = { id: "chroma", family: "chroma", loraCompatibility: { families: ["chroma"] } };
      const fluxModel = { id: "flux", family: "flux", loraCompatibility: { families: ["flux"] } };
      expect(loraMatchesModel({ id: "l", family: "flux" }, chroma)).toBe(true);
      expect(loraMatchesModel({ id: "l", family: "chroma" }, fluxModel)).toBe(false);
      const klein = { id: "flux2_klein", family: "flux2-klein", loraCompatibility: { families: ["flux2-klein"] } };
      expect(loraMatchesModel({ id: "l", family: "flux2" }, klein)).toBe(true);
    });
  });
});

describe("acceptedLoraFamilies", () => {
  it("is the declared families plus the extra-compatible ones, de-duplicated", () => {
    expect(acceptedLoraFamilies({ loraCompatibility: { families: ["krea-realtime"] } })).toEqual([
      "krea-realtime",
      "wan-video",
    ]);
    // A model with no extras is unchanged — the expansion is not applied blindly.
    expect(acceptedLoraFamilies({ loraCompatibility: { families: ["wan-video"] } })).toEqual([
      "wan-video",
    ]);
    // The underscore spelling normalizes first, so the registry lookup still hits.
    expect(acceptedLoraFamilies({ loraCompatibility: { families: ["krea_realtime"] } })).toEqual([
      "krea-realtime",
      "wan-video",
    ]);
    // Already-present extras are not duplicated.
    expect(
      acceptedLoraFamilies({ loraCompatibility: { families: ["krea-realtime", "wan-video"] } }),
    ).toEqual(["krea-realtime", "wan-video"]);
  });

  it("is empty for a model that declares nothing (callers treat that as 'cannot gate')", () => {
    expect(acceptedLoraFamilies(null)).toEqual([]);
    expect(acceptedLoraFamilies({ id: "m", loraCompatibility: { families: [] } })).toEqual([]);
  });

  // sc-15017: the web table is a MIRROR of the Rust registry, and a mirror that is only kept in
  // sync by a comment drifts. Parse the shipped `extra_compatible_lora_families` match arms and
  // require set equality — an entry added on either side alone is red here.
  it("mirrors the Rust extra_compatible_lora_families registry exactly", () => {
    const source = readFileSync(
      resolve(dirname(fileURLToPath(import.meta.url)), "../../../crates/sceneworks-core/src/lora_family.rs"),
      "utf8",
    );
    const body = source.match(
      /fn extra_compatible_lora_families\([^)]*\)[^{]*\{\s*match normalized_family \{([\s\S]*?)\n {4}\}/,
    );
    expect(body, "the Rust registry fn must still be findable — update this parser if it moved").toBeTruthy();
    const arms = body[1];
    // 🔴 Match across newlines, not line-by-line. `cargo fmt` wraps a long arm, and a line-anchored
    // regex would SKIP the wrapped arm silently — then, if the JS table happened to be missing the
    // same entry, the equality below would pass VACUOUSLY on both sides. A non-empty-parse guard
    // does not cover a per-arm drop, so the parsed-arm count is reconciled against the `=>` count:
    // any arm shape this parser cannot read fails loudly and names itself as the thing to update.
    const rust = {};
    let parsedArms = 0;
    for (const arm of arms.matchAll(/((?:"[^"]+"\s*\|\s*)*"[^"]+")\s*=>\s*&\[([\s\S]*?)\]\s*,/g)) {
      parsedArms += 1;
      const extras = [...arm[2].matchAll(/"([^"]+)"/g)].map((m) => m[1]);
      for (const [, key] of arm[1].matchAll(/"([^"]+)"/g)) {
        rust[key] = extras;
      }
    }
    // The `_ => &[],` catch-all carries no families and is deliberately not a table entry, but it
    // still consumes one `=>`, so it is counted here or the reconciliation is off by one.
    const catchAll = /(^|\n)\s*_\s*=>\s*&\[\s*\]\s*,/.test(arms) ? 1 : 0;
    const totalArrows = (arms.match(/=>/g) ?? []).length;
    expect(
      parsedArms + catchAll,
      `parsed ${parsedArms} literal arm(s) + ${catchAll} catch-all, but the registry body has ${totalArrows} \`=>\`. An arm shape this parser cannot read (a wrapped/reformatted arm?) would be dropped silently — update the parser in this test.`,
    ).toBe(totalArrows);
    expect(parsedArms).toBeGreaterThan(0);
    expect(rust["krea-realtime"]).toEqual(["wan-video"]);
    expect(rust).toEqual(EXTRA_COMPATIBLE_LORA_FAMILIES);
  });
});

describe("loraHasResolvableFamily", () => {
  it("is true when the LoRA declares a family", () => {
    expect(loraHasResolvableFamily({ id: "l", family: "sdxl" })).toBe(true);
  });

  it("is false when no family can be resolved (the unusable/escape-hatch guard)", () => {
    expect(loraHasResolvableFamily({ id: "l", name: "mystery.safetensors" })).toBe(false);
    expect(loraHasResolvableFamily({ id: "l", families: [] })).toBe(false);
  });
});

describe("workflowForMode", () => {
  it("folds the character mode into text_to_image", () => {
    expect(workflowForMode("text_to_image")).toBe("text_to_image");
    expect(workflowForMode("character_image")).toBe("text_to_image");
  });

  it("maps the single-mode workflows to themselves", () => {
    expect(workflowForMode("edit_image")).toBe("edit_image");
    expect(workflowForMode("image_to_video")).toBe("image_to_video");
    expect(workflowForMode("text_to_video")).toBe("text_to_video");
    expect(workflowForMode("first_last_frame")).toBe("first_last_frame");
  });

  it("returns an unknown mode unchanged", () => {
    expect(workflowForMode("something_else")).toBe("something_else");
  });
});

describe("slugifyPresetId", () => {
  it("lowercases and replaces runs of invalid characters with a single underscore", () => {
    expect(slugifyPresetId("Atrium Portraits!")).toBe("atrium_portraits");
  });

  it("trims leading and trailing separators so the id starts alphanumeric", () => {
    expect(slugifyPresetId("  --Neon Noir--  ")).toBe("neon_noir");
    expect(slugifyPresetId("neon-noir")).toBe("neon-noir");
  });

  it("returns empty string for non-sluggable input", () => {
    expect(slugifyPresetId("日本語")).toBe("");
    expect(slugifyPresetId("")).toBe("");
  });
});

describe("presetNameTaken", () => {
  const presets = [
    { id: "atrium_portraits", name: "Atrium Portraits" },
    { id: "neon-noir", name: "Neon Noir" },
  ];

  it("matches an existing name case-insensitively", () => {
    expect(presetNameTaken("atrium portraits", presets)).toBe(true);
  });

  it("matches when a different name slugs to an existing id", () => {
    expect(presetNameTaken("Atrium  Portraits!", presets)).toBe(true);
  });

  it("is false for a fresh name and for blank input", () => {
    expect(presetNameTaken("Sunset Pier", presets)).toBe(false);
    expect(presetNameTaken("   ", presets)).toBe(false);
  });
});

describe("finiteNumberOrUndefined", () => {
  it("coerces numeric strings and numbers", () => {
    expect(finiteNumberOrUndefined("30")).toBe(30);
    expect(finiteNumberOrUndefined("4.5")).toBe(4.5);
    expect(finiteNumberOrUndefined(0)).toBe(0);
  });

  it("returns undefined for blank or non-numeric input", () => {
    expect(finiteNumberOrUndefined("")).toBeUndefined();
    expect(finiteNumberOrUndefined(null)).toBeUndefined();
    expect(finiteNumberOrUndefined("abc")).toBeUndefined();
    expect(finiteNumberOrUndefined(undefined)).toBeUndefined();
  });
});

describe("loraWeight", () => {
  it("defaults a generic LoRA to 0.8", () => {
    expect(loraWeight({ id: "sdxl_style", family: "sdxl" })).toBe(0.8);
    expect(loraWeight(null)).toBe(0.8);
  });

  it("defaults a krea-2-family LoRA higher (1.5) for the distilled-Turbo attenuation (sc-7932)", () => {
    // The family token is normalized (krea_2 -> krea-2), and the bump applies via any of the
    // family-bearing shapes extractFamilies() reads.
    expect(loraWeight({ id: "k", family: "krea_2" })).toBe(1.5);
    expect(loraWeight({ id: "k", compatibility: { families: ["krea_2"] } })).toBe(1.5);
  });

  it("lets an explicit weight win over the krea-2 family default", () => {
    expect(loraWeight({ id: "k", family: "krea_2", defaultWeight: 1.0 })).toBe(1.0);
    expect(loraWeight({ id: "k", family: "krea_2", weight: 0.7 })).toBe(0.7);
    expect(loraWeight({ id: "k", family: "krea_2" }, { weight: 2.0 })).toBe(2.0);
  });

  it("falls back to the family default when an explicit value is non-finite", () => {
    expect(loraWeight({ id: "k", family: "krea_2", defaultWeight: "nope" })).toBe(1.5);
    expect(loraWeight({ id: "g", family: "sdxl", defaultWeight: "nope" })).toBe(0.8);
  });
});

describe("cleanPresetDefaults", () => {
  it("drops null/undefined/empty-string but keeps 0 and false", () => {
    expect(
      cleanPresetDefaults({ steps: 0, guidanceScale: "", sampler: null, upscaleEnabled: false, resolution: "1024x1024", motion: undefined }),
    ).toEqual({ steps: 0, upscaleEnabled: false, resolution: "1024x1024" });
  });
});

describe("buildStudioPresetPayload", () => {
  it("snapshots a character-mode image config without the seed", () => {
    const payload = buildStudioPresetPayload({
      name: "Atrium Portraits!",
      scope: "project",
      mode: "character_image",
      model: "instantid_sdxl",
      loras: [{ id: "kelsie", weight: 0.75 }],
      defaults: {
        prompt: "a portrait in the atrium",
        negativePrompt: "",
        resolution: "1024x1024",
        count: 4,
        guidanceScale: 5,
        steps: "",
        sampler: "default",
      },
    });
    expect(payload).toMatchObject({
      id: "atrium_portraits",
      name: "Atrium Portraits!",
      scope: "project",
      workflow: "text_to_image",
      model: "instantid_sdxl",
      loras: [{ id: "kelsie", weight: 0.75 }],
    });
    // modes carry every entry point the picker should surface the preset under.
    expect(payload.modes).toContain("character_image");
    // empty-string knobs are omitted; the literal prompt is preserved.
    expect(payload.defaults).toEqual({
      prompt: "a portrait in the atrium",
      resolution: "1024x1024",
      count: 4,
      guidanceScale: 5,
      sampler: "default",
    });
    expect(payload.defaults).not.toHaveProperty("seed");
  });

  it("coerces a non-finite lora weight to the lora's fallback", () => {
    const payload = buildStudioPresetPayload({
      name: "x",
      mode: "text_to_video",
      model: "ltx_2_3",
      loras: [{ id: "wobble", weight: "not-a-number", defaultWeight: 0.6 }],
      defaults: {},
    });
    expect(payload.workflow).toBe("text_to_video");
    expect(payload.loras).toEqual([{ id: "wobble", weight: 0.6 }]);
  });
});

describe("applyPresetDefault + clearPresetDefault round-trip", () => {
  // Mirrors how the studios drive a state setter: the setter receives either a
  // value or an updater, and the snapshots ref is what the studio keeps in useRef.
  function makeSetter(initial) {
    let value = initial;
    return {
      setter: (updater) => {
        value = typeof updater === "function" ? updater(value) : updater;
      },
      get: () => value,
    };
  }

  it("applies a preset value then restores the user's prior value on clear", () => {
    const snapshots = { current: {} };
    const box = makeSetter("user prompt");
    applyPresetDefault(snapshots, "prompt", box.setter, "preset prompt");
    expect(box.get()).toBe("preset prompt");
    clearPresetDefault(box.setter, snapshots, "prompt");
    expect(box.get()).toBe("user prompt");
  });

  it("leaves a user override in place when they changed the value after applying", () => {
    const snapshots = { current: {} };
    const box = makeSetter(4);
    applyPresetDefault(snapshots, "count", box.setter, 8);
    box.setter(2); // user manually edits after the preset applied
    clearPresetDefault(box.setter, snapshots, "count");
    expect(box.get()).toBe(2);
  });
});

// sc-4162 regression: editModelForAsset previously lived in App.jsx and called
// modelLoraFamilies without importing it — a ReferenceError on the family-sibling
// path (any asset whose generating model can't edit).
describe("editModelForAsset", () => {
  const t2iOnly = { id: "z_image_turbo", family: "z-image", capabilities: ["text_to_image"] };
  const editSibling = { id: "z_image_edit", family: "z-image", capabilities: ["edit_image"] };
  const editSelf = { id: "qwen_image_edit", family: "qwen-image", capabilities: ["image_edit"] };
  const models = [t2iOnly, editSibling, editSelf];

  it("prefers the generating model when it is edit-capable", () => {
    expect(editModelForAsset({ recipe: { model: "qwen_image_edit" } }, models)).toBe("qwen_image_edit");
  });

  it("falls back to a same-family edit-capable sibling when the source model cannot edit", () => {
    expect(editModelForAsset({ recipe: { model: "z_image_turbo" } }, models)).toBe("z_image_edit");
  });

  it("matches a family sibling when the generating model is not in the catalog", () => {
    expect(editModelForAsset({ recipe: { model: "z-image" } }, models)).toBe("z_image_edit");
  });

  it("returns null when no family-matched edit model exists", () => {
    expect(editModelForAsset({ recipe: { model: "z_image_turbo" } }, [t2iOnly, editSelf])).toBe(null);
  });

  it("returns null for assets without a recipe model", () => {
    expect(editModelForAsset({ recipe: {} }, models)).toBe(null);
    expect(editModelForAsset(null, models)).toBe(null);
  });
});

// epic 10871 (Krea image edit): the payload must carry the conditioning role so the worker's
// edit lane can see it (it reads the role straight off `request.loras`, no catalog re-lookup).
describe("serializeLora round-trips the conditioning role (epic 10871)", () => {
  it("forwards conditioningRole and icLora", () => {
    const out = serializeLora({ id: "krea2_identity_edit", family: "krea_2", conditioningRole: "image_edit" });
    expect(out.conditioningRole).toBe("image_edit");
    expect(out.icLora).toBe(false);
  });

  it("preserves an explicit icLora flag (LTX IC-LoRA)", () => {
    const out = serializeLora({ id: "ltx_ic", conditioningRole: "ic_lora", icLora: true });
    expect(out.conditioningRole).toBe("ic_lora");
    expect(out.icLora).toBe(true);
  });

  it("defaults role to null / flag to false for a plain style LoRA", () => {
    const out = serializeLora({ id: "plain" });
    expect(out.conditioningRole).toBe(null);
    expect(out.icLora).toBe(false);
  });
});

describe("serializeLora round-trips the sampling-regime role (sc-13883)", () => {
  it("forwards role: accelerator so the worker can select the turbo regime", () => {
    // The Krea 2 turbo accelerator LoRA (sc-13882). Without this round-trip the worker never sees the
    // `role` marker and a picked accelerator runs only as a plain additive residual — Raw stays 52-step.
    const out = serializeLora({ id: "krea2_turbo_accel", family: "krea_2", role: "accelerator" });
    expect(out.role).toBe("accelerator");
  });

  it("defaults role to null for a LoRA that declares no sampling-regime role", () => {
    expect(serializeLora({ id: "plain" }).role).toBe(null);
    // The conditioning-axis sibling does not backfill `role` (they are orthogonal markers).
    expect(serializeLora({ id: "edit", conditioningRole: "image_edit" }).role).toBe(null);
  });
});

describe("loraLooksLikeImageEditLora (epic 10871)", () => {
  it("matches the image_edit conditioning role (case / separator insensitive)", () => {
    expect(loraLooksLikeImageEditLora({ conditioningRole: "image_edit" })).toBe(true);
    expect(loraLooksLikeImageEditLora({ conditioningRole: "Image-Edit" })).toBe(true);
    expect(loraLooksLikeImageEditLora({ imageEditLora: true })).toBe(true);
  });

  it("does not match an IC-LoRA or a plain LoRA (role is the only signal)", () => {
    expect(loraLooksLikeImageEditLora({ conditioningRole: "ic_lora" })).toBe(false);
    expect(loraLooksLikeImageEditLora({ id: "krea2_identity_edit", name: "Krea 2 Identity Edit" })).toBe(false);
    expect(loraLooksLikeImageEditLora({})).toBe(false);
  });
});

describe("findModelEditLora (epic 10871)", () => {
  const kreaEdit = { id: "krea2_identity_edit", family: "krea_2", conditioningRole: "image_edit" };
  const kreaStyle = { id: "krea_style", family: "krea_2" };
  const catalogLoras = [kreaStyle, kreaEdit];

  it("finds the compatible image_edit LoRA for the model", () => {
    expect(findModelEditLora(catalogLoras, { id: "krea_2_raw", family: "krea_2" })).toBe(kreaEdit);
  });

  it("finds the same image_edit LoRA for Krea 2 Turbo (sc-11640 — shared family, CFG-free edit)", () => {
    // Turbo edit reuses the exact `krea2_identity_edit` LoRA (family krea_2, no base gating), so the
    // auto-apply resolves the same adapter as Raw — the Studio surfaces it for either edit variant.
    expect(findModelEditLora(catalogLoras, { id: "krea_2_turbo", family: "krea_2" })).toBe(kreaEdit);
  });

  it("returns null for a model with no compatible image_edit LoRA (Qwen/FLUX edit)", () => {
    expect(findModelEditLora(catalogLoras, { id: "qwen_image_edit", family: "qwen-image" })).toBe(null);
  });

  it("returns null on missing inputs", () => {
    expect(findModelEditLora(null, { family: "krea_2" })).toBe(null);
    expect(findModelEditLora(catalogLoras, null)).toBe(null);
  });
});

describe("loraIsInstalled (epic 10871)", () => {
  it("is false only when the catalog row is explicitly missing", () => {
    expect(loraIsInstalled({ installState: "installed" })).toBe(true);
    expect(loraIsInstalled({})).toBe(true); // installed local rows carry no installState
    expect(loraIsInstalled({ installState: "missing" })).toBe(false);
    expect(loraIsInstalled(null)).toBe(false);
  });
});
