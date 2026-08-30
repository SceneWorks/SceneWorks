// The drop path's client-side half (sc-15951, epic 15945): what is worth uploading, and how a
// `WorkflowShare` becomes the recipe the Image Studio's existing injection effect already reads.
import { describe, expect, it } from "vitest";

import { readFileSync } from "node:fs";
import { dirname, resolve } from "node:path";
import { fileURLToPath } from "node:url";

import {
  ADVANCED_PREFILL,
  assetImportedWorkflow,
  looksLikeWorkflowCandidate,
  MAX_WORKFLOW_INSPECT_BYTES,
  PREFILL_CONTROL,
  PREFILL_NONE,
  PREFILL_PROMPT,
  recipeFromWorkflowShare,
  workflowHasMissingInputs,
  workflowInputSlots,
  workflowInstallable,
  workflowLoras,
  workflowModelId,
  workflowModelUnknown,
  workflowNotRestored,
  workflowPhases,
  workflowReplaySeed,
  workflowResolutionLabel,
  workflowSettingRows,
  workflowUnresolved,
} from "./workflowShare.js";

const PNG_HEAD = [0x89, 0x50, 0x4e, 0x47, 0x0d, 0x0a, 0x1a, 0x0a];

function pngFile(name = "shared.png", trailing = 64) {
  const bytes = new Uint8Array([...PNG_HEAD, ...new Array(trailing).fill(0)]);
  return new File([bytes], name, { type: "image/png" });
}

function share(overrides = {}) {
  return {
    sceneworksWorkflow: "image",
    schemaVersion: 1,
    producer: { name: "SceneWorks", url: "https://example.invalid", version: "0.8.1" },
    mode: "text_to_image",
    model: "krea_2_turbo",
    prompt: "a lighthouse in heavy fog, cinematic",
    negativePrompt: "text, watermark",
    seed: 880412,
    width: 1024,
    height: 1536,
    count: 4,
    advanced: { steps: 28, guidanceScale: 3.5, sampler: "euler", resolution: "1024x1536" },
    ...overrides,
  };
}

function report(overrides = {}) {
  return {
    model: {
      slug: "krea_2_turbo",
      state: "resolved",
      catalogId: "krea_2_turbo",
      name: "Krea 2 Turbo",
      detail: "Krea 2 Turbo is installed on this machine.",
    },
    loras: [],
    styles: [],
    inputs: [],
    omitted: [],
    runnable: true,
    inputImagesRequired: 0,
    ...overrides,
  };
}

describe("looksLikeWorkflowCandidate", () => {
  it("accepts a PNG", async () => {
    await expect(looksLikeWorkflowCandidate(pngFile())).resolves.toBe(true);
  });

  it("refuses a non-PNG without asking the server", async () => {
    // The whole point of the prescreen: the endpoint stages the WHOLE body to disk before it
    // looks at a byte, so a 400 MB TIFF would be uploaded just to be told it is not a PNG.
    const tiff = new File([new Uint8Array([0x49, 0x49, 0x2a, 0x00, 1, 2, 3, 4, 5])], "scan.tiff", {
      type: "image/tiff",
    });
    await expect(looksLikeWorkflowCandidate(tiff)).resolves.toBe(false);
    const text = new File(["a plain note, dropped by accident"], "notes.txt", {
      type: "text/plain",
    });
    await expect(looksLikeWorkflowCandidate(text)).resolves.toBe(false);
  });

  it("refuses a file over the endpoint's own cap rather than uploading it to be 413'd", async () => {
    const huge = pngFile("huge.png");
    Object.defineProperty(huge, "size", { value: MAX_WORKFLOW_INSPECT_BYTES + 1 });
    await expect(looksLikeWorkflowCandidate(huge)).resolves.toBe(false);
  });

  it("refuses a file too short to hold a signature, and a non-file", async () => {
    await expect(looksLikeWorkflowCandidate(new File([new Uint8Array([0x89])], "x.png"))).resolves.toBe(
      false,
    );
    await expect(looksLikeWorkflowCandidate(null)).resolves.toBe(false);
    await expect(looksLikeWorkflowCandidate({})).resolves.toBe(false);
  });

  it("answers false rather than throwing when the file cannot be read", async () => {
    const unreadable = {
      size: 4096,
      slice: () => ({
        arrayBuffer: () => Promise.reject(new Error("permission revoked")),
      }),
    };
    await expect(looksLikeWorkflowCandidate(unreadable)).resolves.toBe(false);
  });
});

describe("recipeFromWorkflowShare", () => {
  it("populates the fields the studio's injection effect reads", () => {
    const recipe = recipeFromWorkflowShare(share(), report());
    expect(recipe.mode).toBe("text_to_image");
    expect(recipe.model).toBe("krea_2_turbo");
    expect(recipe.prompt).toBe("a lighthouse in heavy fog, cinematic");
    expect(recipe.negativePrompt).toBe("text, watermark");
    expect(recipe.normalizedSettings.width).toBe(1024);
    expect(recipe.normalizedSettings.height).toBe(1536);
    // `recipeResolution` builds "1024x1536" out of these, which is what setResolution takes.
    expect(recipe.rawAdapterSettings.steps).toBe(28);
    expect(recipe.rawAdapterSettings.guidanceScale).toBe(3.5);
    expect(recipe.rawAdapterSettings.sampler).toBe("euler");
  });

  it("drops invalid numeric controls and restores finite values as numbers", () => {
    const recipe = recipeFromWorkflowShare(
      share({
        advanced: {
          schedulerShift: "2.5",
          steps: 0,
          guidanceScale: -1.25,
          ipAdapterScale: "not a number",
          controlnetConditioningScale: Infinity,
          trueCfgScale: "NaN",
          strength: "",
          textStyleGain: {},
          controlScale: [],
        },
      }),
      report(),
    );

    expect(recipe.rawAdapterSettings).toMatchObject({
      schedulerShift: 2.5,
      steps: 0,
      guidanceScale: -1.25,
    });
    for (const key of [
      "ipAdapterScale",
      "controlnetConditioningScale",
      "trueCfgScale",
      "strength",
      "textStyleGain",
      "controlScale",
    ]) {
      expect(recipe.rawAdapterSettings).not.toHaveProperty(key);
    }
  });

  it("always asks for ONE image, whatever batch the shared file came out of", () => {
    // The envelope's seed identifies one image; `count` came from the batch the run requested.
    // Replaying both would reproduce the shared image as the first of a 4-image batch.
    const recipe = recipeFromWorkflowShare(share({ count: 4 }), report());
    expect(recipe.normalizedSettings.count).toBe(1);
  });

  it("carries sanitized pose coordinates for replay and does not invent an omitted selection", () => {
    const poses = [
      { keypoints: [[0.1, 0.2]], hands: [[[0.3, 0.4]]], face: [[0.5, 0.6]] },
      { keypoints: [[0.7, 0.8]] },
    ];
    const shared = share({ advanced: { poses, faceRestore: true } });
    const recipe = recipeFromWorkflowShare(shared, report());
    expect(recipe.rawAdapterSettings.poses).toEqual(poses);
    expect(workflowSettingRows(shared, report()).find((row) => row.key === "poses")).toMatchObject({
      restored: true,
      value: "2 poses",
    });

    const omitted = recipeFromWorkflowShare(share({ advanced: { faceRestore: true } }), report());
    expect(omitted.rawAdapterSettings).not.toHaveProperty("poses");
  });

  it("restores an explicit alternate decoder and does not invent the native default", () => {
    const selected = share({ advanced: { decoder: "wan_2_1_vae" } });
    const recipe = recipeFromWorkflowShare(selected, report());
    expect(recipe.rawAdapterSettings.decoder).toBe("wan_2_1_vae");
    expect(workflowSettingRows(selected, report()).find((row) => row.key === "decoder")).toMatchObject({
      label: "Alternate decoder",
      restored: true,
      value: "wan_2_1_vae",
    });

    const native = recipeFromWorkflowShare(share({ advanced: { steps: 28 } }), report());
    expect(native.rawAdapterSettings).not.toHaveProperty("decoder");
  });

  it("does not prefill a model this install cannot resolve", () => {
    const missing = report({
      model: {
        slug: "some_private_model",
        state: "missing",
        detail: "No model matching `some_private_model` is in this install's catalog…",
      },
      runnable: false,
    });
    const recipe = recipeFromWorkflowShare(share({ model: "some_private_model" }), missing);
    expect(recipe.model).toBeUndefined();
    expect(recipe.prompt).toBe("a lighthouse in heavy fog, cinematic");
  });

  it("does not prefill a model that is merely installable — the picker would drop it", () => {
    // `generationModelsForType` filters on `modelInstallComplete`, so an id with no row behind
    // it would set a select to a value that has no option. Named in the panel instead.
    const installable = report({
      model: {
        slug: "krea_2_turbo",
        state: "installable",
        catalogId: "krea_2_turbo",
        name: "Krea 2 Turbo",
        install: { method: "POST", path: "/api/v1/models/krea_2_turbo/download" },
        detail: "Krea 2 Turbo is in the model catalog but is not downloaded…",
      },
      runnable: false,
    });
    expect(recipeFromWorkflowShare(share(), installable).model).toBeUndefined();
    expect(workflowModelId(installable)).toBeNull();
  });

  it("selects resolved LoRAs by catalog id with their weights, and drops the rest", () => {
    const withLoras = report({
      loras: [
        {
          name: "Film Grain",
          repo: "acme/film-grain",
          weight: 0.65,
          state: "resolved",
          catalogId: "film_grain",
          detail: "Film Grain is installed.",
        },
        {
          name: "Someone's private LoRA",
          state: "missing",
          detail: "No LoRA on this install matches `Someone's private LoRA`…",
        },
        {
          name: "Aurora",
          state: "installable",
          catalogId: "aurora_v3",
          detail: "Aurora is in the LoRA catalog but is not downloaded…",
        },
      ],
      runnable: false,
    });
    expect(workflowLoras(withLoras)).toEqual([{ id: "film_grain", weight: 0.65 }]);
    expect(recipeFromWorkflowShare(share(), withLoras).loras).toEqual([
      { id: "film_grain", weight: 0.65 },
    ]);
  });

  it("carries a resolved LoRA with no recorded weight as a bare id", () => {
    const withLora = report({
      loras: [{ name: "Film Grain", state: "resolved", catalogId: "film_grain", detail: "ok" }],
    });
    expect(workflowLoras(withLora)).toEqual([{ id: "film_grain" }]);
  });

  it("keeps the style round-trip only when the style catalog has the id", () => {
    const styled = share({
      styleId: "ghibli-style",
      advanced: { styleId: "ghibli-style", stylePrompt: "a lighthouse", steps: 20 },
    });
    const resolvedStyle = report({
      styles: [
        {
          field: "styleId",
          id: "ghibli-style",
          state: "resolved",
          name: "Ghibli Style",
          detail: "ok",
        },
      ],
    });
    const kept = recipeFromWorkflowShare(styled, resolvedStyle);
    expect(kept.rawAdapterSettings.styleId).toBe("ghibli-style");
    expect(kept.rawAdapterSettings.stylePrompt).toBe("a lighthouse");

    // Unresolved: dropping both keys is what makes the studio use the already-COMPOSED prompt
    // instead of re-selecting a picker entry this catalog does not have.
    const missingStyle = report({
      styles: [
        {
          field: "styleId",
          id: "ghibli-style",
          state: "missing",
          detail: "No style matching `ghibli-style`…",
        },
      ],
      runnable: false,
    });
    const dropped = recipeFromWorkflowShare(styled, missingStyle);
    expect(dropped.rawAdapterSettings.styleId).toBeUndefined();
    expect(dropped.rawAdapterSettings.stylePrompt).toBeUndefined();
    expect(dropped.prompt).toBe("a lighthouse in heavy fog, cinematic");
  });

  it("moves the top-level upscale and fitMode into the raw settings the effect reads", () => {
    const edit = share({
      mode: "edit_image",
      fitMode: "contain",
      upscale: { enabled: true, factor: 2, engine: "seedvr2", softness: 0.2 },
    });
    const recipe = recipeFromWorkflowShare(edit, report());
    expect(recipe.mode).toBe("edit_image");
    expect(recipe.rawAdapterSettings.fitMode).toBe("contain");
    expect(recipe.rawAdapterSettings.upscale).toEqual({
      enabled: true,
      factor: 2,
      engine: "seedvr2",
      softness: 0.2,
    });
  });

  it("does not alias the envelope's own advanced map", () => {
    // The panel renders the envelope beside the recipe; a shared reference would let the
    // studio-facing edits (the style strip, the upscale graft) rewrite what is displayed.
    const source = share({ styleId: "x", advanced: { styleId: "x", stylePrompt: "raw", steps: 4 } });
    recipeFromWorkflowShare(source, report({ runnable: false }));
    expect(source.advanced.styleId).toBe("x");
    expect(source.advanced.stylePrompt).toBe("raw");
  });

  it("prefills nothing that had to be looked up when no report was supplied", () => {
    const recipe = recipeFromWorkflowShare(share());
    expect(recipe.model).toBeUndefined();
    expect(recipe.loras).toEqual([]);
    expect(recipe.prompt).toBe("a lighthouse in heavy fog, cinematic");
  });

  it("is null for a null share", () => {
    expect(recipeFromWorkflowShare(null, report())).toBeNull();
  });
});

describe("workflowReplaySeed", () => {
  it("reproduces this image's own seed, including zero", () => {
    expect(workflowReplaySeed(share({ seed: 880412 }))).toBe("880412");
    expect(workflowReplaySeed(share({ seed: 0 }))).toBe("0");
  });

  it("is null when the file recorded no seed", () => {
    expect(workflowReplaySeed(share({ seed: null }))).toBeNull();
    expect(workflowReplaySeed({})).toBeNull();
  });
});

describe("workflowUnresolved", () => {
  it("names every requirement that does not resolve, omissions included", () => {
    const rows = workflowUnresolved(
      report({
        model: { slug: "x", state: "missing", detail: "no model" },
        loras: [{ name: "y", state: "installable", detail: "not downloaded" }],
        styles: [{ field: "styleId", id: "z", state: "missing", detail: "no style" }],
        inputs: [
          { kind: "source", count: 1, state: "userSupplied", detail: "Needs a source image." },
        ],
        omitted: [{ field: "loras", detail: "The LoRA list was not recorded." }],
        runnable: false,
      }),
    );
    // `input` is absent on purpose: `ResolutionReport::runnable` excludes input images (nothing on
    // this machine could ever have supplied someone else's source image), so counting them here
    // made a perfectly runnable edit share read "2 things … cannot be reproduced on this install".
    // The panel still RENDERS every input row — this list is what the verdict counts.
    expect(rows.map((row) => row.kind)).toEqual(["model", "lora", "style", "omitted"]);
    expect(rows.map((row) => row.detail)).toContain("The LoRA list was not recorded.");
  });

  it("counts nothing for a runnable recipe that merely needs an image from the user", () => {
    const rows = workflowUnresolved(
      report({
        inputs: [
          { kind: "source", count: 1, state: "userSupplied", detail: "Needs a source image." },
        ],
        runnable: true,
      }),
    );
    expect(rows).toEqual([]);
  });

  it("is empty for a fully resolved report", () => {
    expect(workflowUnresolved(report())).toEqual([]);
    expect(workflowUnresolved(null)).toEqual([]);
  });
});

describe("display helpers", () => {
  it("labels the geometry, or nothing when the envelope carried none", () => {
    expect(workflowResolutionLabel(share())).toBe("1024 × 1536");
    expect(workflowResolutionLabel(share({ width: null, height: null }))).toBeNull();
  });

  it("shows every advanced knob in the file, labelled where a label exists", () => {
    const rows = workflowSettingRows(
      share({ advanced: { steps: 28, someFutureKnob: "on", usePid: true, stylePrompt: "hidden" } }),
    );
    const byKey = Object.fromEntries(rows.map((row) => [row.key, row]));
    expect(byKey.steps.label).toBe("Steps");
    expect(byKey.usePid.value).toBe("On");
    // A knob nobody has written a label for is still a knob that changed the image.
    expect(byKey.someFutureKnob.label).toBe("someFutureKnob");
    // Prose and structures are shown elsewhere, not squeezed into the scalar grid.
    expect(byKey.stylePrompt).toBeUndefined();
  });
});

// ---- The one table both halves read -----------------------------------------------------------

const HERE = dirname(fileURLToPath(import.meta.url));
const ENVELOPE_DOC = resolve(HERE, "../../../docs/workflow-share-envelope.md");

// The `advanced` keys the ENVELOPE can carry, read out of the pinned table in
// `docs/workflow-share-envelope.md`. That table is itself pinned to `ADVANCED_KEY_RULES` in both
// directions by `crates/sceneworks-core/tests/workflow_share_doc.rs`, so reading it here fails the
// moment a key is added to the contract and not to `ADVANCED_PREFILL` — which is what stops a new
// knob being displayed as restored while reaching no control.
function sharedAdvancedKeysFromTheContract() {
  const doc = readFileSync(ENVELOPE_DOC, "utf8");
  const block = doc.split("<!-- PINNED: advanced-shared -->")[1]?.split("<!-- END PINNED")[0] ?? "";
  const keys = [...block.matchAll(/^\|\s*`([A-Za-z][\w.]*)`\s*\|/gm)].map((match) => match[1]);
  expect(keys.length).toBeGreaterThan(20); // the parse still works
  return keys;
}

// A plausible value per key, so one envelope can carry the whole table.
function sampleValueFor(key) {
  switch (key) {
    case "poses":
      return [{ keypoints: [1, 2] }, { keypoints: [3, 4] }];
    case "phases":
      return [{ steps: 4, guidance: 3.5, loras: [] }];
    case "structuredPrompt":
      return { intent: "a lighthouse" };
    case "enhancePrompt":
    case "usePid":
    case "faceRestore":
      return true;
    case "pidTarget":
      return "2k";
    case "controlMode":
      return "depth";
    case "sampler":
      return "euler";
    case "scheduler":
      return "shift";
    case "guidanceMethod":
      return "cfg";
    case "resolution":
      return "1024x1536";
    case "styleId":
      return "ghibli";
    case "stylePrompt":
      return "a lighthouse";
    case "systemMessage":
      return "You are a storyboard artist.";
    case "viewAngle":
      return "three_quarter";
    case "angleSet":
      return "turnaround_4";
    default:
      return 0.75;
  }
}

describe("ADVANCED_PREFILL is the source of truth for both the prefill and the panel", () => {
  it("classifies every advanced key the envelope contract can carry, and no others", () => {
    expect(Object.keys(ADVANCED_PREFILL).sort()).toEqual(sharedAdvancedKeysFromTheContract().sort());
  });

  it("gives every entry a known disposition, and every unapplied one a reason", () => {
    for (const entry of Object.values(ADVANCED_PREFILL)) {
      expect([PREFILL_CONTROL, PREFILL_PROMPT, PREFILL_NONE]).toContain(entry.prefill);
      expect(entry.label.length).toBeGreaterThan(0);
      if (entry.prefill === PREFILL_NONE) {
        // "Not restored" with no reason reads like a bug rather than a decision.
        expect(entry.detail?.length ?? 0).toBeGreaterThan(20);
      } else {
        expect(entry.detail).toBeUndefined();
      }
    }
  });

  // THE ANTI-DRIFT TEST. Marking is worthless if it can disagree with the prefill, so both are
  // derived from the same table against ONE envelope carrying every key: whatever the recipe
  // carries is exactly what the panel does not mark, and vice versa.
  it("marks exactly the rows the recipe does not carry", () => {
    const everyKey = Object.fromEntries(
      Object.keys(ADVANCED_PREFILL).map((key) => [key, sampleValueFor(key)]),
    );
    const here = report({
      styles: [{ field: "styleId", id: "ghibli", name: "Ghibli", state: "resolved", detail: "ok" }],
    });
    const recipe = recipeFromWorkflowShare(share({ advanced: everyKey }), here);
    const rows = workflowSettingRows(share({ advanced: everyKey }), here);

    const carried = new Set(Object.keys(recipe.rawAdapterSettings));
    for (const row of rows) {
      expect(row.restored).toBe(carried.has(row.key));
      if (!row.restored) {
        expect(row.detail.length).toBeGreaterThan(20);
      }
    }
    // …and the unrestored ones are exactly the lanes an envelope carries that THIS studio has no
    // control for, plus `structuredPrompt`, whose caption object does not travel (sc-15952 — the
    // prompt is restored as prose instead, and the row says so). Pose coordinates now hydrate
    // session-only picker records through the ordinary pose-selection path (sc-16132).
    //
    // The remaining video-only keys (sc-15956) are here for the first reason, not the second: they
    // travel intact in a shared MP4 and replay in Video Studio, which is a different panel.
    // `textEncoderModel` is no longer in that set because Image Studio now has the same authored
    // selector and restores its opaque id.
    expect(
      rows
        .filter((row) => !row.restored)
        .map((row) => row.key)
        .sort(),
    ).toEqual([
      "angleSet",
      "bridgeRightVideoConditioningStrength",
      "cnScale",
      "distilledVariant",
      "imageGuidanceScale",
      "lightning",
      "ltxPipeline",
      "motion",
      "structuredPrompt",
      "systemMessage",
      "timelineAction",
      "videoCfgGuidanceScale",
      "videoConditioningStrength",
      "videoRescaleScale",
      "videoStgGuidanceScale",
    ]);
  });

  // `styleId` is the one CONDITIONAL entry: it is applied, but only when this install has the
  // style — so its row follows the report rather than the table's optimism.
  it("marks a styleId this install does not have, and restores one it does", () => {
    const styled = share({ advanced: { styleId: "ghibli", stylePrompt: "a lighthouse" } });
    const absent = workflowSettingRows(styled, report());
    expect(absent[0].key).toBe("styleId");
    expect(absent[0].restored).toBe(false);
    expect(recipeFromWorkflowShare(styled, report()).rawAdapterSettings.styleId).toBeUndefined();

    const here = report({
      styles: [{ field: "styleId", id: "ghibli", name: "Ghibli", state: "resolved", detail: "ok" }],
    });
    expect(workflowSettingRows(styled, here)[0].restored).toBe(true);
    expect(recipeFromWorkflowShare(styled, here).rawAdapterSettings.styleId).toBe("ghibli");
  });

  it("keeps a key the panel has never heard of out of the recipe, and marks it", () => {
    const unknown = share({ advanced: { someFutureKnob: "on" } });
    const rows = workflowSettingRows(unknown);
    expect(rows.map((row) => row.key)).toEqual(["someFutureKnob"]);
    expect(rows[0].restored).toBe(false);
    expect(recipeFromWorkflowShare(unknown, report()).rawAdapterSettings.someFutureKnob).toBeUndefined();
  });
});

describe("workflowNotRestored", () => {
  it("does not report replayable pose coordinates as unrestored", () => {
    const rows = workflowNotRestored(
      share({
        advanced: { steps: 28, cnScale: 0.4, poses: [{ keypoints: [] }, { keypoints: [] }] },
      }),
    );
    expect(rows.map((row) => row.key)).toEqual(["cnScale"]);
  });

  it("names a RESOLVED style preset, which is installed here and still not replayed", () => {
    const styles = [
      { field: "stylePreset", id: "cine", name: "Cinematic", state: "resolved", detail: "ok" },
    ];
    const rows = workflowNotRestored(share({ advanced: {} }), report({ styles }));
    expect(rows.map((row) => row.key)).toEqual(["stylePreset"]);
    // …and the recipe no longer carries the dead `stylePreset` assignment nothing ever read.
    expect(recipeFromWorkflowShare(share(), report({ styles })).stylePreset).toBeUndefined();
  });

  it("leaves an UNRESOLVED style preset to the report, which already blocks on it", () => {
    const styles = [
      { field: "stylePreset", id: "cine", name: "Cinematic", state: "missing", detail: "gone" },
    ];
    expect(workflowNotRestored(share({ advanced: {} }), report({ styles }))).toEqual([]);
    expect(workflowUnresolved(report({ styles, runnable: false })).map((row) => row.kind)).toEqual([
      "style",
    ]);
  });
});

describe("workflowPhases", () => {
  const loraRequirement = (name, catalogId) => ({
    name,
    state: catalogId ? "resolved" : "missing",
    catalogId: catalogId ?? undefined,
    detail: "…",
  });

  it("reindexes each phase's LoRA references onto the resolved subset", () => {
    // The envelope's indices point into the ORIGINAL request's LoRA list. `workflowLoras` drops the
    // unresolved middle entry, so index 2 has to become index 1 — leaving it alone would silently
    // activate a different adapter, which is worse than not replaying the schedule at all.
    const withLoras = report({
      loras: [
        loraRequirement("Grain", "film_grain"),
        loraRequirement("A private LoRA", null),
        loraRequirement("Turbo", "krea2_turbo_accel"),
      ],
      runnable: false,
    });
    const phases = workflowPhases(
      share({
        advanced: {
          phases: [
            { steps: 4, guidance: 3.5, loras: [] },
            { steps: 4, guidance: 0, loras: [{ index: 2, weight: 0.9 }, { index: 1 }] },
            { steps: 2, guidance: 1, loras: [{ index: 0 }] },
          ],
        },
      }),
      withLoras,
    );
    expect(phases).toEqual([
      { steps: 4, guidance: 3.5, loras: [] },
      { steps: 4, guidance: 0, loras: [{ index: 1, weight: 0.9 }] },
      { steps: 2, guidance: 1, loras: [{ index: 0 }] },
    ]);
    expect(workflowLoras(withLoras)).toEqual([{ id: "film_grain" }, { id: "krea2_turbo_accel" }]);
  });

  it("is null when the envelope carried no schedule", () => {
    expect(workflowPhases(share(), report())).toBeNull();
    expect(workflowPhases(share({ advanced: { phases: [] } }), report())).toBeNull();
    expect(recipeFromWorkflowShare(share(), report()).rawAdapterSettings.phases).toBeUndefined();
  });
});

// ---- The envelope on an imported asset (sc-15952) --------------------------------------------

describe("assetImportedWorkflow", () => {
  it("finds the envelope import recorded, and nothing else", () => {
    const envelope = { sceneworksWorkflow: "image", schemaVersion: 1 };
    expect(assetImportedWorkflow({ extra: { importedWorkflow: envelope } })).toBe(envelope);
    // The common case: an image imported from any of the billions of PNGs with no chunk in them.
    // `extra` exists; the key does not. A truthiness check on `extra` would answer yes here.
    expect(assetImportedWorkflow({ extra: { source: "upload" } })).toBeNull();
    expect(assetImportedWorkflow({ extra: {} })).toBeNull();
    expect(assetImportedWorkflow({})).toBeNull();
    expect(assetImportedWorkflow(null)).toBeNull();
    // A scalar under the key is not an envelope.
    expect(assetImportedWorkflow({ extra: { importedWorkflow: "yes" } })).toBeNull();
  });
});

// ---- What the missing-inputs view is derived from (sc-15952) ---------------------------------

describe("the missing-inputs derivations", () => {
  it("offers an install only for the state that publishes one", () => {
    // `install_for` in `workflow_resolution.rs` clears the action for every state but
    // `installable`, so a button anywhere else would be a button that cannot work.
    expect(workflowInstallable(report())).toEqual([]);
    const rows = workflowInstallable(
      report({
        model: {
          slug: "krea_2_turbo",
          state: "installable",
          catalogId: "krea_2_turbo",
          name: "Krea 2 Turbo",
          detail: "not downloaded",
          install: { method: "POST", path: "/api/v1/models/krea_2_turbo/download" },
        },
        loras: [
          { name: "Film Grain", state: "installable", catalogId: "film_grain", detail: "d" },
          { name: "Aurora v3", state: "missing", detail: "user-trained" },
          { name: "Here", state: "resolved", catalogId: "here", detail: "d" },
        ],
      }),
    );
    expect(rows.map((row) => `${row.kind}:${row.id}`)).toEqual([
      "model:krea_2_turbo",
      "lora:film_grain",
    ]);
  });

  it("tells an unknown model apart from one that merely needs downloading", () => {
    // Installing the model the file NAMES reproduces the image; substituting one does not. Only
    // the first of those gets a substitute picker.
    expect(workflowModelUnknown(report())).toBe(false);
    expect(
      workflowModelUnknown(report({ model: { slug: "x", state: "installable", detail: "d" } })),
    ).toBe(false);
    expect(workflowModelUnknown(report({ model: { slug: "x", state: "missing", detail: "d" } }))).toBe(
      true,
    );
  });

  it("expands an input requirement into one slot per image", () => {
    // `reference` travels as ONE entry carrying a count — never as N entries — so a reader that
    // maps over `inputs` renders half a two-reference recipe.
    const slots = workflowInputSlots(
      report({
        inputs: [
          { kind: "source", count: 1, state: "userSupplied", detail: "Needs a source image." },
          { kind: "reference", count: 2, state: "userSupplied", detail: "Needs 2 reference images." },
          { kind: "mask", count: 1, state: "userSupplied", detail: "Needs a mask." },
        ],
      }),
    );
    expect(slots).toHaveLength(4);
    expect(slots.map((slot) => slot.label)).toEqual([
      "Source image",
      "Reference image 1 of 2",
      "Reference image 2 of 2",
      "Mask",
    ]);
    // Only the kinds the launch seam can actually carry get a picker.
    expect(slots.map((slot) => slot.pickable)).toEqual([true, true, true, false]);
    // The report's sentence is shown once per requirement, not once per slot.
    expect(slots.map((slot) => slot.detail)).toEqual([
      "Needs a source image.",
      "Needs 2 reference images.",
      "",
      "Needs a mask.",
    ]);
    expect(slots[3].elsewhere).toContain("Image Editor");
    // The mask row names the OUTCOME of proceeding without it, not only where the control lives:
    // Image Studio has no mask control, so the CTA edits the whole frame instead of the region.
    expect(slots[3].elsewhere).toMatch(/whole image/i);
    expect(slots[3].elsewhere).toMatch(/different edit from the one you were sent/i);
  });

  it("floors an absent count at one slot, the way the report floors its own total", () => {
    const slots = workflowInputSlots(
      report({ inputs: [{ kind: "source", count: 0, state: "userSupplied", detail: "d" }] }),
    );
    expect(slots).toHaveLength(1);
  });

  it("has nothing to show for a recipe this install can run outright", () => {
    // The acceptance criterion behind the component's `return null`.
    expect(workflowHasMissingInputs(report())).toBe(false);
    expect(workflowHasMissingInputs(null)).toBe(false);
    // …including one whose only problems are ones nobody can act on. `omitted` makes the report
    // unrunnable and belongs to the report rows, not to a section of things to do.
    expect(
      workflowHasMissingInputs(
        report({
          runnable: false,
          omitted: [{ field: "loras", detail: "dropped whole" }],
          loras: [{ name: "Aurora v3", state: "missing", detail: "user-trained" }],
          styles: [{ field: "styleId", id: "ghibli", state: "missing", detail: "absent" }],
        }),
      ),
    ).toBe(false);
    // …and something for each of the three it CAN act on.
    expect(
      workflowHasMissingInputs(report({ model: { slug: "x", state: "missing", detail: "d" } })),
    ).toBe(true);
    expect(
      workflowHasMissingInputs(
        report({ inputs: [{ kind: "source", count: 1, state: "userSupplied", detail: "d" }] }),
      ),
    ).toBe(true);
  });
});

// ---- Prefilling a partially-resolvable recipe (sc-15952) -------------------------------------

describe("recipeFromWorkflowShare — the user's own answers", () => {
  const unknownModel = report({
    runnable: false,
    model: { slug: "someones_private_model", state: "missing", detail: "not here" },
    loras: [{ name: "Aurora v3", state: "missing", detail: "user-trained" }],
  });

  it("prefills everything that resolved and names no model when nothing did", () => {
    // "Partially-resolvable prefills the resolved fields and lists the rest": the prompt, the
    // geometry and the allow-listed knobs all land; the model and the LoRA do not, and neither is
    // reached for.
    const recipe = recipeFromWorkflowShare(share(), unknownModel);
    expect(recipe.prompt).toBe("a lighthouse in heavy fog, cinematic");
    expect(recipe.normalizedSettings).toEqual({ width: 1024, height: 1536, count: 1 });
    expect(recipe.rawAdapterSettings.steps).toBe(28);
    expect(recipe.rawAdapterSettings.sampler).toBe("euler");
    expect(recipe.model).toBeUndefined();
    expect(recipe.loras).toEqual([]);
  });

  it("carries a model the USER chose, and never one it chose itself", () => {
    expect(
      recipeFromWorkflowShare(share(), unknownModel, { substituteModelId: "z_image_turbo" }).model,
    ).toBe("z_image_turbo");
    // No substitute → no model, whatever the picker's placeholder happens to be.
    for (const empty of ["", null, undefined]) {
      expect(
        recipeFromWorkflowShare(share(), unknownModel, { substituteModelId: empty }).model,
      ).toBeUndefined();
    }
  });

  it("never lets a substitute displace a model this install actually resolved", () => {
    // The resolved id is what the FILE asked for. A substitute is only ever reached for when there
    // was no resolution — that is the difference between a choice and a fallback.
    expect(
      recipeFromWorkflowShare(share(), report(), { substituteModelId: "z_image_turbo" }).model,
    ).toBe("krea_2_turbo");
  });

  it("carries the reference images the user picked, in order", () => {
    const recipe = recipeFromWorkflowShare(share(), report(), {
      referenceAssetIds: ["ref_one", "ref_two"],
    });
    expect(recipe.rawAdapterSettings.referenceAssetIds).toEqual(["ref_one", "ref_two"]);
    // The single-reference control the studio has always read, so one pick lands even on a model
    // with no plural picker.
    expect(recipe.rawAdapterSettings.referenceAssetId).toBe("ref_one");
    // Nothing picked writes nothing, so a recorded recipe's own selection is never cleared.
    const bare = recipeFromWorkflowShare(share(), report());
    expect(bare.rawAdapterSettings.referenceAssetIds).toBeUndefined();
    expect(bare.rawAdapterSettings.referenceAssetId).toBeUndefined();
  });

  it("leaves a structured recipe's prompt as the prose the model received", () => {
    // The decided answer to "does the builder rehydrate?" — no. The envelope reduces
    // `structuredPrompt` to its scalars and drops the caption object, so there is nothing for the
    // builder to load; the top-level `prompt` is the exact string that made the image. The recipe
    // therefore carries no `structuredPrompt` at all rather than one the studio silently ignores.
    const structured = share({
      advanced: { structuredPrompt: { intent: "a lighthouse", runtimePrompt: "{…}" }, steps: 28 },
    });
    const recipe = recipeFromWorkflowShare(structured, report());
    expect(recipe.rawAdapterSettings.structuredPrompt).toBeUndefined();
    expect(recipe.prompt).toBe("a lighthouse in heavy fog, cinematic");
    // …and the panel says so rather than dropping it silently.
    const row = workflowSettingRows(structured, report()).find(
      (entry) => entry.key === "structuredPrompt",
    );
    expect(row.restored).toBe(false);
    expect(row.detail).toContain("restored as prose");
    expect(
      workflowNotRestored(structured, report()).some((entry) => entry.key === "structuredPrompt"),
    ).toBe(true);
  });
});
