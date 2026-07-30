// The drop path's client-side half (sc-15951, epic 15945): what is worth uploading, and how a
// `WorkflowShare` becomes the recipe the Image Studio's existing injection effect already reads.
import { describe, expect, it } from "vitest";

import {
  looksLikeWorkflowCandidate,
  MAX_WORKFLOW_INSPECT_BYTES,
  recipeFromWorkflowShare,
  workflowLoras,
  workflowModelId,
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

  it("always asks for ONE image, whatever batch the shared file came out of", () => {
    // The envelope's seed identifies one image; `count` came from the batch the run requested.
    // Replaying both would reproduce the shared image as the first of a 4-image batch.
    const recipe = recipeFromWorkflowShare(share({ count: 4 }), report());
    expect(recipe.normalizedSettings.count).toBe(1);
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
    expect(rows.map((row) => row.kind)).toEqual(["model", "lora", "style", "input", "omitted"]);
    expect(rows.map((row) => row.detail)).toContain("The LoRA list was not recorded.");
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
