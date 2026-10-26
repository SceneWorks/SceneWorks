// sc-17161 — the MiniMax-H3 user-facing surface in the Advanced Video Studio.
//
// Every model fixture here is the SHIPPED manifest entry, read from
// `config/manifests/builtin.models.jsonc` at test time. A retyped fixture would let the studio and
// the catalog drift into disagreement about the very numbers this story exists to express — the
// fourteen `17n + 5` duration rungs, the 9/3/3/12 reference caps and the 2-step floor — and a test
// that asserts a control against a hand-written copy of a limit proves nothing about the model the
// user actually picks.
//
// The rule these tests are really guarding is declaration ≠ REACHABILITY: sc-17160 shipped the
// `referenceAudioAssetIds` payload field with no control that could set it, and sc-17159 fixed the
// image-only spelling of the r2v gate in the API while the studio kept its own copy of the same
// bug. A control that renders but cannot produce a submittable job is not delivered.

import { readFileSync } from "node:fs";
import { dirname, resolve } from "node:path";
import { fileURLToPath } from "node:url";

import React, { act } from "react";
import JSON5 from "json5";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { click, mountRoot, setSelect, unmountRoot } from "../testUtils/dom.js";

vi.mock("../api.js", async (importOriginal) => {
  const actual = await importOriginal();
  return { ...actual, apiFetch: vi.fn(async () => ({})) };
});

// sc-19574 — a switch that forces `audioOnlyReferenceSet` FALSE into the real validator, so the
// OTHER guard on the same shape (`hasInputs`' r2v arm, which sc-19574 narrowed to require a VISUAL
// reference) can be asserted on its own. The two are exactly co-extensive on the r2v arm — the
// narrowing changes the answer only when audio is outgoing and no image or clip is, which is the
// precise condition `audioOnlyReferenceSet` names — so through the screen neither can be observed
// while the other holds. That is why re-widening `hasInputs` mutated silently, and why proving it
// individually needs the other guard suppressed rather than a different fixture.
//
// Off by default and reset in `afterEach`, so every other test in this file runs the real thing.
const validationOverride = vi.hoisted(() => ({ suppressAudioOnly: false }));
vi.mock("../videoStudioValidation.js", async (importOriginal) => {
  const actual = await importOriginal();
  return {
    ...actual,
    videoGenerateValidation: (args) =>
      actual.videoGenerateValidation(
        validationOverride.suppressAudioOnly ? { ...args, audioOnlyReferenceSet: false } : args,
      ),
  };
});

import { AppContext } from "../context/AppContext.js";
import { VideoStudio } from "./VideoStudio.jsx";

const HERE = dirname(fileURLToPath(import.meta.url));
const MANIFEST_PATH = resolve(HERE, "../../../../config/manifests/builtin.models.jsonc");
const manifestModels = (() => {
  const parsed = JSON5.parse(readFileSync(MANIFEST_PATH, "utf8"));
  return Array.isArray(parsed) ? parsed : parsed.models;
})();
const shipped = (id) => {
  const entry = manifestModels.find((model) => model.id === id);
  if (!entry) throw new Error(`manifest has no model ${id}`);
  // The catalog serves the manifest entry plus install state; the studio only renders installed,
  // usable models, so the fixture adds exactly what the route adds.
  return { ...entry, installState: "installed", usable: true };
};

const MINIMAX = shipped("minimax_h3");
const MINIMAX_REF = shipped("minimax_h3_ref");
// Bernini is the OTHER model serving `reference_to_video`, and it is the control leg for every
// per-model gate below: its engine encodes reference IMAGES only, so a change that loosened the
// reference rules globally rather than per-model would show up here.
const BERNINI = shipped("bernini");

const asset = (id, type, name) => ({
  id,
  type,
  projectId: "project_1",
  displayName: name,
  status: "active",
  file: { path: `${id}.bin` },
});

function baseContext(overrides = {}) {
  return {
    token: "test-token",
    activeProject: { id: "project_1", name: "My Project" },
    assets: [],
    characters: [],
    createPersonDetectionJob: vi.fn(),
    createPersonTrackJob: vi.fn(),
    createVideoJob: vi.fn(),
    createPreset: vi.fn(async (payload) => ({ id: payload.id })),
    refinePrompt: vi.fn(),
    deleteAsset: vi.fn(),
    purgeAsset: vi.fn(),
    gpuOptions: [],
    importAsset: vi.fn(),
    latestVideoAssets: [],
    recentVideoAssets: [],
    studioLaunch: null,
    loras: [],
    jobs: [],
    videoLocalJobs: [],
    jobAction: vi.fn(),
    rememberLocalGenerationJob: vi.fn(),
    setActiveView: vi.fn(),
    setSelectedAssetId: vi.fn(),
    setPreviewAsset: vi.fn(),
    personTracks: [],
    personReadiness: {},
    presets: [],
    requestedGpu: "",
    saveTrackCorrections: vi.fn(),
    selectedAsset: null,
    setRequestedGpu: vi.fn(),
    updateAssetStatus: vi.fn(),
    models: [],
    videoModels: [MINIMAX],
    ...overrides,
  };
}

describe("MiniMax-H3 in the Video Studio (sc-17161)", () => {
  let container;
  let root;

  beforeEach(() => {
    global.IS_REACT_ACT_ENVIRONMENT = true;
    window.localStorage.clear();
    ({ container, root } = mountRoot());
  });

  afterEach(async () => {
    await unmountRoot(root, container);
    vi.clearAllMocks();
    validationOverride.suppressAudioOnly = false;
  });

  async function render(context) {
    await act(async () => {
      root.render(
        <AppContext.Provider value={context}>
          <VideoStudio />
        </AppContext.Provider>,
      );
    });
  }

  const labelStartingWith = (text) =>
    [...container.querySelectorAll("label")].find((label) => label.textContent.trim().startsWith(text));
  const selectLabelled = (text) => labelStartingWith(text)?.querySelector("select");
  const fieldLabelled = (text) =>
    [...container.querySelectorAll(".asset-picker-field")].find((field) =>
      field.querySelector(".asset-picker-label")?.textContent.trim() === text,
    );
  const modeTab = (label) =>
    [...container.querySelectorAll(".mode-tabs button")].find((b) => b.textContent.trim() === label);
  const openAdvanced = async () => {
    const toggle = container.querySelector(".advanced-section-toggle");
    if (toggle && container.querySelector(".advanced-panel") === null) await click(toggle);
  };

  // ---------------------------------------------------------------- geometry

  it("offers exactly the fourteen lattice rungs and never 15s", async () => {
    await render(baseContext());
    const options = [...selectLabelled("Duration").querySelectorAll("option")];

    // The MENU is the model's declared set, byte for byte — not a range, not a rounded subset.
    expect(options.map((option) => Number(option.value))).toEqual(MINIMAX.limits.durations);
    expect(options).toHaveLength(14);

    // 15.0s is what MiniMax's own documentation advertises. The next lattice rung is 362 frames
    // = 15.083s, so 15s is REFUSED rather than quietly delivered as 14.375s — it must not be
    // offered, and neither must anything above the declared hard maximum.
    expect(options.some((option) => Number(option.value) === 15)).toBe(false);
    for (const option of options) {
      expect(Number(option.value)).toBeLessThanOrEqual(MINIMAX.limits.hardMaxDuration);
      expect(Number(option.value)).toBeGreaterThanOrEqual(MINIMAX.limits.hardMinDuration);
    }

    // Labels name the frame count, because the lattice is a FRAME lattice: every rung's label
    // resolves to a frame count on `17n + 5`, which is what makes "nothing in between" true.
    expect(options[0].textContent).toBe("5.17s · 124 frames");
    expect(options.at(-1).textContent).toBe("14.38s · 345 frames");
    for (const option of options) {
      const frames = Number(option.textContent.match(/· (\d+) frames/)[1]);
      expect(frames % 17, `${option.textContent} is off the 17n+5 lattice`).toBe(5);
    }
  });

  it("tells the user in prose that 15s is refused, not shortened (sc-17162)", async () => {
    // The menu test above proves the studio cannot SUBMIT 15s. It does not prove the user is ever
    // told why the option they came for is missing, and an absent option explains nothing — a
    // user who read MiniMax's advertised "5-15 s" will read the 14.38s cap as a SceneWorks
    // shortcoming rather than a property of these weights.
    //
    // `ui.durationHint` is the field that carries that, and it has exactly ONE reader
    // (`VideoStudio.jsx` → `<p className="helper-copy">`). Set-derived over the family, so a third
    // partition is covered the day it ships, and read out of the DOM rather than off the manifest:
    // pinning the string alone would stay green if the reader were dropped.
    for (const model of manifestModels.filter((entry) => entry.family === "minimax-h3")) {
      const fixture = { ...shipped(model.id), installState: "installed", usable: true };
      await render(baseContext({ videoModels: [fixture] }));
      const helper = [...container.querySelectorAll(".helper-copy")].map((node) => node.textContent);
      expect(
        helper.some((text) => /\b15s is refused\b[^.]{0,60}\bshortened\b/.test(text)),
        `${model.id} must render its durationHint's 15s refusal`,
      ).toBe(true);
    }
  });

  it("preselects the declared default duration, not the first menu entry", async () => {
    // Asserted on BERNINI, because MiniMax-H3's `defaults.duration` and `limits.durations[0]` are
    // the same 5.1667 — on H3 alone the two claims are indistinguishable, and substituting
    // `durations[0]` for `defaults.duration` in the studio leaves every H3 test green. Bernini is
    // the model where they differ, so it is the only place this assertion means anything.
    expect(BERNINI.defaults.duration).not.toBe(BERNINI.limits.durations[0]);
    await render(baseContext({ videoModels: [BERNINI] }));
    expect(Number(selectLabelled("Duration").value)).toBe(BERNINI.defaults.duration);
    expect(Number(selectLabelled("Duration").value)).not.toBe(BERNINI.limits.durations[0]);

    // The H3 leg still runs: the default has to be ON the fourteen-rung lattice, which is a claim
    // about the menu rather than about which entry is preselected.
    await render(baseContext());
    expect(Number(selectLabelled("Duration").value)).toBe(MINIMAX.defaults.duration);
    expect(MINIMAX.limits.durations).toContain(MINIMAX.defaults.duration);
  });

  it("submits an exact lattice value, not a rounded label", async () => {
    // The label is rounded to 2dp; the enqueue gate matches `limits.durations` exactly, so
    // sending 5.17 instead of 5.1667 would be refused for a duration the user picked from the menu.
    const context = baseContext();
    await render(context);
    const target = MINIMAX.limits.durations[3];
    await act(async () => setSelect(selectLabelled("Duration"), String(target)));
    await click([...container.querySelectorAll("button")].find((b) => b.textContent.includes("Render clip")));
    expect(context.createVideoJob.mock.calls[0][0].duration).toBe(target);
  });

  it("takes the Steps floor from the model rather than a hardcoded 1", async () => {
    const context = baseContext({ videoModels: [MINIMAX, BERNINI] });
    await render(context);
    await openAdvanced();
    expect(labelStartingWith("Steps").querySelector("input").getAttribute("min")).toBe(
      String(MINIMAX.limits.hardMinSteps),
    );

    // The other leg: a model that declares no floor keeps the historical 1, so this is a per-model
    // fact and not a new blanket.
    await act(async () => setSelect(selectLabelled("Model"), "bernini"));
    await openAdvanced();
    expect(BERNINI.limits.hardMinSteps).toBeUndefined();
    expect(labelStartingWith("Steps").querySelector("input").getAttribute("min")).toBe("1");
  });

  it("restores and submits H3 last-only FL2VA without inventing a first frame", async () => {
    window.localStorage.setItem("sceneworks-studio-video-project_1", JSON.stringify({mode:"image_to_video",model:"minimax_h3",prompt:"A cyclist rides",lastFrameAssetId:"last",sourceAssetId:"",resolution:"768x768",duration:5.1667}));
    const context=baseContext({assets:[asset("last","image","last"),asset("other","image","other")],selectedAsset:asset("other","image","other")});
    await render(context);expect(fieldLabelled("Last frame")).toBeTruthy();
    const button=[...container.querySelectorAll("button")].find(b=>b.textContent.includes("Render clip"));expect(button.disabled).toBe(false);await click(button);
    const payload=context.createVideoJob.mock.calls[0][0];expect(payload.mode).toBe("image_to_video");expect(payload.model).toBe("minimax_h3");expect(payload.sourceAssetId).toBeNull();expect(payload.lastFrameAssetId).toBe("last");
  });
  it("does not admit last-only Image-to-Video for other families", async () => {
    window.localStorage.setItem("sceneworks-studio-video-project_1", JSON.stringify({mode:"image_to_video",model:"bernini",prompt:"A cyclist rides",lastFrameAssetId:"last",sourceAssetId:""}));
    await render(baseContext({videoModels:[BERNINI],assets:[asset("last","image","last")]}));expect(fieldLabelled("Last frame")).toBeFalsy();
    expect([...container.querySelectorAll("button")].find(b=>b.textContent.includes("Render clip")).disabled).toBe(true);
  });

  // The ambient first frame. The studio adopts the app's current asset as its starting image when
  // the user has not chosen one here (sc-11964) — a convenience for a model that REQUIRES a source
  // image. It must not become an invented first frame on a request the user framed around the LAST
  // frame: the API accepts a last-only request for an imported FL2VA/hybrid model, so the studio
  // has to be able to send one (sc-26687 S8).
  const ambientImported = { ...MINIMAX, id: "imported_ambient_fl2va", family: "minimax-h3", baseModel: "minimax_h3", importSourceShape: "transformer_directory", importPlan: { checkpointId: "checkpoint", planId: "plan", assembly: { schemaVersion: 1, partition: "fl2va" } }, minimaxH3Partition: "fl2va" };

  async function pickAssetFromField(fieldLabel, assetName) {
    const field = fieldLabelled(fieldLabel);
    expect(field, `the ${fieldLabel} picker is rendered`).toBeTruthy();
    const trigger = field.querySelector(".asset-picker-actions button");
    expect(trigger, `the ${fieldLabel} picker opens from its own control`).toBeTruthy();
    await click(trigger);
    const modal = document.querySelector(".asset-picker-modal");
    expect(modal, "the asset picker modal opens").toBeTruthy();
    const option = [...modal.querySelectorAll('[role="option"]')].find((node) => node.textContent.includes(assetName));
    expect(option, `${assetName} is offered in the picker`).toBeTruthy();
    await click(option);
    await click([...modal.querySelectorAll("button")].find((node) => node.textContent.trim() === "Use Selection"));
  }

  it("drops the ambient first frame when the user picks a LAST frame for an imported last-only model", async () => {
    const ambient = asset("other", "image", "Other image");
    const context = baseContext({ videoModels: [ambientImported], assets: [ambient, asset("last", "image", "Last frame")], selectedAsset: ambient });
    await render(context);
    await act(async () => setSelect(selectLabelled("Model"), ambientImported.id));
    await click(modeTab("Image → Video"));
    // Nobody chose this first frame in this studio: it is the app's current asset.
    expect(fieldLabelled("First frame").textContent).toContain("Other image");

    await pickAssetFromField("Last frame", "Last frame");

    // The ambient first frame is dropped rather than sent as an invented first frame.
    expect(fieldLabelled("First frame").textContent).not.toContain("Other image");
    const button = [...container.querySelectorAll("button")].find((b) => b.textContent.includes("Render clip"));
    expect(button.disabled).toBe(false);
    await click(button);
    expect(context.createVideoJob.mock.calls[0][0]).toMatchObject({ model: ambientImported.id, mode: "image_to_video", sourceAssetId: null, lastFrameAssetId: "last" });
  });

  it("keeps the ambient first frame in First -> Last, where the first frame is required", async () => {
    const ambient = asset("other", "image", "Other image");
    const context = baseContext({ videoModels: [ambientImported], assets: [ambient, asset("last", "image", "Last frame")], selectedAsset: ambient });
    await render(context);
    await act(async () => setSelect(selectLabelled("Model"), ambientImported.id));
    await click(modeTab("First → Last"));

    await pickAssetFromField("Last frame", "Last frame");

    // Both frames are required in this mode, so the studio's starting-image default stays.
    expect(fieldLabelled("First frame").textContent).toContain("Other image");
    const button = [...container.querySelectorAll("button")].find((b) => b.textContent.includes("Render clip"));
    expect(button.disabled).toBe(false);
    await click(button);
    expect(context.createVideoJob.mock.calls[0][0]).toMatchObject({ mode: "first_last_frame", sourceAssetId: "other", lastFrameAssetId: "last" });
  });

  // The request is guarded where it is built, not only in the picker: the ambient first frame can
  // still reach a last-only request by another route — the mode can change after the last frame was
  // picked, and the app's ambient selection can land afterwards. This is that route (sc-26687).
  it("does not send an ambient first frame when the mode changes to Image -> Video after the LAST frame was picked", async () => {
    const ambient = asset("other", "image", "Other image");
    const context = baseContext({ videoModels: [ambientImported], assets: [ambient, asset("last", "image", "Last frame")], selectedAsset: ambient });
    await render(context);
    await act(async () => setSelect(selectLabelled("Model"), ambientImported.id));
    await click(modeTab("First → Last"));
    await pickAssetFromField("Last frame", "Last frame");
    // First -> Last REQUIRES a first frame, so the ambient one is kept there.
    expect(fieldLabelled("First frame").textContent).toContain("Other image");

    await click(modeTab("Image → Video"));

    const button = [...container.querySelectorAll("button")].find((b) => b.textContent.includes("Render clip"));
    expect(button.disabled).toBe(false);
    await click(button);
    expect(context.createVideoJob.mock.calls[0][0]).toMatchObject({ model: ambientImported.id, mode: "image_to_video", sourceAssetId: null, lastFrameAssetId: "last" });
  });

  // Stock `minimax_h3` is deliberately OUT of this change: an ambient first frame is part of that
  // model's shipped behaviour, and this epic is about the import platform. Pinning it here means a
  // later change cannot quietly widen the imported-only rule back onto stock.
  it("keeps the ambient first frame for the STOCK minimax_h3", async () => {
    const ambient = asset("other", "image", "Other image");
    const context = baseContext({ videoModels: [MINIMAX], assets: [ambient, asset("last", "image", "Last frame")], selectedAsset: ambient });
    await render(context);
    await act(async () => setSelect(selectLabelled("Model"), MINIMAX.id));
    await click(modeTab("Image → Video"));

    await pickAssetFromField("Last frame", "Last frame");

    const button = [...container.querySelectorAll("button")].find((b) => b.textContent.includes("Render clip"));
    expect(button.disabled).toBe(false);
    await click(button);
    expect(context.createVideoJob.mock.calls[0][0]).toMatchObject({ model: MINIMAX.id, mode: "image_to_video", sourceAssetId: "other", lastFrameAssetId: "last" });
  });

  it("keeps a first frame the user chose when they then pick a LAST frame", async () => {
    const ambient = asset("other", "image", "Other image");
    const context = baseContext({ videoModels: [ambientImported], assets: [ambient, asset("last", "image", "Last frame"), asset("first", "image", "First image")], selectedAsset: ambient });
    await render(context);
    await act(async () => setSelect(selectLabelled("Model"), ambientImported.id));
    await click(modeTab("Image → Video"));

    await pickAssetFromField("First frame", "First image");
    await pickAssetFromField("Last frame", "Last frame");

    const button = [...container.querySelectorAll("button")].find((b) => b.textContent.includes("Render clip"));
    expect(button.disabled).toBe(false);
    await click(button);
    expect(context.createVideoJob.mock.calls[0][0]).toMatchObject({ model: ambientImported.id, mode: "image_to_video", sourceAssetId: "first", lastFrameAssetId: "last" });
  });

  // The snapshot carries the first frame AND how it was chosen (sc-26687). An ambient frame
  // captured before a reload returns as ambient, so it cannot ride along in an imported last-only
  // request — the guard has to survive the studio being remounted, not just the session.
  it("does not send an ambient first frame restored from the snapshot", async () => {
    window.localStorage.setItem("sceneworks-studio-video-project_1", JSON.stringify({
      mode: "image_to_video",
      model: ambientImported.id,
      prompt: "A cyclist rides",
      sourceAssetId: "other",
      sourceAssetOrigin: "ambient",
      lastFrameAssetId: "last",
    }));
    const context = baseContext({
      videoModels: [ambientImported],
      assets: [asset("other", "image", "other"), asset("last", "image", "last")],
      selectedAsset: asset("other", "image", "other"),
    });
    await render(context);
    const button = [...container.querySelectorAll("button")].find((b) => b.textContent.includes("Render clip"));
    expect(button.disabled).toBe(false);
    await click(button);
    expect(context.createVideoJob.mock.calls[0][0]).toMatchObject({ model: ambientImported.id, mode: "image_to_video", sourceAssetId: null, lastFrameAssetId: "last" });
  });

  // A snapshot written before the origin was recorded is not proof that the user chose the frame:
  // the studio's ambient seed is the common producer of a saved sourceAssetId. It is therefore read
  // as ambient, and an ambient frame is withheld from an imported last-only request (sc-26687).
  it("does not send a first frame from a snapshot that records no origin", async () => {
    window.localStorage.setItem("sceneworks-studio-video-project_1", JSON.stringify({
      mode: "image_to_video",
      model: ambientImported.id,
      prompt: "A cyclist rides",
      sourceAssetId: "other",
      lastFrameAssetId: "last",
    }));
    const context = baseContext({
      videoModels: [ambientImported],
      assets: [asset("other", "image", "other"), asset("last", "image", "last")],
      selectedAsset: asset("other", "image", "other"),
    });
    await render(context);
    const button = [...container.querySelectorAll("button")].find((b) => b.textContent.includes("Render clip"));
    expect(button.disabled).toBe(false);
    await click(button);
    expect(context.createVideoJob.mock.calls[0][0]).toMatchObject({ model: ambientImported.id, mode: "image_to_video", sourceAssetId: null, lastFrameAssetId: "last" });
  });

  it("sends a first frame the snapshot records as the user's own pick", async () => {
    window.localStorage.setItem("sceneworks-studio-video-project_1", JSON.stringify({
      mode: "image_to_video",
      model: ambientImported.id,
      prompt: "A cyclist rides",
      sourceAssetId: "other",
      sourceAssetOrigin: "explicit",
      lastFrameAssetId: "last",
    }));
    const context = baseContext({
      videoModels: [ambientImported],
      assets: [asset("other", "image", "other"), asset("last", "image", "last")],
      selectedAsset: asset("other", "image", "other"),
    });
    await render(context);
    const button = [...container.querySelectorAll("button")].find((b) => b.textContent.includes("Render clip"));
    expect(button.disabled).toBe(false);
    await click(button);
    expect(context.createVideoJob.mock.calls[0][0]).toMatchObject({ model: ambientImported.id, mode: "image_to_video", sourceAssetId: "other", lastFrameAssetId: "last" });
  });

  for (const partition of ["fl2va", "hybrid"]) {
    const imported = { ...MINIMAX, id: `imported_${partition}`, family: "minimax-h3", baseModel: "minimax_h3", importSourceShape: "transformer_directory", importPlan: { checkpointId: "checkpoint", planId: "plan", assembly: { schemaVersion: 1, partition } }, minimaxH3Partition: partition };
    it(`restores imported ${partition} last-only without seeding the selected first frame`, async () => {
      window.localStorage.setItem("sceneworks-studio-video-project_1", JSON.stringify({ mode: "image_to_video", model: imported.id, prompt: "A cyclist rides", lastFrameAssetId: "last", sourceAssetId: "" }));
      const context = baseContext({ videoModels: [imported], assets: [asset("last", "image", "last"), asset("other", "image", "other")], selectedAsset: asset("other", "image", "other") });
      await render(context);
      expect(fieldLabelled("Last frame")).toBeTruthy();
      const button = [...container.querySelectorAll("button")].find(b => b.textContent.includes("Render clip"));
      expect(button.disabled).toBe(false);
      await click(button);
      expect(context.createVideoJob.mock.calls[0][0]).toMatchObject({ model: imported.id, mode: "image_to_video", sourceAssetId: null, lastFrameAssetId: "last" });
    });
    it(`replays imported ${partition} last frame and original checkpoint identity`, async () => {
      const importedModelRef = { modelId: imported.id, checkpointId: "original", planId: "plan_original", declaredPartition: partition, baseModel: "minimax_h3", family: "minimax-h3" };
      const context = baseContext({ videoModels: [imported], assets: [asset("last", "image", "last")], studioLaunch: { id: "replay", view: "Video", recipe: { mode: "image_to_video", model: imported.id, prompt: "A cyclist rides", importedModelRef, normalizedSettings: { sourceAssetId: null, lastFrameAssetId: "last", width: 768, height: 768, duration: 5.1667 }, rawAdapterSettings: {} } } });
      await render(context);
      const button = [...container.querySelectorAll("button")].find(b => b.textContent.includes("Render clip"));
      expect(button.disabled).toBe(false);
      await click(button);
      expect(context.createVideoJob.mock.calls[0][0]).toMatchObject({ model: imported.id, sourceAssetId: null, lastFrameAssetId: "last", expectedImportedModelRef: importedModelRef });
      expect(JSON.parse(window.localStorage.getItem("sceneworks-studio-video-project_1")).expectedImportedModelRef).toEqual(importedModelRef);
      await act(async () => setSelect(selectLabelled("Model"), imported.id));
      await click(button);
      expect(context.createVideoJob.mock.calls.at(-1)[0].expectedImportedModelRef).toBeUndefined();
    });
  }

  it("restores the original imported identity from a saved replay snapshot", async () => {
    const imported = { ...MINIMAX, id: "saved-import", family: "minimax-h3", baseModel: "minimax_h3", importSourceShape: "transformer_directory", importPlan: { checkpointId: "checkpoint", planId: "plan" }, minimaxH3Partition: "hybrid" };
    const expectedImportedModelRef = { modelId: imported.id, checkpointId: "original", planId: "original-plan" };
    window.localStorage.setItem("sceneworks-studio-video-project_1", JSON.stringify({ mode: "image_to_video", model: imported.id, sourceAssetId: "", lastFrameAssetId: "last", expectedImportedModelRef }));
    const context = baseContext({ videoModels: [imported], assets: [asset("last", "image", "last")] });
    await render(context);
    await click([...container.querySelectorAll("button")].find(b => b.textContent.includes("Render clip")));
    expect(context.createVideoJob.mock.calls[0][0].expectedImportedModelRef).toEqual(expectedImportedModelRef);
  });
  it("drops a replayed imported identity when the next recipe records none", async () => {
    const imported = { ...MINIMAX, id: "boundary-import", family: "minimax-h3", baseModel: "minimax_h3", importSourceShape: "transformer_directory", importPlan: { checkpointId: "checkpoint" }, minimaxH3Partition: "fl2va" };
    const recipe = { mode: "text_to_video", model: imported.id, prompt: "A cyclist rides", importedModelRef: { modelId: imported.id, checkpointId: "original" }, normalizedSettings: {}, rawAdapterSettings: {} };
    const context = baseContext({ videoModels: [imported], studioLaunch: { id: "first", view: "Video", recipe } });
    await render(context);
    const button = [...container.querySelectorAll("button")].find(b => b.textContent.includes("Render clip"));
    await click(button);
    await act(async () => {});
    expect(context.createVideoJob.mock.calls[0][0].expectedImportedModelRef).toEqual(recipe.importedModelRef);
    await act(async () => {
      root.render(<AppContext.Provider value={{ ...context, studioLaunch: { id: "second", view: "Video", recipe: { ...recipe, importedModelRef: undefined } } }}><VideoStudio /></AppContext.Provider>);
    });
    await act(async () => {});
    const nextButton = [...container.querySelectorAll("button")].find(b => b.textContent.includes("Render clip"));
    await click(nextButton);
    expect(context.createVideoJob).toHaveBeenCalledTimes(2);
    expect(context.createVideoJob.mock.calls[1][0].expectedImportedModelRef).toBeUndefined();
  });

  it("drops a replayed imported identity at the next non-recipe launch boundary", async () => {
    const imported = { ...MINIMAX, id: "boundary-import-2", family: "minimax-h3", baseModel: "minimax_h3", importSourceShape: "transformer_directory", importPlan: { checkpointId: "checkpoint" }, minimaxH3Partition: "fl2va" };
    const recipe = { mode: "text_to_video", model: imported.id, prompt: "A cyclist rides", importedModelRef: { modelId: imported.id, checkpointId: "original" }, normalizedSettings: {}, rawAdapterSettings: {} };
    const context = baseContext({ videoModels: [imported], studioLaunch: { id: "first", view: "Video", recipe } });
    await render(context);
    const snapshot = () => JSON.parse(window.localStorage.getItem("sceneworks-studio-video-project_1") ?? "{}");
    expect(snapshot().expectedImportedModelRef).toEqual(recipe.importedModelRef);
    await act(async () => {
      root.render(<AppContext.Provider value={{ ...context, studioLaunch: { id: "second", view: "Video" } }}><VideoStudio /></AppContext.Provider>);
    });
    await act(async () => {});
    expect(snapshot().expectedImportedModelRef).toBeNull();
  });

  for (const override of [{ minimaxH3Partition: "ref2va" }, { family: "other" }, { baseModel: "minimax_h3_ref" }, { minimaxH3Partition: undefined }, { importSourceShape: undefined }, { importPlan: { checkpointId: " " } }]) {
    it(`does not widen last-only support for mismatched catalog declarations ${JSON.stringify(override)}`, async () => {
      const entry = { ...MINIMAX, id: "other-model", family: "minimax-h3", baseModel: "minimax_h3", importSourceShape: "transformer_directory", importPlan: { checkpointId: "checkpoint" }, minimaxH3Partition: "fl2va", ...override };
      window.localStorage.setItem("sceneworks-studio-video-project_1", JSON.stringify({ mode: "image_to_video", model: entry.id, sourceAssetId: "", lastFrameAssetId: "last" }));
      await render(baseContext({ videoModels: [entry], assets: [asset("last", "image", "last")] }));
      expect(fieldLabelled("Last frame")).toBeFalsy();
      expect([...container.querySelectorAll("button")].find(b => b.textContent.includes("Render clip")).disabled).toBe(true);
    });
  }

  // ------------------------------------------------------------------- modes

  it("drives t2va and fl2va from the base partition and hides the axes it has none of", async () => {
    const context = baseContext({
      assets: [asset("img_first", "image", "first"), asset("img_last", "image", "last")],
    });
    await render(context);

    // Both modes the base checkpoint declares are selectable with `minimax_h3` picked.
    expect(MINIMAX.capabilities).toContain("text_to_video");
    expect(MINIMAX.capabilities).toContain("first_last_frame");
    await click([...container.querySelectorAll("button")].find((b) => b.textContent.includes("Render clip")));
    expect(context.createVideoJob.mock.calls[0][0].mode).toBe("text_to_video");

    // There is no guidance scale and no negative prompt on the pipeline, so neither control is
    // offered — hidden rather than present-and-discarded.
    await openAdvanced();
    expect(labelStartingWith("Guidance")).toBeFalsy();
    expect(labelStartingWith("Negative prompt")).toBeFalsy();
  });

  it("offers no reference picker at all on the text-to-video partition", async () => {
    // `minimax_h3` has no reference-conditioning path: all three caps are 0, so an engine can
    // never be handed conditioning it would silently ignore. The two partitions are separate
    // catalog entries with separate 18.78 GB checkpoints, so this is a wrong-checkpoint guard,
    // not a spare mode. (The picker narrows to models serving the active mode, so reaching r2v
    // with `minimax_h3` selected takes a catalog in which it is the only entry.)
    await render(baseContext({ videoModels: [MINIMAX] }));
    await click(modeTab("Reference → Video"));
    expect(selectLabelled("Model").value).toBe("minimax_h3");
    expect(fieldLabelled("Reference audio")).toBeFalsy();
    expect(fieldLabelled("Reference clips")).toBeFalsy();

    // …and the partition that DOES declare them gets them, so this is a per-entry fact rather
    // than the pickers being unreachable everywhere.
    await render(baseContext({ videoModels: [MINIMAX_REF] }));
    await click(modeTab("Reference → Video"));
    expect(fieldLabelled("Reference audio")).toBeTruthy();
    expect(fieldLabelled("Reference clips")).toBeTruthy();
  });

  // -------------------------------------------------------------- references

  it("renders image, clip and audio reference pickers for Ref2VA", async () => {
    await render(
      baseContext({
        videoModels: [MINIMAX_REF],
        assets: [asset("img_1", "image", "a"), asset("clip_1", "video", "b"), asset("aud_1", "audio", "c")],
      }),
    );
    await click(modeTab("Reference → Video"));
    expect(fieldLabelled("Reference images")).toBeTruthy();
    expect(fieldLabelled("Reference clips")).toBeTruthy();
    expect(fieldLabelled("Reference audio")).toBeTruthy();
  });

  it("gives Bernini's reference→video images only, not clips or audio", async () => {
    // The per-model half of the gate. Bernini declares no reference-clip or audio cap, so the
    // blanket clip default (8) must NOT be read as "this model takes reference clips" — its engine
    // encodes images and would ignore them, which is invisible in the output.
    await render(baseContext({ videoModels: [BERNINI] }));
    await click(modeTab("Reference → Video"));
    expect(fieldLabelled("Reference images")).toBeTruthy();
    expect(fieldLabelled("Reference clips")).toBeFalsy();
    expect(fieldLabelled("Reference audio")).toBeFalsy();
  });

  it("refuses an audio-only reference set with a reason, and submits it alongside an image", async () => {
    // sc-19574. This test asserted the OPPOSITE until now, because `validate_video_job` accepted
    // an audio-only set (sc-17159 widened its arm one list too far). The reference implementation
    // settles it: diffusers `MiniMaxH3` documents `MiniMaxH3AudioReference` as "never on its own …
    // It never reaches the conditioner", and `before_encoder.py` raises on
    // `set(kinds) == {"audio"}`. The worker refused it all along; the Studio enabling Generate for
    // it is how a user built a request the product presented as valid and was then told no.
    //
    // An ERROR, not the silent `inputs` requirement: the audio picker is visibly full, so an empty
    // image zone beside it reads as optional. The user must be told the rule, not left to infer it.
    const context = baseContext({
      videoModels: [MINIMAX_REF],
      assets: [asset("aud_1", "audio", "voice"), asset("img_1", "image", "portrait")],
    });
    await render(context);
    await click(modeTab("Reference → Video"));

    const generate = [...container.querySelectorAll("button")].find((b) =>
      b.textContent.includes("Render clip"),
    );
    expect(generate.disabled, "no references selected yet").toBe(true);

    await click(fieldLabelled("Reference audio").querySelector("button"));
    await click(
      [...document.querySelectorAll(".asset-picker-card")].find((card) => card.textContent.includes("voice")),
    );
    await click(
      [...document.querySelectorAll(".asset-picker-footer button")].find(
        (b) => b.textContent.trim() === "Use Selection",
      ),
    );

    const stillBlocked = [...container.querySelectorAll("button")].find((b) =>
      b.textContent.includes("Render clip"),
    );
    expect(stillBlocked.disabled, "audio alone cannot condition the picture").toBe(true);
    expect(
      container.textContent,
      "the refusal must SAY why — an empty image zone next to a full audio one does not",
    ).toContain("An audio reference can't be the only reference");
    expect(context.createVideoJob).not.toHaveBeenCalled();

    // …and adding an image clears it, with BOTH lists reaching the payload. Without this leg the
    // assertions above would pass on a studio that had stopped sending audio references at all.
    await click(fieldLabelled("Reference images").querySelector("button"));
    await click(
      [...document.querySelectorAll(".asset-picker-card")].find((card) =>
        card.textContent.includes("portrait"),
      ),
    );
    await click(
      [...document.querySelectorAll(".asset-picker-footer button")].find(
        (b) => b.textContent.trim() === "Use Selection",
      ),
    );

    const ready = [...container.querySelectorAll("button")].find((b) => b.textContent.includes("Render clip"));
    expect(ready.disabled, "an image + audio reference set is the shape Ref2VA serves").toBe(false);
    expect(container.textContent).not.toContain("An audio reference can't be the only reference");
    await click(ready);
    const payload = context.createVideoJob.mock.calls[0][0];
    expect(payload.mode).toBe("reference_to_video");
    expect(payload.referenceAudioAssetIds).toEqual(["aud_1"]);
    expect(payload.referenceAssetIds).toEqual(["img_1"]);
  });

  it("keeps `hasInputs` narrowed to a VISUAL r2v reference, with the audio-only refusal suppressed", async () => {
    // sc-19574. The test above proves the PAIR; this one proves `hasInputs`' r2v arm ALONE.
    //
    // The two guards are exactly co-extensive on this arm — the narrowing changes the answer only
    // when audio is outgoing and no image or clip is, which is precisely the condition
    // `audioOnlyReferenceSet` names — so through the screen neither can be observed while the other
    // holds, and re-widening `hasInputs` back to `referenceAssetIds.length >= 0` mutated SILENTLY.
    // Suppressing the validator's arm is what makes the remaining one gradeable.
    validationOverride.suppressAudioOnly = true;
    const context = baseContext({
      videoModels: [MINIMAX_REF],
      assets: [asset("aud_1", "audio", "voice"), asset("img_1", "image", "portrait")],
    });
    await render(context);
    await click(modeTab("Reference → Video"));

    await click(fieldLabelled("Reference audio").querySelector("button"));
    await click(
      [...document.querySelectorAll(".asset-picker-card")].find((card) =>
        card.textContent.includes("voice"),
      ),
    );
    await click(
      [...document.querySelectorAll(".asset-picker-footer button")].find(
        (b) => b.textContent.trim() === "Use Selection",
      ),
    );

    const blocked = [...container.querySelectorAll("button")].find((b) =>
      b.textContent.includes("Render clip"),
    );
    expect(blocked.disabled, "`hasInputs` alone must refuse an audio-only r2v set").toBe(true);
    // Non-vacuity in the direction that matters: the OTHER guard really is suppressed, so what is
    // being graded above is `hasInputs` and not the validator arm leaking through.
    expect(
      container.textContent,
      "the validator arm must be suppressed, or this test grades the wrong guard",
    ).not.toContain("An audio reference can't be the only reference");

    // …and the arm is not simply "always false for r2v": one visual reference admits it, still with
    // the validator arm suppressed.
    await click(fieldLabelled("Reference images").querySelector("button"));
    await click(
      [...document.querySelectorAll(".asset-picker-card")].find((card) =>
        card.textContent.includes("portrait"),
      ),
    );
    await click(
      [...document.querySelectorAll(".asset-picker-footer button")].find(
        (b) => b.textContent.trim() === "Use Selection",
      ),
    );
    const ready = [...container.querySelectorAll("button")].find((b) =>
      b.textContent.includes("Render clip"),
    );
    expect(ready.disabled, "a visual reference is what the arm requires").toBe(false);
  });

  it("does not strand an audio-reference refusal on a form that cannot clear it", async () => {
    // The picker unmounts with the mode; `referenceAudioAssetIds` does not — it is persisted per
    // project and restored on mount. Counting the RAW state therefore disabled Generate on
    // Text → Video with "…takes no reference audio clips, but 1 are selected. Remove them", while
    // the only control that could remove them had just left with the mode. That refusal survived
    // navigation AND a restart, and it is reachable only because this story shipped the picker —
    // the mirror image of the declared-but-undrivable class the story exists to close.
    const context = baseContext({
      videoModels: [MINIMAX, MINIMAX_REF],
      assets: [asset("aud_1", "audio", "voice")],
    });
    await render(context);

    await click(modeTab("Reference → Video"));
    expect(selectLabelled("Model").value).toBe("minimax_h3_ref");
    await click(fieldLabelled("Reference audio").querySelector("button"));
    await click(
      [...document.querySelectorAll(".asset-picker-card")].find((card) => card.textContent.includes("voice")),
    );
    await click(
      [...document.querySelectorAll(".asset-picker-footer button")].find(
        (b) => b.textContent.trim() === "Use Selection",
      ),
    );

    // The primary path: the tab the studio opens on. The model snaps to the base partition, which
    // declares an audio cap of zero, and the picker goes with the mode.
    await click(modeTab("Text → Video"));
    expect(selectLabelled("Model").value).toBe("minimax_h3");
    expect(MINIMAX.limits.maxReferenceAudioAssets).toBe(0);
    expect(fieldLabelled("Reference audio"), "the only control that could clear the selection").toBeFalsy();

    expect(container.textContent).not.toContain("reference audio clips");
    const generate = [...container.querySelectorAll("button")].find((b) => b.textContent.includes("Render clip"));
    expect(generate.disabled, "a cap this form cannot violate must not be able to refuse").toBe(false);

    // …and the selection does not ride along into a job for a model that conditions on no audio.
    await click(generate);
    const payload = context.createVideoJob.mock.calls[0][0];
    expect(payload.mode).toBe("text_to_video");
    expect(payload.referenceAudioAssetIds).toEqual([]);
  });

  // --------------------------------------------------------------- refine

  it("offers Refine for both H3 partitions and names the model the worker keys on (sc-17162)", async () => {
    // sc-17162 ships a TAILORED H3 refiner — an H3-keyed system prompt in the worker
    // (`prompt_refine_jobs.rs`, `build_rewrite_system_prompt`) standing in for the withheld hosted
    // `H3-Context-IR`. That branch is selected by the job payload's `modelId`, so the studio has to
    // (a) render the control at all for this family and (b) hand it the H3 catalog id. Declaration
    // is not reachability: a worker branch nothing routes to is not delivered.
    //
    // The control is gated only on `promptless`, which H3 does not declare — asserted off the
    // shipped manifest rather than assumed, so a future entry that DID declare it would fail here
    // instead of silently losing the refiner.
    //
    // Set-derived over the family, so the day a third partition ships it is covered.
    for (const model of manifestModels.filter((entry) => entry.family === "minimax-h3")) {
      expect(model.promptless, `${model.id} must not be promptless`).toBeFalsy();
      const refinePrompt = vi.fn(async () => "integrated_multimodal_description: a rewritten shot.");
      await render(baseContext({ videoModels: [shipped(model.id)], refinePrompt }));

      const button = [...container.querySelectorAll("button")].find(
        (node) => node.textContent.trim() === "Refine my prompt",
      );
      expect(button, `${model.id} must offer the Refine control`).toBeTruthy();

      const promptBox = container.querySelector('textarea[aria-label="Prompt"]');
      await act(async () => {
        const setter = Object.getOwnPropertyDescriptor(
          window.HTMLTextAreaElement.prototype,
          "value",
        ).set;
        setter.call(promptBox, "a fox on a snowy log");
        promptBox.dispatchEvent(new window.Event("input", { bubbles: true }));
      });
      await click(button);

      expect(refinePrompt).toHaveBeenCalledTimes(1);
      const call = refinePrompt.mock.calls[0][0];
      // The id the worker's family predicate reads. Without it every H3 refine falls back to the
      // generic rewrite rules and the tailored branch is dead code.
      expect(call.modelId).toBe(model.id);
      expect(call.workflow).toBe("video");
      expect(call.prompt).toBe("a fox on a snowy log");

      // Opt-in stays opt-in: the suggestion is offered for review and the composer is UNCHANGED
      // until Apply. This is the story's own acceptance criterion — no silent prompt rewriting.
      expect(container.querySelector(".refine-review")).toBeTruthy();
      expect(container.querySelector('textarea[aria-label="Prompt"]').value).toBe(
        "a fox on a snowy log",
      );
      const apply = [...container.querySelectorAll(".refine-review-actions button")].find(
        (node) => node.textContent.trim() === "Apply",
      );
      expect(apply, "the rewrite is applied by an explicit button, not on arrival").toBeTruthy();
      await click(apply);
      expect(container.querySelector('textarea[aria-label="Prompt"]').value).toBe(
        "integrated_multimodal_description: a rewritten shot.",
      );
    }
  });

  // ------------------------------------------------------------- attribution

  it("carries the licence-required attribution on the studio", async () => {
    // MiniMax H3 Community License §IV.2. Read from the manifest — never a second hard-coded copy
    // of a string the licence specifies.
    await render(baseContext({ videoModels: [MINIMAX, BERNINI] }));
    const line = container.querySelector(".studio-model-attribution");
    expect(line).toBeTruthy();
    expect(line.textContent).toBe(MINIMAX.ui.attribution);
    expect(line.textContent).toContain("MiniMax H3");

    // Per-model, never a global banner: a model that declares none renders no line.
    await act(async () => setSelect(selectLabelled("Model"), "bernini"));
    expect(BERNINI.ui.attribution).toBeUndefined();
    expect(container.querySelector(".studio-model-attribution")).toBeNull();
  });
});
