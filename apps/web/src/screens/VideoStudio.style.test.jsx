import React, { act } from "react";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { click, mountRoot, unmountRoot } from "../testUtils/dom.js";

vi.mock("../api.js", async (importOriginal) => {
  const actual = await importOriginal();
  return {
    ...actual,
    apiFetch: vi.fn(async () => ({})),
  };
});

import { AppContext } from "../context/AppContext.js";
import { VideoStudio } from "./VideoStudio.jsx";
import { composeStyledPrompt } from "../styleComposer.js";
import { STYLE_GROUPS, styleTextForId } from "../data/styleCatalog.js";

// sc-13136 — the Style axis (proven in the Image Studio) mirrored into the Video Studio. These
// tests drive the REAL StylePicker in the rendered screen and assert the outgoing video job the
// client submits, so they cover the DoD end-to-end: a picked style composes the video `prompt` as
// `Subject:/Style:`, sets the client-authoritative flag, records the recipe round-trip fields,
// and clears back to an untouched prompt (identity). Video has no structured-caption/batch modes,
// so the only exclusion is a promptless model (covered elsewhere).

const LTX = {
  id: "ltx_2_3",
  name: "LTX 2.3",
  type: "video",
  family: "ltx-video",
  capabilities: ["image_to_video", "text_to_video", "first_last_frame"],
  defaults: { duration: 6, resolution: "768x512", fps: 25 },
  limits: {},
  quantization: {},
  loraCompatibility: {},
  ui: {},
};

const DEFAULT_PROMPT = "Camera slowly pushes in while the scene comes alive";
// A real catalog group + sub-style (styles.json). The group's first sub-style; both ids resolve
// through styleTextForId. Guarded below so a catalog rename fails loudly instead of silently.
const GROUP_NAME = STYLE_GROUPS[0].name;
const SUBSTYLE = STYLE_GROUPS[0].styles[0];

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
    videoModels: [LTX],
    ...overrides,
  };
}

const buttonWithText = (root, text) =>
  [...root.querySelectorAll("button")].find((b) => b.textContent.trim() === text);
const buttonContaining = (root, sel, text) =>
  [...root.querySelectorAll(sel)].find((b) => b.textContent.includes(text));

async function selectStyle(container, groupName, styleName) {
  await click(container.querySelector('button[aria-label="Style"]'));
  await click(buttonContaining(container, "button.style-picker-group-nav", groupName));
  await click(buttonWithText(container, styleName));
}

async function clearStyle(container) {
  await click(container.querySelector('button[aria-label="Style"]'));
  // The picker reopens at the selected group's level 2; None lives at the group list, so step back.
  const back = container.querySelector("button.style-picker-back");
  if (back) {
    await click(back);
  }
  await click(buttonContaining(container, "button.compact-selector-item", "Pass-through"));
}

describe("VideoStudio — Style Catalog integration (sc-13136)", () => {
  let container;
  let root;

  beforeEach(() => {
    global.IS_REACT_ACT_ENVIRONMENT = true;
    window.localStorage.clear();
    ({ container, root } = mountRoot());
    // Guards against a stale catalog id — the test would otherwise silently pass a null styleText.
    expect(typeof styleTextForId(SUBSTYLE.id)).toBe("string");
  });

  afterEach(async () => {
    await unmountRoot(root, container);
    vi.clearAllMocks();
  });

  async function render(context) {
    await act(async () => {
      root.render(
        <AppContext.Provider value={context}>
          <VideoStudio />
        </AppContext.Provider>,
      );
    });
    await act(async () => {});
  }

  it("no style selected → the outgoing prompt is the untouched user prompt (identity)", async () => {
    const context = baseContext();
    await render(context);
    await click(buttonWithText(container, "Render clip"));

    expect(context.createVideoJob).toHaveBeenCalledTimes(1);
    const payload = context.createVideoJob.mock.calls[0][0];
    expect(payload.prompt).toBe(DEFAULT_PROMPT);
    expect(payload.presetPromptResolvedClientSide).toBeUndefined();
    expect(payload.advanced.styleId).toBeUndefined();
    expect(payload.advanced.stylePrompt).toBeUndefined();
  });

  it("style selected → prompt composes as Subject:/Style: and sets the client flag", async () => {
    const context = baseContext();
    await render(context);
    await selectStyle(container, GROUP_NAME, SUBSTYLE.name);
    await click(buttonWithText(container, "Render clip"));

    expect(context.createVideoJob).toHaveBeenCalledTimes(1);
    const payload = context.createVideoJob.mock.calls[0][0];
    const styleText = styleTextForId(SUBSTYLE.id);
    expect(payload.prompt).toBe(composeStyledPrompt({ styleText, userPrompt: DEFAULT_PROMPT }));
    expect(payload.prompt).toBe(`Subject: ${DEFAULT_PROMPT}\nStyle: ${styleText}`);
    // Client-authoritative: the server must not re-fold the composed prompt.
    expect(payload.presetPromptResolvedClientSide).toBe(true);
    // Recipe round-trip fields ride advanced → rawAdapterSettings.
    expect(payload.advanced.styleId).toBe(SUBSTYLE.id);
    expect(payload.advanced.stylePrompt).toBe(DEFAULT_PROMPT);
    expect(payload.advanced.stylePrompt).not.toBe(payload.prompt);
  });

  it("live preview shows the EXACT composed string that is submitted (no drift)", async () => {
    const context = baseContext();
    await render(context);
    await selectStyle(container, GROUP_NAME, SUBSTYLE.name);

    const preview = container.querySelector('[data-testid="styled-prompt-preview"]');
    expect(preview).toBeTruthy();
    const styleText = styleTextForId(SUBSTYLE.id);
    const composed = composeStyledPrompt({ styleText, userPrompt: DEFAULT_PROMPT });
    expect(preview.textContent).toContain(composed);

    // Submit and prove the previewed string is byte-identical to what is sent.
    await click(buttonWithText(container, "Render clip"));
    expect(context.createVideoJob.mock.calls[0][0].prompt).toBe(composed);
  });

  it("clearing the style back to None → identity again, flag and round-trip fields gone", async () => {
    const context = baseContext();
    await render(context);
    await selectStyle(container, GROUP_NAME, SUBSTYLE.name);
    // The preview appears while a style is active…
    expect(container.querySelector('[data-testid="styled-prompt-preview"]')).toBeTruthy();
    await clearStyle(container);
    // …and disappears once cleared (StyledPromptPreview renders nothing when inactive).
    expect(container.querySelector('[data-testid="styled-prompt-preview"]')).toBeNull();

    await click(buttonWithText(container, "Render clip"));
    const payload = context.createVideoJob.mock.calls[0][0];
    expect(payload.prompt).toBe(DEFAULT_PROMPT);
    expect(payload.presetPromptResolvedClientSide).toBeUndefined();
    expect(payload.advanced.styleId).toBeUndefined();
  });
});

// sc-13136 — the recipe REPLAY/rehydrate side of the Style axis (matching Image Studio's
// imageJobRequest.style.test.js replay cases). A recorded video recipe carries the picked style id
// and the RAW pre-style prompt on `rawAdapterSettings.{styleId, stylePrompt}`. On replay the
// VideoStudio effect (VideoStudio.jsx ~L710-713) re-selects the picker to that id and seeds the box
// with the RAW prompt — so the very next submit recomposes the byte-identical Subject:/Style:
// prompt with EXACTLY ONE Style: block (no double-wrap). These tests drive a real replay through
// context.studioLaunch and assert the outgoing job, for a sub-style id AND a group id, plus the
// hardening case where the id is present but the raw prompt was not recorded.
describe("VideoStudio — Style Catalog recipe replay (sc-13136)", () => {
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
  });

  async function render(context) {
    await act(async () => {
      root.render(
        <AppContext.Provider value={context}>
          <VideoStudio />
        </AppContext.Provider>,
      );
    });
    await act(async () => {});
  }

  const RAW = "Camera drifts over a quiet harbor at dawn";
  // A recorded recipe as the viewer's "Use this recipe" hands it to the studio: the composed
  // top-level `prompt` plus the round-trip facts on rawAdapterSettings.
  const replayLaunch = (rawAdapterSettings, composedPrompt) => ({
    view: "Video",
    id: `replay-${Math.random()}`,
    recipe: {
      mode: "text_to_video",
      prompt: composedPrompt,
      normalizedSettings: {},
      rawAdapterSettings,
    },
  });

  // Both a sub-style id and a group id must round-trip. styleTextForId resolves each; the guard
  // fails loudly if the catalog renames out from under the test.
  for (const [label, styleId] of [
    ["sub-style id", SUBSTYLE.id],
    ["group id", STYLE_GROUPS[0].id],
  ]) {
    it(`replay of a recorded recipe (${label}) recomposes the identical prompt — no double-wrap`, async () => {
      const styleText = styleTextForId(styleId);
      expect(typeof styleText).toBe("string"); // guards a stale catalog id
      const composedOriginal = composeStyledPrompt({ styleText, userPrompt: RAW });
      // Sanity: the "original" carries exactly one Style: block to begin with.
      expect(composedOriginal.match(/^Style:/gm)?.length ?? 0).toBe(1);

      const context = baseContext({
        // The recipe recorded the RAW pre-style prompt, NOT the composed one.
        studioLaunch: replayLaunch({ styleId, stylePrompt: RAW }, composedOriginal),
      });
      await render(context);

      await click(buttonWithText(container, "Render clip"));
      expect(context.createVideoJob).toHaveBeenCalledTimes(1);
      const payload = context.createVideoJob.mock.calls[0][0];

      // The next submit recomposes the byte-identical prompt…
      expect(payload.prompt).toBe(composedOriginal);
      // …with EXACTLY one Style: block — the composer did not wrap an already-composed prompt.
      expect(payload.prompt.match(/^Style:/gm)?.length ?? 0).toBe(1);
      // The picker was re-selected and the round-trip fields ride advanced again (still RAW).
      expect(payload.presetPromptResolvedClientSide).toBe(true);
      expect(payload.advanced.styleId).toBe(styleId);
      expect(payload.advanced.stylePrompt).toBe(RAW);
    });
  }

  it("hardening: styleId present but stylePrompt missing → selection cleared, composed prompt used as-is (no double-wrap)", async () => {
    const styleText = styleTextForId(SUBSTYLE.id);
    const composedOriginal = composeStyledPrompt({ styleText, userPrompt: RAW });

    const context = baseContext({
      // styleId recorded, but the raw pre-style prompt was NOT — the effect must clear the
      // selection and fall back to recipe.prompt rather than re-composing over the composed prompt.
      studioLaunch: replayLaunch({ styleId: SUBSTYLE.id }, composedOriginal),
    });
    await render(context);

    // The picker was cleared, so no live preview is rendered.
    expect(container.querySelector('[data-testid="styled-prompt-preview"]')).toBeNull();

    await click(buttonWithText(container, "Render clip"));
    const payload = context.createVideoJob.mock.calls[0][0];
    // The composed recipe prompt passes through untouched — one Style: block, not two.
    expect(payload.prompt).toBe(composedOriginal);
    expect(payload.prompt.match(/^Style:/gm)?.length ?? 0).toBe(1);
    // No style applied on submit, so neither the client flag nor the round-trip fields are set.
    expect(payload.presetPromptResolvedClientSide).toBeUndefined();
    expect(payload.advanced.styleId).toBeUndefined();
    expect(payload.advanced.stylePrompt).toBeUndefined();
  });
});
