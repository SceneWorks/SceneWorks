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
// `Style:/Description:`, sets the client-authoritative flag, records the recipe round-trip fields,
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

  it("style selected → prompt composes as Style:/Description: and sets the client flag", async () => {
    const context = baseContext();
    await render(context);
    await selectStyle(container, GROUP_NAME, SUBSTYLE.name);
    await click(buttonWithText(container, "Render clip"));

    expect(context.createVideoJob).toHaveBeenCalledTimes(1);
    const payload = context.createVideoJob.mock.calls[0][0];
    const styleText = styleTextForId(SUBSTYLE.id);
    expect(payload.prompt).toBe(composeStyledPrompt({ styleText, userPrompt: DEFAULT_PROMPT }));
    expect(payload.prompt).toBe(`Style: ${styleText}\nDescription: ${DEFAULT_PROMPT}`);
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
