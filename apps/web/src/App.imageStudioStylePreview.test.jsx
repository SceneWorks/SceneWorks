import { act } from "react";
import { createRoot } from "react-dom/client";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";

import { withImageStudioContext, settle } from "./main.testSupport.jsx";
import { STYLE_GROUPS, styleHintForId } from "./data/styleCatalog.js";

// sc-13131 — End-to-end wiring guard for the live Style preview. Renders the real ImageStudio,
// types a prompt, picks a style through the actual StylePicker, and asserts that the on-screen
// preview text equals the `prompt` the run submits — BYTE-FOR-BYTE. This is the DoD's discriminating
// check: it fails if the preview and the payload ever diverge (e.g. the preview re-derived the
// composition instead of reusing buildJobRequest, or the component mangled the string).
describe("Image Studio — live Style preview parity (sc-13131)", () => {
  let container;
  let root;
  let createImageJob;

  const IMAGE_MODEL = { id: "z_image_turbo", name: "Z-Image", type: "image", family: "z-image" };
  const firstStyle = STYLE_GROUPS[0].styles[0]; // { id: "ghibli-style", name: "Ghibli Style", ... }

  beforeEach(() => {
    global.IS_REACT_ACT_ENVIRONMENT = true;
    window.localStorage.clear();
    container = document.createElement("div");
    document.body.appendChild(container);
    createImageJob = vi.fn(() => Promise.resolve({ id: "image-job-1" }));
  });

  afterEach(() => {
    act(() => {
      root?.unmount();
    });
    container.remove();
    vi.restoreAllMocks();
  });

  async function renderStudio() {
    root = createRoot(container);
    await act(async () => {
      root.render(
        withImageStudioContext({
          activeProject: { id: "project-1", name: "Noir" },
          assets: [],
          characters: [],
          createImageJob,
          deleteAsset: () => {},
          purgeAsset: () => {},
          gpuOptions: ["auto"],
          imageModels: [IMAGE_MODEL],
          latestAssets: [],
          localJobs: [],
          loras: [],
          onPreview: () => {},
          requestedGpu: "auto",
          selectedAsset: null,
          setRequestedGpu: () => {},
          updateAssetStatus: () => {},
        }),
      );
    });
    await settle();
  }

  async function typePrompt(value) {
    const textarea = document.body.querySelector('textarea[aria-label="Prompt"]');
    const setter = Object.getOwnPropertyDescriptor(window.HTMLTextAreaElement.prototype, "value").set;
    await act(async () => {
      setter.call(textarea, value);
      textarea.dispatchEvent(new Event("input", { bubbles: true }));
    });
    await settle();
  }

  // sc-13171: the picker is a two-level cascade — open it, drill into the style's group, then pick
  // the sub-style. We resolve the owning group from the shipped catalog so the flow mirrors a user.
  // The menu opens at level 1 (group list), or jumps straight to level 2 when a style is already
  // selected (the studio saved-state persists styleId), so normalize back to level 1 first.
  async function selectStyle(name) {
    const owningGroup = STYLE_GROUPS.find((group) => group.styles.some((style) => style.name === name));
    await act(async () => {
      document.body.querySelector('button[aria-label="Style"]').click();
    });
    await settle();
    const back = document.body.querySelector(".style-picker-back");
    if (back) {
      await act(async () => {
        back.click();
      });
      await settle();
    }
    const groupNav = document.body.querySelector(`button[title="Browse ${owningGroup.name} styles"]`);
    await act(async () => {
      groupNav.click();
    });
    await settle();
    const option = [...document.body.querySelectorAll('[role="option"]')].find(
      (button) => button.textContent.trim() === name,
    );
    await act(async () => {
      option.click();
    });
    await settle();
  }

  const previewText = () =>
    document.body.querySelector('[data-testid="styled-prompt-preview"] .preset-stack-prompt p')?.textContent;

  async function submitAndReadPayloadPrompt() {
    await act(async () => {
      document.body.querySelector(".image-studio form").requestSubmit();
    });
    await settle();
    expect(createImageJob).toHaveBeenCalledTimes(1);
    return createImageJob.mock.calls[0][0].prompt;
  }

  it("shows no style preview until a style is selected, then submits the plain prompt", async () => {
    await renderStudio();
    await typePrompt("a fox in the snow");
    // No style yet → the affordance is hidden (nothing extra to preview).
    expect(document.body.querySelector('[data-testid="styled-prompt-preview"]')).toBeNull();
    const submittedPrompt = await submitAndReadPayloadPrompt();
    expect(submittedPrompt).toBe("a fox in the snow");
  });

  it("previewed prompt equals the submitted prompt byte-for-byte for a selected style", async () => {
    await renderStudio();
    await typePrompt("a fox in the snow");
    await selectStyle(firstStyle.name);

    const shown = previewText();
    expect(shown).toContain("Style: ");
    expect(shown).toContain("\nDescription: a fox in the snow");

    const submittedPrompt = await submitAndReadPayloadPrompt();
    // The DoD: what the user SEES is exactly what is SENT.
    expect(shown).toBe(submittedPrompt);
  });

  it("swaps the suggestion pills for a single tailored style hint when a style is selected (sc-13366)", async () => {
    await renderStudio();
    // No style yet → the usual multi-pill scene-suggestion row.
    const before = [...document.body.querySelectorAll(".suggestion-row .suggestion")];
    expect(before.length).toBeGreaterThan(1);

    await selectStyle(firstStyle.name);

    // Style selected → exactly ONE pill, carrying that style's tailored subject prompt.
    const after = [...document.body.querySelectorAll(".suggestion-row .suggestion")];
    expect(after).toHaveLength(1);
    const hint = styleHintForId(firstStyle.id);
    expect(hint).toEqual(expect.any(String));
    expect(after[0].textContent).toContain(hint);

    // Clicking the hint fills the prompt box (same as any suggestion chip).
    await act(async () => {
      after[0].click();
    });
    await settle();
    expect(document.body.querySelector('textarea[aria-label="Prompt"]').value).toBe(hint);
  });

  it("blocks a styled batch before submit and identifies every over-cap resolved item", async () => {
    const longPrompt = "x".repeat(3500);
    window.localStorage.setItem(
      "sceneworks-studio-image-project-1",
      JSON.stringify({ batchMode: true, batchPromptsText: `${longPrompt}\nshort\n${longPrompt}` }),
    );
    await renderStudio();
    await selectStyle(firstStyle.name);

    const run = [...document.body.querySelectorAll("button")].find((button) => button.textContent.includes("Run batch"));
    await act(async () => {
      run.click();
    });
    await settle();

    expect(createImageJob).not.toHaveBeenCalled();
    const warning = [...document.body.querySelectorAll(".batch-warning")].find((node) =>
      node.textContent.includes("character limit"),
    );
    expect(warning?.textContent).toMatch(/^Batch prompts 1 \(\d+\/4000\), 3 \(\d+\/4000\) exceed/);
  });

  it("updates live as the prompt changes, staying equal to the payload", async () => {
    await renderStudio();
    await typePrompt("a fox in the snow");
    await selectStyle(firstStyle.name);
    const first = previewText();

    await typePrompt("a wolf on a ridge at dusk");
    const second = previewText();
    expect(second).not.toBe(first);
    expect(second).toContain("\nDescription: a wolf on a ridge at dusk");

    const submittedPrompt = await submitAndReadPayloadPrompt();
    expect(second).toBe(submittedPrompt);
  });

  it("MERGE case: the user's own Style: line is visible in the preview and matches the payload", async () => {
    await renderStudio();
    await typePrompt("Style: neon rimlight\na fox in the snow");
    await selectStyle(firstStyle.name);

    const shown = previewText();
    // Catalog style leads, the user's own style words merge after ", " in the same Style block.
    expect(shown).toContain(", neon rimlight");
    expect(shown).toContain("\nDescription: a fox in the snow");

    const submittedPrompt = await submitAndReadPayloadPrompt();
    expect(shown).toBe(submittedPrompt);
  });
});
