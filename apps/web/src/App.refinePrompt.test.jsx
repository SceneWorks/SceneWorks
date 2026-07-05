import { act } from "react";
import { createRoot } from "react-dom/client";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { withImageStudioContext, settle } from "./main.testSupport.jsx";

describe("refine my prompt (sc-2041)", () => {
  let container;
  let root;

  beforeEach(() => {
    global.IS_REACT_ACT_ENVIRONMENT = true;
    container = document.createElement("div");
    document.body.appendChild(container);
    global.fetch = vi.fn(() => Promise.resolve({ ok: true, text: async () => "# Guide\n\nWrite vividly." }));
  });

  afterEach(() => {
    act(() => root?.unmount());
    container.remove();
    vi.restoreAllMocks();
  });

  it("refines the image prompt and applies it to the textarea on Apply", async () => {
    const refinePrompt = vi.fn(async () => "A cinematic neon street at midnight, rain-slick.");
    root = createRoot(container);
    await act(async () => {
      root.render(
        withImageStudioContext({
          activeProject: { id: "project-1", name: "Noir" },
          assets: [],
          characters: [],
          createImageJob: () => {},
          refinePrompt,
          deleteAsset: () => {},
          gpuOptions: ["auto"],
          imageModels: [
            {
              id: "z_image_turbo",
              name: "Z-Image",
              type: "image",
              capabilities: ["text_to_image"],
              ui: { promptGuide: { title: "Z Guide", path: "/prompt-guides/z-image-turbo.md" } },
            },
          ],
          latestAssets: [],
          localJobs: [],
          loras: [],
          onPreview: () => {},
          purgeAsset: () => {},
          requestedGpu: "auto",
          selectedAsset: null,
          setRequestedGpu: () => {},
          updateAssetStatus: () => {},
        }),
      );
    });

    // Prompt tools (UI-refinement 1b): the "Refine my prompt" tile toggles the panel open;
    // the actual refine runs from the RefinePromptControl button revealed inside it.
    const refineTile = [...document.body.querySelectorAll(".prompt-tool")].find((button) =>
      button.textContent.includes("Refine my prompt"),
    );
    await act(async () => {
      refineTile.click();
    });
    await settle();
    const refine = document.body.querySelector(".refine-button");
    await act(async () => {
      refine.click();
    });
    await settle();

    expect(refinePrompt).toHaveBeenCalledWith(expect.objectContaining({
      prompt: "A cinematic frame of a neon street at midnight",
      modelId: "z_image_turbo",
      workflow: "image",
      guide: "# Guide\n\nWrite vividly.",
      signal: expect.any(AbortSignal),
    }));
    expect(document.body.querySelector(".refine-review-text").textContent).toBe("A cinematic neon street at midnight, rain-slick.");
    // Original prompt unchanged until the user applies.
    expect(document.body.querySelector(".prompt-input").value).toBe("A cinematic frame of a neon street at midnight");

    const apply = [...document.body.querySelectorAll("button")].find((button) => button.textContent.trim() === "Apply");
    await act(async () => {
      apply.click();
    });
    await settle();

    expect(document.body.querySelector(".prompt-input").value).toBe("A cinematic neon street at midnight, rain-slick.");
    expect(document.body.querySelector(".refine-review")).toBeNull();
  });
});
