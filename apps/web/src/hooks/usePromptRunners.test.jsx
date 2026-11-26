import React, { act } from "react";
import { createRoot } from "react-dom/client";
import { afterEach, describe, expect, it, vi } from "vitest";
import { usePromptRunners } from "./usePromptRunners.js";
import { useStructuredPrompt } from "../screens/imageStudio/useStructuredPrompt.js";
import { pollJobToCompletion } from "../pollJob.js";

vi.mock("../pollJob.js", () => ({ pollJobToCompletion: vi.fn(async () => " A scene description ") }));

describe("prompt option transport", () => {
  let root;
  let container;
  afterEach(() => {
    act(() => root?.unmount());
    container?.remove();
    vi.clearAllMocks();
  });

  it.each([undefined, "q8"])("carries exclusions from the reference callback to the API, tier %s", async (variant) => {
    global.IS_REACT_ACT_ENVIRONMENT = true;
    container = document.createElement("div");
    document.body.appendChild(container);
    root = createRoot(container);
    let callbacks;
    let runners;
    function Harness() {
      runners = usePromptRunners({ token: "test-token" });
      callbacks = useStructuredPrompt({ imageDescribe: runners.imageDescribe,
        activeProject: { id: "project-1" }, selectedModel: { captionStyle: "tags" },
        setMagicPromptBackend: vi.fn() });
      return null;
    }
    await act(async () => root.render(<Harness />));
    const promptExclusions = ["identifyingCharacteristics", "clothingAndAccessories", "colorMode", "style"];
    const result = await callbacks.onImageDescribe(["ref-1", "ref-2"], variant, promptExclusions);
    expect(result).toBe("A scene description");
    expect(pollJobToCompletion).toHaveBeenLastCalledWith(expect.objectContaining({
      body: expect.objectContaining({ task: "image_describe", sourceAssetIds: ["ref-1", "ref-2"],
        projectId: "project-1", captionStyle: "tags", promptExclusions,
        ...(variant ? { variant, model: "sensenova/SenseNova-Loop-9B" } : {}),
      }),
    }));
    await runners.refinePrompt({ prompt: "a scene", workflow: "image", promptExclusions });
    expect(pollJobToCompletion).toHaveBeenLastCalledWith(expect.objectContaining({
      body: expect.objectContaining({ prompt: "a scene", promptExclusions }),
    }));
    await callbacks.onImageDescribe("ref-1");
    expect(pollJobToCompletion.mock.calls.at(-1)[0].body.promptExclusions).toBeUndefined();
  });
});
