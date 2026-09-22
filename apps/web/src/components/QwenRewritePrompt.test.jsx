import React, { act } from "react";
import { createRoot } from "react-dom/client";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { QwenRewritePromptControl } from "./QwenRewritePromptControl.jsx";

// Qwen-Image 2.1's official prompt rewriter, exercised with a FIXTURE reply (sc-24113).
//
// Every assertion here is one of the story's acceptance criteria rather than a rendering detail:
// the rewrite is EDITABLE, it carries an aspect suggestion the user accepts or ignores SEPARATELY,
// and the user's own prompt is never replaced until they press Apply.

async function settle() {
  await act(async () => {
    for (let index = 0; index < 8; index += 1) {
      await Promise.resolve();
    }
  });
}

function buttonByText(container, text) {
  return [...container.querySelectorAll("button")].find(
    (button) => button.textContent.trim() === text,
  );
}

function buttonContaining(container, text) {
  return [...container.querySelectorAll("button")].find((button) =>
    button.textContent.includes(text),
  );
}

// A T2I reply as the worker hands it to the client: the prompt already parsed out of Qwen's JSON
// envelope, with the aspect suggestion beside it as a legal preset.
const T2I_RESULT = {
  refinedPrompt: "A wide harbour at dusk, fishing boats moored along a stone quay.",
  rewriteSuggestion: {
    whRatio: "16:9",
    ratioFollow: "",
    resolution: "2752x1536",
    rewriter: "t2i",
    rewriterModelId: "qwen_image_2_1_pe_t2i",
  },
};

describe("QwenRewritePromptControl (sc-24113)", () => {
  let container;
  let root;

  beforeEach(() => {
    global.IS_REACT_ACT_ENVIRONMENT = true;
    container = document.createElement("div");
    document.body.appendChild(container);
  });

  afterEach(() => {
    act(() => root?.unmount());
    container.remove();
    vi.restoreAllMocks();
  });

  function render(ui) {
    root = createRoot(container);
    act(() => root.render(ui));
  }

  function draft() {
    return container.querySelector("textarea.qwen-rewrite-draft");
  }

  it("never runs on its own — rewriting is a user action", async () => {
    const rewritePrompt = vi.fn(async () => T2I_RESULT);
    render(
      <QwenRewritePromptControl
        modelId="qwen_image_2_1"
        onApply={() => {}}
        prompt="a cinematic harbour"
        rewritePrompt={rewritePrompt}
      />,
    );
    await settle();
    // Unlike the generic refiner's prompt-tool tile, there is no autoStart. Mounting the panel must
    // not call the model: an automatic rewrite is exactly what the story forbids.
    expect(rewritePrompt).not.toHaveBeenCalled();
    expect(draft()).toBeNull();
    expect(buttonContaining(container, "Rewrite prompt")).toBeTruthy();
  });

  it("offers the rewrite as EDITABLE text and applies what the user edited", async () => {
    const onApply = vi.fn();
    render(
      <QwenRewritePromptControl
        modelId="qwen_image_2_1"
        onApply={onApply}
        prompt="a cinematic harbour"
        rewritePrompt={async () => T2I_RESULT}
      />,
    );
    await act(async () => buttonContaining(container, "Rewrite prompt").click());
    await settle();

    // A TEXTAREA, not a paragraph. This is the difference from `RefinePromptControl`, whose
    // suggestion is read-only take-it-or-leave-it.
    const box = draft();
    expect(box).toBeTruthy();
    expect(box.tagName).toBe("TEXTAREA");
    expect(box.value).toBe(T2I_RESULT.refinedPrompt);

    // Nothing has been applied yet — the user's own prompt is untouched behind this panel.
    expect(onApply).not.toHaveBeenCalled();

    // Edit it, then apply. What lands is the EDITED text, not the model's.
    await act(async () => {
      const setter = Object.getOwnPropertyDescriptor(
        window.HTMLTextAreaElement.prototype,
        "value",
      ).set;
      setter.call(box, "A wide harbour at dawn, my own words.");
      box.dispatchEvent(new Event("input", { bubbles: true }));
    });
    await act(async () => buttonByText(container, "Apply").click());
    expect(onApply).toHaveBeenCalledWith("A wide harbour at dawn, my own words.");
  });

  it("keeps the original when the user declines, without ever calling onApply", async () => {
    const onApply = vi.fn();
    render(
      <QwenRewritePromptControl
        modelId="qwen_image_2_1"
        onApply={onApply}
        prompt="a cinematic harbour"
        rewritePrompt={async () => T2I_RESULT}
      />,
    );
    await act(async () => buttonContaining(container, "Rewrite prompt").click());
    await settle();
    await act(async () => buttonByText(container, "Keep original").click());
    expect(onApply).not.toHaveBeenCalled();
    expect(draft()).toBeNull();
  });

  it("offers the aspect ratio as a SEPARATE accept", async () => {
    const onApply = vi.fn();
    const onApplyResolution = vi.fn();
    render(
      <QwenRewritePromptControl
        modelId="qwen_image_2_1"
        onApply={onApply}
        onApplyResolution={onApplyResolution}
        prompt="a cinematic harbour"
        rewritePrompt={async () => T2I_RESULT}
      />,
    );
    await act(async () => buttonContaining(container, "Rewrite prompt").click());
    await settle();

    expect(container.textContent).toContain("16:9");
    // It names a LEGAL PRESET — one of the model's own seven — so accepting it is the same as
    // picking it from the Aspect menu by hand.
    expect(container.textContent).toContain("2752x1536");
    await act(async () => buttonByText(container, "Use this aspect ratio").click());
    expect(onApplyResolution).toHaveBeenCalledWith("2752x1536");
    // The two decisions are independent: taking the ratio must not apply the prompt.
    expect(onApply).not.toHaveBeenCalled();
  });

  it("selects the editing rewriter from the request, and says which reference the output follows", async () => {
    const rewritePrompt = vi.fn(async () => ({
      refinedPrompt: "Replace the grey backdrop with a sunlit alley.",
      rewriteSuggestion: {
        whRatio: "",
        ratioFollow: "<image2>",
        followsReferenceIndex: 1,
        rewriter: "i2i",
        rewriterModelId: "qwen_image_2_1_pe_i2i",
      },
    }));
    render(
      <QwenRewritePromptControl
        modelId="qwen_image_2_1"
        onApply={() => {}}
        onApplyResolution={() => {}}
        prompt="swap the backdrop"
        referenceAssetIds={["ref_a", "ref_b"]}
        rewritePrompt={rewritePrompt}
      />,
    );
    // References present ⇒ the EDITING rewriter, decided by the request rather than by a picker.
    // The button copy says so before anything runs.
    expect(buttonContaining(container, "Rewrite edit instruction")).toBeTruthy();
    await act(async () => buttonContaining(container, "Rewrite edit instruction").click());
    await settle();

    // The ordered list is sent verbatim: the rewrite's `<imageN>` numbering only means anything
    // against the list the render will condition on.
    expect(rewritePrompt.mock.calls[0][0].sourceAssetIds).toEqual(["ref_a", "ref_b"]);
    // An edit that follows an input image offers NO aspect change — there is no ratio to take.
    expect(buttonByText(container, "Use this aspect ratio")).toBeFalsy();
    // ... and it names the reference in the user's own 1-based numbering, not the raw token.
    expect(container.textContent).toContain("image 2");
  });

  it("offers the download when the checkpoint is missing, and says the model works without it", async () => {
    const onDownloadRewriteModel = vi.fn(async () => ({ id: "job_1" }));
    render(
      <QwenRewritePromptControl
        modelId="qwen_image_2_1"
        onApply={() => {}}
        onDownloadRewriteModel={onDownloadRewriteModel}
        prompt="a cinematic harbour"
        rewriteModel={{
          installState: "missing",
          name: "Qwen Image 2.1 Prompt Rewriter (text-to-image)",
          downloadSizeBytes: 18839810614,
        }}
        rewritePrompt={async () => {
          throw new Error("prompt-refine model path snapshot is not cached");
        }}
      />,
    );
    await act(async () => buttonContaining(container, "Rewrite prompt").click());
    await settle();

    // The copy has to say the image model is unaffected — "no degraded path" means the user is
    // never led to believe generation needs this.
    expect(container.textContent).toContain("generates");
    expect(container.textContent).toContain("18.8 GB");
    await act(async () => buttonByText(container, "Download rewriter").click());
    expect(onDownloadRewriteModel).toHaveBeenCalled();
  });

  it("surfaces a failure instead of offering an empty rewrite to Apply", async () => {
    const onApply = vi.fn();
    render(
      <QwenRewritePromptControl
        modelId="qwen_image_2_1"
        onApply={onApply}
        prompt="a cinematic harbour"
        rewritePrompt={async () => {
          throw new Error("The rewriter returned an empty prompt.");
        }}
      />,
    );
    await act(async () => buttonContaining(container, "Rewrite prompt").click());
    await settle();
    expect(container.textContent).toContain("empty prompt");
    expect(draft()).toBeNull();
    expect(onApply).not.toHaveBeenCalled();
  });
});
