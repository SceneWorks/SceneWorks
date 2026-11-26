import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";

const feedbackState = vi.hoisted(() => ({ value: { state: "valid" } }));
vi.mock("../hooks/usePortableFeedback.js", () => ({ usePortableFeedback: (_kind, _payload, _entry, enabled = true) => enabled ? feedbackState.value : null }));
import { settle } from "../testUtils/dom.js";
import React, { act } from "react";
import { createRoot } from "react-dom/client";
import { RefinePromptControl } from "./RefinePromptControl.jsx";
import { AppContext } from "../context/AppContext.js";



function refineButton(container) {
  return [...container.querySelectorAll("button")].find((button) => button.textContent.includes("Refine my prompt"));
}

function buttonByText(container, text) {
  return [...container.querySelectorAll("button")].find((button) => button.textContent.trim() === text);
}

describe("RefinePromptControl", () => {
  let container;
  let root;

  beforeEach(() => {
    feedbackState.value = { state: "valid" };
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

  function render(ui) {
    root = createRoot(container);
    act(() => root.render(ui));
  }

  it.each(["q4", "q8", "bf16"])("refines with the exact installed Qwen3.8-27B %s tier and exclusions", async (tier) => {
    const refinePrompt = vi.fn(async () => "A rewritten scene");
    const qwenModel = { id: "vision_describe_qwen3_8_27b", variants: [
      { variant: "q4", installState: "installed" },
      { variant: "q8", installState: "installed" },
      { variant: "bf16", installState: "installed" },
    ] };
    render(<RefinePromptControl prompt="a portrait" workflow="image" refinePrompt={refinePrompt} onApply={vi.fn()} qwenModel={qwenModel} />);
    await act(async () => {
      const select = container.querySelector("select");
      select.value = `qwen38:${tier}`;
      select.dispatchEvent(new Event("change", { bubbles: true }));
    });
    await act(async () => container.querySelector('input[type="checkbox"]').click());
    await act(async () => refineButton(container).click());
    expect(refinePrompt).toHaveBeenCalledWith(expect.objectContaining({
      model: "huihui-ai/Huihui-Qwen3.8-27B-abliterated", variant: tier,
      promptExclusions: ["identifyingCharacteristics"], workflow: "image",
    }));
    await act(async () => root.render(<RefinePromptControl prompt="a portrait" workflow="image" refinePrompt={refinePrompt} onApply={vi.fn()} qwenModel={{ ...qwenModel, variants: [] }} />));
    expect(refineButton(container).disabled).toBe(true);
    expect(container.textContent).toContain("no longer installed");
  });

  it("offers only installed Qwen3.8 tiers", () => {
    render(<RefinePromptControl prompt="a portrait" refinePrompt={vi.fn()} onApply={vi.fn()}
      qwenModel={{ variants: [{ variant: "q4", installState: "missing" }, { variant: "q8", installState: "installed" }, { variant: "bf16", installState: "missing" }] }} />);
    expect([...container.querySelectorAll("option")].map((row) => row.value)).toEqual(["default", "qwen38:q8"]);
  });

  it("discovers installed tiers from the live catalog and waits for a model choice before auto-start", () => {
    const refinePrompt = vi.fn();
    render(<AppContext.Provider value={{ models: [{ id: "vision_describe_qwen3_8_27b", variants: [{ variant: "q8", installState: "installed" }] }] }}>
      <RefinePromptControl autoStart prompt="a portrait" refinePrompt={refinePrompt} onApply={vi.fn()} />
    </AppContext.Provider>);
    expect(container.querySelector('option[value="qwen38:q8"]')).not.toBeNull();
    expect(refinePrompt).not.toHaveBeenCalled();
  });

  it("combines optional exclusions and removes deselected options from the rewrite request", async () => {
    const refinePrompt = vi.fn(async () => "A revised scene");
    render(<RefinePromptControl prompt="a portrait in a room" workflow="image" refinePrompt={refinePrompt} onApply={vi.fn()} />);
    const options = [...container.querySelectorAll('input[type="checkbox"]')];
    expect(options).toHaveLength(4);
    expect(options.every((option) => !option.checked)).toBe(true);
    for (const option of options) await act(async () => option.click());
    await act(async () => options[2].click());
    await act(async () => refineButton(container).click());
    expect(refinePrompt).toHaveBeenCalledWith(expect.objectContaining({
      promptExclusions: ["identifyingCharacteristics", "clothingAndAccessories", "style"],
    }));
  });

  it.each(["pending", "server-required", "invalid"])("waits through %s before auto-starting the confirmed draft once", async (state) => {
    feedbackState.value = { state, refusal: { code: "invalid_option", field: "prompt", detail: "Fix this prompt" } };
    const refinePrompt = vi.fn(async () => "A confirmed rewrite");
    const ui = <RefinePromptControl autoStart prompt="neon street" modelId="z" workflow="image" refinePrompt={refinePrompt} onApply={vi.fn()} />;
    render(ui);
    expect(refineButton(container).disabled).toBe(true);
    expect(container.textContent).toContain(state === "pending" ? "Checking options with the server" : state === "server-required" ? "Reconnect to continue" : "Fix this prompt");
    await act(async () => refineButton(container).click());
    expect(refinePrompt).not.toHaveBeenCalled();
    feedbackState.value = { state: "valid" };
    await act(async () => root.render(React.cloneElement(ui)));
    expect(refinePrompt).toHaveBeenCalledTimes(1);
    await act(async () => root.render(React.cloneElement(ui)));
    expect(refinePrompt).toHaveBeenCalledTimes(1);
  });

  it.each(["image", "video"])("reviews a Loop rewrite at the selected tier in %s", async (workflow) => {
    const refinePrompt = vi.fn(async () => "A revised fox prompt");
    const onApply = vi.fn();
    const download = vi.fn(async () => ({ id: "download" }));
    render(<RefinePromptControl autoStart={workflow === "image"} prompt="a fox" modelId="target"
      workflow={workflow} refinePrompt={refinePrompt} onApply={onApply}
      loopModel={{ id: "vision_sensenova_loop_9b", name: "SenseNova-Loop-9B", variants: [
        { variant: "q4", installState: "installed" }, { variant: "bf16", installState: "missing" },
      ] }} onDownloadLoopModel={download} />);
    const choose = async (index, value) => act(async () => {
      const select = container.querySelectorAll("select")[index];
      select.value = value;
      select.dispatchEvent(new Event("change", { bubbles: true }));
    });
    expect(refinePrompt).not.toHaveBeenCalled();
    await choose(0, "loop");
    await choose(1, "bf16");
    expect(refineButton(container).disabled).toBe(true);
    expect(container.textContent).toContain("save a Hugging Face token");
    await act(async () => buttonByText(container, "Download").click());
    expect(download).toHaveBeenCalledWith(expect.objectContaining({ id: "vision_sensenova_loop_9b" }), { variant: "bf16" });
    await choose(1, "q4");
    await act(async () => refineButton(container).click());
    expect(refinePrompt).toHaveBeenCalledWith(expect.objectContaining({
      model: "sensenova/SenseNova-Loop-9B", variant: "q4", modelId: "target", workflow,
    }));
    expect(onApply).not.toHaveBeenCalled();
    await act(async () => buttonByText(container, "Apply").click());
    expect(onApply).toHaveBeenCalledWith("A revised fox prompt");
  });

  it("disables the button when the prompt is empty", () => {
    render(<RefinePromptControl prompt="   " guidePath="/prompt-guides/z.md" modelId="z" workflow="image" refinePrompt={vi.fn()} onApply={vi.fn()} />);
    expect(refineButton(container).disabled).toBe(true);
  });

  it("refines, shows the rewrite, and applies it only on Apply", async () => {
    const refinePrompt = vi.fn(async () => "A vivid neon street at midnight, cinematic.");
    const onApply = vi.fn();
    render(
      <RefinePromptControl
        prompt="neon street"
        guidePath="/prompt-guides/z-image-turbo.md"
        modelId="z_image_turbo"
        workflow="image"
        refinePrompt={refinePrompt}
        onApply={onApply}
      />,
    );

    await act(async () => {
      refineButton(container).click();
    });
    await settle({ microtasks: 8 });

    // Sent the prompt + model + workflow + fetched guide; original not yet applied.
    expect(refinePrompt).toHaveBeenCalledWith(expect.objectContaining({
      prompt: "neon street",
      modelId: "z_image_turbo",
      workflow: "image",
      guide: "# Guide\n\nWrite vividly.",
      signal: expect.any(AbortSignal),
    }));
    expect(onApply).not.toHaveBeenCalled();
    expect(container.querySelector(".refine-review-text").textContent).toBe("A vivid neon street at midnight, cinematic.");

    await act(async () => {
      buttonByText(container, "Apply").click();
    });
    expect(onApply).toHaveBeenCalledWith("A vivid neon street at midnight, cinematic.");
    expect(container.querySelector(".refine-review")).toBeNull();
  });

  it("keeps the original when the user dismisses the rewrite", async () => {
    const refinePrompt = vi.fn(async () => "rewritten");
    const onApply = vi.fn();
    render(<RefinePromptControl prompt="dog" guidePath="" modelId="z" workflow="image" refinePrompt={refinePrompt} onApply={onApply} />);

    await act(async () => {
      refineButton(container).click();
    });
    await settle({ microtasks: 8 });

    await act(async () => {
      buttonByText(container, "Keep original").click();
    });
    expect(onApply).not.toHaveBeenCalled();
    expect(container.querySelector(".refine-review")).toBeNull();
  });

  it("surfaces a failure message and does not apply", async () => {
    const refinePrompt = vi.fn(async () => {
      throw new Error("Prompt refinement runtime is not configured.");
    });
    const onApply = vi.fn();
    render(<RefinePromptControl prompt="dog" guidePath="" modelId="z" workflow="image" refinePrompt={refinePrompt} onApply={onApply} />);

    await act(async () => {
      refineButton(container).click();
    });
    await settle({ microtasks: 8 });

    expect(container.querySelector(".refine-error").textContent).toContain("not configured");
    expect(onApply).not.toHaveBeenCalled();
    expect(container.querySelector(".refine-review")).toBeNull();
  });

  it("still refines (generically) when the guide cannot be fetched", async () => {
    global.fetch = vi.fn(() => Promise.resolve({ ok: false, status: 404, text: async () => "" }));
    const refinePrompt = vi.fn(async () => "rewritten");
    render(
      <RefinePromptControl prompt="dog" guidePath="/prompt-guides/missing.md" modelId="z" workflow="video" refinePrompt={refinePrompt} onApply={vi.fn()} />,
    );

    await act(async () => {
      refineButton(container).click();
    });
    await settle({ microtasks: 8 });

    expect(refinePrompt).toHaveBeenCalledWith(expect.objectContaining({
      prompt: "dog",
      modelId: "z",
      workflow: "video",
      guide: "",
      signal: expect.any(AbortSignal),
    }));
    expect(container.querySelector(".refine-review-text").textContent).toBe("rewritten");
  });

  it("offers to download the refinement model when it isn't installed (sc-5605)", async () => {
    const refinePrompt = vi.fn(async () => {
      throw new Error("prompt-refine model path snapshot is not cached for TheDrummer/Anubis-Mini-8B-v1.");
    });
    const onDownloadRefineModel = vi.fn(async () => ({ id: "job-1" }));
    render(
      <RefinePromptControl
        prompt="dog"
        guidePath=""
        modelId="z"
        workflow="image"
        refinePrompt={refinePrompt}
        onApply={vi.fn()}
        refineModel={{ id: "prompt_refine_anubis_8b", name: "Prompt Refiner", installState: "missing", downloadSizeBytes: 16077901824 }}
        onDownloadRefineModel={onDownloadRefineModel}
      />,
    );

    await act(async () => {
      refineButton(container).click();
    });
    await settle({ microtasks: 8 });

    // The raw worker error is replaced by a download affordance with a size hint.
    expect(container.querySelector(".refine-missing-model").textContent).toContain("isn’t installed");
    expect(container.querySelector(".refine-missing-model").textContent).toContain("16.1 GB");
    const downloadButton = buttonByText(container, "Download refinement model");
    expect(downloadButton).toBeTruthy();

    await act(async () => {
      downloadButton.click();
    });
    await settle({ microtasks: 8 });

    expect(onDownloadRefineModel).toHaveBeenCalledTimes(1);
    expect(container.querySelector(".refine-missing-model").textContent).toContain("Downloading");
  });

  it("shows a plain error (no download CTA) when the refinement model is installed", async () => {
    const refinePrompt = vi.fn(async () => {
      throw new Error("Prompt refinement failed.");
    });
    render(
      <RefinePromptControl
        prompt="dog"
        guidePath=""
        modelId="z"
        workflow="image"
        refinePrompt={refinePrompt}
        onApply={vi.fn()}
        refineModel={{ id: "prompt_refine_anubis_8b", name: "Prompt Refiner", installState: "installed" }}
        onDownloadRefineModel={vi.fn()}
      />,
    );

    await act(async () => {
      refineButton(container).click();
    });
    await settle({ microtasks: 8 });

    expect(container.querySelector(".refine-missing-model")).toBeNull();
    expect(container.querySelector(".refine-error").textContent).toContain("Prompt refinement failed.");
    expect(buttonByText(container, "Download refinement model")).toBeUndefined();
  });
});
