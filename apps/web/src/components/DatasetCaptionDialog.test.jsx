import React, { act } from "react";
import { createRoot } from "react-dom/client";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { DatasetCaptionDialog } from "./DatasetCaptionDialog.jsx";

const baseSettings = {
  captioner: "joy_caption",
  modelNameOrPath: "",
  requestedGpu: "auto",
  captionType: "Descriptive",
  captionLength: "long",
  nameInput: "",
  temperature: 0.6,
  topP: 0.9,
  maxNewTokens: 256,
  lowVram: false,
  recaption: false,
  extraOptions: [],
};

function buttonByText(container, text) {
  return [...document.body.querySelectorAll("button")].find((button) => button.textContent.trim() === text);
}

async function settle() {
  await act(async () => {
    for (let index = 0; index < 8; index += 1) {
      await Promise.resolve();
    }
  });
}

describe("DatasetCaptionDialog", () => {
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

  it("offers a download + blocks Run when the captioning model is missing (sc-5620)", async () => {
    const onDownloadModel = vi.fn(async () => ({ id: "job-1" }));
    render(
      <DatasetCaptionDialog
        settings={baseSettings}
        onChange={vi.fn()}
        onRun={vi.fn()}
        onToggleExtra={vi.fn()}
        onClose={vi.fn()}
        scope={{ type: "all" }}
        modelMissing
        onDownloadModel={onDownloadModel}
        modelSizeLabel="17.0 GB"
        modelName="JoyCaption (beta one)"
      />,
    );

    expect(document.body.querySelector(".caption-missing-model").textContent).toContain("isn’t installed");
    expect(document.body.querySelector(".caption-missing-model").textContent).toContain("17.0 GB");
    // Run is blocked while the model is missing.
    expect(buttonByText(container, "Caption missing").disabled).toBe(true);

    const download = buttonByText(container, "Download captioning model");
    expect(download).toBeTruthy();
    await act(async () => {
      download.click();
    });
    await settle();

    expect(onDownloadModel).toHaveBeenCalledTimes(1);
    expect(document.body.querySelector(".caption-missing-model").textContent).toContain("Downloading");
  });

  it("shows no affordance and enables Run when the captioning model is present", () => {
    render(
      <DatasetCaptionDialog
        settings={baseSettings}
        onChange={vi.fn()}
        onRun={vi.fn()}
        onToggleExtra={vi.fn()}
        onClose={vi.fn()}
        scope={{ type: "all" }}
        modelMissing={false}
      />,
    );

    expect(document.body.querySelector(".caption-missing-model")).toBeNull();
    expect(buttonByText(container, "Caption missing").disabled).toBe(false);
  });

  it("surfaces a model-download enqueue failure and permits a retry", async () => {
    const onDownloadModel = vi
      .fn()
      .mockRejectedValueOnce(new Error("Models service is offline"))
      .mockResolvedValueOnce({ id: "job-2" });
    render(
      <DatasetCaptionDialog
        settings={baseSettings}
        onChange={vi.fn()}
        onRun={vi.fn()}
        onToggleExtra={vi.fn()}
        onClose={vi.fn()}
        scope={{ type: "all" }}
        modelMissing
        onDownloadModel={onDownloadModel}
      />,
    );

    await act(async () => buttonByText(container, "Download captioning model").click());
    await settle();
    expect(document.body.querySelector(".caption-missing-model").textContent).toContain("Models service is offline");
    expect(buttonByText(container, "Download captioning model")).toBeTruthy();

    await act(async () => buttonByText(container, "Download captioning model").click());
    await settle();
    expect(document.body.querySelector(".caption-missing-model").textContent).toContain("Downloading");
    expect(onDownloadModel).toHaveBeenCalledTimes(2);
  });

  // sc-24829: caption modes.
  function labelled(text) {
    const label = [...document.body.querySelectorAll("label")].find((item) => item.childNodes[0]?.textContent.trim() === text);
    return label?.querySelector("input, select, textarea");
  }

  function renderMode(mode, extra = {}) {
    const onChange = vi.fn();
    render(
      <DatasetCaptionDialog
        settings={{ ...baseSettings, mode }}
        onChange={onChange}
        onRun={vi.fn()}
        onToggleExtra={vi.fn()}
        onClose={vi.fn()}
        scope={{ type: "all" }}
        extraOptions={[{ value: "lighting", label: "Mention lighting" }]}
        {...extra}
      />,
    );
    return onChange;
  }

  it("offers the three caption modes and reports a change", async () => {
    const onChange = renderMode("default");
    const select = labelled("Mode");
    expect([...select.options].map((option) => option.value)).toEqual(["default", "subjectOnly", "triggerOnly"]);
    await act(async () => {
      select.value = "subjectOnly";
      select.dispatchEvent(new window.Event("change", { bubbles: true }));
    });
    expect(onChange).toHaveBeenCalledWith("mode", "subjectOnly");
  });

  it("keeps the captioner controls but drops the prompt controls in subject-only mode", () => {
    renderMode("subjectOnly");
    expect(labelled("Model")).toBeTruthy();
    expect(labelled("Temperature")).toBeTruthy();
    expect(labelled("Caption prompt")).toBeUndefined();
    expect(labelled("Type")).toBeUndefined();
    expect(document.body.textContent).not.toContain("Mention lighting");
    expect(document.body.querySelector(".dataset-caption-mode-note").textContent).toContain("clothing, expression, pose");
  });

  it("runs trigger-only without the captioner, even when the captioning model is missing", () => {
    renderMode("triggerOnly", { modelMissing: true, onDownloadModel: vi.fn() });
    expect(labelled("Model")).toBeUndefined();
    expect(labelled("Caption prompt")).toBeUndefined();
    expect(document.body.querySelector(".caption-missing-model")).toBeNull();
    expect(document.body.querySelector(".dataset-caption-mode-note").textContent).toContain("exactly its trigger words");
    expect(buttonByText(container, "Caption missing").disabled).toBe(false);
  });

  it.each(["default", "subjectOnly", "triggerOnly"])("shows the trigger-words field in %s mode and reports an edit", async (mode) => {
    const onChange = renderMode(mode);
    const input = labelled("Trigger words");
    expect(input).toBeTruthy();
    await act(async () => {
      const setter = Object.getOwnPropertyDescriptor(window.HTMLInputElement.prototype, "value").set;
      setter.call(input, "miraStyle");
      input.dispatchEvent(new window.Event("input", { bubbles: true }));
    });
    expect(onChange).toHaveBeenCalledWith("triggerWords", "miraStyle");
  });

  // E6: the dialog enforces the same limits the API does (16 words, 64 characters each).
  it.each([
    ["17 words", Array.from({ length: 17 }, (_, index) => `w${index}`).join(", "), "at most 16 trigger words"],
    ["a 65-character word", "a".repeat(65), "at most 64 characters"],
  ])("blocks Run on %s", (_label, triggerWords, message) => {
    renderMode("triggerOnly", { settings: { ...baseSettings, mode: "triggerOnly", triggerWords } });
    expect(document.body.querySelector("[role=alert]").textContent).toContain(message);
    expect(buttonByText(container, "Caption missing").disabled).toBe(true);
  });

  it("accepts trigger words at the limits", () => {
    // 16 words, the last exactly 64 characters.
    const atLimits = [...Array.from({ length: 15 }, (_, index) => `w${index}`), "a".repeat(64)].join(", ");
    renderMode("triggerOnly", { settings: { ...baseSettings, mode: "triggerOnly", triggerWords: atLimits } });
    expect(document.body.querySelector("[role=alert]")).toBeNull();
    expect(buttonByText(container, "Caption missing").disabled).toBe(false);
  });
});
