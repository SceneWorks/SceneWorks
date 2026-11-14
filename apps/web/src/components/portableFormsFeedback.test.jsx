import React, { act } from "react";
import { createRoot } from "react-dom/client";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";

const feedback = vi.hoisted(() => vi.fn());
vi.mock("../hooks/usePortableFeedback.js", () => ({ usePortableFeedback: feedback }));

import { RefinePromptControl } from "./RefinePromptControl.jsx";
import { VqaPanel } from "./assetPanels.jsx";
import { CharacterTest } from "../screens/characterPanels.jsx";
import { ReplacePersonPanel } from "../screens/ReplacePersonPanel.jsx";
import { VideoUpscalePanel } from "../screens/VideoUpscalePanel.jsx";
import { Yue2ComposePanel } from "../screens/yue2Lab/Yue2ComposePanel.jsx";

const refusal = { state: "invalid", refusal: { code: "invalid_generation_option", field: "prompt", detail: "Prompt is invalid" } };

describe("portable authored form feedback", () => {
  let host;
  let root;
  beforeEach(() => {
    global.IS_REACT_ACT_ENVIRONMENT = true;
    host = document.createElement("div");
    document.body.append(host);
    root = createRoot(host);
    feedback.mockReset();
    feedback.mockReturnValue(refusal);
  });
  afterEach(() => {
    act(() => root.unmount());
    host.remove();
  });

  it("shows a typed refusal for the authored prompt and blocks refinement", () => {
    const refine = vi.fn();
    act(() => root.render(<RefinePromptControl prompt="bad" modelId="z" workflow="image" refinePrompt={refine} onApply={vi.fn()} />));
    expect(feedback).toHaveBeenCalledWith("promptRefine", { prompt: "bad", modelId: "z", workflow: "image" }, null, true);
    const button = host.querySelector("button.refine-button");
    expect(button.disabled).toBe(true);
    expect(host.querySelector('[aria-label="Prompt refinement errors"]').textContent).toContain("Prompt is invalid");
    act(() => button.click());
    expect(refine).not.toHaveBeenCalled();
  });

  it("checks the exact VQA body while leaving model choice to the server", () => {
    const ask = vi.fn();
    act(() => root.render(<VqaPanel asset={{ id: "asset-1" }} entries={[]} pending={false} onAsk={ask} project={{ id: "p", name: "Project" }} requestedGpu="auto" />));
    const question = host.querySelector("textarea");
    act(() => {
      Object.getOwnPropertyDescriptor(HTMLTextAreaElement.prototype, "value").set.call(question, "What is shown?");
      question.dispatchEvent(new Event("input", { bubbles: true }));
    });
    expect(feedback).toHaveBeenLastCalledWith("vqa", {
      projectId: "p", projectName: "Project", sourceAssetId: "asset-1",
      question: "What is shown?", maxNewTokens: 256, requestedGpu: "auto",
    }, null, true);
    expect(host.querySelector("button").disabled).toBe(true);
    expect(host.querySelector('[aria-label="Image question errors"]').textContent).toContain("Prompt is invalid");
    expect(ask).not.toHaveBeenCalled();
  });

  it("shows the refusal beside Character Test and blocks its submit", () => {
    act(() => root.render(<CharacterTest
      selectedCharacter={{ id: "c", looks: [] }} latestAssets={[]} imageModels={[]}
      testPrompt="bad" testModel="z" testCount={4} testResolution="1024x1024" testLookId=""
      setTestPrompt={vi.fn()} setTestModel={vi.fn()} setTestCount={vi.fn()}
      setTestResolution={vi.fn()} setTestLookId={vi.fn()} submitTest={vi.fn()}
      testRefusal={refusal.refusal}
      testFeedback={refusal}
    />));
    expect(host.querySelector('button[type="submit"]').disabled).toBe(true);
    expect(host.querySelector('[aria-label="Character test errors"]').textContent).toContain("Prompt is invalid");
  });

  it("shows a YuE2 refusal next to its generation action", () => {
    act(() => root.render(<Yue2ComposePanel
      settings={{ planSource: "sample", planning: "full", planOnly: false }}
      update={vi.fn()} sessionOnlyNote={() => null} inspect={vi.fn()}
      preview={{}} restorable={[]} submitting={false} composeProblems={[]}
      readyToRun submit={vi.fn()} kind="create" tierProblem="" composeLabel="Generate song"
      portableRefusal={refusal.refusal}
      portableFeedback={refusal}
    />));
    expect(host.querySelector("button.prompt-cta").disabled).toBe(true);
    expect(host.querySelector('[role="alert"]').textContent).toContain("Prompt is invalid");
  });

  it("checks person detection and tracking drafts before their actions", () => {
    act(() => root.render(<ReplacePersonPanel
      sourceClipAssetId="clip-1" representativeFrame={{ id: "frame-1" }}
      selectedDetection={{ id: "person-1" }} trackName="actor"
      personReadiness={{}} matchingTracks={[]} videoAssets={[]} videoModels={[]}
      setSourceClipAssetId={vi.fn()} setSelectedDetectionId={vi.fn()}
      setTrackName={vi.fn()} setPersonTrackId={vi.fn()} setReplacementMode={vi.fn()}
      createPersonDetectionJob={vi.fn()} createPersonTrackJob={vi.fn()}
    />));
    expect(feedback).toHaveBeenCalledWith("personDetect", { sourceAssetId: "clip-1" }, null, true);
    expect(feedback).toHaveBeenCalledWith("personTrack", {
      sourceAssetId: "clip-1", representativeFrameAssetId: "frame-1",
      detection: { id: "person-1" }, trackName: "actor",
    }, null, true);
    expect([...host.querySelectorAll("button")].find((button) => button.textContent.includes("Analyze Source")).disabled).toBe(true);
    expect([...host.querySelectorAll("button")].find((button) => button.textContent.includes("Save Track")).disabled).toBe(true);
    expect(host.querySelectorAll('[role="alert"]')).toHaveLength(2);
  });

  it.each(["pending", "server-required"])("blocks video upscaling while feedback is %s", (state) => {
    feedback.mockReturnValue({ state });
    const createVideoUpscaleJob = vi.fn();
    const clip = { id: "clip-1", type: "video", displayName: "Clip" };
    act(() => root.render(<VideoUpscalePanel
      selectedAsset={clip} videoAssets={[clip]} createVideoUpscaleJob={createVideoUpscaleJob}
    />));
    const button = [...host.querySelectorAll("button")].find((item) => item.textContent === "Upscale clip");
    expect(button.disabled).toBe(true);
    expect(host.querySelector('[role="status"]').textContent).toContain(state === "pending"
      ? "Checking options with the server" : "Reconnect to continue");
    act(() => button.dispatchEvent(new MouseEvent("click", { bubbles: true })));
    expect(createVideoUpscaleJob).not.toHaveBeenCalled();
  });

  it.each(["pending", "server-required"])("blocks person detection and tracking while feedback is %s", (state) => {
    feedback.mockReturnValue({ state });
    const createPersonDetectionJob = vi.fn();
    const createPersonTrackJob = vi.fn();
    act(() => root.render(<ReplacePersonPanel
      sourceClipAssetId="clip-1" representativeFrame={{ id: "frame-1" }}
      selectedDetection={{ id: "person-1" }} trackName="actor"
      personReadiness={{}} matchingTracks={[]} videoAssets={[]} videoModels={[]}
      setSourceClipAssetId={vi.fn()} setSelectedDetectionId={vi.fn()}
      setTrackName={vi.fn()} setPersonTrackId={vi.fn()} setReplacementMode={vi.fn()}
      createPersonDetectionJob={createPersonDetectionJob} createPersonTrackJob={createPersonTrackJob}
    />));
    const buttons = [...host.querySelectorAll("button")];
    const analyze = buttons.find((item) => item.textContent === "Analyze Source");
    const track = buttons.find((item) => item.textContent === "Save Track");
    expect(analyze.disabled).toBe(true);
    expect(track.disabled).toBe(true);
    expect([...host.querySelectorAll('[role="status"]')].map((item) => item.textContent)).toEqual([
      expect.stringContaining(state === "pending" ? "Checking options with the server" : "Reconnect to continue"),
      expect.stringContaining(state === "pending" ? "Checking options with the server" : "Reconnect to continue"),
    ]);
    act(() => {
      analyze.dispatchEvent(new MouseEvent("click", { bubbles: true }));
      track.dispatchEvent(new MouseEvent("click", { bubbles: true }));
    });
    expect(createPersonDetectionJob).not.toHaveBeenCalled();
    expect(createPersonTrackJob).not.toHaveBeenCalled();
  });
});
