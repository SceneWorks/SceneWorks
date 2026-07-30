// "Workflow found" (sc-15951, epic 15945).
//
// The panel's whole job is to be honest about a file that came from a stranger's install, so the
// tests below are almost all about what it REFUSES to imply: that a missing model is fine, that a
// recipe whose LoRAs were omitted had none, or that a batch of four replays as a batch of four.
import React, { act } from "react";
import { createRoot } from "react-dom/client";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";

import { WorkflowDropPanel } from "./WorkflowDropPanel.jsx";

function share(overrides = {}) {
  return {
    sceneworksWorkflow: "image",
    schemaVersion: 1,
    producer: { name: "SceneWorks", url: "https://example.invalid", version: "0.8.1" },
    mode: "text_to_image",
    model: "krea_2_turbo",
    prompt: "a lighthouse in heavy fog, cinematic",
    negativePrompt: "text, watermark",
    seed: 880412,
    width: 1024,
    height: 1024,
    count: 1,
    advanced: { steps: 28, guidanceScale: 3.5 },
    ...overrides,
  };
}

function report(overrides = {}) {
  return {
    model: {
      slug: "krea_2_turbo",
      state: "resolved",
      catalogId: "krea_2_turbo",
      name: "Krea 2 Turbo",
      detail: "Krea 2 Turbo is installed on this machine.",
    },
    loras: [],
    styles: [],
    inputs: [],
    omitted: [],
    runnable: true,
    inputImagesRequired: 0,
    ...overrides,
  };
}

const idleImport = { busy: false, done: false, error: "" };

describe("WorkflowDropPanel", () => {
  let container;
  let root;

  beforeEach(() => {
    global.IS_REACT_ACT_ENVIRONMENT = true;
    container = document.createElement("div");
    document.body.appendChild(container);
    root = createRoot(container);
  });

  afterEach(async () => {
    await act(async () => root.unmount());
    container.remove();
    vi.clearAllMocks();
  });

  // The Modal portals to <body>, so assertions read the document rather than the container.
  const text = () => document.body.textContent;
  const buttonByLabel = (label) =>
    [...document.querySelectorAll("button")].find((node) => node.textContent.trim() === label);

  const render = async (props) =>
    act(async () =>
      root.render(
        <WorkflowDropPanel
          canImport
          importState={idleImport}
          offer={null}
          onDismiss={() => {}}
          onImport={() => {}}
          onUse={() => {}}
          {...props}
        />,
      ),
    );

  it("renders nothing without an offer", async () => {
    await render({});
    expect(document.querySelector(".workflow-drop-modal")).toBeNull();
  });

  it("shows the recipe that came with the file", async () => {
    await render({ offer: { share: share(), report: report(), error: "" } });
    expect(text()).toContain("Workflow found");
    expect(text()).toContain("a lighthouse in heavy fog, cinematic");
    expect(text()).toContain("text, watermark");
    expect(text()).toContain("880412");
    expect(text()).toContain("1024 × 1024");
    expect(text()).toContain("Krea 2 Turbo");
    expect(text()).toContain("Steps");
    expect(text()).toContain("Everything this recipe names is installed here.");
    // No "model not prefilled" warning when the model resolved.
    expect(text()).not.toContain("keeps the model it is already on");
  });

  it("names a missing model instead of quietly running the studio's own", async () => {
    await render({
      offer: {
        share: share({ model: "someones_private_model" }),
        report: report({
          model: {
            slug: "someones_private_model",
            state: "missing",
            detail:
              "No model matching `someones_private_model` is in this install's catalog, so this " +
              "workflow cannot be reproduced here.",
          },
          runnable: false,
        }),
        error: "",
      },
    });
    expect(text()).toContain("No model matching `someones_private_model`");
    expect(text()).toContain("Missing");
    expect(text()).toContain("keeps the model it is already on");
    // "Use this workflow" is still offered — the prompt, seed and knobs are real and the
    // panel has said exactly what will not come with them.
    expect(buttonByLabel("Use this workflow")).toBeTruthy();
  });

  it("does not present an omitted LoRA list as a LoRA-free recipe", async () => {
    // A 6-LoRA envelope whose LoRAs were dropped over the cap and a genuinely LoRA-free one
    // serialize identically; `omitted` is the only thing that tells them apart.
    await render({
      offer: {
        share: share({ omitted: ["loras"] }),
        report: report({
          omitted: [
            {
              field: "loras",
              detail:
                "This image's LoRA list was declared but not recorded, so this recipe is not a " +
                "LoRA-free one.",
            },
          ],
          runnable: false,
        }),
        error: "",
      },
    });
    expect(text()).toContain("Not recorded — loras");
    expect(text()).toContain("not a LoRA-free one");
    expect(text()).toContain("cannot be reproduced on this install");
  });

  it("says a shared edit still needs its source image", async () => {
    await render({
      offer: {
        share: share({ mode: "edit_image" }),
        report: report({
          inputs: [
            {
              kind: "source",
              count: 1,
              state: "userSupplied",
              detail: "Needs a source image.",
            },
          ],
          inputImagesRequired: 1,
        }),
        error: "",
      },
    });
    expect(text()).toContain("Edit image");
    expect(text()).toContain("Needs a source image.");
    expect(text()).toContain("still needs 1 image from you");
  });

  it("reports the batch it came out of, and that replay makes one image", async () => {
    await render({ offer: { share: share({ count: 4 }), report: report(), error: "" } });
    expect(text()).toContain("one of a batch of 4");
    expect(text()).toContain("single image");
  });

  it("wires the three actions", async () => {
    const onUse = vi.fn();
    const onImport = vi.fn();
    const onDismiss = vi.fn();
    await render({
      offer: { share: share(), report: report(), error: "" },
      onUse,
      onImport,
      onDismiss,
    });
    for (const [label, spy] of [
      ["Use this workflow", onUse],
      ["Import the image too", onImport],
      ["Cancel", onDismiss],
    ]) {
      const button = buttonByLabel(label);
      expect(button, label).toBeTruthy();
      await act(async () => button.dispatchEvent(new MouseEvent("click", { bubbles: true })));
      expect(spy, label).toHaveBeenCalledTimes(1);
    }
  });

  it("disables the import when there is no project to import into", async () => {
    await render({ canImport: false, offer: { share: share(), report: report(), error: "" } });
    expect(buttonByLabel("Import the image too").disabled).toBe(true);
  });

  it("confirms an import and keeps the workflow available", async () => {
    await render({
      importState: { busy: false, done: true, error: "" },
      offer: { share: share(), report: report(), error: "" },
    });
    expect(text()).toContain("imported into this project");
    expect(buttonByLabel("Use this workflow")).toBeTruthy();
  });

  it("shows a read failure as its own sentence with nothing to act on", async () => {
    await render({
      offer: {
        share: null,
        report: null,
        error: "This PNG could not be read far enough to look for a workflow.",
      },
    });
    expect(text()).toContain("could not be read far enough");
    expect(buttonByLabel("Use this workflow")).toBeFalsy();
    expect(buttonByLabel("Close")).toBeTruthy();
  });
});

// The mark, and the verdict that counts it (sc-15951 review). A settings grid where
// "PiD decoder — On" and "Steps — 28" look identical, when only one of them reaches a control,
// tells the user a recipe replayed faithfully when it did not.
describe("WorkflowDropPanel — what will not be restored", () => {
  let container;
  let root;

  beforeEach(() => {
    global.IS_REACT_ACT_ENVIRONMENT = true;
    container = document.createElement("div");
    document.body.appendChild(container);
    root = createRoot(container);
  });

  afterEach(async () => {
    await act(async () => root.unmount());
    container.remove();
    vi.clearAllMocks();
  });

  const render = async (props) =>
    act(async () =>
      root.render(
        <WorkflowDropPanel
          canImport
          importState={idleImport}
          onDismiss={() => {}}
          onImport={() => {}}
          onUse={() => {}}
          {...props}
        />,
      ),
    );

  const settingRow = (label) =>
    [...document.querySelectorAll(".workflow-drop-settings li")].find((node) =>
      node.textContent.startsWith(label),
    );

  it("marks the rows that reach no control, and leaves the applied ones unmarked", async () => {
    await render({
      offer: {
        share: share({
          advanced: {
            steps: 28,
            usePid: true,
            cnScale: 0.4,
            poses: [{ keypoints: [] }, { keypoints: [] }, { keypoints: [] }],
          },
        }),
        report: report(),
        error: "",
      },
    });

    expect(settingRow("Steps").querySelector(".workflow-drop-setting-mark")).toBeNull();
    expect(settingRow("PiD decoder").querySelector(".workflow-drop-setting-mark")).toBeNull();
    expect(settingRow("Tile ControlNet").textContent).toContain("not restored");
    expect(settingRow("Poses").textContent).toContain("3 poses");
    expect(settingRow("Poses").textContent).toContain("not restored");
  });

  it("folds them into the verdict, so a fully installed recipe still reads as lossy", async () => {
    await render({
      offer: {
        share: share({ advanced: { steps: 28, poses: [{ keypoints: [] }] } }),
        report: report(),
        error: "",
      },
    });

    // The report itself is clean — the model is installed and nothing was omitted — so without
    // this clause the panel would say "Everything this recipe names is installed here." over a
    // pose selection that is about to be dropped on the floor.
    expect(document.body.textContent).toContain("1 setting will not be restored");
    expect(document.querySelector(".workflow-req-verdict.ok")).toBeNull();
    expect(document.body.textContent).toContain("Poses — 1 pose");
  });

  it("says nothing extra when every setting in the file lands somewhere", async () => {
    await render({
      offer: { share: share({ advanced: { steps: 28, usePid: true } }), report: report(), error: "" },
    });

    expect(document.body.textContent).toContain("Everything this recipe names is installed here.");
    expect(document.body.textContent).not.toContain("will not be restored");
    expect(document.querySelector(".workflow-req-verdict.ok")).toBeTruthy();
  });

  it("does not count an input image the user supplies as something that cannot be reproduced", async () => {
    await render({
      offer: {
        share: share({ mode: "edit_image" }),
        report: report({
          inputs: [
            { kind: "source", count: 1, state: "userSupplied", detail: "Needs a source image." },
          ],
          inputImagesRequired: 1,
          runnable: true,
        }),
        error: "",
      },
    });

    expect(document.body.textContent).toContain("It still needs 1 image from you.");
    expect(document.body.textContent).not.toContain("cannot be reproduced");
    // The row is still rendered — only the verdict's arithmetic excludes it.
    expect(document.body.textContent).toContain("Input image — source");
  });

  it("marks an installed style preset, which reaches no control on this replay path", async () => {
    await render({
      offer: {
        share: share(),
        report: report({
          styles: [
            { field: "stylePreset", id: "cine", name: "Cinematic", state: "resolved", detail: "ok" },
          ],
        }),
        error: "",
      },
    });

    expect(document.body.textContent).toContain("Style preset — Cinematic");
    expect(document.body.textContent).toContain("Not restored");
    // …and NOT as a "Ready" style row, which claimed an axis the studio never receives.
    expect(document.body.textContent).not.toContain("Style — Cinematic");
  });

  it("says why the launch refused rather than closing over a button that did nothing", async () => {
    await render({
      offer: {
        share: share(),
        report: report(),
        error: "",
        launchError: "That opens in the Advanced workspace — switch on a larger screen.",
      },
    });

    expect(document.body.textContent).toContain("switch on a larger screen");
    expect(document.querySelector(".workflow-drop-modal")).toBeTruthy();
  });
});

// The missing-inputs surface, embedded, and the CTA it gates (sc-15952).
//
// The panel is the point where the recipe is handed over, so it is where "generate is blocked with
// the reason named" is enforced. `disabled` and the chips come out of one `useValidation`, which is
// the mechanism Image Studio's own Generate uses — these tests read BOTH on every case, because a
// dead button with nothing to explain it is the failure that vocabulary exists to prevent.
describe("WorkflowDropPanel — what it needs from the user", () => {
  let container;
  let root;

  beforeEach(() => {
    global.IS_REACT_ACT_ENVIRONMENT = true;
    container = document.createElement("div");
    document.body.appendChild(container);
    root = createRoot(container);
  });

  afterEach(async () => {
    await act(async () => root.unmount());
    container.remove();
    vi.clearAllMocks();
  });

  const buttonByLabel = (label) =>
    [...document.querySelectorAll("button")].find((node) => node.textContent.trim() === label);

  const render = async (props) =>
    act(async () =>
      root.render(
        <WorkflowDropPanel
          canImport
          importState={idleImport}
          onDismiss={() => {}}
          onImport={() => {}}
          onUse={() => {}}
          {...props}
        />,
      ),
    );

  const cta = () => buttonByLabel("Use this workflow") ?? buttonByLabel("Use this recipe");

  it("shows no missing-inputs UI at all for a fully-resolvable recipe", async () => {
    await render({ offer: { share: share(), report: report(), error: "" } });
    // The acceptance criterion is an absence, so it is asserted as one: not an empty section, not
    // a zero-state line, not a heading.
    expect(document.querySelector(".workflow-fix")).toBeNull();
    expect(document.body.textContent).not.toContain("What this needs from you");
    expect(cta()?.disabled).toBe(false);
  });

  it("blocks the CTA on an unresolvable model and prints the reason beside it", async () => {
    await render({
      offer: {
        share: share({ model: "someones_private_model" }),
        report: report({
          runnable: false,
          model: {
            slug: "someones_private_model",
            state: "missing",
            detail: "No model matching `someones_private_model` is in this install's catalog.",
          },
        }),
        error: "",
      },
      missing: { models: [{ id: "z_image_turbo", name: "Z-Image Turbo" }] },
    });
    expect(cta()?.disabled).toBe(true);
    // The dead button and its reason come from ONE summarize, so a chip is always there to read.
    const chips = [...document.querySelectorAll(".validation-chip")].map((node) => node.textContent);
    expect(chips.join(" ")).toContain("someones_private_model");
    expect(chips.join(" ")).toContain("nothing is substituted for you");
    // …and the surface that can clear it is on screen.
    expect(document.querySelector(".workflow-fix")).toBeTruthy();
  });

  it("unblocks the CTA once the user has chosen a model", async () => {
    await render({
      offer: {
        share: share({ model: "someones_private_model" }),
        report: report({
          runnable: false,
          model: { slug: "someones_private_model", state: "missing", detail: "…" },
        }),
        error: "",
      },
      missing: {
        models: [{ id: "z_image_turbo", name: "Z-Image Turbo" }],
        substituteModelId: "z_image_turbo",
      },
    });
    expect(cta()?.disabled).toBe(false);
    // The "keeps the model it is already on" warning is now false, and must not be shown.
    expect(document.body.textContent).not.toContain("keeps the model it is already on");
  });

  it("blocks until every input image the recipe needs has been picked", async () => {
    const inputs = report({
      inputImagesRequired: 2,
      inputs: [
        { kind: "source", count: 1, state: "userSupplied", detail: "Needs a source image." },
        { kind: "reference", count: 1, state: "userSupplied", detail: "Needs a reference image." },
      ],
    });
    const assets = [{ id: "asset_a", displayName: "Harbour at dawn" }];
    await render({ assets, offer: { share: share(), report: inputs, error: "" } });
    expect(cta()?.disabled).toBe(true);
    await render({
      assets,
      offer: { share: share(), report: inputs, error: "" },
      missing: { inputPicks: { "source-0-0": "asset_a", "reference-1-0": "asset_a" } },
    });
    expect(cta()?.disabled).toBe(false);
  });

  // ---- The library surface -------------------------------------------------------------------

  it("reads as a recipe from the library, with no import action, when opened from an asset", async () => {
    await render({
      offer: { assetId: "asset_1", file: null, share: share(), report: report(), error: "" },
    });
    expect(document.body.textContent).toContain("The recipe in this image");
    expect(document.body.textContent).not.toContain("Workflow found");
    // "Import the image too" for an image that is already imported is a control describing a
    // state; the panel is not rendered with it at all rather than rendered with it disabled.
    expect(buttonByLabel("Import the image too")).toBeUndefined();
    expect(buttonByLabel("Use this recipe")).toBeTruthy();
  });

  it("keeps the import action for a dropped file", async () => {
    await render({ offer: { assetId: null, share: share(), report: report(), error: "" } });
    expect(buttonByLabel("Import the image too")).toBeTruthy();
    expect(buttonByLabel("Use this workflow")).toBeTruthy();
  });
});
