// The missing-inputs surface (sc-15952, epic 15945).
//
// Two of these carry the weight. `renders nothing at all for a fully-resolvable recipe` is an
// acceptance criterion phrased as an absence, and absences are the easiest thing in a UI to
// accidentally ship as "an empty box with a heading over it". And the substitute-model tests are
// the other half of "never silently substitute": the picker must start empty and must offer only
// models the studio's own list would keep.
import React, { act } from "react";
import { createRoot } from "react-dom/client";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";

import { WorkflowMissingInputs } from "./WorkflowMissingInputs.jsx";

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

const MODELS = [
  { id: "z_image_turbo", name: "Z-Image Turbo" },
  { id: "krea_2_raw", name: "Krea 2 Raw" },
];

const ASSETS = [
  { id: "asset_a", displayName: "Harbour at dawn" },
  { id: "asset_b", displayName: "Lighthouse plate" },
];

describe("WorkflowMissingInputs", () => {
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
      root.render(<WorkflowMissingInputs assets={ASSETS} models={MODELS} {...props} />),
    );

  const text = () => container.textContent;
  const selects = () => [...container.querySelectorAll("select")];

  // ---- The zero state that must not exist -----------------------------------------------

  it("renders nothing at all for a fully-resolvable recipe", async () => {
    await render({ report: report() });
    // Not "renders an empty container": renders NOTHING. A heading with nothing under it, or a
    // "Nothing missing" line, makes a working case look like it needs attention on a screen the
    // user reached by dragging in an image.
    expect(container.innerHTML).toBe("");
  });

  it("renders nothing for a recipe whose only problem is one nobody can fix", async () => {
    // An omitted collection makes the report unrunnable and has no action behind it — the report
    // rows say so, and this surface would have nothing to offer but a heading.
    await render({
      report: report({
        runnable: false,
        omitted: [{ field: "loras", detail: "This workflow named more LoRAs than a job can carry." }],
        styles: [{ field: "styleId", id: "ghibli", state: "missing", detail: "Style `ghibli` is not here." }],
      }),
    });
    expect(container.innerHTML).toBe("");
  });

  it("renders nothing when there is no report at all", async () => {
    await render({ report: null });
    expect(container.innerHTML).toBe("");
  });

  // ---- Install ---------------------------------------------------------------------------

  it("offers an install for a model the catalog knows but has not downloaded", async () => {
    const onInstall = vi.fn();
    await render({
      onInstall,
      report: report({
        runnable: false,
        model: {
          slug: "krea_2_turbo",
          state: "installable",
          catalogId: "krea_2_turbo",
          name: "Krea 2 Turbo",
          detail: "Krea 2 Turbo is in the model catalog but is not downloaded on this machine.",
          install: { method: "POST", path: "/api/v1/models/krea_2_turbo/download" },
        },
      }),
    });
    expect(text()).toContain("Model — Krea 2 Turbo");
    expect(text()).toContain("is not downloaded on this machine");
    const download = [...container.querySelectorAll("button")].find(
      (node) => node.textContent.trim() === "Download",
    );
    expect(download).toBeTruthy();
    await act(async () => download.dispatchEvent(new MouseEvent("click", { bubbles: true })));
    expect(onInstall).toHaveBeenCalledWith(
      expect.objectContaining({ kind: "model", id: "krea_2_turbo" }),
    );
  });

  it("offers no install for a requirement nothing here can fetch", async () => {
    // `missing` is the state the report gives a row it knows about but cannot download, and one it
    // has never heard of. A button either way would be a button that cannot work.
    await render({
      report: report({
        runnable: false,
        loras: [
          {
            name: "Aurora Portrait v3",
            state: "missing",
            detail: "No LoRA on this install matches `Aurora Portrait v3`.",
          },
        ],
      }),
    });
    expect(container.innerHTML).toBe("");
  });

  it("offers a LoRA install and marks it queued once started", async () => {
    const lora = {
      name: "Film Grain",
      state: "installable",
      catalogId: "film_grain",
      detail: "Film Grain is in the LoRA catalog but is not downloaded.",
      install: { method: "POST", path: "/api/v1/loras/film_grain/download" },
    };
    await render({ report: report({ runnable: false, loras: [lora] }) });
    expect(text()).toContain("LoRA — Film Grain");
    await render({
      installed: { "lora:film_grain": true },
      report: report({ runnable: false, loras: [lora] }),
    });
    const queued = [...container.querySelectorAll("button")].find(
      (node) => node.textContent.trim() === "Queued",
    );
    expect(queued?.disabled).toBe(true);
  });

  // ---- The substitute picker -------------------------------------------------------------

  it("names an unknown model and offers a substitute that is not preselected", async () => {
    const onSubstituteModel = vi.fn();
    await render({
      onSubstituteModel,
      report: report({
        runnable: false,
        model: {
          slug: "someones_private_model",
          state: "missing",
          detail:
            "No model matching `someones_private_model` is in this install's catalog, so this " +
            "workflow cannot be reproduced here.",
        },
      }),
    });
    expect(text()).toContain("someones_private_model");
    expect(text()).toContain("nothing is chosen for you");
    const picker = selects()[0];
    // THE test for "never silently substitutes": the control starts on the placeholder, not on the
    // first installed model, not on anything.
    expect(picker.value).toBe("");
    expect([...picker.options].map((option) => option.value)).toEqual([
      "",
      "z_image_turbo",
      "krea_2_raw",
    ]);
    picker.value = "z_image_turbo";
    await act(async () => picker.dispatchEvent(new Event("change", { bubbles: true })));
    expect(onSubstituteModel).toHaveBeenCalledWith("z_image_turbo");
  });

  it("offers no substitute picker for a model that merely needs downloading", async () => {
    // Installing the model the file NAMES reproduces the image; substituting one does not. Offering
    // both side by side invites the wrong one.
    await render({
      report: report({
        runnable: false,
        model: {
          slug: "krea_2_turbo",
          state: "installable",
          catalogId: "krea_2_turbo",
          name: "Krea 2 Turbo",
          detail: "Krea 2 Turbo is in the model catalog but is not downloaded on this machine.",
          install: { method: "POST", path: "/api/v1/models/krea_2_turbo/download" },
        },
      }),
    });
    expect(selects()).toHaveLength(0);
  });

  // ---- Input images ----------------------------------------------------------------------

  it("renders one picker per required image, with the kind and count from the descriptor", async () => {
    const onPickInput = vi.fn();
    await render({
      onPickInput,
      report: report({
        inputImagesRequired: 3,
        inputs: [
          { kind: "source", count: 1, state: "userSupplied", detail: "Needs a source image." },
          {
            kind: "reference",
            count: 2,
            state: "userSupplied",
            detail: "Needs 2 reference images.",
          },
        ],
      }),
    });
    // `reference` travels as ONE entry with a count, never as N entries — a panel that renders one
    // picker for it quietly asks for half the recipe.
    expect(selects()).toHaveLength(3);
    expect(text()).toContain("Source image");
    expect(text()).toContain("Reference image 1 of 2");
    expect(text()).toContain("Reference image 2 of 2");
    expect(text()).toContain("Needs a source image.");
    expect(text()).toContain("Needs 2 reference images.");
    // Every project image is offered, and nothing is preselected.
    expect(selects()[0].value).toBe("");
    expect([...selects()[0].options].map((option) => option.textContent)).toEqual([
      "Choose an image…",
      "Harbour at dawn",
      "Lighthouse plate",
    ]);
    selects()[0].value = "asset_b";
    await act(async () => selects()[0].dispatchEvent(new Event("change", { bubbles: true })));
    expect(onPickInput).toHaveBeenCalledWith(expect.stringContaining("source"), "asset_b");
  });

  it("names a mask without offering a picker whose value would go nowhere", async () => {
    // Image Studio has no mask control — a mask is painted in the Image Editor. Offering a select
    // here would be the inert control this epic exists to prevent, so the row states the
    // requirement and says where it is actually supplied.
    await render({
      report: report({
        inputImagesRequired: 1,
        inputs: [{ kind: "mask", count: 1, state: "userSupplied", detail: "Needs a mask." }],
      }),
    });
    expect(text()).toContain("Needs a mask.");
    expect(text()).toContain("Image Editor");
    expect(selects()).toHaveLength(0);
  });

  it("floors an absent count at one picker rather than rendering none", async () => {
    // The report's own total floors a zero count at 1; rendering zero pickers for a requirement it
    // is simultaneously counting would leave the CTA blocked with nothing to click.
    await render({
      report: report({
        inputImagesRequired: 1,
        inputs: [{ kind: "source", count: 0, state: "userSupplied", detail: "Needs a source image." }],
      }),
    });
    expect(selects()).toHaveLength(1);
  });
});
