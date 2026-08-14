// The CTA gate on "Use this recipe" for a shared workflow (sc-15952, epic 15945).
//
// The rule set is a pure function, so the split this file pins is the one that decides whether the
// feature is usable at all: what BLOCKS (things the user can clear in front of the panel) versus
// what is merely SURFACED (things no action here could ever change). A rule set that blocked on a
// user-trained LoRA would leave "Use this recipe" permanently dead for exactly the recipes this
// feature exists to rescue.
import { describe, expect, it } from "vitest";

import { summarize } from "./validation/issues.js";
import {
  keptSubstituteModelId,
  modelSupportsWorkflowPoseReplay,
  pickedReferenceAssetIds,
  pickedSourceAssetId,
  workflowReplayValidation,
  workflowSubstituteModels,
} from "./workflowReplayValidation.js";

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

const summaryFor = (draft) => summarize(workflowReplayValidation(draft));
const messages = (draft) => summaryFor(draft).surfaced.map((item) => item.message);

describe("workflowReplayValidation", () => {
  it("lets a fully-resolvable recipe through with nothing to say", () => {
    const summary = summaryFor({ report: report() });
    expect(summary.ready).toBe(true);
    expect(summary.surfaced).toEqual([]);
  });

  it("blocks on an unknown model and names it", () => {
    // The acceptance criterion, stated as one assertion: not merely blocked, blocked WITH the
    // reason on screen. The two come from one `summarize`, so they cannot drift.
    const summary = summaryFor({
      report: report({
        runnable: false,
        model: {
          slug: "someones_private_model",
          state: "missing",
          detail: "No model matching `someones_private_model` is in this install's catalog.",
        },
      }),
    });
    expect(summary.ready).toBe(false);
    expect(summary.surfaced.map((item) => item.message).join(" ")).toContain(
      "someones_private_model",
    );
    expect(summary.surfaced.map((item) => item.message).join(" ")).toContain(
      "nothing is substituted for you",
    );
  });

  it("unblocks once the user has chosen a model themselves", () => {
    const draft = {
      report: report({
        runnable: false,
        model: { slug: "someones_private_model", state: "missing", detail: "…" },
      }),
    };
    expect(summaryFor(draft).ready).toBe(false);
    expect(summaryFor({ ...draft, substituteModelId: "z_image_turbo" }).ready).toBe(true);
  });

  it("blocks on a model that is merely not downloaded, and says install OR choose", () => {
    const summary = summaryFor({
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
    expect(summary.ready).toBe(false);
    expect(summary.surfaced[0].message).toContain("Install it");
    expect(summary.surfaced[0].message).toContain("Krea 2 Turbo");
  });

  it("does NOT block on a LoRA nobody can install, but says what the image will lose", () => {
    // The line this rule set draws. A user-trained adapter is on no catalog and behind no
    // download; blocking on it makes a permanently dead button, which is not a safety property.
    const summary = summaryFor({
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
    expect(summary.ready).toBe(true);
    expect(summary.surfaced).toHaveLength(1);
    expect(summary.surfaced[0].kind).toBe("advisory");
    expect(summary.surfaced[0].message).toContain("Aurora Portrait v3");
    expect(summary.surfaced[0].message).toContain("will differ from the one you were sent");
  });

  it("does not block on an omitted collection or an absent style, and surfaces both", () => {
    const summary = summaryFor({
      report: report({
        runnable: false,
        styles: [
          { field: "styleId", id: "ghibli", state: "missing", detail: "Style `ghibli` is not here." },
        ],
        omitted: [{ field: "loras", detail: "This workflow named more LoRAs than a job can carry." }],
      }),
    });
    expect(summary.ready).toBe(true);
    expect(summary.surfaced.map((item) => item.kind)).toEqual(["advisory", "advisory"]);
  });

  it("blocks until every required image is picked, silently", () => {
    const draft = {
      report: report({
        inputImagesRequired: 3,
        inputs: [
          { kind: "source", count: 1, state: "userSupplied", detail: "Needs a source image." },
          { kind: "reference", count: 2, state: "userSupplied", detail: "Needs 2 reference images." },
        ],
      }),
    };
    expect(summaryFor(draft).ready).toBe(false);
    // Requirements are silent by design (`validation/issues.js`): an empty picker is its own
    // message, and a chip repeating "choose a source image" beside it is noise.
    expect(summaryFor(draft).surfaced).toEqual([]);

    const slots = ["source-0-0", "reference-1-0", "reference-1-1"];
    const partly = { ...draft, inputPicks: { [slots[0]]: "asset_a", [slots[1]]: "asset_b" } };
    expect(summaryFor(partly).ready).toBe(false);
    const filled = { ...draft, inputPicks: Object.fromEntries(slots.map((key) => [key, "asset_a"])) };
    expect(summaryFor(filled).ready).toBe(true);
  });

  it("says where a mask comes from without refusing to load the recipe first", () => {
    // A mask is painted in the Image Editor, AFTER a recipe is loaded. Blocking on it would refuse
    // to perform the very step its own message tells the user to take, so it is surfaced instead —
    // the opposite disposition from a missing source image, which a picker right there can fill.
    const summary = summaryFor({
      report: report({
        inputImagesRequired: 1,
        inputs: [{ kind: "mask", count: 1, state: "userSupplied", detail: "Needs a mask." }],
      }),
    });
    expect(summary.ready).toBe(true);
    expect(summary.surfaced[0].kind).toBe("advisory");
    expect(summary.surfaced[0].message).toContain("Needs a mask.");
    expect(summary.surfaced[0].message).toContain("Image Editor");
  });

  it("says what proceeding without the mask actually produces, not just where masks live", () => {
    // The honesty test every advisory in this set has to pass: "does a user who proceeds get an
    // image they believe is a faithful replay but is not?" For a mask the answer is yes, and it is
    // the worst instance of it — `maskAssetId` reaches the worker ONLY from the Image Editor's
    // inpaint brush, Image Studio has no mask control, and its Generate gate checks the source
    // asset alone. So the CTA fires, the model rewrites the whole frame, and nothing downstream
    // notices. A sentence naming the LOCATION of the missing control ("painted in the Image
    // Editor") reads as a note about UI, not as a warning that a different operation is about to
    // run. This asserts the CONSEQUENCE is named.
    const summary = summaryFor({
      report: report({
        inputImagesRequired: 1,
        inputs: [{ kind: "mask", count: 1, state: "userSupplied", detail: "Needs a mask." }],
      }),
    });
    const message = summary.surfaced[0].message;
    expect(message).toMatch(/whole image/i);
    expect(message).toMatch(/different edit from the one you were sent/i);
    // And it still tells them how to actually get the masked version, so the row is a direction
    // rather than a dead end.
    expect(message).toMatch(/load the recipe and then open the image in the Image Editor/i);
    // Naming the consequence must not turn the advisory into a block: the sentence's own
    // instruction is "load the recipe first", so refusing to load it would be incoherent.
    expect(summary.ready).toBe(true);
  });

  it("blocks with a reason when this install was never checked against the recipe", () => {
    const summary = summaryFor({ report: null });
    expect(summary.ready).toBe(false);
    expect(summary.surfaced[0].message).toContain("has not been checked");
  });
});

describe("carrying the user's own answers to the launch", () => {
  it("filters pose-bearing substitutes to the visible pose lane for that workflow mode", () => {
    const models = [
      { id: "plain", capabilities: ["text_to_image"], ui: {} },
      { id: "strict", capabilities: ["text_to_image"], ui: { controlModes: ["pose"] } },
      { id: "edit_pose", capabilities: ["edit_image"], ui: { controlModes: ["pose"] } },
      {
        id: "character",
        capabilities: ["character_image"],
        ui: { poseLibrary: true },
      },
      { id: "library_only", capabilities: ["text_to_image"], ui: { poseLibrary: true } },
    ];
    const poses = [{ keypoints: [[0.2, 0.8]] }];

    expect(
      workflowSubstituteModels({ mode: "text_to_image", advanced: { poses } }, models).map(
        (model) => model.id,
      ),
    ).toEqual(["strict"]);
    expect(
      workflowSubstituteModels({ mode: "character_image", advanced: { poses } }, models).map(
        (model) => model.id,
      ),
    ).toEqual(["character"]);
    expect(workflowSubstituteModels({ mode: "text_to_image", advanced: {} }, models)).toBe(models);
  });

  it("applies Image Studio's whole-model, reference, and pose-feature Mac gates", () => {
    const macCapabilities = { macGatingActive: true };
    const textPose = {
      id: "text_pose",
      capabilities: ["text_to_image"],
      ui: { controlModes: ["pose"] },
      macSupport: { supported: true, features: { pose: true } },
    };
    const characterPose = {
      id: "character_pose",
      capabilities: ["character_image"],
      ui: { poseLibrary: true },
      macSupport: { supported: true, features: { pose: true, reference: true } },
    };

    expect(modelSupportsWorkflowPoseReplay(textPose, "text_to_image", macCapabilities)).toBe(true);
    expect(
      modelSupportsWorkflowPoseReplay(
        { ...textPose, macSupport: { supported: false, features: { pose: true } } },
        "text_to_image",
        macCapabilities,
      ),
    ).toBe(false);
    expect(
      modelSupportsWorkflowPoseReplay(
        { ...textPose, macSupport: { supported: true, features: { pose: false } } },
        "text_to_image",
        macCapabilities,
      ),
    ).toBe(false);
    expect(
      modelSupportsWorkflowPoseReplay(
        {
          ...characterPose,
          macSupport: { supported: true, features: { pose: true, reference: false } },
        },
        "character_image",
        macCapabilities,
      ),
    ).toBe(false);
    // The same declarations are inert when Mac gating is off.
    expect(
      modelSupportsWorkflowPoseReplay(
        { ...textPose, macSupport: { supported: true, features: { pose: false } } },
        "text_to_image",
        { macGatingActive: false },
      ),
    ).toBe(true);
  });

  it("drops a substitute model the catalog no longer offers", () => {
    // The panel can sit open across a catalog refetch: a model deleted (or a row that vanished)
    // must not stay selected, or the recipe seeds an id the studio's picker will drop on the next
    // render — a blank control, which is the silent loss the adapter refuses to produce.
    const models = [{ id: "z_image_turbo" }, { id: "krea_2_raw" }];
    expect(keptSubstituteModelId("z_image_turbo", models)).toBe("z_image_turbo");
    expect(keptSubstituteModelId("gone_model", models)).toBe("");
    expect(keptSubstituteModelId("", models)).toBe("");
    expect(keptSubstituteModelId("z_image_turbo", [])).toBe("");
  });

  it("hands the reference picks over in slot order", () => {
    // The studio's multi-reference lane treats the order as meaningful (image 1, image 2), so it
    // is the slot order and not whatever `Object.values` happens to yield.
    const here = report({
      inputs: [
        { kind: "source", count: 1, state: "userSupplied", detail: "Needs a source image." },
        { kind: "reference", count: 2, state: "userSupplied", detail: "Needs 2 reference images." },
      ],
    });
    const picks = {
      "reference-1-1": "second",
      "source-0-0": "the_source",
      "reference-1-0": "first",
    };
    expect(pickedReferenceAssetIds(here, picks)).toEqual(["first", "second"]);
    expect(pickedSourceAssetId(here, picks)).toBe("the_source");
  });

  it("answers empty rather than guessing when nothing was picked", () => {
    const here = report({
      inputs: [{ kind: "source", count: 1, state: "userSupplied", detail: "Needs a source image." }],
    });
    expect(pickedSourceAssetId(here, {})).toBe("");
    expect(pickedReferenceAssetIds(here, {})).toEqual([]);
    expect(pickedSourceAssetId(report(), {})).toBe("");
  });
});

describe("the messages a panel would show", () => {
  it("never says anything at all about a recipe this install can run outright", () => {
    expect(messages({ report: report() })).toEqual([]);
  });
});
