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
  pickedReferenceAssetIds,
  pickedSourceAssetId,
  workflowReplayValidation,
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

  it("blocks with a reason when this install was never checked against the recipe", () => {
    const summary = summaryFor({ report: null });
    expect(summary.ready).toBe(false);
    expect(summary.surfaced[0].message).toContain("has not been checked");
  });
});

describe("carrying the user's own answers to the launch", () => {
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
