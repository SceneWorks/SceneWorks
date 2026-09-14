// sc-18727 — the turbo-variant derivation, unit-level.
//
// The studio tests next door prove the CHAIN (control → payload). These prove the three predicates
// that chain rests on, against inputs the shipped catalog cannot currently produce: a non-MiniMax
// family that ships an accelerator, an accelerator with no recipe, and a recipe-bearing entry that
// is not an accelerator. Without these the guards would be real but untestable — the shipped catalog
// happens to make every one of them redundant with some other check, which is exactly how a
// defence-in-depth guard rots into a no-op nobody notices.

import { describe, expect, it } from "vitest";

import {
  defaultTurboVariant,
  isTurboVariant,
  modelIsMinimaxH3,
  selectedTurboVariant,
  turboRecipeSummary,
  turboVariantsForModel,
} from "./minimaxH3Turbo.js";

const accelerator = (id, steps, schedulerShift, extra = {}) => ({
  id,
  name: id,
  family: "minimax-h3",
  role: "accelerator",
  sampling: { steps, schedulerShift, audioSchedulerShift: 3.0 },
  ...extra,
});

const MINIMAX = { id: "minimax_h3" };
const MINIMAX_REF = { id: "minimax_h3_ref" };

describe("minimaxH3Turbo", () => {
  it("claims an accelerator only when it also declares a usable recipe", () => {
    expect(isTurboVariant(accelerator("a", 4, 6.0))).toBe(true);
    // `role` alone is the Krea 2 turbo adapter's shape: intent, no numbers. The worker has nothing
    // to apply, so offering it under a control labelled "turbo" would promise a speedup that never
    // arrives.
    expect(isTurboVariant({ id: "b", role: "accelerator" })).toBe(false);
    // A recipe on a non-accelerator is equally not a turbo variant — the two keys mean different
    // things and only the pair is a claim.
    expect(
      isTurboVariant({ id: "c", sampling: { steps: 4, schedulerShift: 6.0 } }),
    ).toBe(false);
    // Degenerate recipes are refused rather than clamped: a 0-step or 0-shift schedule is not a
    // fast render, it is an unrenderable one.
    expect(isTurboVariant(accelerator("d", 0, 6.0))).toBe(false);
    expect(isTurboVariant(accelerator("e", 4, 0))).toBe(false);
    // The `role` spelling is normalised the way the worker's own reader normalises it.
    expect(isTurboVariant({ ...accelerator("f", 4, 6.0), role: " Accelerator " })).toBe(true);
    expect(isTurboVariant(null)).toBe(false);
  });

  it("offers nothing on a family whose accelerators this app cannot apply", () => {
    // No shipped non-MiniMax family declares a `sampling` block today, so this is the ONLY place
    // the family gate is observable. It is not decoration: the worker's turbo resolver returns "no
    // recipe" off this family, so a control offered here would be a knob whose every setting does
    // the same thing.
    const foreign = accelerator("some_other_turbo", 4, 6.0, { family: "wan-video" });
    expect(turboVariantsForModel({ id: "wan_2_2_t2v_14b" }, [foreign])).toEqual([]);
    expect(turboVariantsForModel(null, [foreign])).toEqual([]);
    // …and the same adapter list on a MiniMax-H3 model is offered, so the empty results above are
    // the MODEL gate rather than the adapter being rejected for some other reason.
    expect(turboVariantsForModel(MINIMAX, [foreign])).toEqual([foreign]);
  });

  it("recognises both catalog partitions", () => {
    expect(modelIsMinimaxH3(MINIMAX)).toBe(true);
    expect(modelIsMinimaxH3(MINIMAX_REF)).toBe(true);
    expect(modelIsMinimaxH3({ id: "mochi_1" })).toBe(false);
    expect(modelIsMinimaxH3(undefined)).toBe(false);
  });

  it("defaults to the checkpoint-paired variant, falling back rather than to nothing", () => {
    const p768 = accelerator("minimax_h3_turbo_4step_768p", 4, 6.0);
    const eight = accelerator("minimax_h3_turbo_8step", 8, 12.0);
    const ref2v = accelerator("minimax_h3_ref2v_turbo_4step", 4, 12.0);
    // Order in the list must not decide it: the 8-step file is FIRST here and must still lose to
    // the validated 768p one.
    expect(defaultTurboVariant(MINIMAX, [eight, p768])).toBe(p768);
    // The reference partition takes the ref2v adapter even when an fl2v one is available — the two
    // are one family and nothing else stops a Ref2VA job from folding a checkpoint it was not
    // trained against.
    expect(defaultTurboVariant(MINIMAX_REF, [p768, ref2v])).toBe(ref2v);
    // A user who installed only the 8-step file still gets accelerated.
    expect(defaultTurboVariant(MINIMAX, [eight])).toBe(eight);
    // Nothing installed is an honest null, not a variant that would 400 at enqueue.
    expect(defaultTurboVariant(MINIMAX, [])).toBeNull();
  });

  it("reads the active variant out of the LoRA selection, not a separate flag", () => {
    const p768 = accelerator("minimax_h3_turbo_4step_768p", 4, 6.0);
    const eight = accelerator("minimax_h3_turbo_8step", 8, 12.0);
    expect(selectedTurboVariant([p768, eight], ["some_style", "minimax_h3_turbo_8step"])).toBe(
      eight,
    );
    expect(selectedTurboVariant([p768, eight], ["some_style"])).toBeNull();
    expect(selectedTurboVariant([], ["minimax_h3_turbo_8step"])).toBeNull();
  });

  it("summarises BOTH schedulers, not just the video one", () => {
    // MiniMax-H3 denoises video and audio against two schedules. A summary naming only the video
    // shift would describe half the render — and the audio half is the one the request cannot move,
    // so it is the half a user has no other way to learn.
    expect(turboRecipeSummary(accelerator("x", 4, 6.0))).toBe(
      "4 steps, video shift 6, audio shift 3",
    );
    expect(turboRecipeSummary(null)).toBeNull();
  });
});
