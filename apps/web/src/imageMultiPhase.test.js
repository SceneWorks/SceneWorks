import { describe, expect, it } from "vitest";

import { summarize } from "./validation/issues.js";
import {
  MULTIPHASE_MAX_PHASES,
  MULTIPHASE_MAX_TOTAL_STEPS,
  TURBO_FINISH_PRESET,
  acceleratorLora,
  acceleratorLoraIndex,
  addPhase,
  buildTurboFinishPhases,
  effectiveTotalSteps,
  modelSupportsMultiPhase,
  movePhase,
  multiPhaseIssues,
  newPhase,
  phaseHasLora,
  removePhase,
  serializePhases,
  setPhaseLoraWeight,
  togglePhaseLora,
  updatePhase,
} from "./imageMultiPhase.js";

// Krea 2 multi-phase denoise editor (epic 13879 S5, sc-13885). The load-bearing guarantee is the
// ROUND-TRIP: the editor must emit the EXACT `advanced.phases` shape the worker's
// `parse_multiphase_specs` accepts (crates/sceneworks-worker/src/image_jobs/krea_multiphase.rs), so
// a job built here renders on the worker unchanged. These tests pin that shape and the state ops
// (add/remove/reorder, per-phase LoRA toggle by the selected-LoRA list, reindex/prune, validation).

// Two selected LoRAs — a style adapter and the turbo accelerator (role: "accelerator"). This is the
// job's own LoRA stack; a phase references it by INDEX (== request.loras == LoadSpec::adapters).
const STYLE = { id: "style-lora", name: "Watercolor", role: null };
const TURBO = { id: "krea2_turbo_accel", name: "Krea Turbo", role: "accelerator" };

describe("modelSupportsMultiPhase", () => {
  it("is true only for a model advertising the acceleration LoRA compat (krea_2_raw, sc-13882)", () => {
    expect(
      modelSupportsMultiPhase({ loraCompatibility: { types: ["character", "style", "acceleration"] } }),
    ).toBe(true);
    // Turbo advertises character/style but NOT acceleration → no multi-phase lane.
    expect(modelSupportsMultiPhase({ loraCompatibility: { types: ["character", "style"] } })).toBe(false);
    expect(modelSupportsMultiPhase({})).toBe(false);
    expect(modelSupportsMultiPhase(null)).toBe(false);
  });
});

describe("acceleratorLora / acceleratorLoraIndex", () => {
  it("finds the selected LoRA carrying role accelerator, or -1 / null", () => {
    expect(acceleratorLoraIndex([STYLE, TURBO])).toBe(1);
    expect(acceleratorLora([STYLE, TURBO])).toBe(TURBO);
    expect(acceleratorLoraIndex([STYLE])).toBe(-1);
    expect(acceleratorLora([STYLE])).toBeNull();
  });
});

describe("buildTurboFinishPhases + serializePhases — the canonical S4 (Comfy 4+4) round-trip", () => {
  it("produces EXACTLY the worker-shaped phase list wired to the selected turbo LoRA", () => {
    const phases = buildTurboFinishPhases([STYLE, TURBO]);
    // The emitted shape MUST equal what parse_multiphase_specs expects: phase 1 Raw true-CFG
    // base-only, phase 2 Raw + the turbo LoRA (index 1 in the job's loras) with CFG off.
    expect(serializePhases(phases, [STYLE, TURBO])).toEqual([
      { steps: 4, guidance: 3.5, loras: [] },
      { steps: 4, guidance: 0, loras: [{ index: 1 }] },
    ]);
  });

  it("wires the accelerator at whatever index it occupies in the selected-LoRA list", () => {
    // Turbo first, style second → the finishing phase references index 0.
    const phases = buildTurboFinishPhases([TURBO, STYLE]);
    expect(serializePhases(phases, [TURBO, STYLE])[1].loras).toEqual([{ index: 0 }]);
  });

  it("leaves the finishing phase base-only when no accelerator LoRA is selected", () => {
    const phases = buildTurboFinishPhases([STYLE]);
    expect(serializePhases(phases, [STYLE])).toEqual([
      { steps: 4, guidance: 3.5, loras: [] },
      { steps: 4, guidance: 0, loras: [] },
    ]);
  });

  it("has stable preset identity", () => {
    expect(TURBO_FINISH_PRESET).toEqual({ id: "turbo_finish_4_4", label: "Turbo finish (4+4)" });
  });
});

describe("serializePhases — emit shape matches parse_multiphase_specs", () => {
  it("emits steps as int, guidance as a number (0 kept), and per-lora { index, weight? }", () => {
    // Mirror the worker's own multiphase test fixture (tests.rs): two loras, a base-only CFG-on
    // phase then a CFG-off phase activating adapter 1 @ 0.9 then adapter 0 at load-time scale.
    const phases = [
      { steps: 20, guidance: 3.5, loras: [] },
      {
        steps: 8,
        guidance: 0,
        loras: [
          { id: TURBO.id, weight: 0.9 },
          { id: STYLE.id, weight: null },
        ],
      },
    ];
    expect(serializePhases(phases, [STYLE, TURBO])).toEqual([
      { steps: 20, guidance: 3.5, loras: [] },
      { steps: 8, guidance: 0, loras: [{ index: 1, weight: 0.9 }, { index: 0 }] },
    ]);
  });

  it("truncates a fractional steps and drops a blank weight override", () => {
    const phases = [{ steps: 6, guidance: 2, loras: [{ id: STYLE.id, weight: "" }] }];
    expect(serializePhases(phases, [STYLE])).toEqual([{ steps: 6, guidance: 2, loras: [{ index: 0 }] }]);
  });
});

describe("serializePhases — reindex + prune when the selected-LoRA list changes (no stale index)", () => {
  // A phase built against [STYLE, TURBO] that activates the turbo LoRA (stored by stable id).
  const phases = [
    { steps: 4, guidance: 3.5, loras: [] },
    { steps: 4, guidance: 0, loras: [{ id: TURBO.id, weight: null }] },
  ];

  it("REINDEXES when the selected LoRAs are reordered", () => {
    // Turbo moves from index 1 to index 0 → the emitted index follows the LoRA, not the slot.
    expect(serializePhases(phases, [TURBO, STYLE])[1].loras).toEqual([{ index: 0 }]);
    expect(serializePhases(phases, [STYLE, TURBO])[1].loras).toEqual([{ index: 1 }]);
  });

  it("PRUNES a reference whose LoRA is no longer selected", () => {
    // Turbo deselected → the finishing phase drops it (no dangling index), becomes base-only.
    expect(serializePhases(phases, [STYLE])[1].loras).toEqual([]);
    expect(serializePhases(phases, [])[1].loras).toEqual([]);
  });

  it("reindexes correctly after an insertion shifts positions", () => {
    const extra = { id: "extra-lora", role: null };
    // Insert a new LoRA before turbo → turbo shifts to index 2.
    expect(serializePhases(phases, [STYLE, extra, TURBO])[1].loras).toEqual([{ index: 2 }]);
  });
});

describe("phase list state ops (add / remove / reorder / update / toggle / weight)", () => {
  it("adds and removes phases without mutating the input", () => {
    const base = [newPhase()];
    const added = addPhase(base, newPhase({ steps: 8 }));
    expect(added).toHaveLength(2);
    expect(base).toHaveLength(1); // immutable
    expect(removePhase(added, 0)).toEqual([{ steps: 8, guidance: 3.5, loras: [] }]);
  });

  it("reorders phases with movePhase and no-ops at the ends", () => {
    const phases = [{ steps: 1 }, { steps: 2 }, { steps: 3 }];
    expect(movePhase(phases, 0, 1)).toEqual([{ steps: 2 }, { steps: 1 }, { steps: 3 }]);
    expect(movePhase(phases, 2, -1)).toEqual([{ steps: 1 }, { steps: 3 }, { steps: 2 }]);
    expect(movePhase(phases, 0, -1)).toBe(phases); // top phase can't move up
    expect(movePhase(phases, 2, 1)).toBe(phases); // bottom phase can't move down
  });

  it("updates a single phase's fields", () => {
    const phases = [newPhase(), newPhase()];
    expect(updatePhase(phases, 1, { steps: 12, guidance: 0 })[1]).toEqual({
      steps: 12,
      guidance: 0,
      loras: [],
    });
    expect(updatePhase(phases, 1, { steps: 12 })[0]).toEqual(newPhase()); // other phase untouched
  });

  it("toggles a LoRA on/off for a phase by stable id and edits its weight", () => {
    const phase = newPhase();
    const withTurbo = togglePhaseLora(phase, TURBO.id);
    expect(phaseHasLora(withTurbo, TURBO.id)).toBe(true);
    expect(withTurbo.loras).toEqual([{ id: TURBO.id, weight: null }]);
    const weighted = setPhaseLoraWeight(withTurbo, TURBO.id, 0.8);
    expect(weighted.loras).toEqual([{ id: TURBO.id, weight: 0.8 }]);
    expect(phaseHasLora(togglePhaseLora(weighted, TURBO.id), TURBO.id)).toBe(false); // toggled off
  });
});

describe("effectiveTotalSteps", () => {
  it("sums the phases' steps (the ONE global schedule length)", () => {
    expect(effectiveTotalSteps([{ steps: 4 }, { steps: 8 }, { steps: 2 }])).toBe(14);
    expect(effectiveTotalSteps([{ steps: "" }, { steps: 6 }])).toBe(6); // blank counts as 0
    expect(effectiveTotalSteps([])).toBe(0);
  });
});

describe("multiPhaseIssues — the requirement/error split (epic 10644)", () => {
  const kinds = (issues) => issues.map((i) => i.kind);

  it("is silent (no issues) when the editor is disabled", () => {
    expect(multiPhaseIssues({ enabled: false, phases: [] })).toEqual([]);
    // Even a broken list is silent while disabled — a disabled editor emits no phases.
    expect(multiPhaseIssues({ enabled: false, phases: [{ steps: 0 }] })).toEqual([]);
  });

  it("ERRORS (not a silent requirement) when enabled with no phases", () => {
    const issues = multiPhaseIssues({ enabled: true, phases: [] });
    expect(kinds(issues)).toEqual(["error"]);
    // An error surfaces and blocks; a requirement would block silently.
    const rolled = summarize(issues);
    expect(rolled.ready).toBe(false);
    expect(rolled.surfaced).toHaveLength(1);
  });

  it("ERRORS on a 0-step / blank phase", () => {
    expect(summarize(multiPhaseIssues({ enabled: true, phases: [{ steps: 0, guidance: 3.5 }] })).ready).toBe(false);
    const blank = multiPhaseIssues({ enabled: true, phases: [{ steps: "", guidance: 3.5 }] });
    expect(kinds(blank)).toContain("error");
  });

  it("ERRORS on a negative guidance", () => {
    const issues = multiPhaseIssues({ enabled: true, phases: [{ steps: 4, guidance: -1 }] });
    expect(kinds(issues)).toContain("error");
  });

  it("ERRORS over the phase-count and total-step guardrails (mirrors the worker)", () => {
    const many = Array.from({ length: MULTIPHASE_MAX_PHASES + 1 }, () => ({ steps: 1, guidance: 1 }));
    expect(summarize(multiPhaseIssues({ enabled: true, phases: many })).ready).toBe(false);
    const overBudget = [{ steps: MULTIPHASE_MAX_TOTAL_STEPS + 1, guidance: 1 }];
    expect(summarize(multiPhaseIssues({ enabled: true, phases: overBudget })).ready).toBe(false);
  });

  it("passes a valid canonical list", () => {
    const phases = buildTurboFinishPhases([STYLE, TURBO]);
    expect(summarize(multiPhaseIssues({ enabled: true, phases })).ready).toBe(true);
  });
});
