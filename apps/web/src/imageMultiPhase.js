import { issue } from "./validation/issues.js";

// Krea 2 multi-phase denoise editor (epic 13879 S5, sc-13885). The pure state → payload core for
// the Image Studio phase editor: it builds, mutates, validates and SERIALIZES the ordered phase
// list that rides the image job as `advanced.phases`, the contract the SceneWorks worker consumes in
// `crates/sceneworks-worker/src/image_jobs/krea_multiphase.rs` (`parse_multiphase_specs` /
// `build_generation_phases`, sc-13884 / S4).
//
// The emitted shape MUST match that parser EXACTLY so the request round-trips to the worker
// unchanged:
//
//   advanced.phases = [
//     { steps: <int ≥ 1>, guidance: <number ≥ 0>, loras: [ { index: <int>, weight?: <number> }, … ] },
//     …
//   ]
//
//   - `steps`      contiguous denoise steps for the phase; the schedule length is the SUM.
//   - `guidance`   per-phase true-CFG: 0 = CFG off, > 0 = CFG on. (The worker also accepts an
//                  omitted/null guidance = inherit the request default; the editor always writes an
//                  explicit value, so this module always emits one.)
//   - `loras`      the load-time adapters this phase activates, referencing the job's OWN selected
//                  LoRA list BY INDEX. `index` is the position in `request.loras` — the SAME array
//                  the worker resolves 1:1 to `LoadSpec::adapters`. An empty list = base-only.
//   - `weight`     optional per-phase weight override; omitted = the adapter's load-time scale.
//
// INTERNAL representation vs the CONTRACT: the editor stores each phase's LoRA references by the
// LoRA's stable `id` (`{ id, weight }`), NOT by raw index. `serializePhases` maps those ids to the
// CURRENT index in `selectedLoras` at emit time — so the emitted `index` is always valid, a reorder
// reindexes automatically, and a deselected LoRA is pruned, with no stale-index bookkeeping and no
// async-hydration hazard (indices are never persisted). The canonical S4 workflow (Comfy 4+4) is 4
// steps Raw true-CFG on (base-only), then 4 steps Raw + the turbo accelerator LoRA CFG off —
// `buildTurboFinishPhases` produces exactly that, wired to whichever selected LoRA carries
// `role: "accelerator"` (the `krea2_turbo_accel` builtin, sc-13882).

// Worker guardrails mirrored so an invalid list is caught BEFORE submit (krea_multiphase.rs
// MAX_MULTIPHASE_PHASES / MAX_MULTIPHASE_TOTAL_STEPS). Not creative limits — guardrails.
export const MULTIPHASE_MAX_PHASES = 8;
export const MULTIPHASE_MAX_TOTAL_STEPS = 150;

// The sampling-regime role a LoRA declares to be a step-distill / turbo accelerator (sc-13882,
// serialized by presetUtils.serializeLora as `role`). Phase 2 of the turbo-finish preset toggles it.
export const ACCELERATOR_ROLE = "accelerator";

// The LoRA-compat type only the model with the multi-phase worker lane advertises: krea_2_raw's
// `loraCompatibility.types` includes "acceleration" (config/manifests/builtin.models.jsonc, sc-13882).
// This is the DECLARATIVE gate for showing the editor — no hardcoded model id — and matches the
// worker gate (`request.model == KREA_RAW_MODEL_ID`), since only Raw declares this compat today.
export const MULTIPHASE_COMPAT_TYPE = "acceleration";

// Turbo-finish preset identity (the canonical S4 example).
export const TURBO_FINISH_PRESET = { id: "turbo_finish_4_4", label: "Turbo finish (4+4)" };

// True when the selected model exposes the multi-phase denoise lane. Gated on the acceleration LoRA
// compat the Raw model declares — the same signal that surfaces the turbo accelerator LoRA in the
// picker (sc-13882). The Studio further gates on text-to-image mode (multi-phase renders from pure
// noise; the worker rejects edit/pose/reference/PiD shapes).
export function modelSupportsMultiPhase(model) {
  const types = model?.loraCompatibility?.types;
  return Array.isArray(types) && types.includes(MULTIPHASE_COMPAT_TYPE);
}

// The selected turbo accelerator LoRA (the one carrying `role: "accelerator"`), or null.
export function acceleratorLora(selectedLoras = []) {
  return (
    selectedLoras.find(
      (lora) => String(lora?.role ?? "").trim().toLowerCase() === ACCELERATOR_ROLE,
    ) ?? null
  );
}

// The index of the selected accelerator LoRA in `selectedLoras`, or -1. (Index into the SAME array
// `request.loras` is built from, so it is a valid phase index — used by the editor's "turbo" badge.)
export function acceleratorLoraIndex(selectedLoras = []) {
  return selectedLoras.findIndex(
    (lora) => String(lora?.role ?? "").trim().toLowerCase() === ACCELERATOR_ROLE,
  );
}

// A fresh phase: 4 steps, true-CFG on (guidance 3.5, the Raw default), base-only.
export function newPhase(overrides = {}) {
  return { steps: 4, guidance: 3.5, loras: [], ...overrides };
}

// The canonical S4 "Turbo finish (4+4)" example: phase 1 = 4 steps Raw true-CFG base-only; phase 2 =
// 4 steps Raw + the selected turbo accelerator LoRA, CFG off. When no accelerator LoRA is selected,
// phase 2 is left base-only (the Studio surfaces the "select the turbo LoRA" hint) so the structure
// is still produced — the per-phase toggle can wire the LoRA once it is picked.
export function buildTurboFinishPhases(selectedLoras = []) {
  const accel = acceleratorLora(selectedLoras);
  return [
    { steps: 4, guidance: 3.5, loras: [] },
    { steps: 4, guidance: 0, loras: accel ? [{ id: accel.id, weight: null }] : [] },
  ];
}

export function addPhase(phases = [], phase = newPhase()) {
  return [...phases, phase];
}

export function removePhase(phases = [], at) {
  return phases.filter((_, index) => index !== at);
}

// Move the phase at `at` by `delta` (-1 up / +1 down); a no-op at the ends.
export function movePhase(phases = [], at, delta) {
  const target = at + delta;
  if (at < 0 || at >= phases.length || target < 0 || target >= phases.length) {
    return phases;
  }
  const next = [...phases];
  [next[at], next[target]] = [next[target], next[at]];
  return next;
}

export function updatePhase(phases = [], at, patch) {
  return phases.map((phase, index) => (index === at ? { ...phase, ...patch } : phase));
}

// Toggle a selected LoRA (by its stable id) on/off for one phase.
export function togglePhaseLora(phase, loraId) {
  const loras = phase.loras ?? [];
  const existing = loras.some((ref) => ref.id === loraId);
  return {
    ...phase,
    loras: existing
      ? loras.filter((ref) => ref.id !== loraId)
      : [...loras, { id: loraId, weight: null }],
  };
}

// Set (or clear, with null) a phase LoRA's per-phase weight override.
export function setPhaseLoraWeight(phase, loraId, weight) {
  return {
    ...phase,
    loras: (phase.loras ?? []).map((ref) => (ref.id === loraId ? { ...ref, weight } : ref)),
  };
}

// True when a phase currently activates the given LoRA id.
export function phaseHasLora(phase, loraId) {
  return (phase?.loras ?? []).some((ref) => ref.id === loraId);
}

// The effective total denoise budget = the sum of the phases' steps (the length of the ONE global
// schedule). Surfaced so the single flat "Steps" control isn't misleading when phases are active.
export function effectiveTotalSteps(phases = []) {
  return phases.reduce((sum, phase) => {
    const steps = Number(phase.steps);
    return sum + (Number.isFinite(steps) ? Math.max(0, Math.trunc(steps)) : 0);
  }, 0);
}

// Serialize the editor phase list into the EXACT `advanced.phases` shape the worker's
// `parse_multiphase_specs` accepts, resolving each phase LoRA's stable id to its CURRENT index in
// `selectedLoras` (== the `request.loras` order). This is where reindex + prune happen: a LoRA that
// is no longer selected is dropped, a reorder shifts the index, duplicates collapse — so the emitted
// `index` is always valid against the job's own LoRA stack. `steps` → an integer; `guidance` → a
// finite number (0 = CFG off); `weight` is emitted only when a finite override is set (absent =
// the adapter's load-time scale, the worker's `None`). `loras` is always emitted (an empty array =
// base-only, the canonical phase-1 shape).
export function serializePhases(phases = [], selectedLoras = []) {
  const indexById = new Map(selectedLoras.map((lora, index) => [lora.id, index]));
  return phases.map((phase) => {
    const out = { steps: Math.trunc(Number(phase.steps)) };
    const guidance = Number(phase.guidance);
    if (Number.isFinite(guidance)) {
      out.guidance = guidance;
    }
    const seen = new Set();
    out.loras = (phase.loras ?? []).flatMap((ref) => {
      const index = indexById.get(ref.id);
      if (index == null || seen.has(index)) {
        return []; // deselected LoRA (prune) or already referenced (dedupe).
      }
      seen.add(index);
      const entry = { index };
      const weight = Number(ref.weight);
      if (ref.weight != null && ref.weight !== "" && Number.isFinite(weight)) {
        entry.weight = weight;
      }
      return [entry];
    });
    return out;
  });
}

// The multi-phase rule set, in the app-wide validation vocabulary (epic 10644). Errors, not silent
// requirements: an enabled-but-broken phase list blocks Generate and nothing else on the form
// explains why. Returns [] when the editor is disabled — a disabled editor emits no phases, so a
// single-phase Raw job is byte-for-byte unchanged. Mirrors the worker's own rejects (empty list,
// 0-step phase, > MAX phases, total-step budget) so the request never leaves broken.
export function multiPhaseIssues({ enabled = false, phases = [] } = {}) {
  if (!enabled) {
    return [];
  }
  const issues = [];
  if (!phases.length) {
    issues.push(
      issue.error("multiPhase", "Add at least one phase, or turn off multi-phase denoise."),
    );
    return issues;
  }
  if (phases.length > MULTIPHASE_MAX_PHASES) {
    issues.push(
      issue.error(
        "multiPhase",
        `Multi-phase supports at most ${MULTIPHASE_MAX_PHASES} phases — remove ${
          phases.length - MULTIPHASE_MAX_PHASES
        }.`,
      ),
    );
  }
  phases.forEach((phase, index) => {
    const steps = Number(phase.steps);
    if (!Number.isFinite(steps) || !Number.isInteger(steps) || steps < 1) {
      issues.push(issue.error("multiPhase", `Phase ${index + 1} needs at least 1 step.`));
    }
    const guidance = Number(phase.guidance);
    if (!Number.isFinite(guidance) || guidance < 0) {
      issues.push(
        issue.error(
          "multiPhase",
          `Phase ${index + 1} guidance must be 0 or greater (0 = CFG off).`,
        ),
      );
    }
  });
  const total = effectiveTotalSteps(phases);
  if (total > MULTIPHASE_MAX_TOTAL_STEPS) {
    issues.push(
      issue.error(
        "multiPhase",
        `Multi-phase total steps ${total} exceed the ${MULTIPHASE_MAX_TOTAL_STEPS}-step budget — reduce phase steps.`,
      ),
    );
  }
  return issues;
}
