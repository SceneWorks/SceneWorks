import React from "react";
import { Icon } from "./Icons.jsx";
import {
  MULTIPHASE_MAX_PHASES,
  TURBO_FINISH_PRESET,
  acceleratorLoraIndex,
  addPhase,
  effectiveTotalSteps,
  movePhase,
  newPhase,
  phaseHasLora,
  removePhase,
  setPhaseLoraWeight,
  togglePhaseLora,
  updatePhase,
} from "../imageMultiPhase.js";

// Krea 2 multi-phase denoise editor (epic 13879 S5, sc-13885). A controlled, presentational editor
// for the ordered phase list that rides the image job as `advanced.phases`. Each phase owns a step
// count, a guidance value (0 = true-CFG off), and a per-selected-LoRA on/off + optional weight —
// the per-phase LoRA toggles reference the job's CURRENTLY-SELECTED LoRAs by index, so the emitted
// `loras:[{ index, weight? }]` maps 1:1 to the worker's load-time adapter stack.
//
// It surfaces under an experimental AdvancedSection (default collapsed) in ImageStudio; all state
// mutation goes through the pure helpers in imageMultiPhase.js so the same logic the tests pin also
// drives the UI. Error surfacing (empty list, 0-step phase, …) is the shared ValidationSummary via
// multiPhaseIssues — the app-wide requirement/error split.
export function MultiPhaseEditor({
  enabled,
  onToggleEnabled,
  phases = [],
  onChange,
  selectedLoras = [],
  onApplyTurboFinish,
}) {
  const accelIndex = acceleratorLoraIndex(selectedLoras);
  const total = effectiveTotalSteps(phases);
  const atMaxPhases = phases.length >= MULTIPHASE_MAX_PHASES;

  return (
    <div className="multiphase-editor">
      <label className="checkline multiphase-enable">
        <input
          checked={enabled}
          onChange={(event) => onToggleEnabled(event.target.checked)}
          type="checkbox"
        />
        Enable multi-phase denoise
      </label>
      <p className="field-hint multiphase-intro">
        Split ONE Raw denoise into ordered phases over a single schedule — each phase its own step
        count, guidance (0 = CFG off), and active LoRAs. The canonical flow is a few true-CFG Raw
        steps, then a few CFG-off steps with the turbo LoRA.
      </p>

      <div className="multiphase-presets">
        <button
          className="multiphase-btn multiphase-preset-btn"
          onClick={() => onApplyTurboFinish()}
          type="button"
        >
          {TURBO_FINISH_PRESET.label}
        </button>
        {accelIndex < 0 ? (
          <span className="field-hint multiphase-accel-hint">
            Select the Krea turbo LoRA above to wire it into the CFG-off finishing phase.
          </span>
        ) : null}
      </div>

      {enabled ? (
        <>
          {phases.length ? (
            <p className="field-hint multiphase-effective" role="status">
              Effective steps: <strong>{total}</strong> (sum of {phases.length}{" "}
              {phases.length === 1 ? "phase" : "phases"}). Phases drive the schedule — the Steps and
              Guidance fields above are ignored while multi-phase is on.
            </p>
          ) : null}

          <ol className="multiphase-phase-list">
            {phases.map((phase, phaseIndex) => {
              const guidance = Number(phase.guidance);
              const cfgOn = Number.isFinite(guidance) && guidance > 0;
              return (
                <li className="multiphase-phase" key={phaseIndex}>
                  <div className="multiphase-phase-head">
                    <span className="eyebrow multiphase-phase-title">Phase {phaseIndex + 1}</span>
                    <div className="multiphase-phase-actions">
                      <button
                        aria-label={`Move phase ${phaseIndex + 1} up`}
                        className="icon-btn"
                        disabled={phaseIndex === 0}
                        onClick={() => onChange(movePhase(phases, phaseIndex, -1))}
                        type="button"
                      >
                        ↑
                      </button>
                      <button
                        aria-label={`Move phase ${phaseIndex + 1} down`}
                        className="icon-btn"
                        disabled={phaseIndex === phases.length - 1}
                        onClick={() => onChange(movePhase(phases, phaseIndex, 1))}
                        type="button"
                      >
                        ↓
                      </button>
                      <button
                        aria-label={`Remove phase ${phaseIndex + 1}`}
                        className="icon-btn multiphase-remove"
                        onClick={() => onChange(removePhase(phases, phaseIndex))}
                        type="button"
                      >
                        <Icon.Trash />
                      </button>
                    </div>
                  </div>

                  <div className="multiphase-phase-fields">
                    <label>
                      Steps
                      <input
                        min="1"
                        onChange={(event) =>
                          onChange(
                            updatePhase(phases, phaseIndex, {
                              steps: event.target.value === "" ? "" : Number(event.target.value),
                            }),
                          )
                        }
                        type="number"
                        value={phase.steps}
                      />
                    </label>
                    <label>
                      Guidance
                      <input
                        min="0"
                        onChange={(event) =>
                          onChange(
                            updatePhase(phases, phaseIndex, {
                              guidance:
                                event.target.value === "" ? "" : Number(event.target.value),
                            }),
                          )
                        }
                        step="0.1"
                        type="number"
                        value={phase.guidance}
                      />
                      <span className="field-hint multiphase-cfg-hint">
                        {cfgOn ? "CFG on" : "CFG off"}
                      </span>
                    </label>
                  </div>

                  <div className="multiphase-phase-loras">
                    <span className="eyebrow">Active LoRAs</span>
                    {selectedLoras.length ? (
                      selectedLoras.map((lora, loraIndex) => {
                        const ref = (phase.loras ?? []).find((entry) => entry.id === lora.id);
                        const active = phaseHasLora(phase, lora.id);
                        return (
                          <div className="multiphase-lora-row" key={lora.id ?? loraIndex}>
                            <label className="checkline">
                              <input
                                checked={active}
                                onChange={() =>
                                  onChange(
                                    updatePhase(phases, phaseIndex, togglePhaseLora(phase, lora.id)),
                                  )
                                }
                                type="checkbox"
                              />
                              {lora.name ?? lora.id}
                              {loraIndex === accelIndex ? (
                                <span className="badge multiphase-accel-badge">turbo</span>
                              ) : null}
                            </label>
                            {active ? (
                              <input
                                aria-label={`Phase ${phaseIndex + 1} weight for ${lora.name ?? lora.id}`}
                                className="multiphase-lora-weight"
                                onChange={(event) =>
                                  onChange(
                                    updatePhase(
                                      phases,
                                      phaseIndex,
                                      setPhaseLoraWeight(
                                        phase,
                                        lora.id,
                                        event.target.value === "" ? null : Number(event.target.value),
                                      ),
                                    ),
                                  )
                                }
                                placeholder="load-time"
                                step="0.05"
                                type="number"
                                value={ref?.weight ?? ""}
                              />
                            ) : null}
                          </div>
                        );
                      })
                    ) : (
                      <span className="field-hint">
                        No LoRAs selected — this phase renders base-only. Pick LoRAs above to toggle
                        them per phase.
                      </span>
                    )}
                  </div>
                </li>
              );
            })}
          </ol>

          <button
            className="multiphase-btn multiphase-add"
            disabled={atMaxPhases}
            onClick={() => onChange(addPhase(phases, newPhase()))}
            type="button"
          >
            <Icon.Plus /> Add phase
          </button>
          {atMaxPhases ? (
            <span className="field-hint">Maximum of {MULTIPHASE_MAX_PHASES} phases.</span>
          ) : null}
        </>
      ) : null}
    </div>
  );
}
