import React, { useState } from "react";
import { Icon } from "./Icons.jsx";
import { LoraKeywordSummary } from "./LoraKeywordSummary.jsx";
import { LoraPickerList } from "./LoraPickerList.jsx";
import { StudioUpdateBadge, StudioUpdateNotice } from "./StudioUpdateNotice.jsx";
import { LORA_WEIGHT_MAX, LORA_WEIGHT_MIN, LORA_WEIGHT_STEP, MAX_USER_JOB_LORAS } from "../presetUtils.js";

// Shared LoRA picker for studios, Character, and editor rails.
// Callers own selection and eligibility; the picker owns its open/search state.
export function LoraPickerSection({
  selectedModel,
  label = "LoRAs",
  isLoraDisabled,
  selectedLoras,
  selectedLoraIds,
  compatibleLoras,
  userSelectedLoraCount,
  showIncompatibleLoras,
  setShowIncompatibleLoras,
  toggleLora,
  effectiveLoraWeight,
  setLoraWeight,
  loraEmptyMessage,
  onUpdateLora,
  showIncompatibleControl = false,
  // Optional node appended to the open picker — the studios pass the in-place Import LoRA form
  // (studio-cleanup sc-15373) so "no compatible LoRAs" has a fix right where it is reported. The
  // editor rails omit it and keep a list-only picker.
  importPanel = null,
}) {
  // Add-on-demand picker (UI-refinement 3b): only the LoRAs you've added render as
  // slots; everything else lives behind the "Add LoRA" dropdown. This replaces the
  // checkbox-per-LoRA wall so a large library no longer floods the panel. Slot styles
  // (.lora-stack/.lora-slot/.lora-add/
  // .lora-picker-panel/.lora-pick-row) already live in styles.css.
  const [pickerOpen, setPickerOpen] = useState(false);
  const availableLoras = compatibleLoras.filter((lora) => !selectedLoraIds.includes(lora.id));
  const atUserLimit = userSelectedLoraCount >= MAX_USER_JOB_LORAS;
  const loraMeta = (lora) => {
    const scope = lora.scope ?? "global";
    return lora.family ? `${scope} · ${lora.family}` : scope;
  };

  return (
    <section className="lora-picker" aria-label="LoRA selection">
      {/* Eyebrow + status, not a card heading (studio-cleanup sc-15370): inside the settings bar
          this reads as a peer of the Model and Style sections, so the label uses the same
          `.settings-bar-label` scale they do. */}
      <div className="lora-picker-head">
        <span className="settings-bar-label">{label}</span>
        <span>
          {selectedLoras.length
            ? `${selectedLoras.length} selected · installed and compatible`
            : selectedModel
              ? "Installed and compatible"
              : "Choose a model"}
        </span>
      </div>
      {showIncompatibleControl ? (
        <label className="checkline">
          <input
            checked={showIncompatibleLoras}
            onChange={(event) => setShowIncompatibleLoras(event.target.checked)}
            type="checkbox"
          />
          Show incompatible
        </label>
      ) : null}

      {selectedLoras.length ? (
        <div className="lora-stack">
          {selectedLoras.map((lora) => {
            const weight = effectiveLoraWeight(lora);
            return (
              <div className="lora-slot" key={lora.id}>
                <div className="lora-slot-head">
                  <span className="lora-slot-meta">
                    <strong>{lora.name ?? lora.id}{lora.updateAvailable ? <StudioUpdateBadge item={lora} /> : null}</strong>
                    <small>{loraMeta(lora)}</small>
                  </span>
                  <button
                    aria-label={`Remove ${lora.name ?? lora.id}`}
                    className="lora-slot-remove"
                    onClick={() => toggleLora(lora)}
                    title="Remove"
                    type="button"
                  >
                    ×
                  </button>
                </div>
                <StudioUpdateNotice item={lora} kind="LoRA" onUpdate={onUpdateLora} />
                <LoraKeywordSummary lora={lora} />
                <div className="lora-slot-weight">
                  <label>
                    <span>Weight</span>
                    <span className="lora-slot-weight-value">{weight.toFixed(2)}</span>
                  </label>
                  <input
                    aria-label={`${lora.name ?? lora.id} weight`}
                    max={LORA_WEIGHT_MAX}
                    min={LORA_WEIGHT_MIN}
                    onChange={(event) => setLoraWeight(lora.id, Number(event.target.value))}
                    step={LORA_WEIGHT_STEP}
                    type="range"
                    value={weight}
                  />
                </div>
              </div>
            );
          })}
        </div>
      ) : null}

      {/* Always enabled (studio-cleanup sc-15373): with nothing available the picker is exactly
          where the user needs to be — it explains WHY the list is empty and (in the studios)
          carries the import form that fixes it. Disabling it left a dead end. */}
      <button
        className="lora-add"
        data-count={`· ${availableLoras.length} available`}
        onClick={() => setPickerOpen((open) => !open)}
        type="button"
      >
        <Icon.Plus size={15} />
        <span>Add LoRA</span>
      </button>

      {pickerOpen ? (
        <div className="lora-picker-panel">
          <LoraPickerList
            loras={availableLoras}
            onAdd={(lora) => {
              toggleLora(lora);
              setPickerOpen(false);
            }}
            isDisabled={isLoraDisabled ?? ((lora) => lora.scope !== "builtin" && atUserLimit)}
            emptyMessage={loraEmptyMessage}
          />
          {importPanel}
        </div>
      ) : null}
    </section>
  );
}

