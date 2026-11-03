import React from "react";
import { Icon } from "../../components/Icons.jsx";
import { LORA_WEIGHT_MAX, LORA_WEIGHT_MIN, LORA_WEIGHT_STEP, MAX_PRESET_LORAS, loraMatchesModel } from "../../presetUtils.js";
import { LoraKeywordSummary } from "../../components/LoraKeywordSummary.jsx";

export function LorasSection({
  addLoraById,
  addableLoras,
  editable,
  form,
  loraEmptyMessage,
  loraLabel,
  loras,
  onOpenModels,
  removeLora,
  renderSectionHead,
  selectedModel,
  setShowLoraPicker,
  showLoraEmptyState,
  showLoraPicker,
  updateLoraWeight,
}) {

    return (
      <div className="preset-form-section">
        {renderSectionHead(
          "LoRAs",
          `Up to ${MAX_PRESET_LORAS} fine-tunes layered with this preset, compatible with ${selectedModel?.name ?? "the chosen model"}.`,
          true,
        )}
        <section aria-label="Preset LoRAs" className="lora-stack">
          {form.loras.map((selected) => {
            const lora = loras.find((item) => item.id === selected.id);
            const missing = !lora || lora.installState === "missing";
            const incompatible = lora && selectedModel && !loraMatchesModel(lora, selectedModel);
            const name = lora?.name ?? selected.id;
            const weight = Number(selected.weight);
            return (
              <div className={missing || incompatible ? "lora-slot warning" : "lora-slot"} key={selected.id}>
                <div className="lora-slot-head">
                  <span className="lora-slot-meta">
                    <strong>{name}</strong>
                    <small>
                      {missing
                        ? "Missing or still importing"
                        : incompatible
                          ? `${loraLabel(lora)} | incompatible with ${selectedModel?.name ?? selectedModel?.id}`
                          : loraLabel(lora)}
                    </small>
                  </span>
                  <button
                    aria-label={`Remove ${name}`}
                    className="lora-slot-remove"
                    disabled={!editable}
                    onClick={() => removeLora(selected.id)}
                    title="Remove"
                    type="button"
                  >
                    ×
                  </button>
                </div>
                <LoraKeywordSummary lora={lora} />
                <div className="lora-slot-weight">
                  <label>
                    <span>Weight</span>
                    <span className="lora-slot-weight-value">{Number.isFinite(weight) ? weight.toFixed(2) : "—"}</span>
                  </label>
                  {/* Bidirectional -2..2 (LORA_WEIGHT_*), matching the studio pickers and
                      the preset normalizer: slider LoRAs run negative for the inverse
                      direction, and 0 is a valid neutral. */}
                  <input
                    aria-label={`${name} weight`}
                    disabled={!editable || missing || incompatible}
                    max={LORA_WEIGHT_MAX}
                    min={LORA_WEIGHT_MIN}
                    onChange={(event) => updateLoraWeight(selected.id, event.target.value)}
                    step={LORA_WEIGHT_STEP}
                    type="range"
                    value={Number.isFinite(weight) ? weight : 0}
                  />
                </div>
              </div>
            );
          })}

          {showLoraEmptyState ? (
            <div className="empty-panel compact-panel">
              <span>{loraEmptyMessage}</span>
              <button onClick={onOpenModels} type="button">
                Open Models
              </button>
            </div>
          ) : null}

          {form.loras.length < MAX_PRESET_LORAS ? (
            showLoraPicker ? (
              <div className="lora-picker-panel">
                <strong>Pick a LoRA</strong>
                {addableLoras.length ? (
                  <div className="lora-picker-list">
                    {addableLoras.map((lora) => (
                      <button
                        className="lora-pick-row"
                        key={lora.id}
                        onClick={() => {
                          addLoraById(lora.id);
                          setShowLoraPicker(false);
                        }}
                        type="button"
                      >
                        <span>
                          <strong>{lora.name ?? lora.id}</strong> {lora.family ? <span className="chip">{lora.family}</span> : null}
                        </span>
                        <Icon.Plus size={14} />
                      </button>
                    ))}
                  </div>
                ) : (
                  <p className="lora-pick-empty">{loraEmptyMessage}</p>
                )}
                <div className="lora-picker-actions">
                  <button onClick={() => setShowLoraPicker(false)} type="button">
                    Cancel
                  </button>
                </div>
              </div>
            ) : (
              <button
                className="lora-add"
                data-count={`· ${addableLoras.length} available`}
                disabled={!editable || !addableLoras.length}
                onClick={() => setShowLoraPicker(true)}
                type="button"
              >
                <Icon.Plus size={15} />
                <span>Add LoRA</span>
              </button>
            )
          ) : null}
        </section>
      </div>
    );

}
