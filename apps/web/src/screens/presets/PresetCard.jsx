import React from "react";
import { Icon } from "../../components/Icons.jsx";
import { presetLoras } from "../../presetUtils.js";

export function PresetCard({
  preset,
  allModels,
  archiveOne,
  busy,
  duplicateOne,
  editPreset,
  models,
  presetSegmentKey,
  qualityLabel,
  segmentByKey,
  sendPresetToStudio,
}) {

    const isGeneralPreset = preset.kind === "general";
    const segment = segmentByKey(presetSegmentKey(preset));
    const model = allModels.find((item) => item.id === preset.model);
    // The studio snaps its model back into the installed catalog, so launching a model preset
    // pinned to an uninstalled model would silently land on a different one. A general preset
    // is model-agnostic, so it's always runnable.
    const runnable = isGeneralPreset || models.some((item) => item.id === preset.model);
    const scope = preset.scope ?? "global";
    const builtin = scope === "builtin";
    const defaults = preset.defaults ?? {};
    const loraCount = presetLoras(preset).length;
    const prefix = preset.prompt?.prefix ?? "";
    const suffix = preset.prompt?.suffix ?? "";

    const chips = [];
    if (isGeneralPreset) {
      chips.push(defaults.aspect ? `${defaults.aspect} aspect` : "Any aspect");
      chips.push(defaults.count ? `${defaults.count} variations` : "Default count");
    } else {
      chips.push(defaults.resolution ? defaults.resolution.replace("x", " × ") : "Any size");
      if (segment.type === "video") {
        chips.push(defaults.duration ? `${defaults.duration}s` : "Default length");
        chips.push(defaults.fps ? `${defaults.fps} fps` : "Default fps");
      } else {
        chips.push(defaults.count ? `${defaults.count} variations` : "Default count");
      }
      chips.push(qualityLabel(defaults.quality) ?? "Default quality");
    }

    return (
      <article className="preset-card" key={`${scope}-${preset.id}`}>
        <div className="preset-card-head">
          <div className="preset-card-title">
            <strong>{preset.name ?? preset.id}</strong>
            <div>
              {isGeneralPreset ? (
                <span className="preset-card-model">General · any model</span>
              ) : (
                <>
                  {segment.label} · <span className="preset-card-model">{model?.name ?? preset.model}</span>
                </>
              )}
            </div>
          </div>
          {isGeneralPreset ? <span className="preset-kind-chip">General</span> : null}
          <span className={scope === "global" ? "preset-scope-chip global" : "preset-scope-chip"}>{scope}</span>
        </div>

        <div className="preset-card-preview">
          {prefix ? <span className="faint">{prefix} </span> : null}
          <span className="token">your prompt</span>
          {suffix ? <span className="faint"> {suffix}</span> : null}
        </div>

        <div className="preset-card-chips">
          {chips.map((chip) => (
            <span className="chip" key={chip}>
              {chip}
            </span>
          ))}
          {isGeneralPreset ? null : (
            <span className={loraCount ? "chip accent" : "chip"}>
              {loraCount ? `${loraCount} LoRA${loraCount === 1 ? "" : "s"}` : "No LoRAs"}
            </span>
          )}
        </div>

        <div className="preset-card-actions">
          <button
            className="primary-action preset-card-use"
            disabled={busy || !runnable}
            onClick={() => sendPresetToStudio(preset)}
            title={runnable ? undefined : `Install ${model?.name ?? preset.model} to use this preset`}
            type="button"
          >
            <Icon.Play size={13} /> Use in Studio
          </button>
          {builtin ? null : (
            <button className="secondary-action" onClick={() => editPreset(preset)} type="button">
              <Icon.Pencil size={13} /> Edit
            </button>
          )}
          <span className="preset-card-spacer" />
          <button
            aria-label={`Duplicate ${preset.name ?? preset.id}`}
            className="preset-icon-button"
            disabled={busy}
            onClick={() => duplicateOne(preset)}
            title="Duplicate"
            type="button"
          >
            <Icon.Duplicate size={14} />
          </button>
          {builtin ? null : (
            <button
              aria-label={`Archive ${preset.name ?? preset.id}`}
              className="preset-icon-button danger"
              disabled={busy}
              onClick={() => archiveOne(preset)}
              title="Archive"
              type="button"
            >
              <Icon.Trash size={14} />
            </button>
          )}
        </div>
      </article>
    );

}
