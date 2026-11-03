import React from "react";
import { Icon } from "../../components/Icons.jsx";
import { ValidationSummary } from "../../validation/Validation.jsx";

export function FooterSection({
  backToList,
  busy,
  canSave,
  creating,
  form,
  isGeneral,
  isVideo,
  message,
  saveValidity,
  selectedModel,
}) {

    const loraCount = form.loras.length;
    const chips = isGeneral
      ? [
          "General · any model",
          form.aspect ? `${form.aspect} aspect` : "Any aspect",
          form.count ? `${form.count} variations` : "Default count",
        ]
      : [
          selectedModel?.name ?? form.model,
          form.resolution ? form.resolution.replace("x", " × ") : "Any size",
          isVideo
            ? form.duration
              ? `${form.duration}s`
              : "Default length"
            : form.count
              ? `${form.count} variations`
              : "Default count",
          loraCount ? `${loraCount} LoRA${loraCount === 1 ? "" : "s"}` : "No LoRAs",
        ].filter(Boolean);

    return (
      <>
        {/* Surfaced reasons only — read-only built-in, broken defaults, preset-LoRA
            problems. Name/model are silent requirements. The message banner below is a
            submission result, left as-is (epic 10644, sc-10651). */}
        <ValidationSummary issues={saveValidity.surfaced} label="Save errors" />
        {message.text ? (
          <p className={message.tone === "success" ? "inline-success" : "inline-warning"}>{message.text}</p>
        ) : null}
        <div className="preset-recipe-footer">
          <div className="preset-recipe-chips">
            <span className="eyebrow">Recipe</span>
            {chips.map((chip, index) => (
              <span className={index === 0 ? "chip accent" : "chip"} key={chip}>
                {chip}
              </span>
            ))}
          </div>
          <span className="preset-card-spacer" />
          <button className="secondary-action" onClick={backToList} type="button">
            Cancel
          </button>
          <button className="primary-action" disabled={!canSave || busy} type="submit">
            <Icon.Save size={15} />
            <span>{busy ? "Saving…" : creating ? "Create preset" : "Save preset"}</span>
          </button>
        </div>
      </>
    );

}
