import React from "react";
import { Icon } from "../components/Icons.jsx";
import {
  LORA_WEIGHT_MAX,
  LORA_WEIGHT_MIN,
  LORA_WEIGHT_STEP,
  MAX_USER_JOB_LORAS,
  normalizeLoraFamily,
} from "../presetUtils.js";
import { percent } from "../formatting.js";
import { loraMeta } from "./useSimpleLoras.js";
import { useSimpleUi } from "./SimpleUiContext.js";

// The Simple studios' LoRA control (epic 15404). Rendered as the LAST child of the existing
// `.su-settings-bar`, so it reads as a peer of Model / Resolution / Variations rather than a
// card of its own — the same call `LoraPickerSection` made when it moved into the advanced
// settings bar (sc-15370).

/**
 * Append a trigger keyword to a prompt, unless it's already in there.
 *
 * Surfacing keywords is only useful if they can reach the prompt: a trigger word does
 * nothing to the generation while it sits in a chip. Matching is case-insensitive so a
 * second tap is a no-op rather than a duplicate.
 *
 * @returns {string} the new prompt (the original when the keyword is already present).
 */
export function promptWithKeyword(prompt, keyword) {
  const word = String(keyword ?? "").trim();
  const current = String(prompt ?? "");
  if (!word || current.toLowerCase().includes(word.toLowerCase())) {
    return current;
  }
  return current.trim() ? `${current.trimEnd().replace(/,$/, "")}, ${word}` : word;
}

export function SimpleLoraField({
  selectedModel,
  selectedLoras,
  availableLoras,
  atLimit,
  effectiveLoraWeight,
  onRemove,
  onWeightChange,
  onAddKeyword,
  onOpenPicker,
  pendingImports = [],
  hint = null,
}) {
  const { breakpoint } = useSimpleUi();
  // Pointer-accurate copy: "tap" is wrong on a desktop pointer, "click" is wrong on a phone.
  const keywordHint = breakpoint === "phone" ? "tap to add to prompt" : "click to add to prompt";

  return (
    <div className="su-field su-lora-field">
      <div className="su-lora-head">
        <label>LoRAs</label>
        {/* Status copy lifted verbatim from LoraPickerSection so the two surfaces can't drift. */}
        <span>
          {selectedLoras.length
            ? `${selectedLoras.length} selected · installed and compatible`
            : selectedModel
              ? "Installed and compatible"
              : "Choose a model"}
        </span>
      </div>

      {selectedLoras.length || pendingImports.length ? (
        <div className="su-lora-stack">
          {selectedLoras.map((lora) => {
            const name = lora.name ?? lora.id;
            const weight = effectiveLoraWeight(lora);
            const keywords = Array.isArray(lora.triggerWords) ? lora.triggerWords : [];
            return (
              <div className="su-lora-slot" key={lora.id}>
                <div className="su-lora-slot-head">
                  <span className="su-lora-slot-meta">
                    <strong>{name}</strong>
                    <small>{loraMeta(lora)}</small>
                  </span>
                  <button
                    aria-label={`Remove ${name}`}
                    className="su-lora-remove"
                    onClick={() => onRemove(lora)}
                    title="Remove"
                    type="button"
                  >
                    ×
                  </button>
                </div>
                {/* Notes are deliberately NOT shown here — Simple surfaces only the keywords,
                    because those are the part the user has to act on. Notes stay in the
                    advanced LoraKeywordSummary. */}
                {keywords.length ? (
                  <div className="su-lora-keywords">
                    {keywords.map((keyword) => (
                      <button
                        className="su-lora-keyword"
                        key={keyword}
                        onClick={() => onAddKeyword(keyword)}
                        title="Add to prompt"
                        type="button"
                      >
                        {keyword}
                      </button>
                    ))}
                    <span className="su-lora-keyword-hint">{keywordHint}</span>
                  </div>
                ) : null}
                <div className="su-lora-weight">
                  <label>
                    <span>Weight</span>
                    <span className="su-lora-weight-value">{weight.toFixed(2)}</span>
                  </label>
                  {/* The −2…2 band is deliberate and shared with every other weight surface
                      (presetUtils): slider-style LoRAs are trained to run negative, so clamping
                      to 0…1 here would make half of them unusable from Simple. */}
                  <input
                    aria-label={`${name} weight`}
                    max={LORA_WEIGHT_MAX}
                    min={LORA_WEIGHT_MIN}
                    onChange={(event) => onWeightChange(lora.id, Number(event.target.value))}
                    step={LORA_WEIGHT_STEP}
                    type="range"
                    value={weight}
                  />
                </div>
              </div>
            );
          })}
          {pendingImports.map((job) => (
            <PendingLoraSlot job={job} key={job.id} />
          ))}
        </div>
      ) : null}

      {/* Always enabled, even at 0 available (sc-15373): the sheet is where the reason AND the
          fix (the import form) live, so disabling this would be a dead end. */}
      <button className="su-lora-add" onClick={onOpenPicker} type="button">
        <Icon.Plus size={15} />
        <span>Add LoRA</span>
        <span className="su-lora-add-count">· {availableLoras.length} available</span>
      </button>
      {atLimit ? (
        <p className="su-lora-note">Limit reached — {MAX_USER_JOB_LORAS} LoRAs per run.</p>
      ) : null}
      {hint ? (
        <p className="su-lora-hint">
          <Icon.Info size={13} />
          <span>{hint}</span>
        </p>
      ) : null}
    </div>
  );
}

// An import this studio started that hasn't finished yet. Everything shown is read off the
// live job feed (the same one the Queue renders) — this component holds no state of its own,
// so it can't drift from the Queue's account of the same job.
function PendingLoraSlot({ job }) {
  const name = job.payload?.name ?? job.payload?.loraId ?? job.payload?.fileName ?? "LoRA import";
  const family = job.payload?.manifestEntry?.family;
  const progress = Number(job.progress);
  const hasProgress = Number.isFinite(progress) && progress > 0;
  return (
    <div className="su-lora-slot su-lora-slot--pending">
      <div className="su-lora-slot-head">
        <span className="su-lora-slot-meta">
          <strong>{name}</strong>
          <small>{family ? `importing · family detected: ${normalizeLoraFamily(family)}` : "importing"}</small>
        </span>
        <span className="su-job-badge su-job-badge--running">Queued</span>
      </div>
      {/* Indeterminate until the job reports a percentage, then a real width — the same rule
          SimpleJobCard follows, so a 0% bar never reads as "stuck". */}
      <div className={hasProgress ? "su-bar" : "su-bar su-bar--indeterminate"}>
        <span style={hasProgress ? { width: percent(progress) } : undefined} />
      </div>
      <p className="su-lora-note">Becomes selectable here when the Queue completes.</p>
    </div>
  );
}
