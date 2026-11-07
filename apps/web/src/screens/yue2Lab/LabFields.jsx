import React, { useEffect, useRef, useState } from "react";
import { SAMPLING_FIELDS, parseYue2Lyrics, yue2LyricsText } from "../../yue2Lab.js";
import { YueLyricsEditor } from "../audioYueControls.jsx";

// A text or draft over the durable budget: it survives tab changes but not a relaunch.
export function SessionOnlyText({ testId }) {
  return (
    <p className="yue2-muted" data-testid={testId}>
      Kept for this session only — it is too large to restore after a relaunch.
    </p>
  );
}

export function Segmented({ label, value, options, onChange, disabledValues = [] }) {
  return (
    <div aria-label={label} className="segmented-control yue2-segmented" role="radiogroup">
      {options.map((option) => (
        <button
          aria-checked={value === option.value}
          className={value === option.value ? "active" : ""}
          disabled={disabledValues.includes(option.value)}
          key={option.value}
          onClick={() => onChange(option.value)}
          role="radio"
          type="button"
        >
          {option.label}
        </button>
      ))}
    </div>
  );
}

export const COMPUTE_PRECISION_TIP =
  "Q8/Q4 describe weight storage and quantized matmuls; each quantized matmul briefly uses FP32 input and result tensors before returning to the selected stage precision. This selector governs YuE2 song stages and the VAE, including cached decode; recording transcription runs separately on CPU. Kernels may also use FP32 reductions and audio output formats.";

// Lyrics as section rows (the YuE1 editor) or as raw text for pasting. The persisted value is the
// lyrics string; the rows are local so an empty row being written survives, and they re-parse only
// when the string changes from outside (a preset, a restored run).
export function Yue2LyricsField({ value, onChange }) {
  const [mode, setMode] = useState("sections");
  const [sections, setSections] = useState(() => parseYue2Lyrics(value));
  const emitted = useRef(value);
  useEffect(() => {
    if (value !== emitted.current) {
      emitted.current = value;
      setSections(parseYue2Lyrics(value));
    }
  }, [value]);
  const emit = (text) => {
    emitted.current = text;
    onChange(text);
  };
  return (
    <div className="yue2-lyrics">
      <div className="yue2-inline">
        <span>Lyrics</span>
        <Segmented
          label="Lyrics view"
          onChange={(next) => {
            if (next === "sections") setSections(parseYue2Lyrics(value));
            setMode(next);
          }}
          options={[
            { value: "sections", label: "Sections" },
            { value: "text", label: "Text" },
          ]}
          value={mode}
        />
      </div>
      {mode === "sections" ? (
        <YueLyricsEditor
          onChange={(next) => {
            setSections(next);
            emit(yue2LyricsText(next));
          }}
          placeholder="Lines for this section…"
          sections={sections}
        />
      ) : (
        <textarea
          aria-label="Lyrics"
          onChange={(event) => emit(event.target.value)}
          placeholder={"[verse]\n…\n\n[chorus]\n…"}
          rows={8}
          value={value}
        />
      )}
    </div>
  );
}

export function SamplingFields({ label, value, defaults, onChange, disabled = false, reason = null }) {
  return (
    <fieldset className="yue2-fieldset" disabled={disabled || Boolean(reason)} title={reason ?? undefined}>
      <legend>{label}</legend>
      <div className="yue2-grid">
        {SAMPLING_FIELDS.map((field) => (
          <label key={field.key}>
            {field.label}
            <input
              aria-label={`${label} ${field.label}`}
              max={field.max}
              min={field.min}
              onChange={(event) => onChange({ ...value, [field.key]: event.target.value })}
              placeholder={defaults?.[field.key] != null ? `${defaults[field.key]} (default)` : "Model default"}
              step={field.step}
              type="number"
              value={value?.[field.key] ?? ""}
            />
          </label>
        ))}
      </div>
    </fieldset>
  );
}

export function TriStateSelect({ label, value, onChange, reason = null }) {
  return (
    <label title={reason ?? undefined}>
      {label}
      <select aria-label={label} disabled={Boolean(reason)} onChange={(event) => onChange(event.target.value)} value={value}>
        <option value="">Admission decides</option>
        <option value="on">On</option>
        <option value="off">Off</option>
      </select>
    </label>
  );
}
