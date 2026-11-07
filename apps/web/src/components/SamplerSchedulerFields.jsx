import React from "react";
import { SAMPLER_LABELS, SCHEDULER_LABELS } from "../samplerOptions.js";

function SamplingField({ label, variant, children }) {
  return (
    <label className={variant === "rail" ? "ve-field" : variant === "preset" ? "field" : undefined}>
      {variant === "rail" ? <span className="ve-field-label">{label}</span>
        : variant === "preset" ? <span>{label}</span> : label}
      {children}
    </label>
  );
}

export function SamplerSchedulerFields({
  sampler, scheduler, samplerOptions = [], schedulerOptions = [], onSamplerChange, onSchedulerChange,
  hidden = false, disabled = false, variant, emptyLabel, rawLabels = false,
}) {
  if (hidden) return null;
  const fields = [
    ["Sampler", sampler, samplerOptions, onSamplerChange, SAMPLER_LABELS],
    ["Scheduler", scheduler, schedulerOptions, onSchedulerChange, SCHEDULER_LABELS],
  ];
  return fields.map(([label, value, options, onChange, labels]) => options.length > 0 ? (
    <SamplingField key={label} label={label} variant={variant}>
      <select className={variant === "rail" ? "ve-select" : undefined} disabled={disabled}
        onChange={(event) => onChange(event.target.value)} value={value}>
        {emptyLabel !== undefined ? <option value="">{emptyLabel}</option> : null}
        {options.map((key) => <option key={key} value={key}>{rawLabels ? key : labels[key] ?? key}</option>)}
      </select>
    </SamplingField>
  ) : null);
}

// Blank steps and seeds stay blank strings in the authoring state. Each owner
// supplies its declared range/acceleration policy; only payload builders convert
// blank to an omitted model default or a random seed. Zero remains a real seed.
export function SamplingStepsField({ value, onChange, menu, defaultLabel = "Model default", variant, inputProps, selectProps }) {
  return (
    <SamplingField label="Steps" variant={variant}>
      {menu ? (
        <select {...selectProps} onChange={(event) => onChange(event.target.value)} value={value}>
          <option value="">{defaultLabel}</option>
          {menu.map((steps) => <option key={steps} value={String(steps)}>{steps}</option>)}
        </select>
      ) : (
        <input className={variant === "rail" ? "ve-input" : undefined} {...inputProps}
          onChange={(event) => onChange(event.target.value)} value={value} />
      )}
    </SamplingField>
  );
}

export function SamplingSeedField({ value, onChange, variant, inputProps, suffix }) {
  const input = <input className={variant === "rail" ? "ve-input" : undefined} {...inputProps}
    onChange={(event) => onChange(event.target.value)} value={value} />;
  return <SamplingField label="Seed" variant={variant}>
    {suffix ? <div className="ve-preset-row">{input}{suffix}</div> : input}
  </SamplingField>;
}
