import { useState } from "react";
import {
  aspectRatioById,
  formatMegapixels,
  megapixelsOf,
  nearestAspectRatio,
  resolutionFor,
  stepMegapixels,
} from "../../aspectResolution.js";
import { parseResolution } from "../../resolutionMatch.js";

// Image Studio's size picker: an aspect-ratio menu and a megapixel stepper, with the derived pixel
// size read out underneath. Both controls are views over the "WxH" `resolution` (see
// aspectResolution.js) — changing either writes a new resolution through `onChange`; nothing here
// holds a second copy of the size. The megapixel range is the selected model's own (`envelope`).
export function AspectResolutionPicker({ resolution, onChange, envelope, children }) {
  const size = parseResolution(resolution);
  // What the user ASKED for (ratio + megapixels), remembered against the size it produced. The box
  // shows the request (4.00) while the readout shows the on-grid result (1776 × 2352, 3.98 MP), and
  // repeated steps walk the request — not the rounded result, which would drift. Any outside change
  // to the size (a model switch refit, a preset, a reference snap) drops it, and both controls fall
  // back to reading the size itself.
  const [requested, setRequested] = useState(null);
  const current = requested?.resolution === resolution ? requested : null;
  const ratio = current?.ratio ?? (size ? nearestAspectRatio(size.width, size.height, envelope.ratios) : envelope.ratios[0]);
  const megapixels = current?.megapixels ?? (size ? megapixelsOf(size.width, size.height) : 1);
  // The typed value is a draft until Enter/blur, so a half-typed "4." never resizes the image. It is
  // tagged with the resolution it was typed against, so any outside change to the size drops it.
  const [typed, setTyped] = useState(null);
  const draft = typed?.resolution === resolution ? typed.value : null;
  const setDraft = (value) => setTyped(value === null ? null : { resolution, value });

  if (envelope.fixed) {
    return (
      <div className="settings-field settings-field-aspect aspect-picker">
        Size
        <div className="aspect-picker-readout" role="status" aria-label="Output size">
          <strong>{size ? `${size.width} × ${size.height}` : resolution}</strong>
          <span>Fixed by this model</span>
        </div>
        {children}
      </div>
    );
  }

  const apply = (nextRatio, nextMegapixels) => {
    const shape = typeof nextRatio === "string" ? aspectRatioById(nextRatio, envelope.ratios) : nextRatio;
    const next = shape && resolutionFor(shape, nextMegapixels, envelope);
    if (!next) return;
    const clamped = Math.min(Math.max(nextMegapixels, envelope.minMegapixels), envelope.maxMegapixels);
    setRequested({ resolution: next, ratio: shape, megapixels: clamped });
    if (next !== resolution) onChange(next);
  };
  const commitDraft = () => {
    if (draft === null) return;
    const value = Number(draft);
    setDraft(null);
    if (Number.isFinite(value) && value > 0) apply(ratio, value);
  };
  // A step that cannot change the size (this shape is already at the model's ceiling or floor once
  // it lands on the grid) disables its button rather than leaving a click that does nothing.
  const atMin = megapixels <= envelope.minMegapixels + 1e-6 ||
    resolutionFor(ratio, envelope.minMegapixels, envelope) === resolution;
  const atMax = megapixels >= envelope.maxMegapixels - 1e-6 ||
    resolutionFor(ratio, envelope.maxMegapixels, envelope) === resolution;

  return (
    <div className="settings-field settings-field-aspect aspect-picker">
      <label className="aspect-picker-field">
        Aspect
        <select onChange={(event) => apply(event.target.value, megapixels)} value={ratio.id}>
          {envelope.ratios.map((option) => (
            <option key={option.id} value={option.id}>{option.label}</option>
          ))}
        </select>
      </label>
      <label className="aspect-picker-field">
        Megapixels
        <span className="mp-stepper">
          <button
            aria-label="Decrease megapixels"
            disabled={atMin}
            onClick={() => apply(ratio, stepMegapixels(megapixels, -1, envelope))}
            type="button"
          >
            −
          </button>
          <input
            aria-label="Megapixels"
            inputMode="decimal"
            onBlur={commitDraft}
            onChange={(event) => setDraft(event.target.value)}
            onKeyDown={(event) => {
              if (event.key === "Enter") {
                event.preventDefault();
                commitDraft();
              } else if (event.key === "Escape") {
                setDraft(null);
              }
            }}
            // No min/max/step: this input lives inside the Generate form, and a native range or
            // step mismatch (1.00 is off a 0.25-based 0.1 grid) would silently block submission.
            // The value is clamped and gridded on commit instead; the buttons own stepping.
            step="any"
            type="number"
            value={draft ?? formatMegapixels(megapixels)}
          />
          <button
            aria-label="Increase megapixels"
            disabled={atMax}
            onClick={() => apply(ratio, stepMegapixels(megapixels, 1, envelope))}
            type="button"
          >
            +
          </button>
        </span>
      </label>
      <div className="aspect-picker-readout" role="status" aria-label="Output size">
        <strong>{size ? `${size.width} × ${size.height}` : "—"}</strong>
        <span>{size ? formatMegapixels(megapixelsOf(size.width, size.height)) : "—"} MP</span>
        <span className="aspect-picker-limit">up to {formatMegapixels(envelope.maxMegapixels)} MP on this model</span>
      </div>
      {children}
    </div>
  );
}
