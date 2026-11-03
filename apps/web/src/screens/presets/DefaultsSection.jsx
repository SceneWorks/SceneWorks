import React from "react";
import { qualityChoices } from "../../jobTypes.js";

export function DefaultsSection({
  durationOptions,
  editable,
  form,
  fpsOptions,
  isVideo,
  outOfMenuOption,
  renderSectionHead,
  resolutionOptions,
  updateField,
}) {

    return (
      <div className="preset-form-section">
        {renderSectionHead("Defaults", "The settings the Studio starts on. All still editable per run.")}
        <div className="preset-defaults-grid">
          <label className="field">
            <span>{isVideo ? "Resolution" : "Aspect"}</span>
            <select disabled={!editable} onChange={(event) => updateField("resolution", event.target.value)} value={form.resolution}>
              <option value="">No default</option>
              {outOfMenuOption(form.resolution, resolutionOptions, (value) => value.replace("x", " × "))}
              {resolutionOptions.map((value) => (
                <option key={value} value={value}>
                  {value.replace("x", " × ")}
                </option>
              ))}
            </select>
          </label>
          {isVideo ? (
            <>
              <label className="field">
                <span>Duration</span>
                <select disabled={!editable} onChange={(event) => updateField("duration", event.target.value)} value={form.duration}>
                  <option value="">No default</option>
                  {outOfMenuOption(form.duration, durationOptions, (value) => `${value}s`)}
                  {durationOptions.map((d) => (
                    <option key={d} value={String(d)}>
                      {d}s
                    </option>
                  ))}
                </select>
              </label>
              <label className="field">
                <span>Frames</span>
                <select disabled={!editable} onChange={(event) => updateField("fps", event.target.value)} value={form.fps}>
                  <option value="">No default</option>
                  {outOfMenuOption(form.fps, fpsOptions, (value) => `${value} fps`)}
                  {fpsOptions.map((f) => (
                    <option key={f} value={String(f)}>
                      {f} fps
                    </option>
                  ))}
                </select>
              </label>
            </>
          ) : (
            <label className="field">
              <span>Variations</span>
              <select disabled={!editable} onChange={(event) => updateField("count", event.target.value)} value={form.count}>
                <option value="">No default</option>
                {[1, 2, 3, 4, 6, 8].map((n) => (
                  <option key={n} value={String(n)}>
                    {n}
                  </option>
                ))}
              </select>
            </label>
          )}
          <label className="field preset-quality-field">
            <span>Quality</span>
            <div aria-label="Quality" className="quality-pick" role="radiogroup">
              {qualityChoices.map(([value, label]) => (
                <button
                  aria-checked={form.quality === value}
                  className={form.quality === value ? "active" : ""}
                  disabled={!editable}
                  key={value}
                  onClick={() => updateField("quality", form.quality === value ? "" : value)}
                  role="radio"
                  type="button"
                >
                  {label}
                </button>
              ))}
            </div>
          </label>
        </div>
      </div>
    );

}
