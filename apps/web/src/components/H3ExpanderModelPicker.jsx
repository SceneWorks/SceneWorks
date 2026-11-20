import React, { useState } from "react";

// SW-7208: which model the MiniMax-H3-Image prompt expander runs. Both run MiniMax's own prompt
// files and reply contract; the pick is saved (see `useH3ExpanderModel`). A model that cannot run
// here stays listed, disabled, with what it lacks. When the abliterated describer is installed but
// MiniMax's prompt files are not, the panel offers them alone (no expander weights), behind the
// expander's MiniMax licence like every other MiniMax download.
export function H3ExpanderModelPicker({ choices = [], value, onChange, onDownloadPrompts }) {
  const [download, setDownload] = useState({ status: "idle", error: "" });
  if (choices.length < 2 && !choices.some((choice) => choice.missing === "prompts")) {
    return null;
  }
  const needsPrompts = choices.find((choice) => choice.missing === "prompts" && choice.id === value);

  async function fetchPrompts() {
    setDownload({ status: "busy", error: "" });
    try {
      await onDownloadPrompts();
      setDownload({ status: "queued", error: "" });
    } catch (error) {
      setDownload({ status: "error", error: error?.message || "Could not download the prompt files." });
    }
  }

  return (
    <fieldset className="h3-expander-models">
      <legend>Expander model</legend>
      {choices.map((choice) => (
        <label className="training-toggle-line" key={choice.id}>
          <input
            checked={value === choice.id}
            disabled={choice.missing === "weights"}
            name="h3-expander-model"
            onChange={() => onChange(choice.id)}
            type="radio"
            value={choice.id}
          />
          <span>
            {choice.label}
            {choice.missing === "weights" ? (
              <small className="h3-expander-unavailable"> — not installed</small>
            ) : choice.missing === "prompts" ? (
              <small className="h3-expander-unavailable"> — needs MiniMax’s prompt files</small>
            ) : null}
          </span>
        </label>
      ))}
      {needsPrompts && typeof onDownloadPrompts === "function" ? (
        <div className="h3-expander-prompts">
          <p className="refine-hint">
            The abliterated model expands with MiniMax-H3-Image’s own prompt files (56 KB, no expander
            weights), under the MiniMax-H3-Image Community License you accept on the Models screen.
          </p>
          <button
            className="hero-link"
            disabled={download.status === "busy" || download.status === "queued"}
            onClick={fetchPrompts}
            type="button"
          >
            {download.status === "queued" ? "Downloading prompt files…" : "Download MiniMax’s prompt files"}
          </button>
          {download.status === "error" ? (
            <p className="refine-error" role="alert">
              {download.error}
            </p>
          ) : null}
        </div>
      ) : null}
    </fieldset>
  );
}
