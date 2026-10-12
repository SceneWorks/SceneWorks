import React, { useEffect, useState } from "react";
import { Modal } from "./Modal.jsx";
import { captionModes, captionTriggerWordsError } from "../training/joyCaptionPrompts.js";

// Caption settings modal (sc-2025). One dialog drives both "Caption all" and a
// single image's "Re-Caption" — the dataset editor owns the captioner settings
// state and the job submission; this component only renders the controls and a
// Run button scoped to the target.
//
// When the JoyCaption model isn't provisioned on the native worker (sc-5620), the
// caption job would just fail in the Queue (captioning is fire-and-forget, not polled
// inline), so we surface a proactive "download the captioning model" affordance and
// block Run. `modelMissing` is gated by the parent (Mac + catalog installState
// "missing" + the default model selected); `onDownloadModel` enqueues its download.
export function DatasetCaptionDialog({
  settings,
  onChange,
  gpuOptions = [],
  captionTypes = [],
  captionLengths = [],
  extraOptions = [],
  promptValue = "",
  scope,
  running = false,
  onRun,
  onToggleExtra,
  onClose,
  modelMissing = false,
  onDownloadModel,
  modelSizeLabel = "",
  modelName = "JoyCaption",
}) {
  const single = scope?.type === "item";
  const flagged = scope?.type === "flagged";
  const title = single
    ? `Re-caption ${scope.name ?? "image"}`
    : flagged
      ? `Re-caption ${scope.count ?? 0} flagged ${scope.count === 1 ? "image" : "images"}`
      : "Caption images";
  const runLabel = single
    ? "Re-caption image"
    : flagged
      ? "Re-caption flagged"
      : settings.recaption
        ? "Re-caption all"
        : "Caption missing";
  const joy = settings.captioner === "joy_caption";
  // Caption mode (sc-24829): trigger-only runs no captioner, so its model/sampling controls and
  // the missing-model block do not apply; subject-only uses the worker's own prompt, so the
  // prompt-shaping controls (type, length, name, prompt, extra options) do not apply.
  const mode = settings.mode || "default";
  const usesCaptioner = mode !== "triggerOnly";
  const usesPrompt = mode === "default";
  const [downloadRequested, setDownloadRequested] = useState(false);
  const [downloadError, setDownloadError] = useState("");

  // Once the model finishes downloading (parent's catalog refresh flips modelMissing to
  // false), clear the "downloading" note so Run re-enables.
  useEffect(() => {
    if (!modelMissing) {
      setDownloadRequested(false);
    }
  }, [modelMissing]);

  // Trigger words (sc-24829) apply in every mode; out-of-limit input blocks Run (E6).
  const triggerWordsError = joy ? captionTriggerWordsError(settings.triggerWords) : "";
  const modelBlocked = joy && usesCaptioner && modelMissing;
  const blockRun = modelBlocked || Boolean(triggerWordsError);

  async function handleDownloadModel() {
    if (typeof onDownloadModel !== "function") return;
    setDownloadError("");
    try {
      const job = await onDownloadModel();
      if (job) {
        setDownloadRequested(true);
      }
    } catch (error) {
      setDownloadError(error?.message || "Could not start the model download.");
    }
  }

  return (
    <Modal className="dataset-caption-modal" labelledBy="dataset-caption-title" onClose={onClose}>
      <header className="dataset-caption-head">
        <div>
          <p className="eyebrow">Captioner</p>
          <h2 id="dataset-caption-title">{title}</h2>
        </div>
        <button className="modal-close" onClick={onClose} type="button">
          Close
        </button>
      </header>

      <div className="dataset-caption-body">
        <label>
          Method
          <select onChange={(event) => onChange("captioner", event.target.value)} value={settings.captioner}>
            <option value="joy_caption">Joy Caption</option>
            <option value="metadata">Metadata fallback</option>
          </select>
        </label>

        {modelBlocked ? (
          <div className="caption-missing-model" role="alert">
            {downloadRequested ? (
              <p className="inline-warning">
                Downloading the captioning model… track progress on the Models screen, then caption
                again.
              </p>
            ) : (
              <>
                <p className="inline-warning">
                  The captioning model{modelSizeLabel ? ` (${modelSizeLabel})` : ""} isn’t installed
                  yet.
                </p>
                {typeof onDownloadModel === "function" ? (
                  <>
                    <button className="secondary-action" onClick={handleDownloadModel} type="button">
                      Download captioning model
                    </button>
                    {downloadError ? <p className="inline-warning">{downloadError}</p> : null}
                  </>
                ) : (
                  <p className="inline-warning">Open the Models screen to download “{modelName}”.</p>
                )}
              </>
            )}
          </div>
        ) : null}

        {joy ? (
          <label>
            Mode
            <select onChange={(event) => onChange("mode", event.target.value)} value={mode}>
              {captionModes.map((option) => (
                <option key={option.value} value={option.value}>
                  {option.label}
                </option>
              ))}
            </select>
          </label>
        ) : null}
        {joy ? (
          <label>
            Trigger words
            <input
              aria-invalid={triggerWordsError ? "true" : undefined}
              onChange={(event) => onChange("triggerWords", event.target.value)}
              placeholder="Comma-separated, e.g. miraStyle"
              value={settings.triggerWords ?? ""}
            />
            {triggerWordsError ? (
              <span className="inline-warning" role="alert">
                {triggerWordsError}
              </span>
            ) : (
              <span className="dataset-caption-field-hint">Added to every image that has no trigger words of its own.</span>
            )}
          </label>
        ) : null}
        {joy && mode === "subjectOnly" ? (
          <p className="dataset-caption-mode-note">
            Describes only what changes between images (clothing, expression, pose, accessories) and leaves out the
            background and fixed identity traits. Trigger words are still added to each caption.
          </p>
        ) : null}
        {joy && mode === "triggerOnly" ? (
          <>
            <p className="dataset-caption-mode-note">
              Each image’s caption becomes exactly its trigger words. No captioning model runs.
            </p>
            {single ? null : (
              <label className="training-toggle-line">
                <input checked={settings.recaption} onChange={(event) => onChange("recaption", event.target.checked)} type="checkbox" />
                <span>Re-caption images that already have a caption</span>
              </label>
            )}
          </>
        ) : null}

        {joy && usesCaptioner ? (
          <>
            <label>
              Model
              <input onChange={(event) => onChange("modelNameOrPath", event.target.value)} value={settings.modelNameOrPath} />
            </label>
            <label>
              GPU
              <select onChange={(event) => onChange("requestedGpu", event.target.value)} value={settings.requestedGpu}>
                {gpuOptions.map((gpu) => (
                  <option key={gpu} value={gpu}>
                    {gpu}
                  </option>
                ))}
              </select>
            </label>
            {usesPrompt ? (
              <>
                <div className="dataset-caption-row">
                  <label>
                    Type
                    <select onChange={(event) => onChange("captionType", event.target.value)} value={settings.captionType}>
                      {captionTypes.map((type) => (
                        <option key={type} value={type}>
                          {type}
                        </option>
                      ))}
                    </select>
                  </label>
                  <label>
                    Length
                    <select onChange={(event) => onChange("captionLength", event.target.value)} value={settings.captionLength}>
                      {captionLengths.map((length) => (
                        <option key={length} value={length}>
                          {length}
                        </option>
                      ))}
                    </select>
                  </label>
                </div>
                <label>
                  Character name
                  <input onChange={(event) => onChange("nameInput", event.target.value)} value={settings.nameInput} />
                </label>
                <label>
                  Caption prompt
                  <textarea onChange={(event) => onChange("captionPrompt", event.target.value)} rows={6} value={promptValue} />
                </label>
              </>
            ) : null}
            <div className="dataset-caption-row">
              <label>
                Temperature
                <input
                  max="2"
                  min="0"
                  onChange={(event) => onChange("temperature", event.target.value)}
                  step="0.05"
                  type="number"
                  value={settings.temperature}
                />
              </label>
              <label>
                Top P
                <input
                  max="1"
                  min="0"
                  onChange={(event) => onChange("topP", event.target.value)}
                  step="0.05"
                  type="number"
                  value={settings.topP}
                />
              </label>
              <label>
                Max tokens
                <input
                  max="1024"
                  min="1"
                  onChange={(event) => onChange("maxNewTokens", event.target.value)}
                  step="1"
                  type="number"
                  value={settings.maxNewTokens}
                />
              </label>
            </div>
            <label className="training-toggle-line">
              <input checked={settings.lowVram} onChange={(event) => onChange("lowVram", event.target.checked)} type="checkbox" />
              <span>Low VRAM</span>
            </label>
            {single ? null : (
              <label className="training-toggle-line">
                <input checked={settings.recaption} onChange={(event) => onChange("recaption", event.target.checked)} type="checkbox" />
                <span>Re-caption images that already have a caption</span>
              </label>
            )}
            {usesPrompt ? (
              <div className="dataset-caption-options">
                {extraOptions.map((option) => (
                  <label className="training-toggle-line" key={option.value}>
                    <input
                      checked={settings.extraOptions.includes(option.value)}
                      onChange={() => onToggleExtra(option.value)}
                      type="checkbox"
                    />
                    <span>{option.label}</span>
                  </label>
                ))}
              </div>
            ) : null}
          </>
        ) : null}
      </div>

      <footer className="dataset-caption-footer">
        <button onClick={onClose} type="button">
          Cancel
        </button>
        <button className="primary-action" disabled={running || blockRun} onClick={onRun} type="button">
          {running ? "Queuing…" : runLabel}
        </button>
      </footer>
    </Modal>
  );
}
