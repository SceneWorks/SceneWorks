import React, { useState } from "react";

import { isDesktop, tauriInvoke } from "../runtime.js";
import { Modal } from "./Modal.jsx";
import { errorMessage } from "../errorMessage.js";

const initialSettings = {
  sourcePath: "",
  urlColumn: "URL",
  captionColumn: "TEXT",
  maxItems: 1000,
  minEdge: 512,
  concurrency: 16,
  captionPreset: "all",
  captionIncludes: "",
  captionExcludes: "",
};

const captionPresets = {
  all: "",
  people: "portrait, person, people, woman, man, girl, boy, model",
  faces: "face, facial, portrait, headshot, selfie, close-up, close up",
  objects: "product, object, still life, isolated, sculpture, furniture",
};

export function DatasetParquetImportDialog({ running = false, onClose, onRun }) {
  const [settings, setSettings] = useState(initialSettings);
  const [pickerError, setPickerError] = useState("");
  const [submitError, setSubmitError] = useState("");

  function update(field, value) {
    setSettings((current) => ({ ...current, [field]: value }));
  }

  function applyCaptionPreset(preset) {
    setSettings((current) => ({
      ...current,
      captionPreset: preset,
      captionIncludes: captionPresets[preset] ?? current.captionIncludes,
    }));
  }

  async function chooseSourceFolder() {
    setPickerError("");
    try {
      const selected = await tauriInvoke("choose_folder");
      if (selected) {
        update("sourcePath", selected);
      }
    } catch (error) {
      setPickerError(error?.message ?? String(error));
    }
  }

  async function submit(event) {
    event.preventDefault();
    if (!settings.sourcePath.trim() || running) return;
    setSubmitError("");
    try {
      await onRun({
        sourcePath: settings.sourcePath.trim(),
        urlColumn: settings.urlColumn.trim(),
        captionColumn: settings.captionColumn.trim(),
        maxItems: Number(settings.maxItems),
        minEdge: Number(settings.minEdge),
        concurrency: Number(settings.concurrency),
        captionIncludes: settings.captionIncludes
          .split(",")
          .map((term) => term.trim())
          .filter(Boolean),
        captionExcludes: settings.captionExcludes
          .split(",")
          .map((term) => term.trim())
          .filter(Boolean),
      });
    } catch (error) {
      setSubmitError(errorMessage(error, "Could not start the Parquet import."));
    }
  }

  return (
    <Modal className="dataset-caption-modal" labelledBy="dataset-parquet-title" onClose={onClose}>
      <header className="dataset-caption-head">
        <div>
          <p className="eyebrow">Local metadata</p>
          <h2 id="dataset-parquet-title">Import Parquet dataset</h2>
        </div>
        <button className="modal-close" disabled={running} onClick={onClose} type="button">
          Close
        </button>
      </header>

      <form className="dataset-caption-body" id="dataset-parquet-form" onSubmit={submit}>
        <p className="dataset-parquet-copy">
          Reads LAION-style URL/caption rows, downloads valid images, and adds them to this empty
          dataset. The metadata remains in place.
        </p>
        <label>
          Parquet file or folder
          <span className="dataset-parquet-path-row">
            <input
              autoFocus
              onChange={(event) => update("sourcePath", event.target.value)}
              placeholder="D:\laion2B-en-aesthetic-metadata"
              value={settings.sourcePath}
            />
            {isDesktop ? (
              <button className="secondary-action" onClick={chooseSourceFolder} type="button">
                Browse
              </button>
            ) : null}
          </span>
        </label>
        {!isDesktop ? (
          <small className="dataset-parquet-hint">
            Enter an absolute path on the machine running SceneWorks.
          </small>
        ) : null}
        {pickerError ? <p className="inline-warning">{pickerError}</p> : null}
        {submitError ? <p className="inline-warning">{submitError}</p> : null}

        <div className="dataset-caption-row">
          <label>
            URL column
            <input onChange={(event) => update("urlColumn", event.target.value)} value={settings.urlColumn} />
          </label>
          <label>
            Caption column
            <input
              onChange={(event) => update("captionColumn", event.target.value)}
              value={settings.captionColumn}
            />
          </label>
        </div>
        <label>
          Caption category preset (heuristic)
          <select
            onChange={(event) => applyCaptionPreset(event.target.value)}
            value={settings.captionPreset}
          >
            <option value="all">All image types</option>
            <option value="people">People</option>
            <option value="faces">Faces and portraits</option>
            <option value="objects">Objects and products</option>
            <option value="custom">Custom terms</option>
          </select>
        </label>
        <div className="dataset-caption-row">
          <label>
            Include any caption term
            <input
              onChange={(event) =>
                setSettings((current) => ({
                  ...current,
                  captionPreset: "custom",
                  captionIncludes: event.target.value,
                }))
              }
              placeholder="portrait, person, headshot"
              value={settings.captionIncludes}
            />
          </label>
          <label>
            Exclude any caption term
            <input
              onChange={(event) => update("captionExcludes", event.target.value)}
              placeholder="logo, illustration, watermark"
              value={settings.captionExcludes}
            />
          </label>
        </div>
        <p className="dataset-parquet-hint">
          Type filters search the LAION caption before download; they are fast but not semantic
          classifiers. Use Dataset Doctor’s face check after a face-oriented import.
        </p>
        <div className="dataset-caption-row">
          <label>
            Images to import
            <input
              max="100000"
              min="1"
              onChange={(event) => update("maxItems", event.target.value)}
              type="number"
              value={settings.maxItems}
            />
          </label>
          <label>
            Minimum edge
            <input
              max="8192"
              min="0"
              onChange={(event) => update("minEdge", event.target.value)}
              step="64"
              type="number"
              value={settings.minEdge}
            />
          </label>
          <label>
            Concurrent fetches
            <input
              max="64"
              min="1"
              onChange={(event) => update("concurrency", event.target.value)}
              type="number"
              value={settings.concurrency}
            />
          </label>
        </div>
        <p className="dataset-parquet-hint">
          Imports are resumable. Failed or undersized URLs are skipped, so the final count may be
          lower when the candidate window is exhausted. Large imports can take hours and use
          substantial disk space.
        </p>
      </form>

      <footer className="dataset-caption-footer">
        <button className="secondary-action" disabled={running} onClick={onClose} type="button">
          Cancel
        </button>
        <button
          className="primary-action"
          disabled={!settings.sourcePath.trim() || running}
          form="dataset-parquet-form"
          type="submit"
        >
          {running ? "Queueing…" : "Start import"}
        </button>
      </footer>
    </Modal>
  );
}
