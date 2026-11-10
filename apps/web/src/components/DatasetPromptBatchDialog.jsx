import React, { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { Modal } from "./Modal.jsx";
import { RequiredModelsNotice } from "./RequiredModelsNotice.jsx";
import { useAppContext } from "../context/AppContext.js";
import { trainingApi } from "../api/trainingResources.js";
import { jobsApi } from "../api/jobsResources.js";
import { abortableDelay } from "../pollJob.js";
import { isAbortError } from "../api.js";
import { errorMessage } from "../errorMessage.js";
import { JOY_CAPTION_MODEL_ID, VISION_CAPTION_MODEL_ID, VISION_CAPTION_MODEL_REPO, terminalStatuses } from "../constants.js";
import { joyCaptionModel } from "../training/joyCaptionPrompts.js";
import { missingRequiredModels, modelInstallComplete } from "../modelEligibility.js";
import { formatPromptBatch, snapRatio } from "../promptBatchDescribe.js";
import { extractKeys } from "../promptBatch.js";
import { serializePromptBatchExport } from "../promptBatchIO.js";
import { saveExportFile } from "../saveExportFile.js";

// "Create Prompt Batch from Dataset" (SW-7085, feature SW-F362). A vision model describes each
// dataset image as six fields; the review table lets the user edit them, re-run one image, and save
// the result as an Image Studio prompt batch (one `[W:H]` block per image, see promptBatchDescribe.js).
//
// The describe job (`dataset_batch_describe`) streams its partial result on every running post, so
// the dialog polls the job and merges `result.items` into the table as images finish. A cancel keeps
// every finished row: the canceled job's result carries them.

// The describers the job accepts: catalog id (install state) and the repo the job's `model` names.
export const DESCRIBE_MODELS = [
  { id: VISION_CAPTION_MODEL_ID, repo: VISION_CAPTION_MODEL_REPO, label: "Vision Captioner (Qwen3-VL-8B)" },
  { id: JOY_CAPTION_MODEL_ID, repo: joyCaptionModel, label: "JoyCaption beta-one" },
];

export const DESCRIBE_FIELDS = [
  ["description", "Description"],
  ["style", "Style"],
  ["camera", "Camera"],
  ["composition", "Composition"],
  ["lighting", "Lighting"],
  ["environment", "Environment"],
];

const POLL_MS = 1000;

function rowFromResultItem(item) {
  return {
    itemId: item.itemId,
    imagePath: item.imagePath ?? "",
    assetId: item.assetId ?? null,
    width: item.width ?? null,
    height: item.height ?? null,
    fields: Object.fromEntries(DESCRIBE_FIELDS.map(([key]) => [key, typeof item[key] === "string" ? item[key] : ""])),
    failed: Boolean(item.failed),
    error: item.error ?? null,
  };
}

// Merge result items into the rows, replacing a row with the same item id and keeping dataset order.
export function mergeDescribeRows(rows, items, datasetOrder = []) {
  const byId = new Map(rows.map((row) => [row.itemId, row]));
  for (const item of items ?? []) {
    if (item?.itemId) byId.set(item.itemId, rowFromResultItem(item));
  }
  const rank = new Map(datasetOrder.map((id, index) => [id, index]));
  return [...byId.values()].sort(
    (left, right) => (rank.get(left.itemId) ?? Infinity) - (rank.get(right.itemId) ?? Infinity),
  );
}

// The prompt-batch create payload for the reviewed rows: one formatter block per usable row, and
// each block's exact source pixel size in `sourceImages` (same order as `prompts`).
export function promptBatchPayload({ rows, name, dataset, model }) {
  const inputs = rows.map((row) => ({ ...row }));
  const { text, skipped } = formatPromptBatch(inputs);
  const prompts = text ? text.split("\n---\n") : [];
  const included = inputs.filter((input) => !skipped.includes(input));
  return {
    name: name.trim(),
    scope: "global",
    prompts,
    variables: extractKeys(prompts).map((key) => ({ key, values: [] })),
    sourceImages: included.map((row) => ({
      itemId: row.itemId,
      width: row.width,
      height: row.height,
      ratio: snapRatio(row.width, row.height),
    })),
    source: {
      kind: "training_dataset",
      datasetId: dataset?.id ?? null,
      datasetName: dataset?.name ?? "",
      datasetVersion: dataset?.version ?? null,
      model,
    },
  };
}

export function DatasetPromptBatchDialog({ dataset, dirty = false, renderThumbnail, onClose }) {
  const {
    token,
    activeProject,
    models = [],
    jobs = [],
    createModelDownloadJob,
    jobAction,
    setActiveView,
    createPromptBatch,
    sendPromptBatchToImage,
  } = useAppContext();
  const projectId = activeProject?.id;
  const datasetOrder = useMemo(() => (dataset?.items ?? []).map((item) => item.id), [dataset]);

  const [modelId, setModelId] = useState(VISION_CAPTION_MODEL_ID);
  const [defaultPrompt, setDefaultPrompt] = useState(null);
  const [prompt, setPrompt] = useState("");
  const [promptError, setPromptError] = useState("");
  const [rows, setRows] = useState([]);
  const [job, setJob] = useState(null);
  const [rerunItemId, setRerunItemId] = useState(null);
  const [runError, setRunError] = useState("");
  const [runMessage, setRunMessage] = useState("");
  const [batchName, setBatchName] = useState(() => `${dataset?.name ?? "Dataset"} prompts`);
  const [saving, setSaving] = useState(false);
  const [savedBatch, setSavedBatch] = useState(null);
  const [saveError, setSaveError] = useState("");
  const pollAbort = useRef(null);

  // Prefill the editable prompt with the worker's default instruction (served by the API from the
  // worker's own prompt asset, so the web app holds no second copy).
  useEffect(() => {
    const controller = new AbortController();
    trainingApi
      .defaultDescribePrompt(token, { signal: controller.signal })
      .then((body) => {
        const text = typeof body?.prompt === "string" ? body.prompt : "";
        setDefaultPrompt(text);
        setPrompt((current) => (current ? current : text));
      })
      .catch((error) => {
        if (!isAbortError(error)) setPromptError(errorMessage(error, "Could not load the default describing prompt."));
      });
    return () => controller.abort();
  }, [token]);

  useEffect(() => () => pollAbort.current?.abort(), []);

  const model = DESCRIBE_MODELS.find((entry) => entry.id === modelId) ?? DESCRIBE_MODELS[0];
  const catalogEntry = (id) => models.find((entry) => entry.id === id);
  const modelAvailable = (id) => {
    const entry = catalogEntry(id);
    return Boolean(entry) && modelInstallComplete(entry);
  };
  const missingModels = missingRequiredModels(models, [model.id]);
  const selectedAvailable = modelAvailable(model.id);
  const downloadJobs = useMemo(() => jobs.filter((entry) => entry.type === "model_download"), [jobs]);

  const running = Boolean(job) && !terminalStatuses.has(job.status);
  const result = job?.result ?? null;
  const itemCount = result?.itemCount ?? (rerunItemId ? 1 : datasetOrder.length);
  const processedCount = result?.processedCount ?? 0;
  const progressPercent = Math.round(Math.max(0, Math.min(1, Number(job?.progress) || 0)) * 100);

  const pollJob = useCallback(async (jobId, signal) => {
    for (;;) {
      await abortableDelay(POLL_MS, signal);
      const snapshot = await jobsApi.item(jobId, token, { signal });
      setJob(snapshot);
      if (Array.isArray(snapshot?.result?.items)) {
        setRows((current) => mergeDescribeRows(current, snapshot.result.items, datasetOrder));
      }
      if (terminalStatuses.has(snapshot?.status)) {
        return snapshot;
      }
    }
  }, [datasetOrder, token]);

  async function submit(itemIds = null) {
    if (running || !dataset?.id || !projectId) return;
    setRunError("");
    setRunMessage("");
    setSavedBatch(null);
    const body = { model: model.repo, requestedGpu: "auto" };
    // The built-in default is sent as "no prompt" so the worker uses its own default instruction.
    if (prompt.trim() && prompt.trim() !== (defaultPrompt ?? "").trim()) body.prompt = prompt;
    if (itemIds) body.itemIds = itemIds;
    pollAbort.current?.abort();
    const controller = new AbortController();
    pollAbort.current = controller;
    try {
      const created = await trainingApi.batchDescribeJobs(projectId, dataset.id, token, {
        method: "POST",
        body: JSON.stringify(body),
        signal: controller.signal,
      });
      if (!itemIds) setRows([]);
      setRerunItemId(itemIds ? itemIds[0] : null);
      setJob(created);
      const finished = await pollJob(created.id, controller.signal);
      if (finished.status === "canceled") {
        const kept = finished.result?.processedCount ?? 0;
        setRunMessage(`Canceled. Kept ${kept} finished ${kept === 1 ? "image" : "images"}.`);
      } else if (finished.status !== "completed") {
        setRunError(finished.error || finished.message || "The describe job failed.");
      } else if (finished.result?.failedCount) {
        setRunMessage(`Described ${finished.result.describedCount} images; ${finished.result.failedCount} flagged.`);
      }
    } catch (error) {
      if (!isAbortError(error)) setRunError(errorMessage(error, "Could not run the describe job."));
    } finally {
      setRerunItemId(null);
    }
  }

  async function cancel() {
    if (!job?.id || !running) return;
    try {
      if (typeof jobAction === "function") {
        await jobAction(job, "cancel");
      } else {
        await jobsApi.cancel(job.id, token, { method: "POST" });
      }
    } catch (error) {
      setRunError(errorMessage(error, "Could not cancel the describe job."));
    }
  }

  function updateField(itemId, key, value) {
    setSavedBatch(null);
    setRows((current) =>
      current.map((row) => (row.itemId === itemId ? { ...row, fields: { ...row.fields, [key]: value } } : row)),
    );
  }

  const payload = () => promptBatchPayload({ rows, name: batchName, dataset, model: model.repo });
  const usableCount = useMemo(() => rows.length - formatPromptBatch(rows).skipped.length, [rows]);

  async function save() {
    if (saving || !usableCount || !batchName.trim() || typeof createPromptBatch !== "function") return;
    setSaving(true);
    setSaveError("");
    try {
      const created = await createPromptBatch(payload());
      setSavedBatch(created ?? null);
    } catch (error) {
      setSaveError(errorMessage(error, "Could not save the prompt batch."));
    } finally {
      setSaving(false);
    }
  }

  async function exportJson() {
    setSaveError("");
    const batch = savedBatch ?? payload();
    const slug = (batch.name || "prompt-batch").trim().toLowerCase().replace(/[^a-z0-9]+/g, "-").replace(/^-+|-+$/g, "");
    const file = new File([serializePromptBatchExport(batch)], `${slug || "prompt-batch"}.json`, { type: "application/json" });
    try {
      await saveExportFile(file);
    } catch (error) {
      setSaveError(`Export failed: ${error?.message ?? String(error)}`);
    }
  }

  function openInImageStudio() {
    if (!savedBatch || typeof sendPromptBatchToImage !== "function") return;
    sendPromptBatchToImage(savedBatch);
    onClose();
  }

  const runBlocked = running || dirty || !selectedAvailable || defaultPrompt === null || !datasetOrder.length;

  return (
    <Modal className="dataset-caption-modal dataset-prompt-batch-modal" labelledBy="dataset-prompt-batch-title" onClose={onClose}>
      <header className="dataset-caption-head">
        <div>
          <p className="eyebrow">{dataset?.name ?? "Dataset"}</p>
          <h2 id="dataset-prompt-batch-title">Create Prompt Batch from Dataset</h2>
        </div>
        <button className="modal-close" onClick={onClose} type="button">
          Close
        </button>
      </header>

      <div className="dataset-caption-body">
        <fieldset className="dataset-prompt-batch-models">
          <legend>Describing model</legend>
          {DESCRIBE_MODELS.map((entry) => {
            const available = modelAvailable(entry.id);
            return (
              <label className="training-toggle-line" key={entry.id}>
                <input
                  checked={modelId === entry.id}
                  disabled={running}
                  name="dataset-prompt-batch-model"
                  onChange={() => setModelId(entry.id)}
                  type="radio"
                  value={entry.id}
                />
                <span>
                  {catalogEntry(entry.id)?.name ?? entry.label}
                  {available ? null : <small className="dataset-prompt-batch-unavailable"> — not installed</small>}
                </span>
              </label>
            );
          })}
        </fieldset>
        {!selectedAvailable ? (
          missingModels.length ? (
            <RequiredModelsNotice
              downloadJobs={downloadJobs}
              feature="Describing the dataset"
              models={missingModels}
              onCancelJob={typeof jobAction === "function" ? (download) => jobAction(download, "cancel") : undefined}
              onDownload={typeof createModelDownloadJob === "function" ? createModelDownloadJob : undefined}
              onOpenModels={typeof setActiveView === "function" ? () => setActiveView("Models") : undefined}
            />
          ) : (
            <p className="inline-warning" role="status">
              {model.label} is unavailable on this install.
            </p>
          )
        ) : null}

        <label>
          Describing prompt
          <textarea
            aria-describedby="dataset-prompt-batch-prompt-hint"
            disabled={running || defaultPrompt === null}
            onChange={(event) => setPrompt(event.target.value)}
            rows={8}
            value={prompt}
          />
          <span className="dataset-caption-field-hint" id="dataset-prompt-batch-prompt-hint">
            The tool adds the JSON output format after this text, so edits never change the six fields.
          </span>
        </label>
        <div className="dataset-prompt-batch-prompt-actions">
          <button
            className="secondary-action"
            disabled={running || defaultPrompt === null || prompt === defaultPrompt}
            onClick={() => setPrompt(defaultPrompt ?? "")}
            type="button"
          >
            Reset to default
          </button>
        </div>
        {promptError ? <p className="inline-warning" role="alert">{promptError}</p> : null}
        {dirty ? (
          <p className="inline-warning" role="status">Save the dataset first so the describer sees its current images.</p>
        ) : null}

        {job ? (
          <div className="dataset-prompt-batch-progress" role="status">
            <progress aria-label="Describe progress" max={100} value={progressPercent} />
            <span>
              {running ? "Describing" : job.status === "completed" ? "Described" : `Job ${job.status}`}: {processedCount} of {itemCount}{" "}
              {itemCount === 1 ? "image" : "images"}
            </span>
          </div>
        ) : null}
        {runError ? <p className="inline-warning" role="alert">{runError}</p> : null}
        {runMessage ? <p className="dataset-caption-mode-note" role="status">{runMessage}</p> : null}

        {rows.length ? (
          <div className="dataset-prompt-batch-table" role="table" aria-label="Described images">
            {rows.map((row) => {
              const ratio = snapRatio(row.width, row.height);
              return (
                <div
                  className={row.failed ? "dataset-prompt-batch-row failed" : "dataset-prompt-batch-row"}
                  data-item-id={row.itemId}
                  key={row.itemId}
                  role="row"
                >
                  <div className="dataset-prompt-batch-thumb" role="cell">
                    {renderThumbnail ? renderThumbnail(row.itemId) : null}
                    <span className="dataset-prompt-batch-size">
                      {row.width && row.height ? `${row.width}×${row.height}` : "Size unknown"}
                      {ratio ? ` · ${ratio}` : ""}
                    </span>
                    <button
                      className="secondary-action"
                      disabled={runBlocked}
                      onClick={() => submit([row.itemId])}
                      type="button"
                    >
                      {rerunItemId === row.itemId ? "Re-running…" : "Re-run"}
                    </button>
                  </div>
                  <div className="dataset-prompt-batch-fields" role="cell">
                    {row.failed ? (
                      <p className="inline-warning" role="alert">Flagged: {row.error || "The describer could not describe this image."}</p>
                    ) : null}
                    {DESCRIBE_FIELDS.map(([key, label]) => (
                      <label key={key}>
                        {label}
                        <textarea
                          disabled={running}
                          onChange={(event) => updateField(row.itemId, key, event.target.value)}
                          rows={key === "description" ? 3 : 2}
                          value={row.fields[key]}
                        />
                      </label>
                    ))}
                  </div>
                </div>
              );
            })}
          </div>
        ) : null}

        {rows.length ? (
          <label>
            Prompt batch name
            <input onChange={(event) => setBatchName(event.target.value)} value={batchName} />
          </label>
        ) : null}
        {saveError ? <p className="inline-warning" role="alert">{saveError}</p> : null}
        {savedBatch ? (
          <p className="dataset-caption-mode-note" role="status">
            Saved “{savedBatch.name}” with {savedBatch.prompts?.length ?? 0} prompts.
          </p>
        ) : null}
      </div>

      <footer className="dataset-caption-footer">
        <span className="dataset-prompt-batch-footer-actions">
          {running ? (
            <button className="secondary-action" onClick={cancel} type="button">
              Cancel
            </button>
          ) : null}
          <button className="secondary-action" disabled={!usableCount || running} onClick={exportJson} type="button">
            Export .json
          </button>
        </span>
        <span className="dataset-prompt-batch-footer-actions">
          {savedBatch ? (
            <button className="primary-action" onClick={openInImageStudio} type="button">
              Open in Image Studio
            </button>
          ) : (
            <button
              className="secondary-action"
              disabled={!usableCount || running || saving || !batchName.trim()}
              onClick={save}
              type="button"
            >
              {saving ? "Saving…" : "Save prompt batch"}
            </button>
          )}
          <button className="primary-action" disabled={runBlocked} onClick={() => submit()} type="button">
            {running && !rerunItemId ? "Describing…" : rows.length ? "Describe all again" : "Describe images"}
          </button>
        </span>
      </footer>
    </Modal>
  );
}
