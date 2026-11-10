import React, { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { Modal } from "./Modal.jsx";
import { RequiredModelsNotice } from "./RequiredModelsNotice.jsx";
import { DescribePromptPresets } from "./DescribePromptPresets.jsx";
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
import { appConfirm } from "../appConfirm.jsx";
import { issue, summarize } from "../validation/issues.js";
import { ValidationSummary, invalidProps } from "../validation/Validation.jsx";

// "Create Prompt Batch from Dataset" (SW-7085, feature SW-F362). A vision model describes each
// dataset image as six fields; the review table lets the user edit them, re-run one image, and save
// the result as an Image Studio prompt batch (one `[W:H]` block per image, see promptBatchDescribe.js).
//
// The describe job (`dataset_batch_describe`) stores each image's result as it finishes (the job's
// `/result-items`) and reports only counts in its progress, so the dialog polls the job and, when
// `processedCount` grows, reads the new items in order and merges them into the table. A cancel
// keeps every finished row: the canceled job still serves the items it stored.
//
// Polling survives a dropped request: each read is retried with a bounded backoff, and if it still
// fails the dialog stops monitoring without losing the job or the rows, and offers to resume.

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
// The wait before each retry of a failed job or items read. A read still failing after the last
// retry stops monitoring (the job itself keeps running) and the dialog offers to resume.
export const POLL_RETRY_DELAYS_MS = [1000, 2000, 4000, 8000];

// Run `read`, retrying a failure after each POLL_RETRY_DELAYS_MS wait. An abort is never retried.
async function withRetries(read, signal) {
  for (let attempt = 0; ; attempt += 1) {
    try {
      return await read();
    } catch (error) {
      if (isAbortError(error) || signal?.aborted || attempt >= POLL_RETRY_DELAYS_MS.length) throw error;
      await abortableDelay(POLL_RETRY_DELAYS_MS[attempt], signal);
    }
  }
}

// Characters as the API and worker count them (Unicode scalar values of the trimmed text).
function promptChars(text) {
  return [...(text ?? "").trim()].length;
}

// The describing prompt's issues for the selected model: a text over that model's limit is an
// error that blocks Run (the API would refuse it with a 400).
export function describePromptIssues(prompt, limit, modelLabel) {
  if (!limit) return [];
  const count = promptChars(prompt);
  if (count <= limit) return [];
  return [
    issue.error(
      "prompt",
      `The describing prompt is ${count.toLocaleString()} characters; ${modelLabel} accepts at most ${limit.toLocaleString()}.`,
    ),
  ];
}

// `model` is the repo of the job that produced the item (the job result's `model`), so each row
// remembers which describer wrote it even when rows come from different runs.
function rowFromResultItem(item, model = null) {
  return {
    itemId: item.itemId,
    model: typeof model === "string" && model ? model : null,
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
export function mergeDescribeRows(rows, items, datasetOrder = [], model = null) {
  const byId = new Map(rows.map((row) => [row.itemId, row]));
  for (const item of items ?? []) {
    if (item?.itemId) byId.set(item.itemId, rowFromResultItem(item, model));
  }
  const rank = new Map(datasetOrder.map((id, index) => [id, index]));
  return [...byId.values()].sort(
    (left, right) => (rank.get(left.itemId) ?? Infinity) - (rank.get(right.itemId) ?? Infinity),
  );
}

// The describer recorded for a saved batch: the one model every included row was described with,
// "mixed" when rows came from runs with different models (each `sourceImages` entry then names its
// own), or null when no row recorded one. A single string keeps `source.model` the same shape it
// has always had for the common one-model case; the per-row detail lives beside each image.
export function describedByModel(rows) {
  const models = new Set(rows.map((row) => row.model ?? null));
  if (models.size === 1) return [...models][0];
  return models.size ? "mixed" : null;
}

// The prompt-batch create payload for the reviewed rows: one formatter block per usable row, and
// each block's exact source pixel size and describing model in `sourceImages` (same order as
// `prompts`).
export function promptBatchPayload({ rows, name, dataset }) {
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
      model: row.model ?? null,
    })),
    source: {
      kind: "training_dataset",
      datasetId: dataset?.id ?? null,
      datasetName: dataset?.name ?? "",
      datasetVersion: dataset?.version ?? null,
      model: describedByModel(included),
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
  // Each model's default instruction and limit, by repo: `{ prompt, maxChars }`.
  const [defaults, setDefaults] = useState({});
  const [prompt, setPrompt] = useState("");
  // Whether the editor shows the selected model's untouched default, so a model switch swaps it
  // for the other model's default; an edited text stays as typed.
  const [followDefault, setFollowDefault] = useState(true);
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
  // Set while a describe request is being started (confirm + POST + monitoring). The ref blocks a
  // second click synchronously, before React re-renders the disabled buttons; the state disables them.
  const submittingRef = useRef(false);
  const [submitting, setSubmitting] = useState(false);
  // Why monitoring stopped while the job was still active (its reads kept failing), or "".
  const [monitorError, setMonitorError] = useState("");
  const monitoringRef = useRef(false);
  // How many of a job's stored items the table already holds, so polling reads only new ones.
  const itemsRead = useRef({ jobId: null, count: 0 });
  const pollAbort = useRef(null);

  useEffect(() => () => pollAbort.current?.abort(), []);

  const model = DESCRIBE_MODELS.find((entry) => entry.id === modelId) ?? DESCRIBE_MODELS[0];
  const modelDefault = defaults[model.repo] ?? null;
  const defaultPrompt = modelDefault ? modelDefault.prompt : null;
  const loadedDefault = Boolean(modelDefault);

  // Load the selected model's default instruction and limit (served by the API from the worker's
  // own prompt asset and core's limits, so the web app holds no second copy).
  useEffect(() => {
    if (loadedDefault) return undefined;
    const controller = new AbortController();
    const repo = model.repo;
    trainingApi
      .defaultDescribePrompt(token, { signal: controller.signal }, repo)
      .then((body) => {
        const text = typeof body?.prompt === "string" ? body.prompt : "";
        const maxChars = Number(body?.maxChars) || 0;
        setPromptError("");
        setDefaults((current) => ({ ...current, [repo]: { prompt: text, maxChars } }));
      })
      .catch((error) => {
        if (!isAbortError(error)) setPromptError(errorMessage(error, "Could not load the default describing prompt."));
      });
    return () => controller.abort();
  }, [loadedDefault, model.repo, token]);

  // Show the selected model's default while the text is untouched.
  useEffect(() => {
    if (followDefault && defaultPrompt !== null) setPrompt(defaultPrompt);
  }, [defaultPrompt, followDefault]);

  function chooseModel(id) {
    // An untouched text (or one the user typed back to the default) follows the new model's
    // default; an edited text is kept.
    if (promptChars(prompt) === 0 || (defaultPrompt !== null && prompt.trim() === defaultPrompt.trim())) {
      setFollowDefault(true);
    }
    setModelId(id);
  }

  function editPrompt(text) {
    setFollowDefault(false);
    setPrompt(text);
  }

  function selectPreset(text, { builtin = false } = {}) {
    if (builtin) {
      setFollowDefault(true);
      if (defaultPrompt !== null) setPrompt(defaultPrompt);
    } else {
      editPrompt(text);
    }
  }

  const promptValidation = useMemo(
    () => summarize(describePromptIssues(prompt, modelDefault?.maxChars, model.label)),
    [prompt, modelDefault, model.label],
  );
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

  // Read the job's stored items from `from` on, page by page, merging them into the table. Returns
  // how many items have been read in all.
  const readItems = useCallback(async (jobId, from, model, signal) => {
    let offset = from;
    for (;;) {
      const page = await withRetries(() => jobsApi.resultItems(jobId, token, { offset }, { signal }), signal);
      const items = Array.isArray(page?.items) ? page.items : [];
      if (items.length) setRows((current) => mergeDescribeRows(current, items, datasetOrder, model));
      offset += items.length;
      if (!items.length || page?.nextOffset == null) return offset;
    }
  }, [datasetOrder, token]);

  const pollJob = useCallback(async (jobId, signal) => {
    for (;;) {
      await abortableDelay(POLL_MS, signal);
      const snapshot = await withRetries(() => jobsApi.item(jobId, token, { signal }), signal);
      // The worker stores an item before the post that counts it, so `processedCount` items exist.
      const processed = Number(snapshot?.result?.processedCount) || 0;
      if (itemsRead.current.jobId === jobId && processed > itemsRead.current.count) {
        itemsRead.current.count = await readItems(jobId, itemsRead.current.count, snapshot.result.model, signal);
      }
      setJob(snapshot);
      if (terminalStatuses.has(snapshot?.status)) {
        return snapshot;
      }
    }
  }, [readItems, token]);

  function reportFinished(finished) {
    setRerunItemId(null);
    if (finished.status === "canceled") {
      const kept = finished.result?.processedCount ?? 0;
      setRunMessage(`Canceled. Kept ${kept} finished ${kept === 1 ? "image" : "images"}.`);
    } else if (finished.status !== "completed") {
      setRunError(finished.error || finished.message || "The describe job failed.");
    } else if (finished.result?.failedCount) {
      setRunMessage(`Described ${finished.result.describedCount} images; ${finished.result.failedCount} flagged.`);
    }
  }

  // Poll `jobId` until it ends. If its reads keep failing, stop with the job and rows kept and
  // record why, so the user can resume monitoring (or cancel) instead of facing a frozen dialog.
  async function monitor(jobId) {
    if (monitoringRef.current) return;
    monitoringRef.current = true;
    pollAbort.current?.abort();
    const controller = new AbortController();
    pollAbort.current = controller;
    setMonitorError("");
    try {
      reportFinished(await pollJob(jobId, controller.signal));
    } catch (error) {
      if (!isAbortError(error)) {
        setMonitorError(errorMessage(error, "Lost contact with the describe job."));
      }
    } finally {
      monitoringRef.current = false;
    }
  }

  function resumeMonitoring() {
    if (job?.id && running) monitor(job.id);
  }

  // Start a describe job: every image (`itemIds` null), or exactly `itemIds`. `confirmFirst`, when
  // given, runs first and can decline. Ignored while a start is already in flight or a job runs.
  async function submit(itemIds = null, confirmFirst = null) {
    if (submittingRef.current || running || !dataset?.id || !projectId) return;
    submittingRef.current = true;
    setSubmitting(true);
    try {
      if (confirmFirst && !(await confirmFirst())) return;
      setRunError("");
      setRunMessage("");
      setMonitorError("");
      setSavedBatch(null);
      const body = { model: model.repo, requestedGpu: "auto" };
      // The built-in default is sent as "no prompt" so the worker uses its own default instruction.
      if (prompt.trim() && prompt.trim() !== (defaultPrompt ?? "").trim()) body.prompt = prompt;
      if (itemIds) body.itemIds = itemIds;
      pollAbort.current?.abort();
      const controller = new AbortController();
      pollAbort.current = controller;
      let created;
      try {
        created = await trainingApi.batchDescribeJobs(projectId, dataset.id, token, {
          method: "POST",
          body: JSON.stringify(body),
          signal: controller.signal,
        });
      } catch (error) {
        if (!isAbortError(error)) setRunError(errorMessage(error, "Could not run the describe job."));
        return;
      }
      if (!itemIds) setRows([]);
      itemsRead.current = { jobId: created.id, count: 0 };
      setRerunItemId(itemIds?.length === 1 ? itemIds[0] : null);
      setJob(created);
      await monitor(created.id);
    } finally {
      submittingRef.current = false;
      setSubmitting(false);
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
      return;
    }
    // Monitoring had stopped: resume it so the cancel's outcome (and the kept rows) arrive.
    if (monitorError) monitor(job.id);
  }

  function updateField(itemId, key, value) {
    setSavedBatch(null);
    setRows((current) =>
      current.map((row) => (row.itemId === itemId ? { ...row, edited: true, fields: { ...row.fields, [key]: value } } : row)),
    );
  }

  // Images with no successful result yet: never described (e.g. after a cancel) or flagged.
  const remainingIds = useMemo(() => {
    const described = new Set(rows.filter((row) => !row.failed).map((row) => row.itemId));
    return datasetOrder.filter((id) => !described.has(id));
  }, [rows, datasetOrder]);

  async function describeAll() {
    const confirmFirst = rows.some((row) => row.edited)
      ? () =>
          appConfirm({
            title: "Describe every image again?",
            message: "This replaces the whole table, including the fields you edited. Use “Describe remaining” to describe only images without a result.",
            confirmLabel: "Describe all again",
            cancelLabel: "Keep my edits",
            tone: "danger",
          })
      : null;
    await submit(null, confirmFirst);
  }

  const payload = () => promptBatchPayload({ rows, name: batchName, dataset });
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

  // A job is starting or active: run controls and the inputs it was started from are locked.
  const busy = running || submitting;
  const runBlocked =
    busy || dirty || !selectedAvailable || defaultPrompt === null || !datasetOrder.length || !promptValidation.ready;

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
                  disabled={busy}
                  name="dataset-prompt-batch-model"
                  onChange={() => chooseModel(entry.id)}
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

        <DescribePromptPresets
          disabled={busy || defaultPrompt === null}
          model={model.repo}
          onSelect={selectPreset}
          prompt={prompt}
          token={token}
        />
        <label>
          Describing prompt
          <textarea
            aria-describedby="dataset-prompt-batch-prompt-hint"
            disabled={busy || defaultPrompt === null}
            onChange={(event) => editPrompt(event.target.value)}
            rows={8}
            {...invalidProps(promptValidation, "prompt")}
            value={prompt}
          />
          <span className="dataset-caption-field-hint" id="dataset-prompt-batch-prompt-hint">
            The tool adds the JSON output format after this text, so edits never change the six fields.
          </span>
        </label>
        <div className="dataset-prompt-batch-prompt-actions">
          <button
            className="secondary-action"
            disabled={busy || defaultPrompt === null || prompt === defaultPrompt}
            onClick={() => selectPreset(defaultPrompt ?? "", { builtin: true })}
            type="button"
          >
            Reset to default
          </button>
        </div>
        {promptError ? <p className="inline-warning" role="alert">{promptError}</p> : null}
        <ValidationSummary issues={promptValidation.surfaced} label="Describing prompt errors" />
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
        {monitorError && running ? (
          <div className="inline-warning dataset-prompt-batch-monitor-lost" role="alert">
            <span>Lost contact with the describe job: {monitorError} The job may still be running; finished rows are kept.</span>
            <button className="secondary-action" onClick={resumeMonitoring} type="button">
              Resume monitoring
            </button>
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
                          disabled={busy}
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
          {rows.length && remainingIds.length ? (
            <button className="secondary-action" disabled={runBlocked} onClick={() => submit(remainingIds)} type="button">
              Describe remaining ({remainingIds.length})
            </button>
          ) : null}
          <button className="primary-action" disabled={runBlocked} onClick={describeAll} type="button">
            {running && !rerunItemId ? "Describing…" : rows.length ? "Describe all again" : "Describe images"}
          </button>
        </span>
      </footer>
    </Modal>
  );
}
