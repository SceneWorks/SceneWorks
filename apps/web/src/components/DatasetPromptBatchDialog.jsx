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
import {
  JOY_CAPTION_MODEL_ID,
  VISION_CAPTION_MODEL_ID,
  VISION_CAPTION_MODEL_REPO,
  VISION_DESCRIBE_MODEL_ID,
  VISION_DESCRIBE_MODEL_REPO,
  terminalStatuses,
} from "../constants.js";
import { joyCaptionModel } from "../training/joyCaptionPrompts.js";
import { missingRequiredModels, modelInstallComplete } from "../modelEligibility.js";
import { installedTiers } from "../quantTier.js";
import { formatTierBytes } from "../formatting.js";
import { condenseChunks, condensedFor, condensedSizeText, fitToTargets, formatPromptBatch, snapRatio } from "../promptBatchDescribe.js";
import { extractKeys } from "../promptBatch.js";
import { serializePromptBatchExport } from "../promptBatchIO.js";
import { saveExportFile } from "../saveExportFile.js";
import { promptBudget } from "../styleComposer.js";
import { appConfirm } from "../appConfirm.jsx";
import { issue, summarize } from "../validation/issues.js";
import { ValidationSummary, invalidProps } from "../validation/Validation.jsx";

// "Create Prompt Batch from Dataset" (SW-7085, feature SW-F362). A vision model describes each
// dataset image as seven fields; the review table lets the user edit them, re-run one image, and save
// the result as an Image Studio prompt batch (one `[W:H]` block per image, see promptBatchDescribe.js).
//
// The describe job (`dataset_batch_describe`) stores each image's result as it finishes (the job's
// `/result-items`) and reports only counts in its progress, so the dialog polls the job and, when
// `processedCount` grows, reads the new items in order and merges them into the table. A cancel
// keeps every finished row: the canceled job still serves the items it stored.
//
// Polling survives a dropped request: each read is retried with a bounded backoff, and if it still
// fails the dialog stops monitoring without losing the job or the rows, and offers to resume.
//
// The table is rebuilt from the dataset's earlier describe jobs whenever the dialog opens, so closing
// it, saving, or a restart that interrupts a run never loses finished descriptions: each image gets
// its newest successful result, and a job that is still running is monitored again.

// The describers the job accepts: catalog id (install state) and the repo the job's `model` names.
// A describer with `tiers` runs one quant tier, sent as the job's `variant` (the first is the
// default); the others take no `variant`.
export const DESCRIBE_MODELS = [
// `condenses`: the describer has a text-only step that "Fit to" condenses with (SW-7163).
  { id: VISION_CAPTION_MODEL_ID, repo: VISION_CAPTION_MODEL_REPO, label: "Vision Captioner (Qwen3-VL-8B)", condenses: true },
  {
    id: VISION_DESCRIBE_MODEL_ID,
    repo: VISION_DESCRIBE_MODEL_REPO,
    label: "Vision Describer (Qwen3.8-27B)",
    condenses: true,
    tiers: [
      ["q4", "q4 (20 GB)"],
      ["q8", "q8 (32 GB)"],
      ["bf16", "Full precision (56 GB)"],
    ],
  },
  { id: JOY_CAPTION_MODEL_ID, repo: joyCaptionModel, label: "JoyCaption beta-one", condenses: false },
];

// The tier a describer runs: `tier` when the describer offers it, else its default; null for a
// describer without tiers.
function describerTier(entry, tier) {
  if (!entry?.tiers) return null;
  return entry.tiers.some(([key]) => key === tier) ? tier : entry.tiers[0][0];
}

// Whether the catalog `entry` can run `tier` (null: the model as a whole) on this install.
function describerInstalled(entry, tier) {
  if (!entry) return false;
  if (!tier || !entry.hasVariantMatrix) return modelInstallComplete(entry);
  return installedTiers(entry).includes(tier);
}

// `pose` is one line per person; a result stored before it existed reads as an empty pose.
export const DESCRIBE_FIELDS = [
  ["description", "Description"],
  ["pose", "Pose"],
  ["style", "Style"],
  ["camera", "Camera"],
  ["composition", "Composition"],
  ["lighting", "Lighting"],
  ["environment", "Environment"],
];

const POLL_MS = 1000;
// Describe jobs read per page of the dataset's history while restoring the table.
export const RESTORE_JOBS_PAGE = 50;
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

// `model` is the repo of the job that produced the item (the job result's `model`) and `variant`
// its tier, so each row remembers which describer wrote it even when rows come from different runs.
function rowFromResultItem(item, model = null, variant = null) {
  return {
    itemId: item.itemId,
    model: typeof model === "string" && model ? model : null,
    variant: typeof variant === "string" && variant ? variant : null,
    imagePath: item.imagePath ?? "",
    assetId: item.assetId ?? null,
    width: item.width ?? null,
    height: item.height ?? null,
    fields: Object.fromEntries(DESCRIBE_FIELDS.map(([key]) => [key, typeof item[key] === "string" ? item[key] : ""])),
    failed: Boolean(item.failed),
    error: item.error ?? null,
    // Set when the reply was cut off at the describer's length limit and only part was kept.
    notice: typeof item.notice === "string" && item.notice ? item.notice : null,
    // SW-7163: the prompt condensed for a "Fit to" target; absent on items stored before it.
    condensed: item.condensed && typeof item.condensed === "object" ? { ...item.condensed } : null,
  };
}

// The dialog's remembered Fit to choice per dataset ("" for off), so clearing it sticks even
// though the newest job's payload still names the old target. Browser storage may be unavailable.
const fitToKey = (datasetId) => `sceneworks.datasetPromptBatch.fitTo.${datasetId}`;
function storedFitTo(datasetId) {
  try {
    const value = window.localStorage.getItem(fitToKey(datasetId));
    return typeof value === "string" ? value : null;
  } catch {
    return null;
  }
}
function storeFitTo(datasetId, value) {
  try {
    window.localStorage.setItem(fitToKey(datasetId), value);
  } catch {
    // A private window or blocked storage: the choice lasts for this dialog only.
  }
}

// Record which result positions are provisional (stored before their condensing step) in
// `positions`, for `items` read from `start`; a final item clears its position. Returns the set.
export function trackProvisional(positions, items, start) {
  (items ?? []).forEach((item, index) => {
    if (item?.provisional) positions.add(start + index);
    else positions.delete(start + index);
  });
  return positions;
}

// Whether the describer `repo` runs the "Fit to" condensing step (the `condenses` property of
// DESCRIBE_MODELS; sceneworks-core `describer_condenses` decides the same on the server). A row
// stored before describers were recorded was described by the default, the Vision Captioner.
export function describerCondenses(repo) {
  if (!repo) return true;
  return DESCRIBE_MODELS.some((entry) => entry.repo === repo && entry.condenses);
}

// Whether a row can be condensed: described, by a describer with a text step.
export function condensable(row) {
  return Boolean(row) && !row.failed && describerCondenses(row.model);
}

// The condensable rows that would not save a usable condensed prompt for `fitTo`. This is the ONE
// definition the formatter uses too (`condensedFor`), so Save and the saved blocks cannot
// disagree: never condensed or still provisional, condensing failed, fields changed since, or
// the prompt emptied. Save and Export wait for them; Condense re-runs them.
export function rowsNotFitted(rows, fitTo) {
  if (!fitTo) return [];
  return (rows ?? []).filter((row) => condensable(row) && !condensedFor(row, fitTo));
}

// The rows "Condense" runs for `fitTo`: exactly the rows that are not fitted.
export const rowsToCondense = rowsNotFitted;

// What a condense-only or measure-only request carries for a row: its fields as edited, its size,
// its cut-off notice, and for measuring its edited condensed prompt.
export function condenseRequestRow(row, { measure = false } = {}) {
  return {
    itemId: row.itemId,
    fields: row.fields,
    width: row.width ?? null,
    height: row.height ?? null,
    notice: row.notice ?? null,
    ...(measure ? { condensedText: row.condensed?.text ?? "" } : {}),
  };
}

// Merge result items into the rows, replacing a row with the same item id and keeping dataset order.
export function mergeDescribeRows(rows, items, datasetOrder = [], model = null, variant = null) {
  const byId = new Map(rows.map((row) => [row.itemId, row]));
  for (const item of items ?? []) {
    if (item?.itemId) byId.set(item.itemId, rowFromResultItem(item, model, variant));
  }
  const rank = new Map(datasetOrder.map((id, index) => [id, index]));
  return [...byId.values()].sort(
    (left, right) => (rank.get(left.itemId) ?? Infinity) - (rank.get(right.itemId) ?? Infinity),
  );
}

// Whether a stored result still describes the dataset image as it is now. `source` is the job's
// payload entry for the item (its path and content hash when it was described). An image that was
// cropped or repointed keeps its item id but gets new pixels, so its old result no longer applies.
// Missing facts never disqualify a result: older jobs carry no hash, and some items no size.
export function resultMatchesImage(item, source, image) {
  if (!image) return false;
  if (source?.contentHash && image.contentHash) return source.contentHash === image.contentHash;
  const describedPath = source?.relativePath ?? item?.imagePath;
  if (describedPath && image.path && describedPath !== image.path) return false;
  if (item?.width && item?.height && image.width && image.height) {
    return item.width === image.width && item.height === image.height;
  }
  return true;
}

// Rebuild the table from a dataset's earlier describe jobs, newest first, each `{ model, variant,
// items, sources }` (`sources`: the job's payload entries by item id). Each current dataset image gets
// its newest successful result for its current pixels, or its newest flagged one when no run
// described it. Results for removed or changed images are left out.
export function restoreDescribeRows(runs, datasetItems) {
  const images = new Map((datasetItems ?? []).map((image) => [image.id, image]));
  const restored = new Map();
  for (const run of runs ?? []) {
    for (const item of run?.items ?? []) {
      if (!item?.itemId || !resultMatchesImage(item, run.sources?.get(item.itemId), images.get(item.itemId))) continue;
      const current = restored.get(item.itemId);
      if (!current || (current.failed && !item.failed)) restored.set(item.itemId, rowFromResultItem(item, run.model, run.variant));
    }
  }
  return mergeDescribeRows([...restored.values()], [], (datasetItems ?? []).map((image) => image.id));
}

// The saved global prompt batch a new save would collide with: same name, ignoring case and
// surrounding spaces. Saving again under that name replaces it instead of failing.
export function promptBatchNamed(batches, name) {
  const wanted = name.trim().toLowerCase();
  if (!wanted) return null;
  return (
    (batches ?? []).find(
      (batch) => (batch?.scope ?? "global") === "global" && String(batch?.name ?? "").trim().toLowerCase() === wanted,
    ) ?? null
  );
}

// The replacement for a saved batch keeps the values the user gave its variables (in Image Studio)
// for every key the new prompts still use: from the stored variable, or its `lastValues` entry.
export function keepVariableValues(variables, existing) {
  const stored = new Map((existing?.variables ?? []).map((variable) => [variable?.key, variable?.values]));
  const last = existing?.lastValues ?? {};
  return variables.map((variable) => {
    const values = [stored.get(variable.key), last[variable.key]].find((entry) => Array.isArray(entry) && entry.length);
    return values ? { ...variable, values: [...values] } : variable;
  });
}

// The describer recorded for a saved batch: the one model every included row was described with,
// "mixed" when rows came from runs with different models (each `sourceImages` entry then names its
// own), or null when no row recorded one. A single string keeps `source.model` the same shape it
// has always had for the common one-model case; the per-row detail lives beside each image.
// `key` picks the recorded value: "model", or "variant" for the describer's tier (same rule).
export function describedByModel(rows, key = "model") {
  const models = new Set(rows.map((row) => row[key] ?? null));
  if (models.size === 1) return [...models][0];
  return models.size ? "mixed" : null;
}

// The prompt-batch create payload for the reviewed rows: one formatter block per usable row, and
// each block's exact source pixel size, describing model and (when recorded) tier in
// `sourceImages` (same order as `prompts`).
export function promptBatchPayload({ rows, name, dataset, fitTo = null }) {
  const inputs = rows.map((row) => ({ ...row }));
  const { text, skipped } = formatPromptBatch(inputs, { fitTo });
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
      ...(row.variant ? { variant: row.variant } : {}),
    })),
    source: {
      kind: "training_dataset",
      datasetId: dataset?.id ?? null,
      datasetName: dataset?.name ?? "",
      datasetVersion: dataset?.version ?? null,
      model: describedByModel(included),
      variant: describedByModel(included, "variant"),
    },
  };
}

// One row's "Fit to" state (SW-7163): why a row is not condensed, a failed condense's notice, a
// prompt made stale by a field edit, or the editable condensed prompt with its size against the
// target's limit (with room for each variable value) and an over-limit warning. An edited prompt
// shows its live character count, never its old token count or verdict, and offers to measure it.
function CondensedPrompt({ row, target, busy, measureBlocked, onChange, onMeasure }) {
  if (row.failed) return null;
  if (!condensable(row)) {
    return (
      <p className="dataset-caption-mode-note" data-row-note="not-condensed">
        Described by {DESCRIBE_MODELS.find((entry) => entry.repo === row.model)?.label ?? row.model}, which has no text step, so
        it is not condensed; the seven fields are saved.
      </p>
    );
  }
  const condensed = row.condensed?.target === target.id ? row.condensed : null;
  if (!condensed) {
    return (
      <p className="inline-warning" data-row-note="not-condensed">
        Not condensed for {target.name} yet.
      </p>
    );
  }
  if (condensed.failed) {
    return (
      <p className="inline-warning" data-row-note="condense-failed">
        {condensed.notice || "Condensing failed, so the seven fields are saved."}
      </p>
    );
  }
  if (row.fieldsEdited) {
    return (
      <p className="inline-warning" data-row-note="condense-stale">
        Fields changed since this was condensed, so this prompt is not used. Condense again.
      </p>
    );
  }
  const textarea = (
    <label>
      Condensed prompt
      <textarea disabled={busy} onChange={(event) => onChange(row.itemId, event.target.value)} rows={4} value={condensed.text} />
    </label>
  );
  if (!condensed.text?.trim()) {
    return (
      <>
        {textarea}
        <p className="inline-warning" data-row-note="condensed-empty">
          The condensed prompt is empty, so this row cannot be saved for {target.name}. Type a prompt or condense again.
        </p>
      </>
    );
  }
  if (condensed.edited) {
    const chars = promptBudget(formatPromptBatch([row], { fitTo: target.id }).text);
    return (
      <>
        {textarea}
        <p className={chars.over ? "inline-warning" : "dataset-caption-mode-note"} data-row-note="condensed-size">
          {chars.length.toLocaleString("en-US")} of {chars.max.toLocaleString("en-US")} characters; tokens not measured since your edit.{" "}
          <button className="secondary-action" disabled={busy || measureBlocked} onClick={() => onMeasure(row)} type="button">
            Measure
          </button>
        </p>
      </>
    );
  }
  const size = condensedSizeText(condensed);
  const estimate = condensed.tokenCount !== "exact";
  return (
    <>
      {textarea}
      <p className={condensed.over ? "inline-warning" : "dataset-caption-mode-note"} data-row-note="condensed-size">
        {condensed.over ? "Over the limit: " : ""}
        {size ?? "Tokens not measured"}
        {estimate && condensed.estimateReason ? ` (${size ? "estimated: " : ""}${condensed.estimateReason})` : estimate && size ? " (estimated)" : ""}
        .{condensed.notice ? ` ${condensed.notice}` : ""}
      </p>
    </>
  );
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
    promptBatches = [],
    createPromptBatch,
    updatePromptBatch,
    sendPromptBatchToImage,
  } = useAppContext();
  const projectId = activeProject?.id;
  const datasetOrder = useMemo(() => (dataset?.items ?? []).map((item) => item.id), [dataset]);
  // The images the restore checks, without re-running the restore when the parent passes a new
  // dataset object for the same dataset (that would replace rows the user is editing).
  const datasetItemsRef = useRef(dataset?.items ?? []);
  datasetItemsRef.current = dataset?.items ?? [];

  const [modelId, setModelId] = useState(VISION_CAPTION_MODEL_ID);
  // The tier chosen for a describer with tiers (null: its default).
  const [tierChoice, setTierChoice] = useState(null);
  // SW-7163: the "Fit to" target's catalog id, "" for off. Restored from the newest run.
  const [fitTo, setFitTo] = useState("");
  // Which part of a multi-part condense is running ({ index, total }), or null.
  const [condenseProgress, setCondenseProgress] = useState(null);
  // The terminal status of the last monitored job, so multi-part condensing stops on a failure.
  const lastFinished = useRef(null);
  // Each model's default instruction and limit, by repo: `{ prompt, maxChars }`.
  const [defaults, setDefaults] = useState({});
  const [prompt, setPrompt] = useState("");
  // Whether the editor shows the selected model's untouched default, so a model switch swaps it
  // for the other model's default; an edited text stays as typed.
  const [followDefault, setFollowDefault] = useState(true);
  // A failed default load's message, by repo.
  const [promptErrors, setPromptErrors] = useState({});
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
  // True until the earlier runs' results are loaded into the table; runs stay blocked meanwhile.
  const [restoring, setRestoring] = useState(true);
  const monitoringRef = useRef(false);
  // How many of a job's stored items the table already holds, so polling reads only new ones.
  const itemsRead = useRef({ jobId: null, count: 0, provisional: new Set() });
  // When the dialog resumes an older running job, the images newer runs already described: that
  // job's later results for them are older work and must not replace the restored rows.
  const supersededItems = useRef({ jobId: null, ids: new Set() });
  const pollAbort = useRef(null);

  useEffect(() => () => pollAbort.current?.abort(), []);

  const model = DESCRIBE_MODELS.find((entry) => entry.id === modelId) ?? DESCRIBE_MODELS[0];
  const tier = describerTier(model, tierChoice);
  const modelDefault = defaults[model.repo] ?? null;
  const defaultPrompt = modelDefault ? modelDefault.prompt : null;

  // The describer and tier that wrote a row (its re-run uses them), or the selected ones when the
  // row records no known describer. A row with no recorded tier uses the model's default tier; a
  // recorded tier the model does not offer is kept as `unknownTier` (its re-run is blocked), never
  // replaced by another tier.
  const rowDescriber = (row) => {
    const entry = DESCRIBE_MODELS.find((candidate) => candidate.repo === row.model);
    if (!entry) return { entry: model, tier };
    const unknownTier = Boolean(entry.tiers && row.variant && !entry.tiers.some(([key]) => key === row.variant));
    return { entry, tier: unknownTier ? row.variant : describerTier(entry, row.variant), unknownTier };
  };

  // The models whose default instruction and limit are needed: the selected one, and each row's
  // describer (a re-run checks the prompt against that model's limit). Joined into a string so the
  // effect below depends on the set by value.
  const neededRepos = [...new Set([model.repo, ...rows.map((row) => rowDescriber(row).entry.repo)])].join("\u0000");
  // The repos whose default is loading or loaded, and the signal that aborts their requests: both
  // live until unmount or a token change, so one repo's load never cancels another's.
  const defaultsRequested = useRef(new Set());
  const defaultsAbort = useRef(null);
  useEffect(() => {
    const controller = new AbortController();
    defaultsAbort.current = controller;
    defaultsRequested.current = new Set();
    return () => controller.abort();
  }, [token]);

  // Load each needed default once (served by the API from the worker's own prompt asset and core's
  // limits, so the web app holds no second copy). A failed load is retried the next time the needed
  // set or the selected model changes.
  useEffect(() => {
    const { signal } = defaultsAbort.current;
    for (const repo of neededRepos.split("\u0000")) {
      if (defaultsRequested.current.has(repo)) continue;
      defaultsRequested.current.add(repo);
      trainingApi
        .defaultDescribePrompt(token, { signal }, repo)
        .then((body) => {
          const text = typeof body?.prompt === "string" ? body.prompt : "";
          const maxChars = Number(body?.maxChars) || 0;
          setPromptErrors(({ [repo]: _cleared, ...rest }) => rest);
          setDefaults((current) => ({ ...current, [repo]: { prompt: text, maxChars } }));
        })
        .catch((error) => {
          if (isAbortError(error) || signal.aborted) return;
          defaultsRequested.current.delete(repo);
          const message = errorMessage(error, "Could not load the default describing prompt.");
          setPromptErrors((current) => ({ ...current, [repo]: message }));
        });
    }
  }, [neededRepos, token]);
  const promptError = promptErrors[model.repo] ?? "";

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
  const modelAvailable = (id, modelTier = null) => describerInstalled(catalogEntry(id), modelTier);
  const tierLabel = tier ? model.tiers.find(([key]) => key === tier)[1] : "";
  const selectedEntry = catalogEntry(model.id);
  const selectedAvailable = modelAvailable(model.id, tier);
  // What the not-installed notice offers: the model, or for a tiered describer the selected tier
  // (named with its tier), whose download installs only that tier.
  // The tier's own size, as the Model Manager's tier panel reads it; none rather than the model-level
  // size, which describes the default tier.
  const tierVariant = tier ? selectedEntry?.variants?.find((variant) => variant?.variant === tier) : null;
  const tierBytes = tierVariant?.footprint?.diskSizeBytes ?? tierVariant?.downloadSizeBytes;
  const missingModels = tier
    ? selectedEntry && !selectedAvailable
      ? [
          {
            ...selectedEntry,
            name: `${selectedEntry.name ?? model.label} ${tierLabel}`,
            downloadSizeLabel: typeof tierBytes === "number" && tierBytes > 0 ? formatTierBytes(tierBytes) : null,
            downloadSizeEstimated: false,
          },
        ]
      : []
    : missingRequiredModels(models, [model.id]);
  const downloadJobs = useMemo(
    () => jobs.filter((entry) => entry.type === "model_download" && (!tier || entry.payload?.variant === tier)),
    [jobs, tier],
  );
  const downloadMissing =
    typeof createModelDownloadJob !== "function"
      ? undefined
      : tier
        ? () => createModelDownloadJob(selectedEntry, { variant: tier })
        : createModelDownloadJob;
  const fitTargets = useMemo(() => fitToTargets(models), [models]);
  const fitTarget = fitTargets.find((target) => target.id === fitTo) ?? null;
  // A remembered target the catalog no longer offers (a removed model, or one below the minimum)
  // is cleared once the catalog and the earlier runs are loaded, so it is never sent.
  useEffect(() => {
    if (restoring || !fitTo || !models.length || fitTarget) return;
    setFitTo("");
    if (dataset?.id) storeFitTo(dataset.id, "");
  }, [dataset?.id, fitTarget, fitTo, models.length, restoring]);

  const running = Boolean(job) && !terminalStatuses.has(job.status);
  const result = job?.result ?? null;
  const itemCount = result?.itemCount ?? (rerunItemId ? 1 : datasetOrder.length);
  const processedCount = result?.processedCount ?? 0;
  const progressPercent = Math.round(Math.max(0, Math.min(1, Number(job?.progress) || 0)) * 100);

  // Read the job's stored items from `from` on, page by page, merging them into the table. Returns
  // how many items have been read in all.
  const readItemPages = useCallback(async (jobId, from, onItems, signal) => {
    let offset = from;
    for (;;) {
      const page = await withRetries(() => jobsApi.resultItems(jobId, token, { offset }, { signal }), signal);
      const items = Array.isArray(page?.items) ? page.items : [];
      if (items.length) onItems(items, offset);
      offset += items.length;
      if (!items.length || page?.nextOffset == null) return offset;
    }
  }, [token]);

  // Merge items read from the job into the table and track which positions are still
  // provisional (an image stored before its condensing step, replaced in place once condensed).
  const mergeJobItems = useCallback(
    (jobId, items, start, model, variant = null) => {
      if (itemsRead.current.jobId === jobId) trackProvisional(itemsRead.current.provisional, items, start);
      const superseded = supersededItems.current.jobId === jobId ? supersededItems.current.ids : null;
      const fresh = superseded ? items.filter((item) => !superseded.has(item?.itemId)) : items;
      if (fresh.length) setRows((current) => mergeDescribeRows(current, fresh, datasetOrder, model, variant));
    },
    [datasetOrder],
  );

  const readItems = useCallback(
    (jobId, from, model, variant, signal) =>
      readItemPages(jobId, from, (items, start) => mergeJobItems(jobId, items, start, model, variant), signal),
    [mergeJobItems, readItemPages],
  );

  // Re-read only the positions that were provisional, one item each, until they are final.
  const rereadProvisional = useCallback(
    async (jobId, model, variant, signal) => {
      if (itemsRead.current.jobId !== jobId) return;
      for (const position of [...itemsRead.current.provisional]) {
        const page = await withRetries(
          () => jobsApi.resultItems(jobId, token, { offset: position, limit: 1 }, { signal }),
          signal,
        );
        const items = Array.isArray(page?.items) ? page.items.slice(0, 1) : [];
        if (items.length) mergeJobItems(jobId, items, position, model, variant);
      }
    },
    [mergeJobItems, token],
  );

  const pollJob = useCallback(async (jobId, signal) => {
    for (;;) {
      await abortableDelay(POLL_MS, signal);
      const snapshot = await withRetries(() => jobsApi.item(jobId, token, { signal }), signal);
      // The worker stores an item before the post that counts it, so `processedCount` items exist.
      const processed = Number(snapshot?.result?.processedCount) || 0;
      if (itemsRead.current.jobId === jobId && itemsRead.current.provisional.size) {
        await rereadProvisional(jobId, snapshot?.result?.model, snapshot?.result?.variant, signal);
      }
      if (itemsRead.current.jobId === jobId && processed > itemsRead.current.count) {
        itemsRead.current.count = await readItems(
          jobId,
          itemsRead.current.count,
          snapshot.result.model,
          snapshot.result.variant,
          signal,
        );
      }
      setJob(snapshot);
      if (terminalStatuses.has(snapshot?.status)) {
        return snapshot;
      }
    }
  }, [readItems, rereadProvisional, token]);

  function reportFinished(finished) {
    lastFinished.current = finished?.status ?? null;
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

  // Rebuild the table from this dataset's earlier describe jobs, and monitor one still running.
  useEffect(() => {
    if (!dataset?.id || !projectId) {
      setRestoring(false);
      return undefined;
    }
    const controller = new AbortController();
    const { signal } = controller;
    const datasetItems = datasetItemsRef.current;
    setRestoring(true);
    (async () => {
      try {
        // Read runs newest first, a page of the history at a time, so a newer result always
        // wins. Stop once every image has a current successful result, but never before the
        // active job is read (its count seeds polling).
        const images = new Map(datasetItems.map((image) => [image.id, image]));
        const missing = new Set(images.keys());
        const runs = [];
        const jobsNewestFirst = [];
        // The newest job still queued or running, found apart from the history scan, which may
        // have to page past many newer jobs to reach it.
        const unfinished = await withRetries(
          () => trainingApi.listBatchDescribeJobs(projectId, dataset.id, token, { unfinished: true, limit: 1 }, { signal }),
          signal,
        );
        const activeId = (Array.isArray(unfinished) ? unfinished[0]?.id : null) ?? null;
        let active = null;
        let activeCount = 0;
        let activeProvisional = new Set();
        let activeRead = !activeId;
        const newerDescribed = new Set();
        let offset = 0;
        let exhausted = false;
        const nextJob = async () => {
          if (offset >= jobsNewestFirst.length && !exhausted) {
            const page = await withRetries(
              () =>
                trainingApi.listBatchDescribeJobs(
                  projectId,
                  dataset.id,
                  token,
                  { offset: jobsNewestFirst.length, limit: RESTORE_JOBS_PAGE },
                  { signal },
                ),
              signal,
            );
            const listed = Array.isArray(page) ? page : [];
            jobsNewestFirst.push(...listed);
            if (listed.length < RESTORE_JOBS_PAGE) exhausted = true;
          }
          return offset < jobsNewestFirst.length ? jobsNewestFirst[offset++] : null;
        };
        for (let entry = await nextJob(); entry; entry = await nextJob()) {
          const items = [];
          const count = await readItemPages(entry.id, 0, (page) => items.push(...page), signal);
          const provisional = trackProvisional(new Set(), items, 0);
          if (entry.id === activeId) {
            active = entry;
            activeCount = count;
            activeProvisional = provisional;
            activeRead = true;
          }
          const sources = new Map(
            (Array.isArray(entry.payload?.items) ? entry.payload.items : []).map((source) => [source?.itemId, source]),
          );
          for (const item of items) {
            if (item?.itemId && !item.failed && resultMatchesImage(item, sources.get(item.itemId), images.get(item.itemId))) {
              missing.delete(item.itemId);
              // Described by a run newer than the active one, which therefore must not replace it.
              if (!activeRead) newerDescribed.add(item.itemId);
            }
          }
          runs.push({
            model: entry.result?.model ?? entry.payload?.model ?? null,
            variant: entry.result?.variant ?? entry.payload?.variant ?? null,
            items,
            sources,
          });
          if (activeRead && !missing.size) break;
        }
        if (signal.aborted) return;
        const restored = restoreDescribeRows(runs, datasetItems);
        setRows(restored);
        // The dialog keeps the Fit to choice of its newest run.
        const lastFitTo = storedFitTo(dataset.id) ?? jobsNewestFirst[0]?.payload?.fitTo;
        if (typeof lastFitTo === "string") setFitTo(lastFitTo);
        if (active) {
          itemsRead.current = { jobId: active.id, count: activeCount, provisional: activeProvisional };
          supersededItems.current = { jobId: active.id, ids: newerDescribed };
          setJob(active);
          monitor(active.id);
        } else if (restored.length) {
          const last = jobsNewestFirst[0];
          const stopped =
            last && last.status !== "completed"
              ? ` The last run stopped (${last.status}) after ${last.result?.processedCount ?? 0} of ${last.result?.itemCount ?? "?"} images.`
              : "";
          setRunMessage(`Restored ${restored.length} described ${restored.length === 1 ? "image" : "images"} from earlier runs.${stopped}`);
        }
      } catch (error) {
        if (!isAbortError(error) && !signal.aborted) {
          setRunError(errorMessage(error, "Could not load the results of earlier runs."));
        }
      } finally {
        if (!signal.aborted) setRestoring(false);
      }
    })();
    return () => controller.abort();
    // Restores once per dataset; the images are read through a ref (see datasetItemsRef).
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [dataset?.id, projectId, token]);

  // Start a describe job: every image (`itemIds` null), or exactly `itemIds`. `confirmFirst`, when
  // given, runs first and can decline. `describer` is the model and tier to run (a re-run passes the
  // row's own). Ignored while a start is already in flight or a job runs.
  // `condense` ({ fitTo, model, variant, rows, measure }) starts a condense-only job instead: the
  // rows' own describer, at the tier they were described with, condenses those already-described
  // rows (their fields as edited here) for the target, describing nothing; with `measure` it only
  // re-counts their edited condensed prompts.
  async function submit(itemIds = null, confirmFirst = null, describer = { entry: model, tier }, condense = null) {
    if (submittingRef.current || running || !dataset?.id || !projectId) return;
    submittingRef.current = true;
    setSubmitting(true);
    try {
      if (confirmFirst && !(await confirmFirst())) return;
      setRunError("");
      setRunMessage("");
      setMonitorError("");
      setSavedBatch(null);
      let body;
      if (condense) {
        body = {
          model: condense.model ?? VISION_CAPTION_MODEL_REPO,
          ...(condense.variant ? { variant: condense.variant } : {}),
          requestedGpu: "auto",
          fitTo: condense.fitTo,
          condenseRows: condense.rows,
          ...(condense.measure ? { measureOnly: true } : {}),
        };
      } else {
        body = { model: describer.entry.repo, requestedGpu: "auto" };
        if (describer.tier) body.variant = describer.tier;
        // The built-in default is sent as "no prompt" so the worker uses its own default instruction.
        if (prompt.trim() && prompt.trim() !== (defaultPrompt ?? "").trim()) body.prompt = prompt;
        if (itemIds) body.itemIds = itemIds;
        // A describer without a text step (JoyCaption) never condenses.
        // Only a target the catalog still offers: a remembered id of a removed model is never sent.
        if (fitTarget && describer.entry.condenses) body.fitTo = fitTarget.id;
      }
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
      if (!itemIds && !condense) setRows([]);
      itemsRead.current = { jobId: created.id, count: 0, provisional: new Set() };
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
      current.map((row) =>
        row.itemId === itemId ? { ...row, edited: true, fieldsEdited: true, fields: { ...row.fields, [key]: value } } : row,
      ),
    );
  }

  // An edited condensed prompt keeps the size measured before the edit, marked as such.
  function updateCondensed(itemId, value) {
    setSavedBatch(null);
    setRows((current) =>
      current.map((row) =>
        row.itemId === itemId && row.condensed ? { ...row, edited: true, condensed: { ...row.condensed, text: value, edited: true } } : row,
      ),
    );
  }

  // Run only the condensing step for `target` on `pending` rows (never a describe), as one job per
  // part (each part one describer, bounded in size and rows), one after another; stop at a part
  // that does not complete. `measure` re-counts edited condensed prompts instead.
  async function runCondenseParts(target, pending, { measure = false } = {}) {
    const parts = condenseChunks(pending, (row) => condenseRequestRow(row, { measure }));
    for (let index = 0; index < parts.length; index += 1) {
      setCondenseProgress(parts.length > 1 ? { index, total: parts.length } : null);
      lastFinished.current = null;
      await submit(null, null, undefined, {
        fitTo: target, model: parts[index].model, variant: parts[index].variant, rows: parts[index].rows, measure,
      });
      if (lastFinished.current !== "completed") break;
    }
    setCondenseProgress(null);
  }

  async function condenseRows(target = fitTo) {
    const pending = rowsToCondense(rows, target);
    if (!target || !pending.length) return;
    await runCondenseParts(target, pending);
  }

  // Re-count one edited condensed prompt with the target's tokenizer; nothing is generated.
  async function measureRow(row) {
    if (!fitTo) return;
    await runCondenseParts(fitTo, [row], { measure: true });
  }

  // Changing the target condenses the rows already described for it; clearing it needs no job
  // (the saved blocks go back to the seven fields). The choice is remembered for this dataset.
  function chooseFitTo(id) {
    setSavedBatch(null);
    setFitTo(id);
    if (dataset?.id) storeFitTo(dataset.id, id);
    if (id && !condenseBlocked) condenseRows(id);
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

  const activeFitTo = fitTarget ? fitTo : null;
  const payload = () => promptBatchPayload({ rows, name: batchName, dataset, fitTo: activeFitTo });
  // Counted on the blocks that would be saved for the active target.
  const usableCount = useMemo(
    () => rows.length - formatPromptBatch(rows, { fitTo: activeFitTo }).skipped.length,
    [rows, activeFitTo],
  );
  const pendingCondense = useMemo(() => rowsToCondense(rows, activeFitTo), [rows, activeFitTo]);
  // Rows that would save without a condensed prompt matching their fields: Save/Export wait.
  const unfitted = useMemo(() => rowsNotFitted(rows, activeFitTo), [rows, activeFitTo]);
  // Saved with a condensed prompt still over the limit after the retry: reported, never fitting.
  const overLimit = useMemo(
    () => (activeFitTo ? rows.filter((row) => { const c = condensedFor(row, activeFitTo); return c?.over && !c.edited; }).length : 0),
    [rows, activeFitTo],
  );
  // Saved with a condensed prompt edited since it was measured.
  const unmeasured = useMemo(
    () => (activeFitTo ? rows.filter((row) => condensedFor(row, activeFitTo)?.edited).length : 0),
    [rows, activeFitTo],
  );

  // Save the batch. A saved batch with the same name is replaced after a confirm, so saving again
  // (after describing more images) updates it instead of failing on the name.
  async function save() {
    if (saving || !usableCount || unfitted.length || !batchName.trim() || typeof createPromptBatch !== "function") return;
    setSaving(true);
    setSaveError("");
    try {
      const body = payload();
      const existing = typeof updatePromptBatch === "function" ? promptBatchNamed(promptBatches, body.name) : null;
      if (existing) {
        const replace = await appConfirm({
          title: `Replace “${existing.name}”?`,
          message: `A prompt batch named “${existing.name}” already exists. Replace its prompts with these ${body.prompts.length}, or keep it and change the name.`,
          confirmLabel: "Replace",
          cancelLabel: "Keep it",
          tone: "danger",
        });
        if (!replace) return;
        const replacement = { ...body, variables: keepVariableValues(body.variables, existing) };
        setSavedBatch((await updatePromptBatch(existing.id, replacement, "global")) ?? null);
      } else {
        setSavedBatch((await createPromptBatch(body)) ?? null);
      }
    } catch (error) {
      setSaveError(errorMessage(error, "Could not save the prompt batch."));
    } finally {
      setSaving(false);
    }
  }

  // Closing keeps described images (the next open restores them) but not edits made here, so
  // unsaved edits are confirmed first.
  async function requestClose() {
    if (rows.some((row) => row.edited) && !savedBatch) {
      const close = await appConfirm({
        title: "Close without saving your edits?",
        message: "Described images come back when you reopen this dialog, but fields you edited are kept only in a saved prompt batch.",
        confirmLabel: "Close",
        cancelLabel: "Keep editing",
        tone: "danger",
      });
      if (!close) return;
    }
    onClose();
  }

  async function exportJson() {
    if (!savedBatch && unfitted.length) return;
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
    busy || restoring || dirty || !selectedAvailable || defaultPrompt === null || !datasetOrder.length || !promptValidation.ready;
  // Condensing runs on each row's own describer, which must be installed.
  const condenseBlocked =
    busy || restoring || dirty || !DESCRIBE_MODELS.some((entry) => entry.condenses && modelAvailable(entry.id));
  // The block each row will be saved as (its condensed prompt when Fit to applies to it).
  const fitToId = activeFitTo;

  // Why a row's re-run, which runs the row's own describer and tier, cannot start: that model or
  // tier is not installed, or an edited prompt is over that model's limit. "" when it can.
  function rerunIssue(row) {
    const { entry, tier: rowTier, unknownTier } = rowDescriber(row);
    if (unknownTier) return `${entry.label} has no tier “${rowTier}”, the tier that described this image.`;
    const name = rowTier ? `${entry.label} ${entry.tiers.find(([key]) => key === rowTier)[1]}` : entry.label;
    if (!modelAvailable(entry.id, rowTier)) return `${name} is not installed.`;
    const rowDefault = defaults[entry.repo];
    if (!rowDefault) return promptErrors[entry.repo] || "Loading the describing prompt…";
    const sendsPrompt = prompt.trim() && prompt.trim() !== (defaultPrompt ?? "").trim();
    return sendsPrompt ? (describePromptIssues(prompt, rowDefault.maxChars, entry.label)[0]?.message ?? "") : "";
  }

  return (
    <Modal className="dataset-caption-modal dataset-prompt-batch-modal" labelledBy="dataset-prompt-batch-title" onClose={requestClose}>
      <header className="dataset-caption-head">
        <div>
          <p className="eyebrow">{dataset?.name ?? "Dataset"}</p>
          <h2 id="dataset-prompt-batch-title">Create Prompt Batch from Dataset</h2>
        </div>
        <button className="modal-close" onClick={requestClose} type="button">
          Close
        </button>
      </header>

      <div className="dataset-caption-body">
        <fieldset className="dataset-prompt-batch-models">
          <legend>Describing model</legend>
          {DESCRIBE_MODELS.map((entry) => {
            const available = modelAvailable(entry.id, entry.id === model.id ? tier : null);
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
        {model.tiers ? (
          <fieldset className="dataset-prompt-batch-models">
            <legend>Tier</legend>
            {model.tiers.map(([key, label]) => (
              <label className="training-toggle-line" key={key}>
                <input
                  checked={tier === key}
                  disabled={busy}
                  name="dataset-prompt-batch-tier"
                  onChange={() => setTierChoice(key)}
                  type="radio"
                  value={key}
                />
                <span>
                  {label}
                  {modelAvailable(model.id, key) ? null : (
                    <small className="dataset-prompt-batch-unavailable"> — not installed</small>
                  )}
                </span>
              </label>
            ))}
          </fieldset>
        ) : null}
        {!selectedAvailable ? (
          missingModels.length ? (
            <RequiredModelsNotice
              // Every tier's entry shares the model id, so each tier gets its own notice state.
              key={tier ?? model.id}
              downloadJobs={downloadJobs}
              feature="Describing the dataset"
              models={missingModels}
              onCancelJob={typeof jobAction === "function" ? (download) => jobAction(download, "cancel") : undefined}
              onDownload={downloadMissing}
              onOpenModels={typeof setActiveView === "function" ? () => setActiveView("Models") : undefined}
            />
          ) : (
            <p className="inline-warning" role="status">
              {model.label}
              {tier ? ` ${tierLabel}` : ""} is unavailable on this install.
            </p>
          )
        ) : null}

        <label>
          Fit to
          <select
            aria-describedby="dataset-prompt-batch-fit-to-hint"
            disabled={busy || restoring}
            onChange={(event) => chooseFitTo(event.target.value)}
            value={fitTarget ? fitTo : ""}
          >
            <option value="">Off — save the seven fields</option>
            {fitTargets.map((target) => (
              <option key={target.id} value={target.id}>
                {target.name} — {target.limit.toLocaleString("en-US")} tokens
              </option>
            ))}
          </select>
          <span className="dataset-caption-field-hint" id="dataset-prompt-batch-fit-to-hint">
            Rewrites each described image into one prompt that fits this model's prompt limit. The seven fields are kept.
          </span>
        </label>
        {fitTarget && !model.condenses ? (
          <p className="dataset-caption-mode-note" role="status">
            JoyCaption rows are not condensed: Fit to needs the Vision Captioner's text step.
          </p>
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
            The tool adds the JSON output format after this text, so edits never change the seven fields.
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
              {(() => {
                const [doing, done] = job.payload?.measureOnly
                  ? ["Measuring", "Measured"]
                  : job.payload?.condenseOnly
                    ? ["Condensing", "Condensed"]
                    : ["Describing", "Described"];
                return running ? doing : job.status === "completed" ? done : `Job ${job.status}`;
              })()}
              : {processedCount} of {itemCount} {itemCount === 1 ? "image" : "images"}
              {condenseProgress ? ` (part ${condenseProgress.index + 1} of ${condenseProgress.total})` : ""}
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
        {restoring ? <p className="dataset-caption-mode-note" role="status">Loading results of earlier runs…</p> : null}
        {runError ? <p className="inline-warning" role="alert">{runError}</p> : null}
        {runMessage ? <p className="dataset-caption-mode-note" role="status">{runMessage}</p> : null}

        {rows.length ? (
          <div className="dataset-prompt-batch-table" role="table" aria-label="Described images">
            {rows.map((row) => {
              const ratio = snapRatio(row.width, row.height);
              const rerunBlockedBy = rerunIssue(row);
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
                      disabled={busy || restoring || dirty || defaultPrompt === null || Boolean(rerunBlockedBy)}
                      onClick={() => submit([row.itemId], null, rowDescriber(row))}
                      title={rerunBlockedBy || undefined}
                      type="button"
                    >
                      {rerunItemId === row.itemId ? "Re-running…" : "Re-run"}
                    </button>
                  </div>
                  <div className="dataset-prompt-batch-fields" role="cell">
                    {row.failed ? (
                      <p className="inline-warning" role="alert">Flagged: {row.error || "The describer could not describe this image."}</p>
                    ) : null}
                    {!row.failed && row.notice ? (
                      <p className="dataset-caption-mode-note" data-row-note="cut-off" role="status">{row.notice}</p>
                    ) : null}
                    {fitTarget ? (
                      <CondensedPrompt
                        busy={busy}
                        measureBlocked={condenseBlocked}
                        onChange={updateCondensed}
                        onMeasure={measureRow}
                        row={row}
                        target={fitTarget}
                      />
                    ) : null}
                    {(() => {
                      // Image generation refuses a prompt over PROMPT_MAX_CHARS; warn, never block.
                      // Measured on the block that will be saved (the condensed prompt when used).
                      const budget = promptBudget(formatPromptBatch([row], { fitTo: fitToId }).text);
                      return budget.over ? (
                        <p className="inline-warning" data-row-note="too-long">
                          This prompt is {budget.length.toLocaleString()} characters; image generation accepts at most{" "}
                          {budget.max.toLocaleString()}, so shorten it before generating.
                        </p>
                      ) : null;
                    })()}
                    {DESCRIBE_FIELDS.map(([key, label]) => (
                      <label key={key}>
                        {label}
                        <textarea
                          disabled={busy}
                          onChange={(event) => updateField(row.itemId, key, event.target.value)}
                          rows={key === "description" || key === "pose" ? 3 : 2}
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

      </div>

      {/* Outside the scrolling table, so the name and the save outcome are always in view. */}
      <div className="dataset-prompt-batch-bottom">
        {rows.length ? (
          <div className="dataset-prompt-batch-save">
            <label>
              Prompt batch name
              <input onChange={(event) => setBatchName(event.target.value)} value={batchName} />
            </label>
            {saveError ? <p className="inline-warning" role="alert">{saveError}</p> : null}
            {unfitted.length ? (
              <p className="inline-warning" data-save-note="not-fitted" role="status">
                {unfitted.length} {unfitted.length === 1 ? "row has" : "rows have"} no usable condensed prompt for {fitTarget.name}{" "}
                (not condensed yet, condensing failed, fields changed since, or the prompt is empty). Condense{" "}
                {unfitted.length === 1 ? "it" : "them"}, or turn Fit to off to save the seven fields.
              </p>
            ) : null}
            {overLimit ? (
              <p className="inline-warning" data-save-note="over" role="status">
                {overLimit} condensed {overLimit === 1 ? "prompt is" : "prompts are"} still over the limit for {fitTarget.name}; the image
                model will refuse {overLimit === 1 ? "it" : "them"} until shortened.
              </p>
            ) : null}
            {unmeasured ? (
              <p className="dataset-caption-mode-note" data-save-note="unmeasured" role="status">
                {unmeasured} condensed {unmeasured === 1 ? "prompt was" : "prompts were"} edited and not measured since.
              </p>
            ) : null}
            {savedBatch ? (
              <p className="dataset-caption-mode-note" role="status">
                Saved “{savedBatch.name}” with {savedBatch.prompts?.length ?? 0} prompts.
              </p>
            ) : null}
          </div>
        ) : null}

        <footer className="dataset-caption-footer">
          <span className="dataset-prompt-batch-footer-actions">
            {running ? (
              <button className="secondary-action" onClick={cancel} type="button">
                Cancel
              </button>
            ) : null}
            <button
              className="secondary-action"
              disabled={!usableCount || running || (!savedBatch && unfitted.length > 0)}
              onClick={exportJson}
              type="button"
            >
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
                disabled={!usableCount || running || saving || unfitted.length > 0 || !batchName.trim()}
                onClick={save}
                type="button"
              >
                {saving ? "Saving…" : "Save prompt batch"}
              </button>
            )}
            {pendingCondense.length ? (
              <button className="secondary-action" disabled={condenseBlocked} onClick={() => condenseRows()} type="button">
                Condense for {fitTarget.name} ({pendingCondense.length})
              </button>
            ) : null}
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
      </div>
    </Modal>
  );
}
