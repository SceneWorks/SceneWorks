import React, { useCallback, useEffect, useMemo, useState } from "react";
import { Icon } from "../components/Icons.jsx";
import { WorkPanel } from "../components/WorkPanel.jsx";
import { AdvancedSection } from "../components/AdvancedSection.jsx";
import { PromptGuideModal } from "../components/PromptGuideModal.jsx";
import { useAppContext } from "../context/AppContext.js";
import {
  durableTextFits,
  loadStudioSettings,
  sessionOnlyFields,
  useStudioSettingsWriter,
} from "../hooks/useStudioSettings.js";
import { writeLicenseAck } from "../licenseAcknowledgment.js";
import { terminalStatuses } from "../jobTypes.js";
import {
  MAX_DECODE_TILE_FRAMES,
  MAX_TAKES,
  MIN_ATTENTION_CHUNK_ELEMENTS,
  SAMPLING_FIELDS,
  SAMPLING_TOKEN_DEFAULTS,
  YUE2_MODEL_ID,
  buildYue2JobRequest,
  commercialAlternatives,
  installJobStatusLabel,
  isCoverComponentDownload,
  composeKind,
  restoreYue2Settings,
  stripYue2ExportHeader,
  yue2ModelIdentity,
  yue2ModelInstalled,
  yue2PlanRequest,
  yue2ProjectRuns,
  yue2FieldDisabledReason,
  yue2RequestProblems,
  yue2RunView,
  yue2TierRows,
} from "../yue2Lab.js";
import {
  deleteLicenseAcknowledgment,
  getLicenseAcknowledgment,
  getYue2ScoreVersion,
  inspectYue2Score,
  listYue2Comparisons,
  listYue2ScoreVersions,
  putLicenseAcknowledgment,
  submitYue2Jobs,
} from "../api/yue2.js";
import {
  ErrorNotice,
  PolicyChips,
  RegenerationNotice,
  ScorePreview,
  Yue2CompareWorkbench,
  Yue2RunCard,
  Yue2ScoreWorkbench,
} from "./audioYue2Parts.jsx";
import { Yue2RecordingCover } from "./audioYue2Recording.jsx";

// YuE2 Song Lab (sc-23000, epic 22988) — the Audio Studio's EXPERIMENTAL surface for YuE2.
//
// Why a separate surface rather than a fifth AUDIO_MODES tab: YuE2 serves none of the four standard
// modes (`audioModelServesMode` is false for it by design, sc-22998) and is submitted through its own
// route (`POST /projects/:id/yue2/jobs` — the generic audio route refuses it). Its request is a
// style + lyrics + symbolic score plan with job kinds, score versions and comparisons that none of
// the standard modes builds. Keeping it out of AUDIO_MODES leaves every standard mode, its picker,
// its defaults and YuE1 (Music) exactly as they are; the lab is entered only by an explicit click on
// the separately-marked "Song Lab · Experimental" tab, and it opens behind an experimental opt-in
// plus the server-side licence acknowledgment.
//
// Persistence: the lab's settings live in their own studio snapshot (`yue2lab`), which the
// studio-settings seam mirrors to the SERVER's ui-preferences (desktop localStorage does not
// survive a relaunch). The acknowledgment is the server's own record; runs are server jobs; score
// versions and comparisons are server records. Nothing the lab needs after a relaunch lives only
// in the browser.

const LAB_TABS = [
  { id: "compose", label: "Compose" },
  { id: "cover", label: "Cover" },
  { id: "scores", label: "Scores" },
  { id: "compare", label: "Compare" },
];

const TIER_LABELS = { bf16: "BF16 (original)", q8: "Q8 (derived here)", q4: "Q4 (derived here)" };
const PRESET_LIMIT = 12;
// Preset snapshots carry CONTROLS only — never free text (lyrics, style, scores, notes, drafts).
// Twelve presets each holding a full lyric sheet would blow the durable snapshot budget, and a
// preset is a reusable setup, not a song.
const PRESET_EXCLUDED = new Set([
  "presets",
  // Run-specific: the saved plan a restore renders, the version a cover follows, and the plan
  // source that points at them belong to one song, not to a reusable setup.
  "planSource",
  "restorePlanJobId",
  "coverVersionId",
  "optIn",
  "tab",
  "lyrics",
  "style",
  "suppliedScore",
  "coverScore",
  "coverTranslatedFrom",
  "transcribeAssetId",
  "transcriptionId",
  "selectedVersionId",
  "compareA",
  "compareB",
  "compareRenderA",
  "compareRenderB",
  "compareNotes",
  "advancedOpen",
  "editDraft",
  "importDraft",
]);

// A text or draft over the durable budget: it survives tab changes but not a relaunch.
function SessionOnlyText({ testId }) {
  return (
    <p className="yue2-muted" data-testid={testId}>
      Kept for this session only — it is too large to restore after a relaunch.
    </p>
  );
}

function Segmented({ label, value, options, onChange, disabledValues = [] }) {
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

function SamplingFields({ label, value, defaults, onChange, disabled = false, reason = null }) {
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

function TriStateSelect({ label, value, onChange, reason = null }) {
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

// ---- gate -----------------------------------------------------------------------------------

function Yue2Gate({ model, models, identity, ack, onAccept, busy, error, optIn, onRetry }) {
  const [understood, setUnderstood] = useState(Boolean(optIn));
  const [accepted, setAccepted] = useState(Boolean(ack.acknowledged));
  const alternatives = commercialAlternatives(model, models);
  const canEnable = understood && accepted && !busy && ack.status === "ready";
  return (
    <div className="yue2-gate" data-testid="yue2-gate">
      <div className="yue2-gate__head">
        <strong>Before you use {identity.name}</strong>
        <PolicyChips model={model} />
      </div>
      <p>
        YuE2 is a separate, <strong>experimental</strong> song model — not a newer YuE1 and never a substitute for
        it. Its weights are licensed for <strong>noncommercial use only</strong>. Nothing about the standard Audio
        Studio modes changes when you enable it.
      </p>
      {model.licenseNotice ? <p className="model-license-terms">{model.licenseNotice}</p> : null}
      {identity.licenseUrl ? (
        <a href={identity.licenseUrl} rel="noreferrer" target="_blank">
          Read the full licence
        </a>
      ) : null}
      {model.commercialUse && model.commercialUse.eligible === false ? (
        <div className="yue2-notice yue2-notice--warn" data-testid="yue2-commercial-note">
          <Icon.Warning size={15} />
          <span>
            Not for commercial use: {model.commercialUse.reason}
            {/[.!?]$/.test(String(model.commercialUse.reason ?? "").trim()) ? "" : "."}
            {alternatives.length ? ` For commercial work use ${alternatives.map((alt) => alt.name).join(", ")}.` : ""}
            {model.commercialUse.alternativeNote ? ` ${model.commercialUse.alternativeNote}` : ""}
          </span>
        </div>
      ) : null}
      {ack.status === "error" ? (
        <ErrorNotice error={ack.error} testId="yue2-ack-error">
          <button className="secondary-action" onClick={onRetry} type="button">
            Retry
          </button>
        </ErrorNotice>
      ) : null}
      <label className="model-license-ack">
        <input checked={understood} onChange={(event) => setUnderstood(event.target.checked)} type="checkbox" />I
        understand YuE2 is experimental and separate from YuE1
      </label>
      <label className="model-license-ack">
        <input checked={accepted} onChange={(event) => setAccepted(event.target.checked)} type="checkbox" />I accept
        the licence terms above — noncommercial use only
      </label>
      <button className="prompt-cta" disabled={!canEnable} onClick={onAccept} type="button">
        Enable the experimental Song Lab
      </button>
      <ErrorNotice error={error} testId="yue2-gate-error" />
    </div>
  );
}

// ---- install --------------------------------------------------------------------------------

function Yue2Install({ model, downloads, onInstall, decoder }) {
  const rows = yue2TierRows(model);
  return (
    <div className="yue2-install" data-testid="yue2-install">
      <span className="eyebrow">Install YuE2</span>
      <p className="yue2-muted">
        BF16 is the released checkpoint. Q8 and Q4 are derived on this machine from it (the BF16 original is fetched
        first); they are never downloaded pre-quantized.
      </p>
      <ul className="yue2-tier-list">
        {rows.map((row) => (
          <li className="yue2-tier" key={row.tier}>
            <strong>{TIER_LABELS[row.tier] ?? row.tier}</strong>
            <span className={row.installed ? "status-badge installed" : "status-badge"}>{row.state}</span>
            {row.installable ? (
              <button
                className="secondary-action"
                disabled={downloads.some((job) => [job.payload?.variant, job.payload?.localDerivation?.variant].includes(row.tier))}
                onClick={() => onInstall(row.tier, decoder === "legacy" ? { decoder: "legacy" } : undefined)}
                type="button"
              >
                {row.derived ? `Derive ${row.tier} here` : `Download ${row.tier}`}
              </button>
            ) : null}
          </li>
        ))}
      </ul>
      {downloads.map((job) => (
        <p className="yue2-muted" key={job.id}>
          {job.payload?.localDerivation?.variant ?? job.payload?.variant ?? "install"}: {installJobStatusLabel(job)}
          {job.message ? ` — ${job.message}` : ""}
        </p>
      ))}
    </div>
  );
}

// ---- lab ------------------------------------------------------------------------------------

export function Yue2SongLab({ header }) {
  const {
    token,
    activeProject,
    models = [],
    jobs = [],
    assets = [],
    jobAction,
    importAsset,
    createModelDownloadJob,
    requestedGpu,
    visibleWorkers = [],
    preferencesHydrated,
  } = useAppContext();
  const projectId = activeProject?.id ?? null;
  const model = useMemo(() => (models ?? []).find((entry) => entry?.id === YUE2_MODEL_ID) ?? null, [models]);
  const identity = useMemo(() => yue2ModelIdentity(model), [model]);

  const saved = useMemo(() => restoreYue2Settings(loadStudioSettings("yue2lab", projectId)), [projectId]);
  const [settings, setSettings] = useState(saved);
  const update = useCallback((patch) => setSettings((current) => ({ ...current, ...patch })), []);
  useStudioSettingsWriter("yue2lab", projectId, settings, Boolean(preferencesHydrated) && Boolean(model));
  // Texts and drafts the durable copy leaves out (over their cap, or dropped by the snapshot
  // budget) survive only this session; each surface says so beside the text.
  const sessionOnly = useMemo(() => sessionOnlyFields("yue2lab", settings), [settings]);

  // Server-side licence acknowledgment (sc-22999): the lab reads the server's record, never a
  // browser flag, so the acceptance survives a relaunch and lapses when the terms change.
  const [ack, setAck] = useState({ status: "loading", acknowledged: false, error: null });
  const [ackBusy, setAckBusy] = useState(false);
  const [gateError, setGateError] = useState(null);
  // Keyed on the model id and its licence TERMS, not the catalog object: a catalog refresh hands
  // back a new object for the same entry, and re-reading on identity would reset the lab to
  // "checking" on every refresh. Changed terms do re-read (the server lapses the old acceptance).
  const modelId = model?.id ?? null;
  const termsKey = model ? `${model.licenseNotice ?? ""}|${model.licenseUrl ?? ""}` : "";
  const loadAck = useCallback(() => {
    if (!modelId) return undefined;
    let live = true;
    setAck((current) => ({ ...current, status: "loading", error: null }));
    getLicenseAcknowledgment(modelId, token)
      .then((view) => {
        if (!live) return;
        setAck({ status: "ready", acknowledged: view?.acknowledged === true, error: null, view });
        // Mirror the server's record into the download choke point's cache so an install from
        // here (or the Models card) is not refused for a stale browser flag.
        // …and clear it when the server says the acceptance lapsed or was withdrawn.
        writeLicenseAck(modelId, view?.acknowledged === true);
      })
      .catch((error) => {
        if (live) setAck({ status: "error", acknowledged: false, error });
      });
    return () => {
      live = false;
    };
    // termsKey re-reads the acknowledgment when the licence terms change.
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [modelId, termsKey, token]);
  useEffect(() => loadAck(), [loadAck]);

  const enabled = settings.optIn && ack.status === "ready" && ack.acknowledged;

  async function accept() {
    setAckBusy(true);
    setGateError(null);
    try {
      if (!ack.acknowledged) {
        const view = await putLicenseAcknowledgment(model.id, token);
        setAck({ status: "ready", acknowledged: view?.acknowledged === true, error: null, view });
      }
      writeLicenseAck(model.id, true);
      update({ optIn: true });
    } catch (error) {
      setGateError(error);
    } finally {
      setAckBusy(false);
    }
  }

  async function withdraw() {
    setGateError(null);
    try {
      await deleteLicenseAcknowledgment(model.id, token);
      writeLicenseAck(model.id, false);
      setAck({ status: "ready", acknowledged: false, error: null });
      update({ optIn: false });
    } catch (error) {
      setGateError(error);
    }
  }

  // Score versions and comparisons (server records).
  const [versions, setVersions] = useState([]);
  const [versionsError, setVersionsError] = useState(null);
  const [regenerationNotice, setRegenerationNotice] = useState("");
  const [comparisons, setComparisons] = useState([]);
  const [comparisonsError, setComparisonsError] = useState(null);
  const reloadVersions = useCallback(async () => {
    if (!projectId) return;
    try {
      const listing = await listYue2ScoreVersions(projectId, token);
      setVersions([...(listing?.items ?? [])].sort((a, b) => Date.parse(b.createdAt ?? 0) - Date.parse(a.createdAt ?? 0)));
      setRegenerationNotice(listing?.renderNotice ?? "");
      setVersionsError(
        Array.isArray(listing?.unreadable) && listing.unreadable.length
          ? `Some score versions could not be read: ${listing.unreadable.join(", ")}`
          : null,
      );
    } catch (error) {
      setVersionsError(error);
    }
  }, [projectId, token]);
  const reloadComparisons = useCallback(async () => {
    if (!projectId) return;
    try {
      const listing = await listYue2Comparisons(projectId, token);
      setComparisons([...(listing?.items ?? [])].sort((a, b) => Date.parse(b.createdAt ?? 0) - Date.parse(a.createdAt ?? 0)));
      setComparisonsError(
        Array.isArray(listing?.unreadable) && listing.unreadable.length
          ? `Some comparisons could not be read: ${listing.unreadable.join(", ")}`
          : null,
      );
    } catch (error) {
      setComparisonsError(error);
    }
  }, [projectId, token]);
  useEffect(() => {
    if (enabled) {
      reloadVersions();
      reloadComparisons();
    }
  }, [enabled, reloadVersions, reloadComparisons]);

  // Runs: the project's YuE2 jobs from the live job list, plus any this lab just submitted that
  // the list has not caught up with (the server's copy wins once it arrives).
  const [submitted, setSubmitted] = useState([]);
  const runs = useMemo(() => {
    const live = yue2ProjectRuns(jobs, projectId);
    const known = new Set(live.map((job) => job.id));
    return [...yue2ProjectRuns(submitted.filter((job) => !known.has(job.id)), projectId), ...live];
  }, [jobs, submitted, projectId]);
  // A finished plan becomes a score version server-side; refresh the history when one lands.
  const finishedKey = runs.filter((job) => job.status === "completed").map((job) => job.id).join(",");
  useEffect(() => {
    if (enabled && finishedKey) reloadVersions();
  }, [enabled, finishedKey, reloadVersions]);

  const [submitError, setSubmitError] = useState(null);
  const [submitting, setSubmitting] = useState(false);
  const [guideOpen, setGuideOpen] = useState(false);

  // The version a cover follows: a transcribed one must be covered in its own mode (sc-23002).
  const coverVersion = versions.find((version) => version.id === settings.coverVersionId) ?? null;
  const [reviewedTranscriptionVersionId, setReviewedTranscriptionVersionId] = useState("");
  const transcriptionReviewProblem =
    settings.coverSource === "version" && coverVersion?.transcription &&
    reviewedTranscriptionVersionId !== settings.coverVersionId
      ? "Review the transcription and score, then confirm your review before covering it."
      : null;
  const selectedGpuCapabilities = visibleWorkers.find((worker) => worker.gpuId === requestedGpu)?.capabilities ?? [];
  const problemContext = { hasProject: Boolean(projectId), coverVersion, requestedGpu, selectedGpuCapabilities };

  // The server says the acceptance lapsed (the terms changed, or it was withdrawn): re-gate.
  function licenseLapsed() {
    writeLicenseAck(model.id, false);
    setAck({ status: "ready", acknowledged: false, error: null });
  }

  async function submit(kind, target = {}) {
    if (submitting) return;
    const problems = yue2RequestProblems(kind, settings, target, problemContext);
    if (kind === "cover" && transcriptionReviewProblem) problems.push(transcriptionReviewProblem);
    if (problems.length) {
      setSubmitError(problems.join(" "));
      return;
    }
    setSubmitting(true);
    setSubmitError(null);
    try {
      const body = buildYue2JobRequest(kind, settings, target, requestedGpu);
      const response = await submitYue2Jobs(projectId, body, token);
      const created = Array.isArray(response?.jobs) ? response.jobs : [];
      setSubmitted((current) => [...created, ...current]);
    } catch (error) {
      setSubmitError(error);
      if (error?.code === "license_acknowledgment_required") {
        licenseLapsed();
      }
    } finally {
      setSubmitting(false);
    }
  }

  // Score inspection (supplied ABC / pasted cover score / chosen cover version).
  const [preview, setPreview] = useState({ key: "", inspection: null, error: null });
  async function inspect(key, abc) {
    setPreview({ key, inspection: null, error: null });
    try {
      // A lab-exported score opens with the `% SceneWorks YuE2 export:` licence header, which the
      // native dialect (it requires `X:1` first) refuses — strip it, as a submission does.
      const inspection = await inspectYue2Score(stripYue2ExportHeader(abc), token);
      setPreview({ key, inspection, error: null });
    } catch (error) {
      setPreview({ key, inspection: null, error });
    }
  }
  async function inspectCoverVersion(versionId) {
    setPreview({ key: "coverVersion", inspection: null, error: null });
    try {
      const detail = await getYue2ScoreVersion(projectId, versionId, token);
      const inspection = await inspectYue2Score(detail?.version?.score?.abc ?? "", token);
      setPreview({ key: "coverVersion", inspection, error: null });
    } catch (error) {
      setPreview({ key: "coverVersion", inspection: null, error });
    }
  }

  const fetchVersionAbc = useCallback(
    async (versionId) => (await getYue2ScoreVersion(projectId, versionId, token))?.version?.score?.abc ?? null,
    [projectId, token],
  );

  // Presets: named control snapshots that record the model, version and licence they were made
  // with, stored with the lab's settings (server-side).
  const [presetName, setPresetName] = useState("");
  function savePreset() {
    const name = presetName.trim();
    if (!name) return;
    const snapshot = Object.fromEntries(Object.entries(settings).filter(([key]) => !PRESET_EXCLUDED.has(key)));
    const preset = {
      id: `yue2preset_${Date.now().toString(36)}`,
      name,
      modelId: identity.id,
      version: identity.version,
      license: identity.license,
      experimental: identity.experimental,
      nonCommercial: identity.nonCommercial,
      settings: snapshot,
    };
    update({ presets: [preset, ...settings.presets.filter((item) => item.name !== name)].slice(0, PRESET_LIMIT) });
    setPresetName("");
  }
  function applyPreset(preset) {
    // A preset saved by an earlier build may still carry excluded keys; they never apply.
    const controls = Object.fromEntries(Object.entries(preset.settings).filter(([key]) => !PRESET_EXCLUDED.has(key)));
    // An old preset cannot silently inherit the current run's compute choice.
    if (!("computePolicy" in controls)) controls.computePolicy = controls.precision === "fp32" ? "fp32" : "";
    setSettings((current) => restoreYue2Settings({ ...current, ...controls }));
  }

  if (!model) {
    return (
      <WorkPanel className="studio-work-panel yue2-lab">
        {header}
        <p className="yue2-muted" data-testid="yue2-missing">
          YuE2 is not in this model catalog.
        </p>
      </WorkPanel>
    );
  }

  const installed = yue2ModelInstalled(model);
  const tierRows = yue2TierRows(model);
  const installedTiers = tierRows.filter((row) => row.installed).map((row) => row.tier);
  // YuE2's own installs; the cover closure's downloads are shown by the recording flow.
  const downloads = (jobs ?? []).filter(
    (job) =>
      job.type === "model_download" &&
      job.payload?.modelId === model.id &&
      !terminalStatuses.has(job.status) &&
      !isCoverComponentDownload(job, model),
  );
  const kind = composeKind(settings);
  const hasProject = problemContext;
  const composeProblems = yue2RequestProblems(kind, settings, {}, hasProject);
  const coverProblems = yue2RequestProblems("cover", settings, {}, hasProject);
  if (transcriptionReviewProblem) coverProblems.push(transcriptionReviewProblem);
  // The job kind the visible surface submits: its controls are the only ones that reach a request,
  // so a control the kind does not read is disabled with core's reason rather than silently dropped.
  const activeKind =
    settings.tab === "compose" ? kind : settings.tab === "cover" ? "cover" : settings.tab === "scores" ? "renderVersion" : null;
  const why = (field) => yue2FieldDisabledReason(field, activeKind);
  const renderProblems = yue2RequestProblems("renderVersion", settings, { versionId: "selected" }, hasProject);
  // A score text over its durable cap survives only this session; say so and offer the durable home.
  const sessionOnlyNote = (text) =>
    text && !durableTextFits("yue2lab", "suppliedScore", text) ? (
      <p className="yue2-muted" data-testid="yue2-session-only-note">
        Kept for this session only — it is too large to restore after a relaunch.{" "}
        <button
          className="audio-link"
          onClick={() => update({ tab: "scores", importDraft: { ...settings.importDraft, open: true, abc: text } })}
          type="button"
        >
          Import it as a score version
        </button>{" "}
        to keep it.
      </p>
    ) : null;
  const restorable = runs.map(yue2RunView).filter((view) => view.restorable);
  // A restored plan renders its own words: the Compose fields show them read-only and the request
  // carries them verbatim (the server refuses any other style or lyrics for a saved plan).
  const restoring = settings.tab === "compose" && kind === "fromPlan";
  const planWords = restoring
    ? yue2PlanRequest(restorable.find((view) => view.id === settings.restorePlanJobId))
    : null;
  const tierProblem =
    settings.tier && !installedTiers.includes(settings.tier) ? `The ${settings.tier} tier is not installed yet.` : "";
  const readyToRun = installed && !tierProblem;
  const composeLabel = kind === "plan" ? "Plan the score" : kind === "fromPlan" ? "Render from the saved plan" : "Generate song";

  return (
    <div className="yue2-lab-shell">
      <WorkPanel className="studio-work-panel yue2-lab" data-testid="yue2-song-lab">
        {header}
        <div className="yue2-lab__banner">
          <span className="yue2-lab__title">
            <Icon.Audio size={15} /> {identity.name}
          </span>
          <PolicyChips model={model} />
          <div className="prompt-hero-links">
            <button className="hero-link" onClick={() => setGuideOpen(true)} type="button">
              <Icon.Book size={14} /> Prompt guide
            </button>
            {enabled ? (
              <button className="hero-link" onClick={withdraw} type="button">
                Withdraw licence acceptance
              </button>
            ) : null}
          </div>
        </div>

        {!enabled && ack.status === "loading" ? (
          <p className="yue2-muted" data-testid="yue2-ack-loading">
            Checking your licence acceptance…
          </p>
        ) : !enabled ? (
          <Yue2Gate
            // Remount once the server's acknowledgment has loaded so the checkbox starts from it.
            key={`${ack.status}:${ack.acknowledged}`}
            ack={ack}
            busy={ackBusy}
            error={gateError}
            identity={identity}
            model={model}
            models={models}
            onAccept={accept}
            onRetry={loadAck}
            optIn={settings.optIn}
          />
        ) : (
          <>
            {!installed || tierProblem ? (
              <Yue2Install
                decoder={settings.decoder}
                downloads={downloads}
                model={model}
                onInstall={(tier, choices) => createModelDownloadJob?.(model, { variant: tier, choices })}
              />
            ) : null}

            <div className="settings-bar audio-settings-bar yue2-settings">
              <div className="settings-bar-row">
                <label className="settings-field settings-field-model">
                  Model
                  <select aria-label="YuE2 model" disabled value={identity.id}>
                    <option value={identity.id}>
                      {identity.name} · {identity.version} · {identity.license || "licence restricted"}
                    </option>
                  </select>
                </label>
                <label className="settings-field settings-field-tier">
                  Weight tier
                  <select aria-label="Weight tier" disabled={Boolean(why("tier"))} title={why("tier") ?? undefined} onChange={(event) => update({ tier: event.target.value })} value={settings.tier}>
                    <option value="">Default (BF16 weights)</option>
                    {tierRows.map((row) => (
                      <option key={row.tier} value={row.tier}>
                        {TIER_LABELS[row.tier] ?? row.tier}
                        {row.installed ? "" : " — not installed"}
                      </option>
                    ))}
                  </select>
                </label>
                <label className="settings-field">
                  Compute precision
                  <select aria-label="Compute precision" disabled={Boolean(why("computePolicy"))} title={why("computePolicy") ?? undefined} onChange={(event) => update({ computePolicy: event.target.value })} value={settings.computePolicy}>
                    <option value="">Choose compute precision</option>
                    <option value="auto">Auto (BF16 model + FP32 VAE on GPU; FP32 on CPU)</option>
                    <option value="bf16">BF16 (model + VAE on GPU)</option>
                    <option value="fp32">FP32 (model + VAE)</option>
                  </select>
                  <small>Q8/Q4 describe weight storage and quantized matmuls; each quantized matmul briefly uses FP32 input and result tensors before returning to the selected stage precision. This selector governs YuE2 song stages and the VAE, including cached decode; recording transcription runs separately on CPU. Kernels may also use FP32 reductions and audio output formats.</small>
                </label>
                <label className="settings-field">
                  Decoder
                  <select aria-label="Decoder" disabled={Boolean(why("decoder"))} title={why("decoder") ?? undefined} onChange={(event) => update({ decoder: event.target.value })} value={settings.decoder}>
                    <option value="">Default (standard)</option>
                    <option value="standard">Standard</option>
                    <option value="legacy">Legacy (add-on install)</option>
                  </select>
                </label>
                <label className="settings-field settings-field-count">
                  Takes
                  <input
                    aria-label="Takes"
                    disabled={Boolean(why("count"))}
                    title={why("count") ?? undefined}
                    max={MAX_TAKES}
                    min="1"
                    onChange={(event) => update({ count: event.target.value })}
                    type="number"
                    value={settings.count}
                  />
                </label>
              </div>
              {settings.decoder === "legacy" ? (
                <p className="yue2-muted">
                  The legacy decoder is an add-on.{" "}
                  <button
                    className="audio-link"
                    onClick={() =>
                      createModelDownloadJob?.(model, {
                        variant: settings.tier || undefined,
                        choices: { decoder: "legacy" },
                      })
                    }
                    type="button"
                  >
                    Install the legacy decoder
                  </button>
                </p>
              ) : null}
            </div>

            <div className="yue2-presets" data-testid="yue2-presets">
              <span className="eyebrow">Presets</span>
              {settings.presets.map((preset) => (
                <span className="preset-chip yue2-preset" key={preset.id}>
                  <button className="audio-link" onClick={() => applyPreset(preset)} type="button">
                    {preset.name}
                  </button>
                  <span className="yue2-muted">
                    {preset.version} · {preset.license || "licence restricted"}
                    {preset.nonCommercial ? " · Noncommercial" : ""}
                  </span>
                  <button
                    aria-label={`Delete preset ${preset.name}`}
                    className="audio-link"
                    onClick={() => update({ presets: settings.presets.filter((item) => item.id !== preset.id) })}
                    type="button"
                  >
                    <Icon.Close size={12} />
                  </button>
                </span>
              ))}
              <input
                aria-label="Preset name"
                onChange={(event) => setPresetName(event.target.value)}
                placeholder="Preset name"
                value={presetName}
              />
              <button className="secondary-action" disabled={!presetName.trim()} onClick={savePreset} type="button">
                <Icon.Save size={13} /> Save preset
              </button>
            </div>

            <div className="mode-tabs yue2-lab-tabs" role="tablist" aria-label="Song Lab">
              {LAB_TABS.map((tab) => (
                <button
                  aria-selected={settings.tab === tab.id}
                  className={settings.tab === tab.id ? "mode-tab active" : "mode-tab"}
                  key={tab.id}
                  onClick={() => update({ tab: tab.id })}
                  role="tab"
                  type="button"
                >
                  {tab.label}
                </button>
              ))}
            </div>

            {settings.tab === "compose" || settings.tab === "cover" ? (
              <div className="yue2-text-fields">
                <label>
                  Style
                  <textarea
                    aria-label="Style"
                    onChange={(event) => update({ style: event.target.value })}
                    placeholder="Genre, instruments, mood, vocal timbre…"
                    readOnly={restoring}
                    rows={2}
                    value={restoring ? (planWords?.style ?? "") : settings.style}
                  />
                </label>
                {!restoring && sessionOnly.has("style") ? <SessionOnlyText testId="yue2-style-session-only" /> : null}
                <label>
                  Lyrics
                  <textarea
                    aria-label="Lyrics"
                    onChange={(event) => update({ lyrics: event.target.value })}
                    placeholder={"[verse]\n…\n\n[chorus]\n…"}
                    readOnly={restoring}
                    rows={6}
                    value={restoring ? (planWords?.lyrics ?? "") : settings.lyrics}
                  />
                </label>
                {!restoring && sessionOnly.has("lyrics") ? <SessionOnlyText testId="yue2-lyrics-session-only" /> : null}
                {restoring ? (
                  <p className="yue2-muted" data-testid="yue2-plan-words">
                    {planWords
                      ? "The saved plan renders its own style and lyrics. To change them, plan again."
                      : "Choose a saved plan — it renders its own style and lyrics."}
                  </p>
                ) : null}
              </div>
            ) : null}

            {settings.tab === "compose" ? (
              <div className="yue2-panel" data-testid="yue2-compose">
                <div className="yue2-inline">
                  <span className="eyebrow">Plan</span>
                  <Segmented
                    label="Plan source"
                    onChange={(value) => update({ planSource: value })}
                    options={[
                      { value: "sample", label: "Sample a new plan" },
                      { value: "supplied", label: "Supply an ABC score" },
                      { value: "restore", label: "Restore a saved plan" },
                    ]}
                    value={settings.planSource}
                  />
                </div>
                {settings.planSource !== "restore" ? (
                  <div className="yue2-inline">
                    <span className="eyebrow">Planning</span>
                    <Segmented
                      disabledValues={settings.planSource === "supplied" || settings.planOnly ? ["off"] : []}
                      label="Planning"
                      onChange={(value) => update({ planning: value })}
                      options={[
                        { value: "full", label: "Full (melody + harmony)" },
                        { value: "melody", label: "Melody only" },
                        { value: "off", label: "Off (no score)" },
                      ]}
                      value={settings.planning}
                    />
                    <label className="checkline">
                      <input
                        checked={settings.planOnly}
                        onChange={(event) => update({ planOnly: event.target.checked })}
                        type="checkbox"
                      />
                      Plan only (stop after the score)
                    </label>
                  </div>
                ) : null}
                {settings.planSource === "supplied" ? (
                  <div className="yue2-abc-block">
                    <textarea
                      aria-label="Supplied ABC score"
                      className="yue2-abc"
                      onChange={(event) => update({ suppliedScore: event.target.value })}
                      placeholder="X:1 … two voices (Vocal / Ins) in the native YuE2 dialect"
                      rows={8}
                      spellCheck={false}
                      value={settings.suppliedScore}
                    />
                    {sessionOnlyNote(settings.suppliedScore)}
                    <button
                      className="secondary-action"
                      disabled={!settings.suppliedScore.trim()}
                      onClick={() => inspect("supplied", settings.suppliedScore)}
                      type="button"
                    >
                      Preview score
                    </button>
                    {preview.key === "supplied" ? (
                      <>
                        <ErrorNotice error={preview.error} testId="yue2-preview-error" />
                        <ScorePreview inspection={preview.inspection} />
                      </>
                    ) : null}
                  </div>
                ) : null}
                {settings.planSource === "restore" ? (
                  <label>
                    Saved plan
                    <select
                      aria-label="Saved plan"
                      onChange={(event) => update({ restorePlanJobId: event.target.value })}
                      value={settings.restorePlanJobId}
                    >
                      <option value="">Choose a finished plan or song…</option>
                      {restorable.map((view) => (
                        <option key={view.id} value={view.id}>
                          {view.kindLabel} · {view.style || view.id}
                        </option>
                      ))}
                    </select>
                  </label>
                ) : null}
                <div className="prompt-cta-stack yue2-cta">
                  <button
                    className="prompt-cta"
                    disabled={submitting || composeProblems.length > 0 || !readyToRun}
                    onClick={() => submit(kind, kind === "fromPlan" ? { plan: planWords } : {})}
                    title={composeProblems.join(" ") || tierProblem || undefined}
                    type="button"
                  >
                    <Icon.Sparkle size={14} /> {composeLabel}
                  </button>
                  {composeProblems.length ? <span className="yue2-muted">{composeProblems.join(" ")}</span> : null}
                </div>
              </div>
            ) : null}

            {settings.tab === "cover" ? (
              <div className="yue2-panel" data-testid="yue2-cover">
                <Yue2RecordingCover
                  assets={assets}
                  importAsset={importAsset}
                  jobs={jobs}
                  model={model}
                  onCancel={(target) => jobAction?.(target, "cancel")}
                  onLicenseLapsed={licenseLapsed}
                  onTranscribe={(target) => submit("transcribe", target)}
                  projectId={projectId}
                  refreshKey={finishedKey}
                  requestedGpu={requestedGpu}
                  runs={runs}
                  settings={settings}
                  submitting={submitting}
                  token={token}
                  update={update}
                  versions={versions}
                />
                <div className="yue2-inline">
                  <span className="eyebrow">Cover score</span>
                  <Segmented
                    label="Cover score source"
                    onChange={(value) => update({ coverSource: value })}
                    options={[
                      { value: "version", label: "Score version" },
                      { value: "inline", label: "Paste ABC" },
                    ]}
                    value={settings.coverSource}
                  />
                </div>
                {settings.coverSource === "version" ? (
                  <div className="yue2-inline">
                    <select
                      aria-label="Cover score version"
                      onChange={(event) => update({ coverVersionId: event.target.value })}
                      value={settings.coverVersionId}
                    >
                      <option value="">Choose a reviewed score version…</option>
                      {versions.map((version) => (
                        <option key={version.id} value={version.id}>
                          {version.id} · {version.editOperation ?? version.origin} · {version.cot}
                          {version.transcription ? ` · from a recording (${version.transcription.mode})` : ""}
                          {version.nonCommercial ? " · Noncommercial" : ""}
                        </option>
                      ))}
                    </select>
                    <button
                      className="secondary-action"
                      disabled={!settings.coverVersionId}
                      onClick={() => inspectCoverVersion(settings.coverVersionId)}
                      type="button"
                    >
                      Review score
                    </button>
                    <button
                      className="secondary-action"
                      disabled={!settings.coverVersionId}
                      onClick={() => update({ tab: "scores", selectedVersionId: settings.coverVersionId })}
                      type="button"
                    >
                      Edit in Scores
                    </button>
                  </div>
                ) : (
                  <div className="yue2-abc-block">
                    <textarea
                      aria-label="Cover ABC score"
                      className="yue2-abc"
                      onChange={(event) => update({ coverScore: event.target.value })}
                      rows={8}
                      spellCheck={false}
                      value={settings.coverScore}
                    />
                    {sessionOnlyNote(settings.coverScore)}
                    <button
                      className="secondary-action"
                      disabled={!settings.coverScore.trim()}
                      onClick={() => inspect("coverInline", settings.coverScore)}
                      type="button"
                    >
                      Review score
                    </button>
                  </div>
                )}
                {preview.key === "coverVersion" || preview.key === "coverInline" ? (
                  <div data-testid="yue2-cover-review">
                    <ErrorNotice error={preview.error} testId="yue2-preview-error" />
                    <ScorePreview inspection={preview.inspection} />
                  </div>
                ) : null}
                <div className="yue2-grid">
                  <label>
                    Cover mode
                    <select aria-label="Cover mode" onChange={(event) => update({ coverMode: event.target.value })} value={settings.coverMode}>
                      <option value="melody">Melody</option>
                      <option value="full">Full</option>
                    </select>
                  </label>
                  <label title={settings.coverMode === "melody" ? undefined : "Only a melody cover chooses which melodies it keeps."}>
                    Keep
                    <select aria-label="Cover keep" disabled={settings.coverMode !== "melody"} onChange={(event) => update({ coverKeep: event.target.value })} value={settings.coverKeep}>
                      <option value="">Model default</option>
                      <option value="both">Vocal and instrumental</option>
                      <option value="vocal">Vocal only</option>
                      <option value="instrumental">Instrumental only</option>
                    </select>
                  </label>
                </div>
                <label>
                  Source lyrics (when the lyrics above are their translation)
                  <textarea
                    aria-label="Translated from"
                    onChange={(event) => update({ coverTranslatedFrom: event.target.value })}
                    rows={3}
                    value={settings.coverTranslatedFrom}
                  />
                </label>
                <RegenerationNotice text={regenerationNotice} />
                {settings.coverSource === "version" && coverVersion?.transcription ? (
                  <label className="model-license-ack" data-testid="yue2-transcription-review-confirmation">
                    <input
                      checked={reviewedTranscriptionVersionId === settings.coverVersionId}
                      onChange={(event) => setReviewedTranscriptionVersionId(event.target.checked ? settings.coverVersionId : "")}
                      type="checkbox"
                    />
                    I reviewed this transcription, its warnings and octave evidence, and this score version
                  </label>
                ) : null}
                <div className="prompt-cta-stack yue2-cta">
                  <button
                    className="prompt-cta"
                    disabled={submitting || coverProblems.length > 0 || !readyToRun}
                    onClick={() => submit("cover")}
                    title={coverProblems.join(" ") || tierProblem || undefined}
                    type="button"
                  >
                    <Icon.Sparkle size={14} /> Generate cover
                  </button>
                  {coverProblems.length ? <span className="yue2-muted">{coverProblems.join(" ")}</span> : null}
                </div>
              </div>
            ) : null}

            {settings.tab === "scores" ? (
              <Yue2ScoreWorkbench
                assets={assets}
                onReloadVersions={reloadVersions}
                onRender={(versionId) => submit("renderVersion", { versionId })}
                onSelectVersion={(versionId) => update({ selectedVersionId: versionId })}
                onUseForCover={(versionId) => {
                  const version = versions.find((item) => item.id === versionId);
                  update({
                    tab: "cover", coverSource: "version", coverVersionId: versionId,
                    ...(version?.transcription && (version.cot === "full" || version.cot === "melody")
                      ? { coverMode: version.cot } : {}),
                  });
                }}
                projectId={projectId}
                regenerationNotice={regenerationNotice}
                draft={settings.editDraft}
                draftSessionOnly={sessionOnly.has("editDraft")}
                importDraft={settings.importDraft}
                importDraftSessionOnly={sessionOnly.has("importDraft")}
                // A score version's export is marked from the catalog's own declaration of the model.
                policy={{
                  nonCommercial: identity.nonCommercial,
                  experimental: identity.experimental,
                  license: { license: identity.license, url: identity.licenseUrl },
                }}
                refreshKey={finishedKey}
                renderProblem={
                  !readyToRun
                    ? tierProblem || "Install YuE2 first."
                    : submitting
                      ? "Submitting…"
                      : renderProblems.join(" ")
                }
                seedSettings={settings}
                setDraft={(next) =>
                  setSettings((current) => ({
                    ...current,
                    editDraft: typeof next === "function" ? next(current.editDraft) : next,
                  }))
                }
                setImportDraft={(next) =>
                  setSettings((current) => ({
                    ...current,
                    importDraft: typeof next === "function" ? next(current.importDraft) : next,
                  }))
                }
                selectedVersionId={settings.selectedVersionId}
                token={token}
                versions={versions}
                versionsError={versionsError}
              />
            ) : null}

            {settings.tab === "compare" ? (
              <Yue2CompareWorkbench
                assets={assets}
                comparisons={comparisons}
                comparisonsError={comparisonsError}
                onReloadComparisons={reloadComparisons}
                projectId={projectId}
                refreshKey={finishedKey}
                settings={settings}
                token={token}
                update={update}
                versions={versions}
              />
            ) : null}

            <ErrorNotice error={submitError} testId="yue2-submit-error" />

            <AdvancedSection
              hint="Sampling, guidance, seed, synthesis steps, AR mode, compute precision and memory"
              onToggle={() => update({ advancedOpen: !settings.advancedOpen })}
              open={settings.advancedOpen}
            >
              <div className="yue2-grid">
                <label>
                  Seed
                  <input aria-label="Seed" disabled={Boolean(why("seed"))} title={why("seed") ?? undefined} onChange={(event) => update({ seed: event.target.value })} placeholder="Random" step="1" type="number" value={settings.seed} />
                </label>
                <label>
                  Guidance (CFG)
                  <input aria-label="Guidance" disabled={Boolean(why("cfgScale"))} title={why("cfgScale") ?? undefined} onChange={(event) => update({ cfgScale: event.target.value })} placeholder="Model default" step="0.1" type="number" value={settings.cfgScale} />
                </label>
                <label>
                  Acoustic ODE steps
                  <input aria-label="ODE steps" disabled={Boolean(why("steps"))} title={why("steps") ?? undefined} max="10000" min="1" onChange={(event) => update({ steps: event.target.value })} placeholder="Model default" step="1" type="number" value={settings.steps} />
                </label>
                <label>
                  AR mode
                  <select aria-label="AR mode" disabled={Boolean(why("arMode"))} title={why("arMode") ?? undefined} onChange={(event) => update({ arMode: event.target.value })} value={settings.arMode}>
                    <option value="">Native (default)</option>
                    <option value="experimentalFp8">Experimental FP8 AR (CUDA sm_89+, bf16 only)</option>
                  </select>
                </label>
                <label>
                  Offload
                  <select aria-label="Offload" disabled={Boolean(why("offloadPolicy"))} title={why("offloadPolicy") ?? undefined} onChange={(event) => update({ offloadPolicy: event.target.value })} value={settings.offloadPolicy}>
                    <option value="">Default (resident)</option>
                    <option value="resident">Resident</option>
                    <option value="sequential">Sequential (offload the AR stage)</option>
                  </select>
                </label>
              </div>
              <SamplingFields
                defaults={SAMPLING_TOKEN_DEFAULTS.scoreSampling}
                disabled={settings.planSource !== "sample" || settings.planning === "off"}
                label="Score planning sampling"
                reason={why("scoreSampling")}
                onChange={(value) => update({ scoreSampling: value })}
                value={settings.scoreSampling}
              />
              <SamplingFields
                defaults={SAMPLING_TOKEN_DEFAULTS.semanticSampling}
                label="Semantic sampling"
                reason={why("semanticSampling")}
                onChange={(value) => update({ semanticSampling: value })}
                value={settings.semanticSampling}
              />
              <fieldset className="yue2-fieldset">
                <legend>Memory</legend>
                <div className="yue2-grid">
                  <TriStateSelect label="Stage residency" reason={why("memory.acoustic")} onChange={(value) => update({ stageResidency: value })} value={settings.stageResidency} />
                  <TriStateSelect label="Chunked attention" reason={why("memory.acoustic")} onChange={(value) => update({ chunkAttention: value })} value={settings.chunkAttention} />
                  <label>
                    Attention chunk size
                    <input aria-label="Attention chunk size" disabled={settings.chunkAttention !== "on" || Boolean(why("memory.acoustic"))} title={why("memory.acoustic") ?? undefined} min={MIN_ATTENTION_CHUNK_ELEMENTS} onChange={(event) => update({ attentionChunkSize: event.target.value })} placeholder={`≥ ${MIN_ATTENTION_CHUNK_ELEMENTS} (with chunked attention on)`} type="number" value={settings.attentionChunkSize} />
                  </label>
                  <TriStateSelect label="Tiled decode" reason={why("memory.decode")} onChange={(value) => update({ tileVaeDecode: value })} value={settings.tileVaeDecode} />
                  <label>
                    Decode tile (frames)
                    <input aria-label="Decode tile" disabled={settings.tileVaeDecode !== "on" || Boolean(why("memory.decode"))} title={why("memory.decode") ?? undefined} max={MAX_DECODE_TILE_FRAMES} min="1" onChange={(event) => update({ decodeTileEdge: event.target.value })} placeholder="1–1024 (with tiled decode on)" type="number" value={settings.decodeTileEdge} />
                  </label>
                </div>
              </fieldset>
            </AdvancedSection>
          </>
        )}
      </WorkPanel>

      {enabled ? (
        <div className="studio-results yue2-runs" data-testid="yue2-runs">
          <section className="review-panel">
            <div className="review-panel-head">
              <div className="review-panel-head-title">
                <h2>Song Lab runs</h2>
              </div>
            </div>
            {runs.length === 0 ? <div className="empty-panel">No YuE2 runs in this project yet</div> : null}
            {runs.map((job) => (
              <Yue2RunCard
                assets={assets}
                job={job}
                key={job.id}
                model={model}
                onCancel={(target) => jobAction?.(target, "cancel")}
                decodeBlocked={readyToRun ? null : tierProblem || "Install YuE2 first."}
                onDecodeAgain={(sourceJobId, decoder) => submit("decode", { sourceJobId, decoder })}
                onFetchVersionAbc={fetchVersionAbc}
                onOpenVersion={(versionId) => update({ tab: "scores", selectedVersionId: versionId })}
                onOpenTranscription={(transcriptionId) => update({ tab: "cover", transcriptionId })}
                onRestorePlan={(jobId) => update({ tab: "compose", planSource: "restore", restorePlanJobId: jobId })}
              />
            ))}
          </section>
        </div>
      ) : null}

      {guideOpen && model.ui?.promptGuide ? (
        <PromptGuideModal guide={model.ui.promptGuide} modelName={model.name} onClose={() => setGuideOpen(false)} />
      ) : null}
    </div>
  );
}
