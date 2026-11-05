import { Yue2ReviewPanel } from "./yue2Lab/Yue2ReviewPanel.jsx";
import { Yue2AdvancedPanel } from "./yue2Lab/Yue2AdvancedPanel.jsx";
import { Yue2TextFields } from "./yue2Lab/Yue2TextFields.jsx";
import { Yue2PresetPanel } from "./yue2Lab/Yue2PresetPanel.jsx";
import { Yue2GenerationSettings } from "./yue2Lab/Yue2GenerationSettings.jsx";
import { Yue2ComposePanel } from "./yue2Lab/Yue2ComposePanel.jsx";
import { Yue2CoverPanel } from "./yue2Lab/Yue2CoverPanel.jsx";
import { useYue2Presets } from "./yue2Lab/useYue2Presets.js";
import { useYue2Inspection } from "./yue2Lab/useYue2Inspection.js";
import { useYue2Submission } from "./yue2Lab/useYue2Submission.js";
import { useYue2History } from "./yue2Lab/useYue2History.js";
import { useYue2License } from "./yue2Lab/useYue2License.js";
import { useYue2Draft } from "./yue2Lab/useYue2Draft.js";
import { LAB_TABS } from "./yue2Lab/labConstants.js";

import { Yue2Gate, Yue2Install } from "./yue2Lab/Yue2LicensePanels.jsx";
import React, { useMemo, useState } from "react";
import { Icon } from "../components/Icons.jsx";
import { WorkPanel } from "../components/WorkPanel.jsx";

import { PromptGuideModal } from "../components/PromptGuideModal.jsx";
import { useAppContext } from "../context/AppContext.js";
import { durableTextFits } from "../hooks/useStudioSettings.js";

import { terminalStatuses } from "../jobTypes.js";
import { YUE2_MODEL_ID, isCoverComponentDownload, composeKind, yue2ModelIdentity, yue2ModelInstalled, yue2PlanRequest, yue2FieldDisabledReason, yue2RequestProblems, yue2RunView, yue2TierRows } from "../yue2Lab.js";

import { ErrorNotice, PolicyChips, Yue2CompareWorkbench, Yue2ScoreWorkbench } from "./audioYue2Parts.jsx";

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

  const {
    settings,
    setSettings,
    update,
    sessionOnly
  } = useYue2Draft({
    projectId,
  preferencesHydrated,
  model
  });

  const {
    ack,
    setAck,
    ackBusy,
    gateError,
    loadAck,
    enabled,
    accept,
    withdraw
  } = useYue2License({
    model,
  token,
  settings,
  update
  });

  const {
    versions,
    versionsError,
    regenerationNotice,
    comparisons,
    comparisonsError,
    reloadVersions,
    reloadComparisons,
    setSubmitted,
    runs,
    finishedKey
  } = useYue2History({
    projectId,
  token,
  enabled,
  jobs
  });

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

  const {
    submitError,
    submitting,
    licenseLapsed,
    submit
  } = useYue2Submission({
    model,
  setAck,
  settings,
  problemContext,
  transcriptionReviewProblem,
  requestedGpu,
  projectId,
  token,
  setSubmitted
  });

  const {
    preview,
    inspect,
    inspectCoverVersion,
    fetchVersionAbc
  } = useYue2Inspection({
    token,
  projectId
  });

  const {
    presetName,
    setPresetName,
    savePreset,
    applyPreset
  } = useYue2Presets({
    settings,
  identity,
  update,
  setSettings
  });

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

            <Yue2GenerationSettings
            identity={identity}
            why={why}
            update={update}
            settings={settings}
            tierRows={tierRows}
            createModelDownloadJob={createModelDownloadJob}
            model={model}
          />

            <Yue2PresetPanel
            settings={settings}
            applyPreset={applyPreset}
            update={update}
            setPresetName={setPresetName}
            presetName={presetName}
            savePreset={savePreset}
          />

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
              <Yue2TextFields
            update={update}
            restoring={restoring}
            planWords={planWords}
            settings={settings}
            sessionOnly={sessionOnly}
          />
            ) : null}

            {settings.tab === "compose" ? (
              <Yue2ComposePanel
            update={update}
            settings={settings}
            sessionOnlyNote={sessionOnlyNote}
            inspect={inspect}
            preview={preview}
            restorable={restorable}
            submitting={submitting}
            composeProblems={composeProblems}
            readyToRun={readyToRun}
            submit={submit}
            kind={kind}
            planWords={planWords}
            tierProblem={tierProblem}
            composeLabel={composeLabel}
          />
            ) : null}

            {settings.tab === "cover" ? (
              <Yue2CoverPanel
            assets={assets}
            importAsset={importAsset}
            jobs={jobs}
            model={model}
            jobAction={jobAction}
            licenseLapsed={licenseLapsed}
            submit={submit}
            projectId={projectId}
            finishedKey={finishedKey}
            requestedGpu={requestedGpu}
            runs={runs}
            settings={settings}
            submitting={submitting}
            token={token}
            update={update}
            versions={versions}
            inspectCoverVersion={inspectCoverVersion}
            sessionOnlyNote={sessionOnlyNote}
            inspect={inspect}
            preview={preview}
            regenerationNotice={regenerationNotice}
            coverVersion={coverVersion}
            reviewedTranscriptionVersionId={reviewedTranscriptionVersionId}
            setReviewedTranscriptionVersionId={setReviewedTranscriptionVersionId}
            coverProblems={coverProblems}
            readyToRun={readyToRun}
            tierProblem={tierProblem}
          />
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

            <Yue2AdvancedPanel
            update={update}
            settings={settings}
            why={why}
          />
          </>
        )}
      </WorkPanel>

      {enabled ? (
        <Yue2ReviewPanel
            runs={runs}
            assets={assets}
            model={model}
            jobAction={jobAction}
            readyToRun={readyToRun}
            tierProblem={tierProblem}
            submit={submit}
            fetchVersionAbc={fetchVersionAbc}
            update={update}
          />
      ) : null}

      {guideOpen && model.ui?.promptGuide ? (
        <PromptGuideModal guide={model.ui.promptGuide} modelName={model.name} onClose={() => setGuideOpen(false)} />
      ) : null}
    </div>
  );
}
