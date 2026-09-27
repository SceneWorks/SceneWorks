import React, { useEffect, useMemo, useRef, useState } from "react";
import { Icon } from "../components/Icons.jsx";
import { AssetPickerField } from "../components/AssetPicker.jsx";
import { assetCanRenderAsAudio } from "../components/assetMedia.jsx";
import { formatRelativeTime } from "../audioTakes.js";
import { formatBytes, percent } from "../formatting.js";
import { terminalStatuses } from "../jobTypes.js";
import {
  TRANSCRIPTION_DEFAULT_LOOKAHEAD_SECONDS,
  TRANSCRIPTION_DEFAULT_OVERLAP_SECONDS,
  installJobStatusLabel,
  isCoverComponentDownload,
  midiNoteLabel,
  stripYue2ExportHeader,
  yue2CoverSetup,
  yue2RequestProblems,
  yue2RunView,
  yue2TranscriptionView,
} from "../yue2Lab.js";
import {
  downloadYue2ConditionalComponents,
  fetchYue2TranscriptionFile,
  getYue2Transcription,
  inspectYue2Score,
  listYue2Transcriptions,
  yue2TranscriptionFileUrl,
} from "../api/yue2.js";
import { ErrorNotice, PolicyChips, ScorePreview } from "./audioYue2Parts.jsx";

// Cover from a recording (sc-23002, epic 22988 AT2) — the Song Lab's recording → transcription →
// cover flow. Three user-visible steps, never one click:
//
//   1. transcribe a project recording with SheetSage2 + MERT-v2-FullSong (a `transcribe` job; its
//      closure is installed once through the conditional-components door, under YuE2's licence
//      acknowledgment — the lab only opens behind that gate);
//   2. review the transcription (warnings, octave evidence and per-mode readiness up front) and
//      review / edit the score version it imported in the lab's Scores workbench;
//   3. cover that (possibly edited) version — the existing cover job with `cover.versionId`.
//
// Every server refusal and failed job is rendered where it happened; nothing falls back to an
// empty state.

const MODES = [
  { mode: "melody", label: "Melody-only score" },
  { mode: "full", label: "Full score (melody + chords)" },
];

function downloadBlob(blob, filename) {
  const url = URL.createObjectURL(blob);
  const anchor = document.createElement("a");
  anchor.href = url;
  anchor.download = filename;
  document.body.appendChild(anchor);
  anchor.click();
  anchor.remove();
  URL.revokeObjectURL(url);
}

function NoncommercialChips({ license }) {
  return (
    <span className="yue2-chips" data-testid="yue2-cover-closure-chips">
      <span className="yue2-chip yue2-chip--warn">Noncommercial</span>
      {license ? <span className="yue2-chip">{license}</span> : null}
    </span>
  );
}

// A microphone take becomes a normal project audio asset before it can be transcribed. The
// recorder and stream never survive leaving this panel, including during the permission prompt.
function MicrophoneTake({ importAsset, onImported }) {
  const recorder = useRef(null);
  const stream = useRef(null);
  const live = useRef(true);
  const discard = useRef(false);
  const [state, setState] = useState("idle");
  const [error, setError] = useState(null);
  const [importedName, setImportedName] = useState("");

  function release() {
    stream.current?.getTracks().forEach((track) => track.stop());
    stream.current = null;
    recorder.current = null;
  }

  useEffect(() => {
    live.current = true;
    return () => {
      live.current = false;
      discard.current = true;
      const active = recorder.current;
      if (active?.state === "recording") active.stop();
      release();
    };
  }, []);

  async function start() {
    setError(null);
    setImportedName("");
    if (!navigator.mediaDevices?.getUserMedia || typeof MediaRecorder === "undefined") {
      setError("Microphone recording is unavailable in this browser. Upload an audio file instead.");
      return;
    }
    setState("requesting");
    let acquired;
    try {
      acquired = await navigator.mediaDevices.getUserMedia({ audio: true });
      if (!live.current) {
        acquired.getTracks().forEach((track) => track.stop());
        return;
      }
      stream.current = acquired;
      const next = new MediaRecorder(acquired);
      const chunks = [];
      discard.current = false;
      recorder.current = next;
      next.ondataavailable = (event) => { if (event.data?.size) chunks.push(event.data); };
      next.onerror = (event) => {
        discard.current = true;
        release();
        if (live.current) { setError(event.error || "The microphone recording failed."); setState("idle"); }
      };
      next.onstop = async () => {
        release();
        if (!live.current || discard.current) { if (live.current) setState("idle"); return; }
        setState("saving");
        try {
          if (!chunks.length) throw new Error("The microphone recording was empty.");
          const mime = next.mimeType || chunks[0].type || "audio/webm";
          const ext = mime.includes("mp4") ? "m4a" : mime.includes("ogg") ? "ogg" : "webm";
          const file = new File(chunks, `Song Lab recording ${new Date().toISOString().replaceAll(":", "-")}.${ext}`, { type: mime });
          const asset = await importAsset(file, { select: false, throwOnError: true });
          if (!asset?.id || !assetCanRenderAsAudio(asset)) throw new Error("The recording could not be imported as an audio asset.");
          if (live.current) {
            setImportedName(file.name);
            onImported(asset.id);
          }
        } catch (cause) {
          if (live.current) setError(cause);
        } finally {
          if (live.current) setState("idle");
        }
      };
      next.start();
      setState("recording");
    } catch (cause) {
      acquired?.getTracks().forEach((track) => track.stop());
      release();
      if (live.current) { setError(cause); setState("idle"); }
    }
  }

  function stop(keep) {
    discard.current = !keep;
    if (recorder.current?.state === "recording") recorder.current.stop();
    stream.current?.getTracks().forEach((track) => track.stop());
    if (!keep) setState("idle");
  }

  return (
    <div className="yue2-microphone" data-testid="yue2-microphone" data-state={state}>
      {state === "recording" ? (
        <div className="yue2-inline">
          <span role="status">Recording from the microphone…</span>
          <button className="secondary-action" onClick={() => stop(true)} type="button">Stop and use recording</button>
          <button className="secondary-action" onClick={() => stop(false)} type="button">Discard recording</button>
        </div>
      ) : (
        <button className="secondary-action" disabled={!importAsset || state !== "idle"} onClick={start} type="button">
          {state === "requesting" ? "Requesting microphone…" : state === "saving" ? "Saving recording…" : "Record with microphone"}
        </button>
      )}
      {importedName ? <p className="yue2-muted" role="status">{importedName} was saved to the project and is ready to transcribe.</p> : null}
      <ErrorNotice error={error} testId="yue2-microphone-error" />
    </div>
  );
}

// ---- step 0: the cover closure --------------------------------------------------------------

function CoverSetup({ model, setup, jobs, token, requestedGpu, onLicenseLapsed }) {
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState(null);
  const [queued, setQueued] = useState(null);
  // The live job list wins; the POST's snapshots cover the moment before it catches up.
  const setupJobs = useMemo(() => {
    const live = (jobs ?? []).filter((job) => isCoverComponentDownload(job, model));
    const known = new Set(live.map((job) => job.id));
    return [...live, ...(queued?.jobs ?? []).filter((job) => !known.has(job.id))];
  }, [jobs, model, queued]);
  const jobFor = (repo) =>
    setupJobs
      .filter((job) => job.payload?.repo === repo)
      .sort((a, b) => Date.parse(b.createdAt ?? 0) - Date.parse(a.createdAt ?? 0))[0] ?? null;
  const active = setupJobs.some((job) => !terminalStatuses.has(job.status));

  async function setUp() {
    setBusy(true);
    setError(null);
    try {
      const body = { licenseAcknowledged: true };
      if (requestedGpu) body.requestedGpu = requestedGpu;
      const response = await downloadYue2ConditionalComponents(model.id, "cover", body, token);
      setQueued({
        jobs: Array.isArray(response?.jobs) ? response.jobs : [],
        components: Array.isArray(response?.components) ? response.components : [],
      });
    } catch (err) {
      setError(err);
      if (err?.code === "license_acknowledgment_required") onLicenseLapsed?.();
    } finally {
      setBusy(false);
    }
  }

  const license = setup.components.find((row) => row.license)?.license ?? "";
  const basis = [...new Set(setup.components.map((row) => row.licenseBasis).filter(Boolean))];
  return (
    <div className="yue2-cover-setup" data-testid="yue2-cover-setup" data-state={setup.installState}>
      <div className="yue2-inline">
        <strong>Set up covers from a recording</strong>
        <NoncommercialChips license={license} />
      </div>
      <p>
        Transcribing a recording needs SheetSage2 and MERT-v2-FullSong ({formatBytes(setup.totalBytes)} in total). They
        are downloaded only for covers — never as part of the YuE2 install — under the licence you accepted for the
        Song Lab: <strong>noncommercial use only</strong>, with attribution to their authors.
      </p>
      {basis.map((text) => (
        <p className="yue2-muted" data-testid="yue2-cover-license-basis" key={text}>
          {text}
        </p>
      ))}
      <ul className="yue2-tier-list">
        {setup.components.map((row) => {
          const job = jobFor(row.repo);
          const reported = queued?.components.find((item) => item.componentId === row.componentId);
          const state =
            row.installState === "installed"
              ? "installed"
              : job
                ? installJobStatusLabel(job)
                : reported?.status === "installed"
                  ? "installed"
                  : row.installState;
          return (
            <li className="yue2-tier" data-testid="yue2-cover-component" key={row.componentId}>
              <strong>{row.componentId}</strong>
              <span className="yue2-muted">
                {row.repo}
                {row.revision ? `@${row.revision.slice(0, 8)}` : ""} · {formatBytes(row.estimatedSizeBytes)} · {row.license}
                {row.nonCommercial ? " · Noncommercial" : ""}
              </span>
              <span
                className={state === "installed" ? "status-badge installed" : "status-badge"}
                data-testid="yue2-cover-component-state"
              >
                {state}
              </span>
              {job && !terminalStatuses.has(job.status) ? (
                <span className="yue2-inline">
                  <span aria-label={`${row.componentId} download progress`} className="progress-track">
                    <span style={{ width: percent(Number(job.progress) || 0) }} />
                  </span>
                  {job.message ? <span className="yue2-muted">{job.message}</span> : null}
                </span>
              ) : null}
              {job && (job.status === "failed" || job.status === "interrupted") && row.installState !== "installed" ? (
                <ErrorNotice error={job.error || "The download failed without an error message."} testId="yue2-cover-download-error" />
              ) : null}
              {row.blocked ? (
                <span className="yue2-error-text">
                  Blocked: {row.blocked.reason}
                  {row.blocked.unblock ? ` Unblocks when: ${row.blocked.unblock}` : ""}
                </span>
              ) : null}
            </li>
          );
        })}
      </ul>
      <button
        className="secondary-action strong"
        disabled={busy || active || setup.blocked}
        onClick={setUp}
        type="button"
      >
        <Icon.Download size={14} /> {active ? "Setting up covers…" : "Set up covers"}
      </button>
      <ErrorNotice error={error} testId="yue2-cover-setup-error" />
    </div>
  );
}

// ---- step 1: the transcription job ----------------------------------------------------------

function TranscriptionJob({ job, assetName, onCancel, onReview }) {
  const view = yue2RunView(job);
  return (
    <div className="yue2-run__progress" data-testid="yue2-transcription-job" data-status={view.status}>
      <span className="yue2-run__stage">
        <strong>{assetName}</strong> —{" "}
        {view.running
          ? view.message || "Queued — waiting for a worker."
          : view.status === "completed"
            ? "Transcribed."
            : view.status}
      </span>
      {view.running ? (
        <>
          <div aria-label="Transcription progress" className="progress-track">
            <span style={{ width: percent(view.progress) }} />
          </div>
          <button className="secondary-action" disabled={view.cancelRequested} onClick={() => onCancel(job)} type="button">
            {view.cancelRequested ? "Canceling…" : "Cancel"}
          </button>
        </>
      ) : null}
      {view.error ? <ErrorNotice error={view.error} testId="yue2-transcription-job-error" /> : null}
      {view.sideEffectErrors.length ? (
        <ErrorNotice
          error={`The transcription finished, but recording it failed: ${view.sideEffectErrors.join("; ")}`}
          testId="yue2-transcription-import-error"
        />
      ) : null}
      {view.transcriptionId ? (
        <button className="secondary-action" onClick={() => onReview(view.transcriptionId)} type="button">
          Review this transcription
        </button>
      ) : null}
    </div>
  );
}

// ---- step 2: the review ---------------------------------------------------------------------

function VoiceFacts({ label, voice }) {
  return (
    <div>
      <dt>{label}</dt>
      <dd>
        {voice.notes} notes
        {voice.notes
          ? ` · ${midiNoteLabel(voice.minPitch)} – ${midiNoteLabel(voice.maxPitch)} · median ${midiNoteLabel(voice.medianPitch)}`
          : ""}
      </dd>
    </div>
  );
}

function OctaveEvidence({ octave }) {
  if (!octave) {
    return (
      <p className="yue2-muted" data-testid="yue2-octave-evidence">
        No octave check was recorded for this transcription (no vocal notes long enough to check).
      </p>
    );
  }
  return (
    <div className="yue2-octave" data-testid="yue2-octave-evidence">
      <p>
        <strong>Octave check:</strong> {octave.f0HalfDominant ?? "?"} of {octave.checked} checked vocal notes have more
        energy one octave below the transcribed pitch
        {octave.fraction !== null ? ` (${Math.round(octave.fraction * 100)}%)` : ""}
        {octave.range ? ` · transcribed range ${midiNoteLabel(octave.range[0])} – ${midiNoteLabel(octave.range[1])}` : ""}.
      </p>
      {octave.method ? <p className="yue2-muted">Method: {octave.method}</p> : null}
      {octave.notes.length ? (
        <details className="yue2-inspect" data-testid="yue2-octave-notes">
          <summary>Per-note octave evidence ({octave.notes.length})</summary>
          <table className="yue2-octave__table">
            <thead>
              <tr>
                <th>Start (s)</th>
                <th>Note</th>
                <th>Energy f0/2</th>
                <th>Energy f0</th>
                <th>Energy 2·f0</th>
              </tr>
            </thead>
            <tbody>
              {octave.notes.map((note, index) => (
                <tr className={note.energy_f0_half > note.energy_f0 ? "is-flagged" : ""} key={`${note.start}-${index}`}>
                  <td>{Number(note.start).toFixed(2)}</td>
                  <td>{midiNoteLabel(note.midi)}</td>
                  <td>{Number(note.energy_f0_half).toFixed(3)}</td>
                  <td>{Number(note.energy_f0).toFixed(3)}</td>
                  <td>{Number(note.energy_2f0).toFixed(3)}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </details>
      ) : null}
    </div>
  );
}

function ExportLinks({ projectId, transcriptionId, groups, token }) {
  const [error, setError] = useState(null);
  async function save(event, file) {
    // The files route needs the session token, which a bare link cannot carry in remote-auth mode.
    event.preventDefault();
    setError(null);
    try {
      const blob = await fetchYue2TranscriptionFile(projectId, transcriptionId, file.path, token);
      downloadBlob(blob, String(file.path).split("/").pop());
    } catch (err) {
      setError(err);
    }
  }
  if (!groups.length) {
    return <p className="yue2-muted">This transcription recorded no export files.</p>;
  }
  return (
    <div className="yue2-exports" data-testid="yue2-transcription-exports">
      {groups.map((group) => (
        <div className="yue2-inline" data-testid={`yue2-exports-${group.kind}`} key={group.kind}>
          <span className="eyebrow">{group.label}</span>
          {group.files.map((file) => (
            <a
              className="audio-link"
              download={String(file.path).split("/").pop()}
              href={yue2TranscriptionFileUrl(projectId, transcriptionId, file.path)}
              key={file.path}
              onClick={(event) => save(event, file)}
            >
              <Icon.Download size={13} /> {file.path}
            </a>
          ))}
        </div>
      ))}
      <ErrorNotice error={error} testId="yue2-export-download-error" />
    </div>
  );
}

function ModeCard({ mode, label, view, abc, versions, token, onEdit, onUseForCover, coverVersionId }) {
  const readiness = view.readiness[mode];
  const versionId = view.versions[mode];
  const versionError = view.versionErrors.find((row) => row.mode === mode);
  const abcError = view.abcErrors[mode];
  // The imported (source) version and every edit derived from it — all retained, each coverable.
  const lineage = versions.filter(
    (version) => version.transcription?.transcriptionId === view.id && version.transcription?.mode === mode,
  );
  if (versionId && !lineage.some((version) => version.id === versionId)) {
    lineage.push({ id: versionId, origin: "transcription", editOperation: null, cot: mode });
  }
  const [preview, setPreview] = useState({ inspection: null, error: null, busy: false });
  async function inspect() {
    setPreview({ inspection: null, error: null, busy: true });
    try {
      setPreview({ inspection: await inspectYue2Score(stripYue2ExportHeader(abc), token), error: null, busy: false });
    } catch (error) {
      setPreview({ inspection: null, error, busy: false });
    }
  }
  return (
    <div className="yue2-transcription-mode" data-testid={`yue2-transcription-mode-${mode}`}>
      <div className="yue2-inline">
        <strong>{label}</strong>
        {readiness ? (
          <span
            className={readiness.ready ? "status-badge installed" : "status-badge danger"}
            data-testid={`yue2-readiness-${mode}`}
          >
            {readiness.ready ? `Ready for a ${mode} cover` : "Refused as a cover plan"}
          </span>
        ) : (
          <span className="status-badge" data-testid={`yue2-readiness-${mode}`}>
            readiness not recorded
          </span>
        )}
      </div>
      {readiness && !readiness.ready ? (
        <p className="yue2-notice yue2-notice--error" data-testid={`yue2-readiness-reason-${mode}`} role="alert">
          <Icon.Warning size={15} />
          <span>
            A {mode} cover cannot use this transcription: {readiness.reason || "the review refused it without a reason."}
          </span>
        </p>
      ) : null}
      {versionError ? (
        <ErrorNotice
          error={`The ${mode} score could not be imported as a score version: ${versionError.message}`}
          testId={`yue2-version-error-${mode}`}
        />
      ) : null}
      {abcError ? (
        <ErrorNotice error={`The ${mode} score could not be built: ${abcError}`} testId={`yue2-abc-error-${mode}`} />
      ) : null}
      {lineage.length ? (
        <ul className="yue2-version-list" data-testid={`yue2-transcription-versions-${mode}`}>
          {lineage.map((version) => (
            <li className="yue2-inline" key={version.id}>
              <span>
                <strong>{version.id}</strong>{" "}
                <span className="yue2-muted">
                  {version.editOperation ? `edit · ${version.editOperation}${version.editBrief ? ` — ${version.editBrief}` : ""}` : "imported from the recording"}
                </span>
              </span>
              <button className="secondary-action" onClick={() => onEdit(version.id)} type="button">
                Review &amp; edit in Scores
              </button>
              <button
                aria-pressed={coverVersionId === version.id}
                className="secondary-action"
                disabled={!readiness?.ready || Boolean(versionError)}
                onClick={() => onUseForCover(version.id, version.cot === "full" || version.cot === "melody" ? version.cot : mode)}
                title={!readiness?.ready ? readiness?.reason || "The transcription is not ready for this cover mode." : versionError ? versionError.message : undefined}
                type="button"
              >
                Use for the cover
              </button>
            </li>
          ))}
        </ul>
      ) : readiness?.ready ? null : (
        <p className="yue2-muted">No score version was imported for this mode.</p>
      )}
      {abc ? (
        <>
          <button className="secondary-action" disabled={preview.busy} onClick={inspect} type="button">
            Preview {mode === "melody" ? "melody-only" : "full"} score
          </button>
          <ErrorNotice error={preview.error} testId={`yue2-transcription-preview-error-${mode}`} />
          <ScorePreview inspection={preview.inspection} />
          <details className="yue2-inspect">
            <summary>ABC text</summary>
            <pre className="yue2-abc" data-testid={`yue2-transcription-abc-${mode}`}>{abc}</pre>
          </details>
        </>
      ) : (
        <p className="yue2-muted">No {mode === "melody" ? "melody-only" : "full"} score was built.</p>
      )}
    </div>
  );
}

function TranscriptionReview({ projectId, token, detail, versions, assetName, onEdit, onUseForCover, coverVersionId }) {
  const view = useMemo(() => yue2TranscriptionView(detail?.transcription), [detail]);
  const scores = detail?.scores ?? {};
  return (
    <div className="yue2-transcription-review" data-testid="yue2-transcription-review">
      <div className="yue2-inline">
        <strong>Review the transcription</strong>
        <PolicyChips policy={view.usagePolicy} />
      </div>
      <p className="yue2-muted" data-testid="yue2-transcription-source">
        {assetName(view.sourceAudioAssetId, view.sourceName)}
        {view.durationSeconds !== null ? ` · ${Math.round(view.durationSeconds)} s` : ""} · transcribed on{" "}
        <strong>{view.device || "an unrecorded device"}</strong>
        {view.createdAt ? ` · ${formatRelativeTime(view.createdAt)}` : ""}
      </p>

      {view.warnings.length ? (
        <ul className="yue2-notice yue2-notice--warn yue2-transcription-warnings" data-testid="yue2-transcription-warnings">
          {view.warnings.map((warning, index) => (
            <li key={`${warning.code}-${index}`}>
              <Icon.Warning size={14} /> {warning.code ? <code>{warning.code}</code> : null} {warning.message}
            </li>
          ))}
        </ul>
      ) : (
        <p className="yue2-muted" data-testid="yue2-transcription-warnings">
          The review raised no warnings.
        </p>
      )}
      {view.decodeWarnings.length ? (
        <ul className="yue2-notice yue2-notice--warn" data-testid="yue2-transcription-decode-warnings">
          {view.decodeWarnings.map((warning, index) => (
            <li key={`${warning.code}-${index}`}><Icon.Warning size={14} /> {warning.code ? <code>{warning.code}</code> : null} {warning.message}</li>
          ))}
        </ul>
      ) : null}
      <OctaveEvidence octave={view.octave} />

      <dl className="yue2-facts" data-testid="yue2-transcription-facts">
        <VoiceFacts label="Vocal melody" voice={view.vocal} />
        <VoiceFacts label="Instrumental melody" voice={view.instrumental} />
        <div>
          <dt>Key</dt>
          <dd>{view.keys.length ? view.keys.join(" → ") : "—"}</dd>
        </div>
        <div>
          <dt>Structure</dt>
          <dd>
            {view.bars ?? "—"} bars
            {view.sections.length ? ` · ${view.sections.join(" · ")}` : ""}
          </dd>
        </div>
      </dl>
      <p className="yue2-chords" data-testid="yue2-transcription-chords">
        <span className="eyebrow">Chords</span>{" "}
        {view.chords.length ? (
          <>
            {view.chords.map((chord) => (
              <span className="yue2-chord" key={chord}>
                {chord}
              </span>
            ))}
            {view.chordRoots !== null ? <span className="yue2-muted"> · {view.chordRoots} distinct roots</span> : null}
          </>
        ) : (
          <span className="yue2-muted">No chords were transcribed.</span>
        )}
      </p>
      {view.diagnostics.length ? (
        <div data-testid="yue2-transcription-diagnostics">
          <span className="eyebrow">Beat and meter notes</span>
          <ul className="yue2-run__warnings">
            {view.diagnostics.map((line, index) => (
              <li key={`${index}-${line}`}>{line}</li>
            ))}
          </ul>
        </div>
      ) : null}

      <div className="yue2-transcription-modes">
        {MODES.map(({ mode, label }) => (
          <ModeCard
            abc={scores[mode] ?? null}
            coverVersionId={coverVersionId}
            key={`${view.id}-${mode}`}
            label={label}
            mode={mode}
            onEdit={onEdit}
            onUseForCover={onUseForCover}
            token={token}
            versions={versions}
            view={view}
          />
        ))}
      </div>

      <ExportLinks groups={view.exportGroups} projectId={projectId} token={token} transcriptionId={view.id} />

      <details className="yue2-inspect">
        <summary>Transcription settings and provenance</summary>
        <pre data-testid="yue2-transcription-provenance">
          {JSON.stringify(
            {
              settings: view.settings,
              device: view.device,
              closure: view.closure,
              replay: view.replay,
              unload: view.unload,
              decodeWarnings: view.decodeWarnings,
              usagePolicy: view.usagePolicy,
            },
            null,
            2,
          )}
        </pre>
      </details>
    </div>
  );
}

// ---- the flow -------------------------------------------------------------------------------

export function Yue2RecordingCover({
  model,
  projectId,
  token,
  assets,
  importAsset,
  jobs,
  runs,
  versions,
  settings,
  update,
  submitting,
  onTranscribe,
  onCancel,
  onLicenseLapsed,
  requestedGpu,
  refreshKey = "",
}) {
  const setup = useMemo(() => yue2CoverSetup(model), [model]);
  const audioAssets = useMemo(() => (assets ?? []).filter(assetCanRenderAsAudio), [assets]);
  const assetName = (id, fallback = "") => {
    const asset = (assets ?? []).find((item) => item.id === id);
    return asset?.displayName || fallback || id || "recording";
  };

  // Transcriptions: server records, re-read whenever a run finishes.
  const [listing, setListing] = useState({ items: [], error: null, loading: true });
  const [readRetry, setReadRetry] = useState(0);
  useEffect(() => {
    let live = true;
    if (!projectId) return undefined;
    listYue2Transcriptions(projectId, token)
      .then((result) => {
        if (!live) return;
        const items = [...(result?.items ?? [])].sort((a, b) => Date.parse(b.createdAt ?? 0) - Date.parse(a.createdAt ?? 0));
        const unreadable = Array.isArray(result?.unreadable) ? result.unreadable : [];
        setListing({
          items,
          error: unreadable.length ? `Some transcriptions could not be read: ${unreadable.join(", ")}` : null,
          loading: false,
        });
      })
      .catch((error) => {
        if (live) setListing((current) => ({ ...current, error, loading: false }));
      });
    return () => {
      live = false;
    };
  }, [projectId, token, refreshKey, readRetry]);

  const selectedId = listing.loading
    ? settings.transcriptionId || ""
    : listing.items.some((item) => item.id === settings.transcriptionId)
      ? settings.transcriptionId
      : listing.items[0]?.id || "";
  const [detail, setDetail] = useState({ id: "", data: null, error: null });
  useEffect(() => {
    let live = true;
    if (!projectId || !selectedId) {
      setDetail({ id: "", data: null, error: null });
      return undefined;
    }
    getYue2Transcription(projectId, selectedId, token)
      .then((data) => {
        if (live) setDetail({ id: selectedId, data, error: null });
      })
      .catch((error) => {
        if (live) setDetail({ id: selectedId, data: null, error });
      });
    return () => {
      live = false;
    };
  }, [projectId, selectedId, token, refreshKey, readRetry]);

  // The newest transcription job, and a just-finished one opens its own review.
  const transcribeRuns = useMemo(() => runs.filter((job) => job.payload?.yue2?.kind === "transcribe"), [runs]);
  const latestRun = transcribeRuns[0] ?? null;
  const doneKey = transcribeRuns
    .filter((job) => job.status === "completed" && job.result?.yue2?.transcriptionId)
    .map((job) => job.id)
    .join(",");
  const seenDone = useRef(null);
  useEffect(() => {
    const done = transcribeRuns.filter((job) => job.status === "completed" && job.result?.yue2?.transcriptionId);
    if (seenDone.current === null) {
      seenDone.current = new Set(done.map((job) => job.id));
      return;
    }
    const fresh = done.find((job) => !seenDone.current.has(job.id));
    for (const job of done) seenDone.current.add(job.id);
    if (fresh) update({ transcriptionId: fresh.result.yue2.transcriptionId });
    // doneKey changes exactly when the set of finished transcriptions does.
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [doneKey]);

  const target = { sourceAudioAssetId: settings.transcribeAssetId };
  const problems = yue2RequestProblems("transcribe", settings, target, { hasProject: Boolean(projectId) });
  const blockedReason = !setup.installed ? "Set up covers first — the transcriber is not installed." : "";
  const canTranscribe = !submitting && !problems.length && !blockedReason;

  if (!setup.declared) {
    return (
      <div className="yue2-transcription" data-testid="yue2-recording-cover">
        <p>Recording transcription is not available in this build.</p>
      </div>
    );
  }

  return (
    <div className="yue2-recording" data-testid="yue2-recording-cover">
      <div className="yue2-inline">
        <span className="eyebrow">Cover from a recording</span>
        <NoncommercialChips license={setup.components.find((row) => row.license)?.license ?? ""} />
      </div>
      <ol className="yue2-steps">
        <li>Transcribe a recording into a score.</li>
        <li>Review the transcription, then review or edit the score version it imported.</li>
        <li>Cover that score version with your own style and lyrics.</li>
      </ol>

      {!setup.installed ? (
        <CoverSetup
          jobs={jobs}
          model={model}
          onLicenseLapsed={onLicenseLapsed}
          requestedGpu={requestedGpu}
          setup={setup}
          token={token}
        />
      ) : (
        <p className="yue2-muted" data-testid="yue2-cover-setup-installed">
          SheetSage2 and MERT-v2-FullSong are installed (noncommercial use only).
        </p>
      )}

      <div className="yue2-transcribe" data-testid="yue2-transcribe">
        <AssetPickerField
          assets={audioAssets}
          buttonLabel="Choose or upload a recording"
          changeLabel="Change"
          emptyLabel="No recording selected"
          importAsset={importAsset}
          label="Recording"
          mediaKind="audio"
          onChange={(id) => update({ transcribeAssetId: id })}
          showCategories={false}
          value={settings.transcribeAssetId}
        />
        <MicrophoneTake importAsset={importAsset} onImported={(id) => update({ transcribeAssetId: id })} />
        <details className="yue2-inspect" data-testid="yue2-transcribe-settings">
          <summary>Transcription settings</summary>
          <div className="yue2-grid">
            <label>
              Length limit (s)
              <input
                aria-label="Transcription length limit"
                min="0"
                onChange={(event) => update({ transcribeMaxSeconds: event.target.value })}
                placeholder="Whole recording"
                step="1"
                type="number"
                value={settings.transcribeMaxSeconds}
              />
            </label>
            <label>
              Window overlap (s)
              <input
                aria-label="Transcription window overlap"
                max="299"
                min="0"
                onChange={(event) => update({ transcribeOverlapSeconds: event.target.value })}
                placeholder={`${TRANSCRIPTION_DEFAULT_OVERLAP_SECONDS} (default)`}
                step="1"
                type="number"
                value={settings.transcribeOverlapSeconds}
              />
            </label>
            <label>
              Look-ahead (s)
              <input
                aria-label="Transcription look-ahead"
                min="0"
                onChange={(event) => update({ transcribeLookaheadSeconds: event.target.value })}
                placeholder={`${TRANSCRIPTION_DEFAULT_LOOKAHEAD_SECONDS} (default)`}
                step="1"
                type="number"
                value={settings.transcribeLookaheadSeconds}
              />
            </label>
          </div>
          <p className="yue2-muted">SheetSage2 reads 300-second windows: 0 ≤ look-ahead ≤ overlap &lt; 300.</p>
        </details>
        <div className="prompt-cta-stack yue2-cta">
          <button
            className="prompt-cta"
            disabled={!canTranscribe}
            onClick={() => onTranscribe(target)}
            title={blockedReason || problems.join(" ") || undefined}
            type="button"
          >
            <Icon.Audio size={14} /> Transcribe the recording
          </button>
          {blockedReason || problems.length ? (
            <span className="yue2-muted" data-testid="yue2-transcribe-problems">
              {[blockedReason, ...problems].filter(Boolean).join(" ")}
            </span>
          ) : null}
        </div>
        {latestRun ? (
          <TranscriptionJob
            assetName={assetName(latestRun.payload?.yue2?.sourceAudioAssetId)}
            job={latestRun}
            onCancel={onCancel}
            onReview={(transcriptionId) => update({ transcriptionId })}
          />
        ) : null}
      </div>

      {listing.error ? (
        <ErrorNotice error={listing.error} testId="yue2-transcriptions-error">
          <button className="secondary-action" onClick={() => setReadRetry((value) => value + 1)} type="button">Retry transcriptions</button>
        </ErrorNotice>
      ) : null}
      {listing.items.length > 1 ? (
        <label>
          Transcription
          <select
            aria-label="Transcription to review"
            onChange={(event) => update({ transcriptionId: event.target.value })}
            value={selectedId}
          >
            {listing.items.map((item) => (
              <option key={item.id} value={item.id}>
                {assetName(item.sourceAudioAssetId, item.source?.name)} · {formatRelativeTime(item.createdAt)} ·{" "}
                {item.device}
              </option>
            ))}
          </select>
        </label>
      ) : null}
      {selectedId && detail.error ? (
        <ErrorNotice error={detail.error} testId="yue2-transcription-error">
          <button className="secondary-action" onClick={() => setReadRetry((value) => value + 1)} type="button">Retry transcription</button>
        </ErrorNotice>
      ) : null}
      {detail.data && detail.id === selectedId ? (
        <TranscriptionReview
          assetName={assetName}
          coverVersionId={settings.coverSource === "version" ? settings.coverVersionId : ""}
          detail={detail.data}
          onEdit={(versionId) => update({ tab: "scores", selectedVersionId: versionId })}
          onUseForCover={(versionId, mode) => update({ coverSource: "version", coverVersionId: versionId, coverMode: mode })}
          projectId={projectId}
          token={token}
          versions={versions}
        />
      ) : !selectedId && listing.loading ? (
        <p className="yue2-muted">Loading transcriptions…</p>
      ) : !selectedId && !listing.error ? (
        <p className="yue2-muted">No transcriptions in this project yet.</p>
      ) : null}
    </div>
  );
}
