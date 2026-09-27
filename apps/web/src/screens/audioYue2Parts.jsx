import React, { useEffect, useMemo, useState } from "react";
import { Icon } from "../components/Icons.jsx";
import { assetUrl } from "../components/assetMedia.jsx";
import { AudioDownloadButton } from "../components/audioTakeParts.jsx";
import { jobAudioResultAssets } from "../jobResultAssets.js";
import { formatRelativeTime } from "../audioTakes.js";
import { percent } from "../formatting.js";
import {
  EDIT_OPERATIONS,
  UI_PROVENANCE,
  buildEditOperation,
  defaultEditDraft,
  usagePolicyChips,
  yue2ExportStem,
  yue2RunExport,
  yue2RunView,
} from "../yue2Lab.js";
import {
  createYue2Comparison,
  createYue2ScoreVersion,
  editYue2ScoreVersion,
  getYue2ScoreVersion,
  inspectYue2ScoreVersion,
} from "../api/yue2.js";

// YuE2 Song Lab parts (sc-23000): run cards, the score preview, the score-version workbench
// (history, bounded edits with their invariant reports, renders) and the A/B listening comparisons.
// Every server failure is rendered where it happened — never replaced with an empty state.

export function downloadText(text, filename, type = "application/json") {
  const blob = new Blob([text], { type });
  const url = URL.createObjectURL(blob);
  const anchor = document.createElement("a");
  anchor.href = url;
  anchor.download = filename;
  document.body.appendChild(anchor);
  anchor.click();
  anchor.remove();
  URL.revokeObjectURL(url);
}

export function ErrorNotice({ error, children = null, testId = "yue2-error" }) {
  if (!error) {
    return null;
  }
  const message = typeof error === "string" ? error : error.message || "The request failed.";
  const blocked = Array.isArray(error?.context?.blocked) ? error.context.blocked : [];
  const violations = Array.isArray(error?.context?.violations) ? error.context.violations : [];
  return (
    <div className="yue2-notice yue2-notice--error" data-testid={testId} role="alert">
      <Icon.Warning size={15} />
      <div className="yue2-notice__body">
        <p>{message}</p>
        {blocked.length ? (
          <ul className="yue2-notice__list">
            {blocked.map((row) => (
              <li key={row.componentId}>
                <strong>{row.componentId}</strong>: {row.reason}
                {row.unblock ? <span className="yue2-muted"> Unblocks when: {row.unblock}</span> : null}
              </li>
            ))}
          </ul>
        ) : null}
        {violations.length ? (
          <ul className="yue2-notice__list">
            {violations.map((violation) => (
              <li key={violation}>{violation}</li>
            ))}
          </ul>
        ) : null}
        {children}
      </div>
    </div>
  );
}

export function RegenerationNotice({ text }) {
  if (!text) {
    return null;
  }
  return (
    <p className="yue2-notice yue2-notice--info" data-testid="yue2-regeneration-notice">
      <Icon.Info size={15} />
      <span>{text}</span>
    </p>
  );
}

export function PolicyChips({ policy, model, className = "" }) {
  return (
    <span className={`yue2-chips ${className}`.trim()} data-testid="yue2-policy-chips">
      {usagePolicyChips(policy, model).map((chip) => (
        <span className={chip === "Noncommercial" || chip === "Experimental" ? "yue2-chip yue2-chip--warn" : "yue2-chip"} key={chip}>
          {chip}
        </span>
      ))}
    </span>
  );
}

// ---- score preview --------------------------------------------------------------------------

function fraction(value) {
  const [num, den] = String(value ?? "0").split("/").map(Number);
  return den ? num / den : num || 0;
}

export function ScorePreview({ inspection }) {
  if (!inspection) {
    return null;
  }
  const sections = Array.isArray(inspection.sections) ? inspection.sections : [];
  const totalBars = sections.reduce((sum, section) => sum + (section.barCount ?? 0), 0) || 1;
  const vocal = inspection.voices?.Vocal ?? {};
  const ins = inspection.voices?.Ins ?? {};
  const chords = Array.isArray(vocal.chords) ? vocal.chords : [];
  const seconds = Number(inspection.nominalDurationSeconds);
  return (
    <div className="yue2-score-preview" data-testid="yue2-score-preview">
      <dl className="yue2-facts">
        <div>
          <dt>Tempo</dt>
          <dd>{inspection.bpm} BPM</dd>
        </div>
        <div>
          <dt>Length</dt>
          <dd>
            {fraction(inspection.durationQuarters)} quarters
            {Number.isFinite(seconds) ? ` · ~${Math.round(seconds)} s` : ""}
          </dd>
        </div>
        <div>
          <dt>Vocal</dt>
          <dd>
            {vocal.soundingNotes ?? 0} notes · {vocal.measures ?? 0} bars
          </dd>
        </div>
        <div>
          <dt>Instrumental</dt>
          <dd>
            {ins.soundingNotes ?? 0} notes · {ins.measures ?? 0} bars
          </dd>
        </div>
      </dl>
      <div className="yue2-sections" aria-label="Sections">
        {sections.map((section) => (
          <span
            className="yue2-section"
            key={section.index}
            style={{ flexGrow: section.barCount ?? 1, flexBasis: `${((section.barCount ?? 1) / totalBars) * 100}%` }}
            title={`Section ${section.index}: bars ${section.firstBar}–${section.firstBar + (section.barCount ?? 1) - 1}`}
          >
            <strong>{section.index}</strong> {section.label || "untitled"}
            <span className="yue2-muted"> · {section.barCount} bars</span>
          </span>
        ))}
      </div>
      {chords.length ? (
        <p className="yue2-chords">
          <span className="eyebrow">Chords</span>{" "}
          {chords.slice(0, 32).map((chord, index) => (
            <span className="yue2-chord" key={`${chord.onsetQuarters}-${index}`}>
              {chord.chord}
            </span>
          ))}
          {chords.length > 32 ? <span className="yue2-muted"> +{chords.length - 32} more</span> : null}
        </p>
      ) : (
        <p className="yue2-muted">No chord symbols (melody-only score).</p>
      )}
    </div>
  );
}

export function InvariantReport({ report }) {
  if (!report) {
    return null;
  }
  const checks = Array.isArray(report.checks) ? report.checks : [];
  return (
    <div className="yue2-invariants" data-testid="yue2-invariant-report">
      <p className={report.match ? "yue2-invariants__verdict is-ok" : "yue2-invariants__verdict is-bad"}>
        {report.match ? "Invariants hold: only what the edit declares changed." : "The edit changes what its contract fixes."}
      </p>
      <ul>
        {checks.map((check) => (
          <li className={`yue2-check yue2-check--${check.status}`} key={check.name}>
            <strong>{check.name}</strong> — {check.status}
            {check.detail ? <span className="yue2-muted"> ({check.detail})</span> : null}
          </li>
        ))}
      </ul>
      {Array.isArray(report.violations) && report.violations.length ? (
        <ul className="yue2-notice__list">
          {report.violations.map((violation) => (
            <li key={violation}>{violation}</li>
          ))}
        </ul>
      ) : null}
    </div>
  );
}

// ---- runs -----------------------------------------------------------------------------------

export function Yue2RunCard({
  job,
  assets,
  model,
  onCancel,
  onOpenVersion,
  onRestorePlan,
  onDecodeAgain,
  onFetchVersionAbc,
}) {
  const view = useMemo(() => yue2RunView(job), [job]);
  const takes = useMemo(() => jobAudioResultAssets(job, assets), [job, assets]);
  const [exportError, setExportError] = useState("");

  async function exportScore() {
    setExportError("");
    try {
      let abc = view.score?.abc ?? null;
      if (!abc && view.scoreVersionId && onFetchVersionAbc) {
        abc = await onFetchVersionAbc(view.scoreVersionId);
      }
      if (!abc) {
        setExportError("This run recorded no score to export.");
        return;
      }
      downloadText(abc, `${yue2ExportStem("yue2-score", view.id, view.usagePolicy)}.abc`, "text/vnd.abc");
    } catch (error) {
      setExportError(error?.message || "The score could not be read.");
    }
  }

  function exportRun() {
    downloadText(
      JSON.stringify(yue2RunExport(job, takes), null, 2),
      `${yue2ExportStem("yue2-run", view.id, view.usagePolicy)}.json`,
    );
  }

  const hasScore = Boolean(view.score?.abc || view.scoreVersionId);
  return (
    <article className={`yue2-run yue2-run--${view.status}`} data-testid="yue2-run-card" data-job-id={view.id}>
      <header className="yue2-run__head">
        <span className="audio-mode-chip audio-mode-chip--yue2">{view.kindLabel}</span>
        <span className={`status-badge ${view.status}`}>{view.cancelRequested && view.running ? "canceling" : view.status}</span>
        <PolicyChips model={model} policy={view.usagePolicy} />
        {view.batch?.count > 1 ? (
          <span className="yue2-muted">
            take {view.batch.index + 1} of {view.batch.count}
          </span>
        ) : null}
        <span className="yue2-run__time">{view.createdAt ? formatRelativeTime(view.createdAt) : ""}</span>
      </header>
      {view.style ? <p className="yue2-run__style">{view.style}</p> : null}

      {view.running ? (
        <div className="yue2-run__progress" data-testid="yue2-run-progress">
          <span className="yue2-run__stage">{view.message || "Queued — waiting for a worker."}</span>
          <div aria-label="Run progress" className="progress-track">
            <span style={{ width: percent(view.progress) }} />
          </div>
          <button
            className="secondary-action"
            disabled={view.cancelRequested}
            onClick={() => onCancel?.(job)}
            type="button"
          >
            {view.cancelRequested ? "Canceling…" : "Cancel"}
          </button>
        </div>
      ) : null}

      {view.error ? <ErrorNotice error={view.error} testId="yue2-run-error" /> : null}

      {view.truncations.map((truncation) => (
        <p className="yue2-notice yue2-notice--warn" data-testid="yue2-run-truncated" key={truncation.key}>
          <Icon.Warning size={15} />
          <span>
            <strong>Truncated.</strong> {truncation.text}
          </span>
        </p>
      ))}
      {view.warnings.length ? (
        <ul className="yue2-run__warnings" data-testid="yue2-run-warnings">
          {view.warnings.map((warning, index) => (
            <li key={`${warning.code}-${index}`}>
              {warning.code ? <code>{warning.code}</code> : null} {warning.message}
            </li>
          ))}
        </ul>
      ) : null}
      {view.scoreVersionSkipped ? (
        <p className="yue2-notice yue2-notice--warn" data-testid="yue2-run-version-skipped">
          <Icon.Warning size={15} />
          <span>{view.scoreVersionSkipped}</span>
        </p>
      ) : null}
      {view.sideEffectErrors.length ? (
        <ErrorNotice
          error={`The run finished, but recording it failed: ${view.sideEffectErrors.join("; ")}`}
          testId="yue2-run-side-effect-error"
        />
      ) : null}

      {takes.length ? (
        <div className="yue2-run__takes">
          {takes.map((asset) => (
            <div className="yue2-take" key={asset.id}>
              <audio controls preload="none" src={assetUrl(asset)} aria-label={`Play ${asset.displayName ?? "take"}`} />
              <AudioDownloadButton asset={asset} className="secondary-action" iconSize={14} label="Audio" />
            </div>
          ))}
        </div>
      ) : null}

      {!view.running && view.status === "completed" ? (
        <div className="yue2-run__actions">
          {hasScore ? (
            <button className="secondary-action" onClick={exportScore} type="button">
              <Icon.Download size={14} /> Score (.abc)
            </button>
          ) : null}
          <button className="secondary-action" onClick={exportRun} type="button">
            <Icon.Download size={14} /> Run record (.json)
          </button>
          {view.scoreVersionId ? (
            <button className="secondary-action" onClick={() => onOpenVersion?.(view.scoreVersionId)} type="button">
              Open score version
            </button>
          ) : null}
          {view.restorable ? (
            <button className="secondary-action" onClick={() => onRestorePlan?.(view.id)} type="button">
              Restore this plan
            </button>
          ) : null}
          {view.decodable ? (
            <button className="secondary-action" onClick={() => onDecodeAgain?.(view.id)} type="button">
              Decode again
            </button>
          ) : null}
        </div>
      ) : null}
      {exportError ? <ErrorNotice error={exportError} testId="yue2-export-error" /> : null}

      {view.effectiveSettings || view.model || view.run ? (
        <details className="yue2-inspect">
          <summary>Inspect settings and provenance</summary>
          <pre data-testid="yue2-effective-settings">
            {JSON.stringify(
              {
                effectiveSettings: view.effectiveSettings,
                model: view.model,
                decoder: view.decoder,
                run: view.run,
                usagePolicy: view.usagePolicy,
              },
              null,
              2,
            )}
          </pre>
        </details>
      ) : null}
    </article>
  );
}

// ---- score versions -------------------------------------------------------------------------

function VersionRow({ version, selected, onSelect }) {
  return (
    <li>
      <button
        className={selected ? "yue2-version is-selected" : "yue2-version"}
        data-testid="yue2-version-row"
        onClick={() => onSelect(version.id)}
        type="button"
      >
        <span className="yue2-version__origin">{version.origin}</span>
        <span className="yue2-version__op">{version.editOperation ?? "root"}</span>
        <span className="yue2-version__brief">{version.editBrief ?? `${version.cot} score`}</span>
        <span className="yue2-muted">
          {version.renderCount} {version.renderCount === 1 ? "render" : "renders"} ·{" "}
          {formatRelativeTime(version.createdAt)}
        </span>
      </button>
    </li>
  );
}

function EditFields({ draft, setDraft, sections }) {
  const set = (patch) => setDraft((current) => ({ ...current, ...patch }));
  switch (draft.op) {
    case "reharmonize":
      return (
        <div className="yue2-edit-fields">
          {draft.chordChanges.map((row, index) => (
            <div className="yue2-chord-change" key={index}>
              <label>
                Bar
                <input
                  aria-label={`Chord change ${index + 1} bar`}
                  min="1"
                  onChange={(event) =>
                    set({
                      chordChanges: draft.chordChanges.map((item, i) =>
                        i === index ? { ...item, bar: event.target.value } : item,
                      ),
                    })
                  }
                  type="number"
                  value={row.bar}
                />
              </label>
              <label>
                Onset (quarters)
                <input
                  aria-label={`Chord change ${index + 1} onset`}
                  onChange={(event) =>
                    set({
                      chordChanges: draft.chordChanges.map((item, i) =>
                        i === index ? { ...item, onsetQuarters: event.target.value } : item,
                      ),
                    })
                  }
                  placeholder="0 or 3/2"
                  value={row.onsetQuarters}
                />
              </label>
              <label>
                Chord
                <input
                  aria-label={`Chord change ${index + 1} chord`}
                  onChange={(event) =>
                    set({
                      chordChanges: draft.chordChanges.map((item, i) =>
                        i === index ? { ...item, chord: event.target.value } : item,
                      ),
                    })
                  }
                  placeholder="Am7 (empty removes)"
                  value={row.chord}
                />
              </label>
            </div>
          ))}
          <button
            className="secondary-action"
            onClick={() => set({ chordChanges: [...draft.chordChanges, { bar: "", onsetQuarters: "", chord: "" }] })}
            type="button"
          >
            <Icon.Plus size={13} /> Add chord change
          </button>
        </div>
      );
    case "strip_chords":
      return (
        <label className="yue2-edit-fields">
          Keep voices
          <select onChange={(event) => set({ keepVoice: event.target.value })} value={draft.keepVoice}>
            <option value="both">Both voices</option>
            <option value="Vocal">Vocal only</option>
            <option value="Ins">Instrumental only</option>
          </select>
        </label>
      );
    case "set_tempo":
      return (
        <div className="yue2-edit-fields">
          <label>
            Tempo (BPM)
            <input min="1" onChange={(event) => set({ bpm: event.target.value })} type="number" value={draft.bpm} />
          </label>
          <label>
            Style (optional)
            <input onChange={(event) => set({ style: event.target.value })} value={draft.style} />
          </label>
        </div>
      );
    case "arrange_sections":
      return (
        <div className="yue2-edit-fields">
          <label>
            Section order
            <input
              onChange={(event) => set({ sectionOrder: event.target.value })}
              placeholder={sections.length ? sections.map((section) => section.index).join(", ") : "0, 1, 1, 2"}
              value={draft.sectionOrder}
            />
          </label>
          <label>
            Lyrics for the new form
            <textarea onChange={(event) => set({ lyrics: event.target.value })} rows={4} value={draft.lyrics} />
          </label>
          <label>
            Style (optional)
            <input onChange={(event) => set({ style: event.target.value })} value={draft.style} />
          </label>
        </div>
      );
    case "set_lyrics":
      return (
        <label className="yue2-edit-fields">
          New lyrics
          <textarea onChange={(event) => set({ lyrics: event.target.value })} rows={5} value={draft.lyrics} />
        </label>
      );
    case "set_style":
      return (
        <label className="yue2-edit-fields">
          New style
          <input onChange={(event) => set({ style: event.target.value })} value={draft.style} />
        </label>
      );
    case "replace_score":
      return (
        <div className="yue2-edit-fields">
          <label>
            Edited ABC score
            <textarea
              className="yue2-abc"
              onChange={(event) => set({ abc: event.target.value })}
              rows={8}
              spellCheck={false}
              value={draft.abc}
            />
          </label>
          <fieldset className="yue2-fieldset">
            <legend>Declared relaxations</legend>
            <label className="checkline">
              <input checked={draft.allowHarmony} onChange={(event) => set({ allowHarmony: event.target.checked })} type="checkbox" />
              Harmony may change
            </label>
            <label className="checkline">
              <input checked={draft.allowTempo} onChange={(event) => set({ allowTempo: event.target.checked })} type="checkbox" />
              Tempo may change
            </label>
            <label className="checkline">
              <input checked={draft.allowMelody} onChange={(event) => set({ allowMelody: event.target.checked })} type="checkbox" />
              Melody may change in bars
            </label>
            {draft.allowMelody ? (
              <div className="yue2-inline">
                <select aria-label="Melody voices" onChange={(event) => set({ melodyVoices: event.target.value })} value={draft.melodyVoices}>
                  <option value="vocal">Vocal</option>
                  <option value="ins">Instrumental</option>
                  <option value="both">Both</option>
                </select>
                <input aria-label="From bar" min="1" onChange={(event) => set({ melodyFromBar: event.target.value })} type="number" value={draft.melodyFromBar} />
                <input aria-label="To bar" min="1" onChange={(event) => set({ melodyToBar: event.target.value })} type="number" value={draft.melodyToBar} />
              </div>
            ) : null}
          </fieldset>
          <label>
            Lyrics (optional)
            <textarea onChange={(event) => set({ lyrics: event.target.value })} rows={3} value={draft.lyrics} />
          </label>
          <label>
            Style (optional)
            <input onChange={(event) => set({ style: event.target.value })} value={draft.style} />
          </label>
          <label>
            Planning mode
            <select onChange={(event) => set({ cot: event.target.value })} value={draft.cot}>
              <option value="">Keep</option>
              <option value="full">Full (melody + harmony)</option>
              <option value="melody">Melody only</option>
            </select>
          </label>
        </div>
      );
    default:
      return null;
  }
}

/**
 * The score-version workbench: version history (write-once versions with parent/root links), a
 * version's score preview, request and renders, and the bounded edit operations with their
 * invariant report. A saved edit is a NEW version; rendering it regenerates the whole recording.
 */
export function Yue2ScoreWorkbench({
  projectId,
  token,
  versions,
  versionsError,
  onReloadVersions,
  selectedVersionId,
  onSelectVersion,
  onRender,
  onUseForCover,
  renderProblem,
  assets,
  regenerationNotice,
  seedSettings,
}) {
  const [detail, setDetail] = useState(null);
  const [inspection, setInspection] = useState(null);
  const [detailError, setDetailError] = useState(null);
  const [draft, setDraft] = useState(defaultEditDraft);
  const [editResult, setEditResult] = useState(null);
  const [editError, setEditError] = useState(null);
  const [editBusy, setEditBusy] = useState(false);
  const [importDraft, setImportDraft] = useState({ open: false, abc: "", cot: "full", seed: "", cfgScale: "" });
  const [importError, setImportError] = useState(null);

  useEffect(() => {
    let live = true;
    setDetail(null);
    setInspection(null);
    setDetailError(null);
    setEditResult(null);
    setEditError(null);
    if (!selectedVersionId) {
      return undefined;
    }
    Promise.all([
      getYue2ScoreVersion(projectId, selectedVersionId, token),
      inspectYue2ScoreVersion(projectId, selectedVersionId, token),
    ])
      .then(([nextDetail, nextInspection]) => {
        if (live) {
          setDetail(nextDetail);
          setInspection(nextInspection);
        }
      })
      .catch((error) => {
        if (live) setDetailError(error);
      });
    return () => {
      live = false;
    };
  }, [projectId, selectedVersionId, token]);

  async function runEdit(dryRun) {
    const built = buildEditOperation(draft);
    if (built.error) {
      setEditError(built.error);
      return;
    }
    if (!draft.brief.trim()) {
      setEditError("Say what the edit changes and why (the brief is recorded on the version).");
      return;
    }
    setEditBusy(true);
    setEditError(null);
    setEditResult(null);
    try {
      const response = await editYue2ScoreVersion(
        projectId,
        selectedVersionId,
        { operation: built.operation, brief: draft.brief.trim(), provenance: UI_PROVENANCE, dryRun },
        token,
      );
      setEditResult(response);
      if (!dryRun && response?.version?.id) {
        await onReloadVersions?.();
        onSelectVersion(response.version.id);
        setDraft(defaultEditDraft());
      }
    } catch (error) {
      setEditError(error);
      if (error?.code === "yue2_invariant_violation" && error.context) {
        setEditResult({ dryRun, rejectedReport: error.context });
      }
    } finally {
      setEditBusy(false);
    }
  }

  async function importVersion() {
    setImportError(null);
    if (!importDraft.abc.trim() || !seedSettings.lyrics.trim()) {
      setImportError("Paste the ABC score and write the lyrics (the Compose lyrics are used).");
      return;
    }
    const request = {
      style: seedSettings.style.trim(),
      lyrics: seedSettings.lyrics.trim(),
      cot: importDraft.cot,
    };
    if (importDraft.seed !== "") request.seed = Math.trunc(Number(importDraft.seed));
    if (importDraft.cfgScale !== "") request.cfgScale = Number(importDraft.cfgScale);
    try {
      const record = await createYue2ScoreVersion(
        projectId,
        { abc: importDraft.abc, request, origin: "import", provenance: UI_PROVENANCE },
        token,
      );
      setImportDraft({ open: false, abc: "", cot: "full", seed: "", cfgScale: "" });
      await onReloadVersions?.();
      onSelectVersion(record.id);
    } catch (error) {
      setImportError(error);
    }
  }

  const version = detail?.version ?? null;
  const renders = Array.isArray(detail?.renders) ? detail.renders : [];
  const report = editResult?.version?.edit?.invariants ?? editResult?.rejectedReport ?? null;
  const assetById = useMemo(() => new Map((assets ?? []).map((asset) => [asset.id, asset])), [assets]);

  return (
    <div className="yue2-workbench" data-testid="yue2-score-workbench">
      <aside className="yue2-workbench__list">
        <div className="yue2-workbench__list-head">
          <span className="eyebrow">Score versions</span>
          <button className="audio-link" onClick={() => setImportDraft((d) => ({ ...d, open: !d.open }))} type="button">
            <Icon.Plus size={13} /> Import ABC
          </button>
        </div>
        {importDraft.open ? (
          <div className="yue2-import" data-testid="yue2-import-version">
            <textarea
              aria-label="ABC score to import"
              className="yue2-abc"
              onChange={(event) => setImportDraft((d) => ({ ...d, abc: event.target.value }))}
              rows={6}
              spellCheck={false}
              value={importDraft.abc}
            />
            <div className="yue2-inline">
              <select aria-label="Import planning mode" onChange={(event) => setImportDraft((d) => ({ ...d, cot: event.target.value }))} value={importDraft.cot}>
                <option value="full">Full</option>
                <option value="melody">Melody</option>
              </select>
              <input aria-label="Import seed" onChange={(event) => setImportDraft((d) => ({ ...d, seed: event.target.value }))} placeholder="Seed" type="number" value={importDraft.seed} />
              <input aria-label="Import guidance" onChange={(event) => setImportDraft((d) => ({ ...d, cfgScale: event.target.value }))} placeholder="Guidance" step="0.1" type="number" value={importDraft.cfgScale} />
              <button className="secondary-action" onClick={importVersion} type="button">
                Save version
              </button>
            </div>
            <ErrorNotice error={importError} testId="yue2-import-error" />
          </div>
        ) : null}
        <ErrorNotice error={versionsError} testId="yue2-versions-error" />
        {!versionsError && versions.length === 0 ? (
          <p className="yue2-muted">No score versions yet. A plan (or a planned song) becomes one when it finishes.</p>
        ) : null}
        <ul className="yue2-version-list">
          {versions.map((item) => (
            <VersionRow key={item.id} onSelect={onSelectVersion} selected={item.id === selectedVersionId} version={item} />
          ))}
        </ul>
      </aside>

      <section className="yue2-workbench__detail">
        {!selectedVersionId ? <p className="yue2-muted">Choose a score version to preview, edit or render it.</p> : null}
        <ErrorNotice error={detailError} testId="yue2-version-error" />
        {version ? (
          <>
            <header className="yue2-version-head">
              <strong>{version.id}</strong>
              <span className="yue2-muted">
                {version.origin}
                {version.parentVersionId ? ` · from ${version.parentVersionId}` : " · root"}
                {version.rootVersionId !== version.id ? ` · root ${version.rootVersionId}` : ""}
              </span>
            </header>
            <ScorePreview inspection={inspection} />
            <dl className="yue2-facts">
              <div>
                <dt>Style</dt>
                <dd>{version.request?.style || "—"}</dd>
              </div>
              <div>
                <dt>Planning</dt>
                <dd>{version.request?.cot}</dd>
              </div>
              <div>
                <dt>Seed</dt>
                <dd>{version.request?.seed}</dd>
              </div>
              <div>
                <dt>Guidance</dt>
                <dd>{version.request?.cfgScale ?? "model default"}</dd>
              </div>
            </dl>
            <details className="yue2-inspect">
              <summary>ABC score and lyrics</summary>
              <pre className="yue2-abc" data-testid="yue2-version-abc">{version.score?.abc}</pre>
              <pre>{version.request?.lyrics}</pre>
            </details>
            <RegenerationNotice text={version.renderNotice || regenerationNotice} />
            <div className="yue2-run__actions">
              <button
                className="prompt-cta"
                disabled={Boolean(renderProblem)}
                onClick={() => onRender(version.id)}
                title={renderProblem || "Regenerates the whole recording from this version's score."}
                type="button"
              >
                <Icon.Sparkle size={14} /> Render this version
              </button>
              <button className="secondary-action" onClick={() => onUseForCover(version.id)} type="button">
                Use as cover score
              </button>
              <button
                className="secondary-action"
                onClick={() =>
                  downloadText(version.score?.abc ?? "", `yue2-score-${version.id}-noncommercial.abc`, "text/vnd.abc")
                }
                type="button"
              >
                <Icon.Download size={14} /> Score (.abc)
              </button>
            </div>

            <div className="yue2-renders" data-testid="yue2-version-renders">
              <span className="eyebrow">Renders</span>
              {renders.length === 0 ? <p className="yue2-muted">Not rendered yet.</p> : null}
              {renders.map((render) => {
                const asset = render.audioAssetId ? assetById.get(render.audioAssetId) : null;
                return (
                  <div className="yue2-render" key={render.id}>
                    <span className={`status-badge ${render.status}`}>{render.status}</span>
                    {asset ? <audio controls preload="none" src={assetUrl(asset)} /> : null}
                    {render.truncated?.abc || render.truncated?.semantic ? (
                      <span className="status-badge warning">truncated</span>
                    ) : null}
                    {render.error ? <span className="yue2-error-text">{render.error}</span> : null}
                  </div>
                );
              })}
            </div>

            <div className="yue2-edit" data-testid="yue2-edit-panel">
              <span className="eyebrow">Edit the score</span>
              <label>
                Operation
                <select
                  aria-label="Edit operation"
                  onChange={(event) => setDraft((current) => ({ ...current, op: event.target.value }))}
                  value={draft.op}
                >
                  {EDIT_OPERATIONS.map((operation) => (
                    <option key={operation.op} value={operation.op}>
                      {operation.label}
                    </option>
                  ))}
                </select>
              </label>
              <EditFields draft={draft} sections={inspection?.sections ?? []} setDraft={setDraft} />
              <label>
                Brief (what changes and why)
                <input
                  aria-label="Edit brief"
                  onChange={(event) => setDraft((current) => ({ ...current, brief: event.target.value }))}
                  value={draft.brief}
                />
              </label>
              <div className="yue2-run__actions">
                <button className="secondary-action" disabled={editBusy} onClick={() => runEdit(true)} type="button">
                  Check edit
                </button>
                <button className="secondary-action strong" disabled={editBusy} onClick={() => runEdit(false)} type="button">
                  Save as new version
                </button>
              </div>
              <ErrorNotice error={editError} testId="yue2-edit-error" />
              <InvariantReport report={report} />
              {editResult?.renderNotice ? <RegenerationNotice text={editResult.renderNotice} /> : null}
            </div>
          </>
        ) : null}
      </section>
    </div>
  );
}

// ---- comparisons ----------------------------------------------------------------------------

function ComparisonSide({ label, side, asset }) {
  return (
    <div className="yue2-compare__side" data-testid={`yue2-compare-side-${label}`}>
      <span className="eyebrow">{label}</span>
      <strong>{side?.versionId}</strong>
      <span className="yue2-muted">
        {side?.origin}
        {side?.editOperation ? ` · ${side.editOperation}` : ""}
        {side?.editBrief ? ` — ${side.editBrief}` : ""}
      </span>
      {asset ? (
        <audio controls preload="none" src={assetUrl(asset)} aria-label={`Listen to ${label}`} />
      ) : (
        <span className="yue2-muted">No render to listen to.</span>
      )}
      {side?.render?.truncated?.abc || side?.render?.truncated?.semantic ? (
        <span className="status-badge warning">truncated</span>
      ) : null}
    </div>
  );
}

/** A/B listening comparisons between two score versions (and, optionally, specific renders). */
export function Yue2CompareWorkbench({
  projectId,
  token,
  versions,
  settings,
  update,
  comparisons,
  comparisonsError,
  onReloadComparisons,
  assets,
}) {
  const [rendersBy, setRendersBy] = useState({});
  const [error, setError] = useState(null);
  const [busy, setBusy] = useState(false);
  const [selectedId, setSelectedId] = useState("");
  const assetById = useMemo(() => new Map((assets ?? []).map((asset) => [asset.id, asset])), [assets]);

  useEffect(() => {
    let live = true;
    for (const versionId of [settings.compareA, settings.compareB]) {
      if (!versionId || rendersBy[versionId]) continue;
      getYue2ScoreVersion(projectId, versionId, token)
        .then((detail) => {
          if (live) setRendersBy((current) => ({ ...current, [versionId]: detail?.renders ?? [] }));
        })
        .catch((err) => {
          if (live) setError(err);
        });
    }
    return () => {
      live = false;
    };
    // rendersBy is a cache keyed by version; re-reading it here would refetch on every write.
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [projectId, token, settings.compareA, settings.compareB]);

  async function compare() {
    setError(null);
    if (!settings.compareA || !settings.compareB) {
      setError("Choose the two versions to compare.");
      return;
    }
    setBusy(true);
    try {
      const body = {
        versionA: settings.compareA,
        versionB: settings.compareB,
        provenance: UI_PROVENANCE,
      };
      if (settings.compareRenderA) body.renderA = settings.compareRenderA;
      if (settings.compareRenderB) body.renderB = settings.compareRenderB;
      if (settings.compareNotes.trim()) body.notes = settings.compareNotes.trim();
      const record = await createYue2Comparison(projectId, body, token);
      await onReloadComparisons?.();
      setSelectedId(record.id);
    } catch (err) {
      setError(err);
    } finally {
      setBusy(false);
    }
  }

  const selected = comparisons.find((item) => item.id === selectedId) ?? comparisons[0] ?? null;
  const renderOptions = (versionId) =>
    (rendersBy[versionId] ?? []).filter((render) => render.status === "completed" && render.audioAssetId);

  return (
    <div className="yue2-compare" data-testid="yue2-compare">
      <div className="yue2-compare__form">
        {["A", "B"].map((side) => {
          const versionKey = side === "A" ? "compareA" : "compareB";
          const renderKey = side === "A" ? "compareRenderA" : "compareRenderB";
          return (
            <div className="yue2-compare__pick" key={side}>
              <label>
                Version {side}
                <select
                  aria-label={`Version ${side}`}
                  onChange={(event) => update({ [versionKey]: event.target.value, [renderKey]: "" })}
                  value={settings[versionKey]}
                >
                  <option value="">Choose…</option>
                  {versions.map((version) => (
                    <option key={version.id} value={version.id}>
                      {version.id} · {version.editOperation ?? version.origin}
                    </option>
                  ))}
                </select>
              </label>
              <label>
                Render {side}
                <select
                  aria-label={`Render ${side}`}
                  onChange={(event) => update({ [renderKey]: event.target.value })}
                  value={settings[renderKey]}
                >
                  <option value="">Latest completed</option>
                  {renderOptions(settings[versionKey]).map((render) => (
                    <option key={render.id} value={render.id}>
                      {render.id}
                    </option>
                  ))}
                </select>
              </label>
            </div>
          );
        })}
        <label className="yue2-compare__notes">
          Listening notes
          <input onChange={(event) => update({ compareNotes: event.target.value })} value={settings.compareNotes} />
        </label>
        <button className="secondary-action strong" disabled={busy} onClick={compare} type="button">
          Compare A / B
        </button>
      </div>
      <ErrorNotice error={error} testId="yue2-compare-error" />
      <ErrorNotice error={comparisonsError} testId="yue2-comparisons-error" />
      {comparisons.length > 1 ? (
        <select aria-label="Saved comparisons" onChange={(event) => setSelectedId(event.target.value)} value={selected?.id ?? ""}>
          {comparisons.map((item) => (
            <option key={item.id} value={item.id}>
              {item.a?.versionId} ↔ {item.b?.versionId} · {formatRelativeTime(item.createdAt)}
            </option>
          ))}
        </select>
      ) : null}
      {selected ? (
        <div className="yue2-compare__result" data-testid="yue2-comparison">
          <div className="yue2-compare__sides">
            <ComparisonSide asset={assetById.get(selected.a?.render?.audioAssetId)} label="A" side={selected.a} />
            <ComparisonSide asset={assetById.get(selected.b?.render?.audioAssetId)} label="B" side={selected.b} />
          </div>
          <p className="yue2-muted">
            Lineage: {String(selected.lineage).replace(/_/g, " ")}
            {selected.identicalMusicAndRequest ? " · identical score and request" : ""}
          </p>
          {Array.isArray(selected.symbolicDifferences) && selected.symbolicDifferences.length ? (
            <ul className="yue2-run__warnings">
              {selected.symbolicDifferences.map((difference) => (
                <li key={difference}>{difference}</li>
              ))}
            </ul>
          ) : null}
          {Array.isArray(selected.warnings) && selected.warnings.length ? (
            <ul className="yue2-run__warnings">
              {selected.warnings.map((warning) => (
                <li key={warning}>{warning}</li>
              ))}
            </ul>
          ) : null}
          {selected.notes ? <p>{selected.notes}</p> : null}
          <RegenerationNotice text={selected.renderNotice} />
        </div>
      ) : !comparisonsError ? (
        <p className="yue2-muted">No comparisons yet. Pick two versions and compare their complete recordings.</p>
      ) : null}
    </div>
  );
}
