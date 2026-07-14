import React, { useEffect, useMemo, useRef, useState } from "react";
import { API_BASE_URL, withMediaTicket } from "../api.js";
import { AssetPickerField } from "../components/AssetPicker.jsx";
import { AssetMedia } from "../components/assetMedia.jsx";

const MASK_STATE_COPY = {
  active: "Per-frame segmentation masks generated for tracked frames.",
  generated: "Per-frame segmentation masks generated for most tracked frames.",
  degraded: "Box-derived masks (segmentation backend unavailable on the worker).",
  missing: "No masks yet — re-run tracking or correct the track.",
  deferred: "Procedural preview track — not real tracking output.",
};

const BOX_FIELDS = [
  ["x", "X"],
  ["y", "Y"],
  ["width", "W"],
  ["height", "H"],
];

function maskStateCopy(track) {
  const state = track?.status?.maskState;
  return MASK_STATE_COPY[state] ?? "Tracked boxes are stored in the track sidecar.";
}

function clamp01(value) {
  if (!Number.isFinite(value)) {
    return 0;
  }
  return Math.max(0, Math.min(1, value));
}

function roundComponent(value) {
  return Math.round(clamp01(value) * 10000) / 10000;
}

function normalizeBox(box) {
  return {
    x: roundComponent(box?.x ?? 0),
    y: roundComponent(box?.y ?? 0),
    width: roundComponent(box?.width ?? 0),
    height: roundComponent(box?.height ?? 0),
  };
}

function boxesEqual(a, b) {
  if (!a || !b) {
    return false;
  }
  return BOX_FIELDS.every(([key]) => Math.abs((a[key] ?? 0) - (b[key] ?? 0)) < 1e-4);
}

// The MEANINGFUL corrections in a drafts working set: only frames whose box
// drifted from the tracked box or that are rejected carry intent worth
// persisting. This is the single source of truth for "what would this drafts set
// save", shared by the render (Save/dirty), the seed effect's dirty measure, and
// the post-save re-baseline (sc-11966). A NO-OP touched draft (e.g. reject
// toggled on then off) collapses out here, so all three agree it is clean.
function pendingCorrectionsFromDrafts(drafts, frames) {
  return Object.entries(drafts ?? {})
    .map(([key, entry]) => {
      const index = Number(key);
      const original = frames[index]?.box;
      const boxChanged = entry?.box && original && !boxesEqual(entry.box, original);
      const isRejected = Boolean(entry?.rejected);
      if (!boxChanged && !isRejected) {
        return null;
      }
      const correction = { frameIndex: index, rejected: isRejected, author: "ui", source: "manual" };
      if (boxChanged) {
        correction.box = normalizeBox(entry.box);
      }
      return correction;
    })
    .filter(Boolean)
    .sort((a, b) => a.frameIndex - b.frameIndex);
}

// Stable comparable form of a correction (ignores stamped author/createdAt/source)
// so equality checks only fire on frameIndex/box/rejected changes.
function comparableCorrection(correction) {
  return JSON.stringify({
    frameIndex: correction.frameIndex,
    box: correction.box ? normalizeBox(correction.box) : null,
    rejected: Boolean(correction.rejected),
  });
}

// Signature of the MEANINGFUL corrections in a drafts working set. The seed
// effect and the post-save re-baseline measure dirtiness against this — NOT the
// raw drafts JSON — so a no-op touched draft reads clean and an external
// correction on the same track is reseeded/shown instead of hidden (sc-11966).
function meaningfulDraftsSignature(drafts, frames) {
  return JSON.stringify(pendingCorrectionsFromDrafts(drafts, frames).map(comparableCorrection).sort());
}

// Signature of a persisted corrections ARRAY in the SAME comparable form as
// meaningfulDraftsSignature. The post-save re-baseline keys off what was actually
// saved (this save's corrections payload) instead of the live drafts, so an edit
// made WHILE the save was in flight stays measured as dirty and survives the
// post-save refetch instead of being folded into the clean baseline (sc-12020).
function meaningfulCorrectionsSignature(corrections) {
  return JSON.stringify(
    (corrections ?? [])
      .filter((correction) => Number.isInteger(correction?.frameIndex))
      .map(comparableCorrection)
      .sort(),
  );
}

function maskUrl(projectId, relPath) {
  if (!projectId || !relPath) {
    return "";
  }
  const normalized = String(relPath).replaceAll("\\", "/");
  return withMediaTicket(`${API_BASE_URL}/api/v1/projects/${projectId}/files/${normalized}`);
}

/**
 * Sparse correction surface for a tracked person (sc-1485). Scrub sampled
 * tracking frames, inspect the box/mask overlay, nudge a box, and reject low
 * quality frames; the corrections persist in the track sidecar and the
 * replacement pipeline regenerates masks from corrected boxes.
 */
function PersonTrackCorrections({ track, sourceClip, saveTrackCorrections }) {
  const frames = useMemo(() => track?.frames ?? [], [track]);
  const [frameIndex, setFrameIndex] = useState(0);
  const [drafts, setDrafts] = useState({});
  const [saving, setSaving] = useState(false);
  const videoRef = useRef(null);

  // Keep the latest drafts and last-seeded snapshot addressable from the seed
  // effect without listing `drafts` as a dep (which would refire it on every
  // keystroke). `draftsRef` is the current working set; `seededSignatureRef` is
  // the MEANINGFUL-corrections signature of the drafts as last seeded (null until
  // the first seed), so "dirty" == the working set's meaningful corrections have
  // diverged from that snapshot — matching the Save/display `dirty` measure below.
  const draftsRef = useRef(drafts);
  draftsRef.current = drafts;
  const seededSignatureRef = useRef(null);
  const seededTrackIdRef = useRef(null);

  const correctionsSignature = JSON.stringify(track?.corrections ?? []);
  // Seed working drafts from the persisted corrections so reopening a track
  // shows its saved adjustments. A successful save re-baselines `seededSignatureRef`
  // (see save()), so the post-save corrections refetch — and any later external
  // correction on the same track — converge cleanly instead of being mistaken for
  // a dirty conflict against the pre-edit seed.
  //
  // sc-11966: a background track refresh (SSE / track-job update) can bump the
  // corrections signature on the SAME track while the user has unsaved
  // per-frame edits. Reseeding then would clobber those dirty drafts. So reseed
  // only when the track IDENTITY changes (a genuine switch — the key remount
  // also resets this), on the first seed, or when a signature change lands while
  // drafts are CLEAN (external-change visibility). Dirty drafts on the same
  // track are preserved.
  useEffect(() => {
    const seeded = {};
    for (const correction of track?.corrections ?? []) {
      const index = correction?.frameIndex;
      if (!Number.isInteger(index) || index < 0 || index >= frames.length) {
        continue;
      }
      seeded[index] = {
        box: correction.box ? normalizeBox(correction.box) : null,
        rejected: Boolean(correction.rejected),
      };
    }

    const trackChanged = seededTrackIdRef.current !== track?.id;
    const firstSeed = seededSignatureRef.current === null;
    // Measure dirtiness on the MEANINGFUL corrections (the same filtered view the
    // Save/display `dirty` uses), NOT the raw drafts JSON. Otherwise a no-op
    // touched draft (reject toggled on then off) reads "dirty" here while the
    // display reads clean, so a same-track external correction would be silently
    // skipped/hidden and later dropped on save (sc-11966).
    const draftsDirty =
      !firstSeed && meaningfulDraftsSignature(draftsRef.current, frames) !== seededSignatureRef.current;

    if (trackChanged || firstSeed || !draftsDirty) {
      const seededSignature = meaningfulDraftsSignature(seeded, frames);
      draftsRef.current = seeded;
      seededSignatureRef.current = seededSignature;
      seededTrackIdRef.current = track?.id;
      setDrafts(seeded);
      setFrameIndex((current) => (current < frames.length ? current : 0));
    }
  }, [track?.id, correctionsSignature, frames.length]);

  useEffect(() => {
    const video = videoRef.current;
    const timestamp = frames[frameIndex]?.timestamp;
    if (!video || !Number.isFinite(timestamp)) {
      return;
    }
    try {
      video.currentTime = timestamp;
    } catch {
      // Seeking before metadata loads is retried by onLoadedMetadata below.
    }
  }, [frameIndex, frames]);

  if (!frames.length) {
    return (
      <div className="empty-panel compact-panel">
        This track has no sampled frames to correct yet.
      </div>
    );
  }

  const safeIndex = Math.min(frameIndex, frames.length - 1);
  const frame = frames[safeIndex];
  const draft = drafts[safeIndex];
  const workingBox = normalizeBox(draft?.box ?? frame.box ?? {});
  const rejected = Boolean(draft?.rejected);
  const flags = frame.flags ?? [];
  const overlayMask = frame.mask ? maskUrl(track.projectId, frame.mask) : "";

  function updateDraft(index, updater) {
    setDrafts((current) => {
      const base = current[index] ?? {
        box: normalizeBox(frames[index]?.box ?? {}),
        rejected: false,
      };
      const next = updater({ box: base.box ? normalizeBox(base.box) : normalizeBox(frames[index]?.box ?? {}), rejected: base.rejected });
      return { ...current, [index]: next };
    });
  }

  function setBoxComponent(key, rawValue) {
    const value = roundComponent(Number.parseFloat(rawValue));
    updateDraft(safeIndex, (entry) => ({ ...entry, box: { ...entry.box, [key]: value } }));
  }

  function setRejected(value) {
    updateDraft(safeIndex, (entry) => ({ ...entry, rejected: value }));
  }

  function resetFrame() {
    setDrafts((current) => {
      const next = { ...current };
      delete next[safeIndex];
      return next;
    });
  }

  // The corrections payload is the UI's full view: only frames whose box drifted
  // from the tracked box or that are rejected carry intent worth persisting.
  const pendingCorrections = pendingCorrectionsFromDrafts(drafts, frames);

  const persistedCount = (track?.corrections ?? []).length;
  const frameCorrected = pendingCorrections.some((correction) => correction.frameIndex === safeIndex);

  // Compare the working set against what is persisted (ignoring stamped
  // author/createdAt/source) so Save only lights up when something changed. This
  // is the same meaningful-corrections measure the seed effect now uses for
  // dirtiness (sc-11966).
  const persistedComparable = (track?.corrections ?? [])
    .filter((correction) => Number.isInteger(correction?.frameIndex))
    .map(comparableCorrection)
    .sort();
  const pendingComparable = pendingCorrections.map(comparableCorrection).sort();
  const dirty = JSON.stringify(persistedComparable) !== JSON.stringify(pendingComparable);

  async function save() {
    if (saving || typeof saveTrackCorrections !== "function") {
      return;
    }
    setSaving(true);
    try {
      const result = await saveTrackCorrections(track.id, pendingCorrections);
      // sc-11966: a successful save makes the just-saved working set the new clean
      // baseline. The save returns the updated track, so the parent refetch bumps
      // correctionsSignature to the persisted value. Without re-baselining here,
      // the seed effect keeps measuring "dirty" against the stale pre-edit seed, so
      // the post-save refetch (and any later external correction on the same track)
      // is treated as a dirty conflict and skipped — hiding the external change and
      // letting the next Save silently drop it. Baseline on the MEANINGFUL-
      // corrections signature so it matches the seed effect's dirty measure.
      //
      // sc-12020: baseline on the corrections THIS save persisted (`pendingCorrections`,
      // the POST body captured when Save was clicked) — NOT draftsRef.current at
      // resolve time. The correction inputs stay enabled during the in-flight save,
      // so draftsRef.current can already hold an edit the user made mid-save. Keying
      // the clean baseline off draftsRef.current would fold that concurrent edit into
      // "clean", and the post-save corrections refetch would then reseed the persisted
      // value over — clobbering — it. Keying off the saved payload keeps a mid-save
      // edit measured as dirty, so it survives the refetch.
      if (result) {
        seededSignatureRef.current = meaningfulCorrectionsSignature(pendingCorrections);
      }
    } finally {
      setSaving(false);
    }
  }

  return (
    <div className="person-track-corrections" aria-label="Track corrections">
      <div className="person-correction-header">
        <strong>Review &amp; correct track</strong>
        <span>
          Frame {safeIndex + 1} / {frames.length}
          {Number.isFinite(frame.timestamp) ? ` • ${frame.timestamp.toFixed(2)}s` : ""}
          {Number.isFinite(frame.confidence) ? ` • ${Math.round(frame.confidence * 100)}% conf` : ""}
        </span>
      </div>

      <div className="person-selection-frame">
        {sourceClip ? (
          <AssetMedia
            asset={sourceClip}
            controls={false}
            muted
            onLoadedMetadata={() => {
              const video = videoRef.current;
              if (video && Number.isFinite(frame.timestamp)) {
                try {
                  video.currentTime = frame.timestamp;
                } catch {
                  // Ignore: some browsers reject seeks until the buffer is ready.
                }
              }
            }}
            ref={videoRef}
          />
        ) : null}
        {overlayMask ? <img alt="" className="person-track-mask-overlay" src={overlayMask} /> : null}
        <div
          className={rejected ? "person-box rejected" : "person-box active"}
          style={{
            left: `${workingBox.x * 100}%`,
            top: `${workingBox.y * 100}%`,
            width: `${workingBox.width * 100}%`,
            height: `${workingBox.height * 100}%`,
          }}
        >
          <span>{rejected ? "rejected" : "corrected box"}</span>
        </div>
      </div>

      <div className="person-correction-scrubber">
        <button
          aria-label="Previous frame"
          disabled={safeIndex <= 0}
          onClick={() => setFrameIndex(Math.max(0, safeIndex - 1))}
          type="button"
        >
          ‹
        </button>
        <input
          aria-label="Scrub tracking frames"
          max={frames.length - 1}
          min={0}
          onChange={(event) => setFrameIndex(Number.parseInt(event.target.value, 10) || 0)}
          step={1}
          type="range"
          value={safeIndex}
        />
        <button
          aria-label="Next frame"
          disabled={safeIndex >= frames.length - 1}
          onClick={() => setFrameIndex(Math.min(frames.length - 1, safeIndex + 1))}
          type="button"
        >
          ›
        </button>
      </div>

      {flags.length ? (
        <div className="person-correction-flags" role="status">
          {flags.map((flag) => (
            <span className="person-correction-flag" key={flag}>
              {flag.replaceAll("_", " ")}
            </span>
          ))}
        </div>
      ) : null}

      <div className="control-grid person-correction-box">
        {BOX_FIELDS.map(([key, label]) => (
          <label key={key}>
            {label}
            <input
              aria-label={`Box ${key}`}
              disabled={rejected}
              max={1}
              min={0}
              onChange={(event) => setBoxComponent(key, event.target.value)}
              step={0.01}
              type="number"
              value={workingBox[key]}
            />
          </label>
        ))}
      </div>

      <label className="person-correction-reject">
        <input checked={rejected} onChange={(event) => setRejected(event.target.checked)} type="checkbox" />
        Reject this frame (low quality — replacement borrows the nearest good frame)
      </label>

      <div className="guidance-strip">
        <span>
          Adjusting a box regenerates that frame&apos;s mask from the corrected box at replacement time. Saved
          corrections record author and time in the track sidecar.
        </span>
      </div>

      <div className="replace-actions">
        <div className="person-correction-actions">
          <button disabled={!frameCorrected} onClick={resetFrame} type="button">
            Reset frame
          </button>
          <button className="primary" disabled={saving || !dirty} onClick={save} type="button">
            {saving ? "Saving…" : "Save corrections"}
          </button>
        </div>
        <span>
          {dirty ? `${pendingCorrections.length} unsaved` : `${persistedCount} saved`}
        </span>
      </div>
    </div>
  );
}

export function ReplacePersonPanel({
  createPersonDetectionJob,
  createPersonTrackJob,
  detectionResult,
  matchingTracks,
  representativeFrame,
  selectedDetection,
  selectedTrack,
  setPersonTrackId,
  setReplacementMode,
  setSelectedDetectionId,
  setSourceClipAssetId,
  setTrackName,
  sourceClipAssetId,
  trackName,
  personTrackId,
  replacementMode,
  saveTrackCorrections,
  videoAssets,
  videoModels = [],
  model,
  setModel,
  personReadiness = {},
}) {
  // The replacement backend = the replace-capable video models. The user picks one here (it drives
  // the job's `model`); SCAIL-2 (scail2_14b) is the native cross-identity engine, the others inpaint
  // the masked region via Wan-VACE (sc-5449 / sc-5452).
  const replacementModels = useMemo(
    () => videoModels.filter((item) => item.capabilities?.includes("replace_person")),
    [videoModels],
  );
  // Default-open: only gate when readiness explicitly reports a backend missing.
  const detectReady = personReadiness?.detect?.ready !== false;
  const trackReady = personReadiness?.track?.ready !== false;
  const replaceReady = personReadiness?.replace?.ready !== false;
  const readinessNotice = !detectReady
    ? "Detection unavailable: no live GPU worker is advertising the detector capability."
    : !trackReady
      ? "Tracking unavailable: no live GPU worker is advertising the tracker capability."
      : !replaceReady
        ? "Replacement unavailable: no live GPU worker can run person replacement yet."
        : "";

  const selectedTrackSourceClip = selectedTrack
    ? videoAssets.find((asset) => asset.id === selectedTrack.sourceAssetId) ?? null
    : null;

  function analyzeSource() {
    if (!sourceClipAssetId) {
      return;
    }
    createPersonDetectionJob({ sourceAssetId: sourceClipAssetId }, { navigateToQueue: false });
  }

  function createTrack() {
    if (!sourceClipAssetId || !representativeFrame || !selectedDetection) {
      return;
    }
    createPersonTrackJob(
      {
        sourceAssetId: sourceClipAssetId,
        representativeFrameAssetId: representativeFrame.id,
        detection: selectedDetection,
        trackName,
      },
      { navigateToQueue: false },
    );
  }

  return (
    <div className="replace-person-panel">
      <AssetPickerField
        assets={videoAssets}
        buttonLabel="Select clip"
        emptyLabel="No source clip selected"
        label="Source clip"
        onChange={setSourceClipAssetId}
        value={sourceClipAssetId}
      />

      <div className="guidance-strip">
        <strong>Real person tracking</strong>
        <span>Detection and tracking run on a GPU worker (YOLO + ByteTrack). Replacement uses per-frame segmentation masks when a segmenter is installed and falls back to box masks otherwise.</span>
      </div>

      {readinessNotice ? (
        <div className="guidance-strip warning" role="status">
          <strong>Not ready</strong>
          <span>{readinessNotice}</span>
        </div>
      ) : null}

      <div className="replace-actions">
        <button disabled={!sourceClipAssetId || !detectReady} onClick={analyzeSource} type="button">
          Analyze Source
        </button>
        <span>{detectionResult ? `${detectionResult.detections?.length ?? 0} candidates` : "No analysis yet"}</span>
      </div>

      {representativeFrame ? (
        <div className="person-selection-frame">
          <AssetMedia asset={representativeFrame} />
          {(detectionResult?.detections ?? []).map((detection) => (
            <button
              aria-label={`Select ${detection.label}`}
              className={selectedDetection?.id === detection.id ? "person-box active" : "person-box"}
              key={detection.id}
              onClick={() => setSelectedDetectionId(detection.id)}
              style={{
                left: `${detection.box.x * 100}%`,
                top: `${detection.box.y * 100}%`,
                width: `${detection.box.width * 100}%`,
                height: `${detection.box.height * 100}%`,
              }}
              type="button"
            >
              <span>{Math.round(detection.confidence * 100)}%</span>
            </button>
          ))}
        </div>
      ) : (
        <div className="empty-panel compact-panel">Analyze a source clip to extract a selection frame.</div>
      )}

      <div className="control-grid compact-controls">
        <label>
          Track name
          <input onChange={(event) => setTrackName(event.target.value)} value={trackName} />
        </label>
        <button disabled={!representativeFrame || !selectedDetection || !trackReady} onClick={createTrack} type="button">
          Save Track
        </button>
      </div>

      <label>
        Person track
        <select onChange={(event) => setPersonTrackId(event.target.value)} value={personTrackId}>
          <option value="">Select tracked person</option>
          {matchingTracks.map((track) => (
            <option key={track.id} value={track.id}>
              {track.name}
            </option>
          ))}
        </select>
      </label>

      {selectedTrack ? (
        <div className="guidance-strip">
          <strong>{selectedTrack.status?.averageConfidence ? `${Math.round(selectedTrack.status.averageConfidence * 100)}% track` : "Reusable track"}</strong>
          <span>{maskStateCopy(selectedTrack)}</span>
        </div>
      ) : null}

      {selectedTrack ? (
        <PersonTrackCorrections
          key={selectedTrack.id}
          saveTrackCorrections={saveTrackCorrections}
          sourceClip={selectedTrackSourceClip}
          track={selectedTrack}
        />
      ) : null}

      {replacementModels.length > 1 && setModel ? (
        <label>
          Replacement engine
          <select onChange={(event) => setModel(event.target.value)} value={model}>
            {replacementModels.map((item) => (
              <option key={item.id} value={item.id}>
                {item.name}
              </option>
            ))}
          </select>
        </label>
      ) : null}

      {model === "scail2_14b" ? (
        <div className="guidance-strip">
          <strong>SCAIL-2 full-character replacement</strong>
          <span>
            SCAIL-2 re-renders the whole tracked person from the character reference, so the
            Replacement mode below (face-only / keep-outfit) does not apply.
          </span>
        </div>
      ) : null}

      <label>
        Replacement mode
        <select onChange={(event) => setReplacementMode(event.target.value)} value={replacementMode}>
          <option value="face_only">Face Only</option>
          <option value="full_person_keep_outfit">Full Person, Keep Outfit</option>
          <option value="full_person_replace_outfit">Full Person, Replace Outfit</option>
        </select>
      </label>
    </div>
  );
}
