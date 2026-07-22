import React, { useEffect, useMemo, useRef, useState } from "react";
import { assetCanRenderAsImage, assetCanRenderAsVideo } from "../components/assetMedia.jsx";
import { formatTimecode } from "../formatting.js";
import {
  ensureItemVersionFields,
  itemDuration,
  sourceTimestampAtPlayhead,
  timelineDuration,
  trackItems,
} from "../timeline.js";
import { useAppStatic } from "../context/AppContext.js";
import { useScreenActive } from "../context/ScreenActiveContext.js";
import { appConfirm } from "../appConfirm.jsx";
import { EditorToolbar } from "../components/editor/EditorToolbar.jsx";
import { MediaBin } from "../components/editor/MediaBin.jsx";
import { ProgramMonitor } from "../components/editor/ProgramMonitor.jsx";
import { GenerationRail } from "../components/editor/GenerationRail.jsx";
import { StoryboardStrip } from "../components/editor/StoryboardStrip.jsx";
import { Timeline } from "../components/editor/Timeline.jsx";
import { useEditorGeneration } from "../components/editor/useEditorGeneration.js";
import { ZOOM_MIN, ZOOM_MAX, ZOOM_STEP, MAIN_TRACK_ID } from "../components/editor/editorUtils.js";

export function EditorScreen() {
  const app = useAppStatic();
  const {
    activeProject,
    activeTimeline,
    mediaAssets,
    setPreviewAsset,
    sendAssetToVideo,
    createTimeline,
    extractTimelineFrame,
    exportTimeline,
    queueTimelineVideoJob,
    saveTimeline,
    selectedTimelineId,
    setActiveTimeline,
    setSelectedTimelineId,
    isActiveTimelineDirty,
    timelines,
  } = app;
  const assets = mediaAssets;
  const gen = useEditorGeneration({ context: app });

  const [selectedItemId, setSelectedItemId] = useState(null);
  const [selectionKind, setSelectionKind] = useState(null); // clip | audio | gap | key | marker
  const [selectedGap, setSelectedGap] = useState(null);
  const [selectedMarker, setSelectedMarker] = useState(null);
  const [playheadSeconds, setPlayheadSeconds] = useState(0);
  const [isPlaying, setIsPlaying] = useState(false);
  const [zoom, setZoom] = useState(1);
  const [snap, setSnap] = useState(true);
  const [history, setHistory] = useState([]);
  const [future, setFuture] = useState([]);
  const [trackVisible, setTrackVisible] = useState({});
  const [trackMuted, setTrackMuted] = useState({});
  const [trackSoloed, setTrackSoloed] = useState({});
  const [markers] = useState([]); // Local UI markers only — no persisted marker model yet (audit).
  const [timelineNotice, setTimelineNotice] = useState("");
  const previewVideoRef = useRef(null);
  const screenActive = useScreenActive();

  const assetsById = useMemo(() => new Map(assets.map((asset) => [asset.id, asset])), [assets]);
  const trackKindById = useMemo(() => {
    const map = new Map();
    (activeTimeline?.tracks ?? []).forEach((track) => map.set(track.id, track.kind));
    return map;
  }, [activeTimeline]);
  const selectedItem = useMemo(() => {
    if (!activeTimeline) {
      return null;
    }
    return activeTimeline.tracks.flatMap((track) => track.items).find((item) => item.id === selectedItemId) ?? null;
  }, [activeTimeline, selectedItemId]);
  const selectedAsset = assetsById.get(selectedItem?.assetId) ?? null;
  const duration = activeTimeline ? timelineDuration(activeTimeline) : 0;
  const mainTrack = activeTimeline?.tracks?.find((track) => track.id === MAIN_TRACK_ID || track.kind === "video") ?? null;
  const mainClips = mainTrack ? trackItems(mainTrack) : [];
  const isSelectedAi = useMemo(() => {
    const history = selectedItem?.versionHistory ?? [];
    return history.some((entry) => ["extension", "bridge", "replacement"].includes(entry?.source));
  }, [selectedItem]);

  useEffect(() => {
    setHistory([]);
    setFuture([]);
    setSelectedItemId(null);
    setSelectionKind(null);
    setPlayheadSeconds(0);
  }, [activeTimeline?.id]);

  // Preview playback: drive the selected clip's <video> only while foregrounded (sc-11961).
  useEffect(() => {
    const video = previewVideoRef.current;
    if (!assetCanRenderAsVideo(selectedAsset) || !video) {
      return;
    }
    if (isPlaying && screenActive) {
      video.play().catch(() => setIsPlaying(false));
      return;
    }
    video.pause();
    // Re-run only when the selected clip changes (by id), not on every asset-object identity.
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [isPlaying, selectedAsset?.id, screenActive]);

  // Playhead transport: a rAF loop advances the playhead across the whole timeline while
  // playing, wrapping to 0 at the end. Only runs while foregrounded.
  useEffect(() => {
    if (!isPlaying || !screenActive || duration <= 0 || typeof window.requestAnimationFrame !== "function") {
      return undefined;
    }
    let raf = 0;
    let last = performance.now();
    const tick = (now) => {
      const dt = (now - last) / 1000;
      last = now;
      setPlayheadSeconds((prev) => {
        const next = prev + dt;
        return next >= duration ? 0 : next;
      });
      raf = requestAnimationFrame(tick);
    };
    raf = requestAnimationFrame(tick);
    return () => cancelAnimationFrame(raf);
  }, [isPlaying, screenActive, duration]);

  const shortcutStateRef = useRef({ undo, redo, removeSelectedItem, selectedItemId, screenActive });
  shortcutStateRef.current = { undo, redo, removeSelectedItem, selectedItemId, screenActive };

  useEffect(() => {
    function onKeyDown(event) {
      const { undo, redo, removeSelectedItem, selectedItemId, screenActive } = shortcutStateRef.current;
      // Under selective keep-alive (sc-11959) this editor stays mounted and this window
      // listener stays subscribed even when another view is foregrounded, so gate every
      // shortcut on the active flag — a backgrounded timeline must never eat Space (play),
      // undo/redo, or Delete meant for the visible screen (sc-13589). Read it from the ref
      // so the once-subscribed listener always sees the latest value.
      if (!screenActive) {
        return;
      }
      const target = event.target;
      if (["INPUT", "TEXTAREA", "SELECT"].includes(target?.tagName)) {
        return;
      }
      if (event.code === "Space") {
        event.preventDefault();
        setIsPlaying((value) => !value);
      }
      if ((event.metaKey || event.ctrlKey) && event.key.toLowerCase() === "z") {
        event.preventDefault();
        event.shiftKey ? redo() : undo();
      }
      if ((event.metaKey || event.ctrlKey) && event.key.toLowerCase() === "y") {
        event.preventDefault();
        redo();
      }
      if ((event.key === "Delete" || event.key === "Backspace") && selectedItemId) {
        event.preventDefault();
        removeSelectedItem();
      }
    }
    window.addEventListener("keydown", onKeyDown);
    return () => window.removeEventListener("keydown", onKeyDown);
  }, []);

  function commit(nextTimeline) {
    if (!activeTimeline) {
      return;
    }
    setHistory((items) => [...items.slice(-24), activeTimeline]);
    setFuture([]);
    setActiveTimeline({ ...nextTimeline, duration: timelineDuration(nextTimeline) });
  }

  function normalizeTimelineItem(item) {
    const start = Number(item.timelineStart) || 0;
    const end = Math.max(start + 0.1, Number(item.timelineEnd) || start + 0.1);
    const sourceIn = Number(item.sourceIn) || 0;
    const sourceOut = Math.max(sourceIn + 0.1, Number(item.sourceOut) || sourceIn + itemDuration(item));
    return {
      ...ensureItemVersionFields(item),
      sourceIn,
      sourceOut,
      timelineStart: Math.max(0, start),
      timelineEnd: end,
      speed: Math.max(0.1, Number(item.speed) || 1),
    };
  }

  function undo() {
    if (!history.length || !activeTimeline) {
      return;
    }
    const previous = history[history.length - 1];
    setHistory((items) => items.slice(0, -1));
    setFuture((items) => [activeTimeline, ...items]);
    setActiveTimeline(previous);
  }

  function redo() {
    if (!future.length || !activeTimeline) {
      return;
    }
    const next = future[0];
    setFuture((items) => items.slice(1));
    setHistory((items) => [...items, activeTimeline]);
    setActiveTimeline(next);
  }

  function addAssetToTrack(asset, trackId = MAIN_TRACK_ID) {
    if (!activeTimeline) {
      return;
    }
    const isStill = asset.type !== "video" && assetCanRenderAsImage(asset);
    const track = activeTimeline.tracks.find((item) => item.id === trackId) ?? activeTimeline.tracks[0];
    const start = Math.max(0, ...track.items.map((item) => item.timelineEnd));
    const sourceDuration = Number(asset.file?.duration) || 4;
    const durationSeconds = isStill ? 4 : sourceDuration;
    const item = normalizeTimelineItem({
      id: `item_${crypto.randomUUID().replaceAll("-", "")}`,
      trackId: track.id,
      assetId: asset.id,
      type: isStill ? "image" : "video",
      displayName: asset.displayName,
      sourceIn: 0,
      sourceOut: Math.max(0.1, sourceDuration),
      timelineStart: start,
      timelineEnd: start + Math.max(0.1, durationSeconds),
      speed: 1,
      fit: "fit",
      volume: 1,
      versionAssetIds: [asset.id],
      currentVersionAssetId: asset.id,
      versionHistory: [{ assetId: asset.id, createdAt: null, source: "original", jobId: null, note: null }],
      transitionIn: { id: `transition_${crypto.randomUUID().replaceAll("-", "")}`, type: "cut", duration: 0 },
      transitionOut: { id: `transition_${crypto.randomUUID().replaceAll("-", "")}`, type: "cut", duration: 0 },
    });
    commit({
      ...activeTimeline,
      tracks: activeTimeline.tracks.map((current) =>
        current.id === track.id ? { ...current, items: [...current.items, item] } : current,
      ),
    });
    selectItem(item);
  }

  function addAudioTrack() {
    if (!activeTimeline) {
      return;
    }
    const count = activeTimeline.tracks.filter((track) => track.kind === "audio").length;
    const newTrack = {
      id: `track_audio_${crypto.randomUUID().replaceAll("-", "")}`,
      name: `Audio ${count + 1}`,
      kind: "audio",
      locked: false,
      muted: false,
      items: [],
    };
    commit({ ...activeTimeline, tracks: [...activeTimeline.tracks, newTrack] });
    setTimelineNotice("Audio track added. Note: audio isn't mixed into exports yet (tracked for the backend audit).");
  }

  function removeSelectedItem() {
    if (!activeTimeline || !selectedItemId) {
      return;
    }
    commit({
      ...activeTimeline,
      tracks: activeTimeline.tracks.map((track) => ({
        ...track,
        items: track.items.filter((item) => item.id !== selectedItemId),
      })),
    });
    setSelectedItemId(null);
    setSelectionKind(null);
  }

  function rippleCloseGap(gap) {
    if (!activeTimeline || !gap) {
      return;
    }
    const shift = Number(gap.rightItem.timelineStart) - Number(gap.leftItem.timelineEnd);
    if (shift <= 0) {
      return;
    }
    commit({
      ...activeTimeline,
      tracks: activeTimeline.tracks.map((track) => {
        if (track.id !== gap.rightItem.trackId) {
          return track;
        }
        return {
          ...track,
          items: track.items.map((item) =>
            Number(item.timelineStart) >= Number(gap.rightItem.timelineStart)
              ? { ...item, timelineStart: item.timelineStart - shift, timelineEnd: item.timelineEnd - shift }
              : item,
          ),
        };
      }),
    });
    setSelectedGap(null);
    setSelectionKind(null);
  }

  // ---- selection ----
  function selectItem(item) {
    setSelectedItemId(item.id);
    setSelectionKind(trackKindById.get(item.trackId) === "audio" ? "audio" : "clip");
    setSelectedGap(null);
    setSelectedMarker(null);
    setPlayheadSeconds(Number(item.timelineStart) || 0);
    setIsPlaying(false);
    setTimelineNotice("");
  }
  function selectStoryboardKey(clip) {
    selectItem(clip);
  }
  function selectKeyframe(item) {
    setSelectedItemId(item.id);
    setSelectionKind("key");
    setSelectedGap(null);
    setPlayheadSeconds(Number(item.timelineStart) || 0);
    setIsPlaying(false);
  }
  function selectGap(gap) {
    setSelectedGap(gap);
    setSelectionKind("gap");
    setSelectedItemId(null);
    setSelectedMarker(null);
    setPlayheadSeconds(Number(gap.leftItem.timelineEnd) || 0);
    setIsPlaying(false);
  }
  function selectMarker(marker) {
    setSelectedMarker(marker);
    setSelectionKind("marker");
    setPlayheadSeconds(Number(marker.time) || 0);
  }

  function stepClip(direction) {
    if (!mainClips.length) {
      return;
    }
    const index = mainClips.findIndex((item) => item.id === selectedItemId);
    const nextIndex = index === -1 ? 0 : Math.min(mainClips.length - 1, Math.max(0, index + direction));
    selectItem(mainClips[nextIndex]);
  }

  // ---- generation context ----
  function generationContext(action, item, extra = {}) {
    return {
      action,
      timelineId: activeTimeline.id,
      timelineName: activeTimeline.name,
      itemId: item.id,
      trackId: item.trackId,
      sourceAssetId: item.assetId,
      sourceTimestamp: sourceTimestampAtPlayhead(item, playheadSeconds),
      ...extra,
    };
  }

  async function extractFrame() {
    if (!selectedItem || selectedItem.type !== "video") {
      setTimelineNotice("Select a video clip before extracting a frame.");
      return;
    }
    const job = await extractTimelineFrame({ timeline: activeTimeline, item: selectedItem, playheadSeconds, intendedUse: "reuse" });
    if (job) {
      setTimelineNotice("Frame extraction queued.");
    }
  }

  async function extendSelectedClip() {
    if (!selectedItem || selectedItem.type !== "video") {
      setTimelineNotice("Select a video clip before extending.");
      return;
    }
    const base = gen.buildBasePayload();
    const job = await queueTimelineVideoJob({
      ...base,
      mode: "extend_clip",
      sourceClipAssetId: selectedItem.assetId,
      advanced: {
        ...base.advanced,
        timelineAction: "extend",
        timelineContext: generationContext("extend", selectedItem, {
          endpointTimestamp: Number(selectedItem.sourceOut) || sourceTimestampAtPlayhead(selectedItem, selectedItem.timelineEnd),
          timelineStart: Number(selectedItem.timelineEnd),
        }),
      },
    });
    if (job) {
      setTimelineNotice("Extension job queued. The new clip lands after the selection when it completes.");
    }
  }

  async function replaceSelectedItem({ variation = false } = {}) {
    if (!selectedItem) {
      return;
    }
    const isStill = selectedItem.type === "image";
    const base = gen.buildBasePayload();
    const job = await queueTimelineVideoJob({
      ...base,
      mode: isStill ? "image_to_video" : "extend_clip",
      duration: itemDuration(selectedItem),
      seed: variation ? Math.floor(Math.random() * 1_000_000_000) : base.seed,
      sourceAssetId: isStill ? selectedItem.assetId : null,
      sourceClipAssetId: isStill ? null : selectedItem.assetId,
      advanced: {
        ...base.advanced,
        timelineAction: "replace",
        timelineContext: generationContext("replace", selectedItem),
      },
    });
    if (job) {
      setTimelineNotice(
        variation
          ? "Variation queued with a fresh seed. The prior asset stays in this item's version history."
          : "Replacement job queued. The prior asset stays in this item's version history.",
      );
    }
  }

  async function bridgeGap(gap) {
    const left = gap?.leftItem;
    const right = gap?.rightItem;
    if (!left || left.type !== "video" || !right || right.type !== "video") {
      setTimelineNotice("A bridge needs a video clip on each side of the gap.");
      return;
    }
    const gapSeconds = Number(right.timelineStart) - Number(left.timelineEnd);
    if (gapSeconds <= 0.05) {
      setTimelineNotice("Create space between the clips before generating a bridge.");
      return;
    }
    const base = gen.buildBasePayload();
    const job = await queueTimelineVideoJob({
      ...base,
      mode: "video_bridge",
      duration: Number(gapSeconds.toFixed(2)),
      sourceClipAssetId: left.assetId,
      bridgeRightClipAssetId: right.assetId,
      advanced: {
        ...base.advanced,
        timelineAction: "bridge",
        timelineContext: generationContext("bridge", left, {
          rightItemId: right.id,
          rightAssetId: right.assetId,
          leftTimestamp: Number(left.sourceOut) || sourceTimestampAtPlayhead(left, left.timelineEnd),
          rightTimestamp: Number(right.sourceIn) || 0,
          timelineStart: Number(left.timelineEnd),
          timelineEnd: Number(right.timelineStart),
        }),
      },
    });
    if (job) {
      setTimelineNotice("Bridge job queued. The generated clip will land in the gap.");
    }
  }

  function noticeUnsupported(feature) {
    setTimelineNotice(`${feature} isn't wired to a backend yet — tracked in the Video Editor audit (epic 12798).`);
  }

  // ---- timeline switching / creation ----
  async function handleSelectTimeline(nextId) {
    if (nextId === selectedTimelineId) {
      return;
    }
    if (isActiveTimelineDirty?.()) {
      const proceed = await appConfirm({
        title: "Discard timeline edits?",
        message: "You have unsaved timeline edits. Switch timelines and discard them?",
        confirmLabel: "Discard & switch",
        cancelLabel: "Keep editing",
        tone: "danger",
      });
      if (!proceed) {
        return;
      }
    }
    setSelectedTimelineId(nextId);
  }

  async function handleNewTimeline() {
    const count = timelines.length + 1;
    await createTimeline({ name: `Timeline ${count}`, aspectRatio: "16:9", fps: 30 });
  }

  // ---- rail contextual header + actions ----
  function buildContextActions() {
    if (selectionKind === "clip" || selectionKind === "key") {
      if (selectedItem?.type === "image") {
        return [
          { id: "animate", label: "Animate", primary: true, onClick: () => replaceSelectedItem() },
          { id: "send-video", label: "Send to Video", onClick: () => sendAssetToVideo(selectedAsset, "image_to_video") },
          { id: "variation", label: "Variation", onClick: () => replaceSelectedItem({ variation: true }) },
          { id: "extract", label: "Extract frame", disabled: true, onClick: () => extractFrame() },
        ];
      }
      return [
        { id: "extend", label: "Extend clip", primary: true, onClick: extendSelectedClip },
        { id: "regenerate", label: "Regenerate", onClick: () => replaceSelectedItem() },
        { id: "extract", label: "Extract frame", onClick: extractFrame },
        { id: "variation", label: "Variation", onClick: () => replaceSelectedItem({ variation: true }) },
      ];
    }
    if (selectionKind === "gap") {
      return [
        { id: "bridge", label: "Generate bridge", primary: true, onClick: () => bridgeGap(selectedGap) },
        { id: "fill", label: "Generate to fill", onClick: () => bridgeGap(selectedGap) },
        { id: "ripple", label: "Ripple close gap", onClick: () => rippleCloseGap(selectedGap) },
      ];
    }
    if (selectionKind === "audio") {
      return [
        { id: "music", label: "Regenerate music", primary: true, disabled: true, onClick: () => noticeUnsupported("Audio generation") },
        { id: "match", label: "Match to cut", disabled: true, onClick: () => noticeUnsupported("Match to cut") },
        { id: "vo", label: "Voiceover", disabled: true, onClick: () => noticeUnsupported("Voiceover") },
        { id: "duck", label: "Ducking", disabled: true, onClick: () => noticeUnsupported("Ducking") },
      ];
    }
    return [];
  }

  const contextActions = buildContextActions();
  const primaryAction = contextActions.find((action) => action.primary && !action.disabled);

  function railHeader() {
    if (selectionKind === "gap" && selectedGap) {
      const seconds = (Number(selectedGap.rightItem.timelineStart) - Number(selectedGap.leftItem.timelineEnd)).toFixed(1);
      return { eyebrow: "GAP", title: `${seconds}s gap` };
    }
    if (selectionKind === "audio") {
      return { eyebrow: "AUDIO", title: selectedItem?.displayName ?? "Audio clip" };
    }
    if (selectionKind === "key") {
      return { eyebrow: "KEYFRAME", title: selectedItem?.displayName ?? "Key image" };
    }
    if (selectionKind === "marker" && selectedMarker) {
      return { eyebrow: "MARKER", title: selectedMarker.label };
    }
    if (selectionKind === "clip" && selectedItem) {
      const eyebrow = isSelectedAi ? "VIDEO · AI CLIP" : selectedItem.type === "image" ? "IMAGE CLIP" : "VIDEO CLIP";
      return { eyebrow, title: selectedItem.displayName };
    }
    return { eyebrow: "NO SELECTION", title: activeTimeline?.name ?? "Timeline" };
  }

  function toggleMap(setter, key) {
    if (!key) {
      return;
    }
    setter((current) => ({ ...current, [key]: !current[key] }));
  }

  if (!activeProject) {
    return (
      <section className="ve-editor ve-editor-empty">
        <div className="empty-panel">Open a project before assembling a timeline.</div>
      </section>
    );
  }

  if (!activeTimeline) {
    return (
      <section className="ve-editor ve-editor-empty">
        <div className="empty-panel">
          <p>Create a timeline to start editing.</p>
          <button className="ve-generate" onClick={handleNewTimeline} type="button">
            New timeline
          </button>
        </div>
      </section>
    );
  }

  const aspectClass = `ve-aspect-${activeTimeline.aspectRatio.replace(":", "-")}`;
  const generateSummary = `${gen.duration}s · ${gen.resolution} · ${gen.fps}fps · queues to your GPU`;

  return (
    <section className="ve-editor">
      <EditorToolbar
        canRedo={future.length > 0}
        canUndo={history.length > 0}
        durationTimecode={formatTimecode(duration, activeTimeline.fps)}
        exportDisabled={!activeTimeline.tracks.some((track) => track.items.length)}
        onExport={() => exportTimeline(activeTimeline, { resolution: activeTimeline.height, fps: activeTimeline.fps })}
        onNewTimeline={handleNewTimeline}
        onRedo={redo}
        onSave={() => saveTimeline(activeTimeline)}
        onSelectTimeline={handleSelectTimeline}
        onUndo={undo}
        saveDisabled={!activeTimeline}
        onZoomIn={() => setZoom((z) => Math.min(ZOOM_MAX, +(z + ZOOM_STEP).toFixed(2)))}
        onZoomOut={() => setZoom((z) => Math.max(ZOOM_MIN, +(z - ZOOM_STEP).toFixed(2)))}
        projectName={activeTimeline.name}
        selectedTimelineId={selectedTimelineId}
        subLabel={`timeline · ${activeTimeline.aspectRatio} · ${activeTimeline.fps}fps`}
        timecode={formatTimecode(playheadSeconds, activeTimeline.fps)}
        timelines={timelines}
        zoomPct={`${Math.round(zoom * 100)}%`}
      />

      <div className="ve-upper">
        <MediaBin assets={assets} onAddToTrack={(asset) => addAssetToTrack(asset)} onPreview={(asset) => setPreviewAsset(asset, assets)} />
        <ProgramMonitor
          aspectClass={aspectClass}
          clipLabel={selectedItem ? `${selectedItem.displayName}${isSelectedAi ? " · AI" : ""}` : null}
          isAi={isSelectedAi}
          isPlaying={isPlaying}
          onEnded={() => setIsPlaying(false)}
          onNext={() => stepClip(1)}
          onPause={() => setIsPlaying(false)}
          onPlay={() => setIsPlaying(true)}
          onPrev={() => stepClip(-1)}
          onTogglePlay={() => setIsPlaying((value) => !value)}
          previewVideoRef={previewVideoRef}
          resolutionLabel={`${activeTimeline.width} × ${activeTimeline.height}`}
          selectedAsset={selectedAsset}
        />
        <GenerationRail
          contextActions={contextActions}
          gen={gen}
          generateDisabled={!activeTimeline}
          generateSummary={generateSummary}
          header={railHeader()}
          onGenerate={() => (primaryAction ? primaryAction.onClick() : setTimelineNotice("Select a clip or gap to generate."))}
        />
      </div>

      {timelineNotice ? <p className="ve-notice">{timelineNotice}</p> : null}

      <StoryboardStrip
        assetsById={assetsById}
        clips={mainClips}
        fps={activeTimeline.fps}
        onAddKey={() => noticeUnsupported("Storyboard key images")}
        onSelectKey={selectStoryboardKey}
        selectedItemId={selectedItemId}
      />

      <Timeline
        assetsById={assetsById}
        duration={duration}
        markers={markers}
        onAddAudioTrack={addAudioTrack}
        onScrub={(seconds) => {
          setIsPlaying(false);
          setPlayheadSeconds(seconds);
        }}
        onSelectGap={selectGap}
        onSelectItem={selectItem}
        onSelectKey={(item) => selectKeyframe(item)}
        onSelectMarker={selectMarker}
        onToggleMute={(id) => toggleMap(setTrackMuted, id)}
        onToggleSnap={() => setSnap((value) => !value)}
        onToggleSolo={(id) => toggleMap(setTrackSoloed, id)}
        onToggleVisible={(id) => toggleMap(setTrackVisible, id)}
        onZoomIn={() => setZoom((z) => Math.min(ZOOM_MAX, +(z + ZOOM_STEP).toFixed(2)))}
        onZoomOut={() => setZoom((z) => Math.max(ZOOM_MIN, +(z - ZOOM_STEP).toFixed(2)))}
        playheadSeconds={playheadSeconds}
        selectedGapId={selectedGap?.id}
        selectedItemId={selectedItemId}
        snap={snap}
        timeline={activeTimeline}
        trackMuted={trackMuted}
        trackSoloed={trackSoloed}
        trackVisible={trackVisible}
        zoom={zoom}
      />
    </section>
  );
}
