import React, { useRef } from "react";
import { Icon } from "../Icons.jsx";
import { trackItems } from "../../timeline.js";
import {
  BASE_PX_PER_SEC,
  MAIN_TRACK_ID,
  OVERLAY_TRACK_ID,
  buildTicks,
  itemGeometry,
  clipHue,
  isAiItem,
  waveformPath,
} from "./editorUtils.js";

// One video clip block on V1/V2 — absolutely positioned by % of duration, gradient fill
// from a per-clip hue, selection ring, and (for AI clips) a violet outline + sparkle plus
// boundary keyframe diamonds.
function ClipBlock({ item, duration, selected, ai, onSelect, onSelectKey }) {
  const { leftPct, widthPct } = itemGeometry(item, duration);
  return (
    <button
      className={`ve-clip${selected ? " selected" : ""}${ai ? " ai" : ""}`}
      onClick={() => onSelect(item)}
      style={{ left: `${leftPct}%`, width: `${widthPct}%`, "--clip-hue": clipHue(item.id) }}
      title={item.displayName}
      type="button"
    >
      {ai ? (
        <span className="ve-clip-ai" aria-hidden="true">
          <Icon.Stars size={9} />
        </span>
      ) : null}
      {/* Boundary keyframe markers for AI clips (no persisted keyframe model yet — these
          represent the first/last-frame conditioning; tracked as a backend-gap story). */}
      {ai
        ? [0.001, 0.999].map((pos, index) => (
            <span
              className="ve-keyframe"
              key={index}
              onClick={(event) => {
                event.stopPropagation();
                onSelectKey?.(item, index);
              }}
              style={{ left: `${pos * 100}%` }}
            />
          ))
        : null}
      <span className="ve-clip-name">{item.displayName}</span>
    </button>
  );
}

// The full NLE timeline (design 2a, epic 12798). Track headers on the left (non-scrolling),
// the %-positioned lanes on the right (horizontal scroll). Zoom scales the content width
// only; everything inside is positioned as a percentage of duration.
export function Timeline({
  timeline,
  duration,
  zoom,
  snap,
  onToggleSnap,
  onZoomIn,
  onZoomOut,
  playheadSeconds,
  onScrub,
  selectedItemId,
  selectedGapId,
  onSelectItem,
  onSelectGap,
  onSelectKey,
  assetsById,
  trackVisible,
  trackMuted,
  trackSoloed,
  onToggleVisible,
  onToggleMute,
  onToggleSolo,
  onAddAudioTrack,
  markers = [],
  onSelectMarker,
}) {
  const laneRef = useRef(null);
  const tracks = timeline?.tracks ?? [];
  const overlayTrack = tracks.find((t) => t.id === OVERLAY_TRACK_ID || t.kind === "overlay") ?? null;
  const mainTrack = tracks.find((t) => t.id === MAIN_TRACK_ID || t.kind === "video") ?? null;
  const audioTracks = tracks.filter((t) => t.kind === "audio");

  const contentWidth = Math.max(640, duration * BASE_PX_PER_SEC * zoom);
  const ticks = buildTicks(duration);
  const playheadPct = duration > 0 ? Math.min(100, (playheadSeconds / duration) * 100) : 0;

  const mainItems = mainTrack ? trackItems(mainTrack) : [];
  const overlayItems = overlayTrack ? trackItems(overlayTrack) : [];

  // Gaps between consecutive main-track clips → dashed "＋ Bridge" zones.
  const gaps = [];
  for (let i = 0; i < mainItems.length - 1; i += 1) {
    const left = mainItems[i];
    const right = mainItems[i + 1];
    const gap = Number(right.timelineStart) - Number(left.timelineEnd);
    if (gap > 0.05 && duration > 0) {
      gaps.push({
        id: `gap_${left.id}`,
        leftItem: left,
        rightItem: right,
        leftPct: (Number(left.timelineEnd) / duration) * 100,
        widthPct: (gap / duration) * 100,
      });
    }
  }

  // Transition badges at contiguous cuts that carry a non-cut transition.
  const transitions = [];
  for (let i = 0; i < mainItems.length - 1; i += 1) {
    const left = mainItems[i];
    const right = mainItems[i + 1];
    const contiguous = Math.abs(Number(right.timelineStart) - Number(left.timelineEnd)) <= 0.05;
    const type = left.transitionOut?.type && left.transitionOut.type !== "cut" ? left.transitionOut.type : right.transitionIn?.type;
    if (contiguous && type && type !== "cut" && duration > 0) {
      transitions.push({ id: `trans_${left.id}`, type, leftPct: (Number(right.timelineStart) / duration) * 100 });
    }
  }

  function handleRulerMouseDown(event) {
    const el = laneRef.current;
    if (!el || duration <= 0) {
      return;
    }
    const rect = el.getBoundingClientRect();
    const toSeconds = (clientX) => {
      const ratio = Math.min(1, Math.max(0, (clientX - rect.left) / rect.width));
      return ratio * duration;
    };
    onScrub?.(toSeconds(event.clientX));
    const onMove = (moveEvent) => onScrub?.(toSeconds(moveEvent.clientX));
    const onUp = () => {
      window.removeEventListener("mousemove", onMove);
      window.removeEventListener("mouseup", onUp);
    };
    window.addEventListener("mousemove", onMove);
    window.addEventListener("mouseup", onUp);
  }

  const audioSubLabel = (track, index) => {
    if (track.name && track.name.toLowerCase() !== "audio") {
      return track.name;
    }
    return ["dialogue", "AI music", "sfx"][index] ?? "audio";
  };
  const audioIsAi = (track, index) => /music/i.test(track.name ?? "") || index === 1;

  return (
    <div className="ve-timeline">
      <div className="ve-timeline-hd">
        <strong className="ve-timeline-title">TIMELINE</strong>
        <div className="ve-timeline-legend">
          <span className="ve-legend-dot ve-dot-accent" />
          Clip
          <span className="ve-legend-dot ve-dot-ai" />
          AI
        </div>
        <div className="ve-timeline-hd-right">
          <button className={`ve-snap${snap ? " on" : ""}`} onClick={onToggleSnap} type="button">
            <span className={`ve-switch${snap ? " on" : ""}`} aria-hidden="true">
              <span className="ve-switch-knob" />
            </span>
            Snap
          </button>
          <button className="ve-ghost-btn" onClick={onZoomOut} title="Zoom out" type="button">
            <Icon.Minus size={14} />
          </button>
          <button className="ve-ghost-btn" onClick={onZoomIn} title="Zoom in" type="button">
            <Icon.Plus size={14} />
          </button>
        </div>
      </div>

      <div className="ve-timeline-body">
        {/* Track headers (non-scrolling) */}
        <div className="ve-track-heads">
          <div className="ve-ruler-spacer" />
          <div className="ve-track-head ve-head-v2">
            <div className="ve-head-name">
              <strong>V2</strong>
              <span className="ve-head-sub">overlay</span>
            </div>
            <button
              className={`ve-track-btn${trackVisible?.[OVERLAY_TRACK_ID] === false ? " off" : ""}`}
              onClick={() => onToggleVisible?.(OVERLAY_TRACK_ID)}
              title="Toggle visibility"
              type="button"
            >
              <Icon.Image size={13} />
            </button>
          </div>
          <div className="ve-track-head ve-head-v1">
            <div className="ve-head-name">
              <strong>V1</strong>
              <span className="ve-head-sub">main video</span>
            </div>
            <button
              className={`ve-track-btn${mainTrack && trackVisible?.[mainTrack.id] === false ? " off" : ""}`}
              onClick={() => onToggleVisible?.(mainTrack?.id)}
              title="Toggle visibility"
              type="button"
            >
              <Icon.Image size={13} />
            </button>
          </div>
          {audioTracks.map((track, index) => (
            <div className="ve-track-head ve-head-audio" key={track.id}>
              <div className="ve-head-name">
                <strong>
                  A{index + 1}
                  {audioIsAi(track, index) ? <Icon.Stars size={10} className="ve-ai-icon" /> : null}
                </strong>
                <span className="ve-head-sub">{audioSubLabel(track, index)}</span>
              </div>
              <div className="ve-track-mschips">
                <button
                  className={`ve-track-chip${trackMuted?.[track.id] ? " on" : ""}`}
                  onClick={() => onToggleMute?.(track.id)}
                  title="Mute"
                  type="button"
                >
                  M
                </button>
                <button
                  className={`ve-track-chip${trackSoloed?.[track.id] ? " on" : ""}`}
                  onClick={() => onToggleSolo?.(track.id)}
                  title="Solo"
                  type="button"
                >
                  S
                </button>
              </div>
            </div>
          ))}
          <button className="ve-add-audio" onClick={onAddAudioTrack} title="Add audio track" type="button">
            <Icon.Plus size={13} />
            Audio
          </button>
        </div>

        {/* Scrollable lanes */}
        <div className="ve-lanes-scroll">
          <div className="ve-lanes" ref={laneRef} style={{ width: `${contentWidth}px` }}>
            {/* Ruler */}
            <div className="ve-ruler" onMouseDown={handleRulerMouseDown}>
              {ticks.map((tick) => (
                <div
                  className={`ve-tick${tick.major ? " major" : ""}`}
                  key={tick.second}
                  style={{ left: `${tick.leftPct}%` }}
                >
                  {tick.major ? <span className="ve-tick-label">{tick.label}</span> : null}
                </div>
              ))}
              {markers.map((marker) => (
                <div
                  className="ve-marker"
                  key={marker.id}
                  onClick={(event) => {
                    event.stopPropagation();
                    onSelectMarker?.(marker);
                  }}
                  style={{ left: `${duration > 0 ? (marker.time / duration) * 100 : 0}%` }}
                >
                  <span className="ve-marker-flag">{marker.label}</span>
                </div>
              ))}
            </div>

            {/* V2 overlay lane */}
            <div className="ve-lane ve-lane-v2">
              {overlayItems.map((item) => (
                <ClipBlock
                  ai={isAiItem(item, assetsById)}
                  duration={duration}
                  item={item}
                  key={item.id}
                  onSelect={onSelectItem}
                  onSelectKey={onSelectKey}
                  selected={selectedItemId === item.id}
                />
              ))}
            </div>

            {/* V1 main video lane */}
            <div className="ve-lane ve-lane-v1">
              {gaps.map((gap) => (
                <button
                  className={`ve-gap${selectedGapId === gap.id ? " selected" : ""}`}
                  key={gap.id}
                  onClick={() => onSelectGap?.(gap)}
                  style={{ left: `${gap.leftPct}%`, width: `${gap.widthPct}%` }}
                  type="button"
                >
                  <span className="ve-gap-label">
                    <Icon.Plus size={12} />
                    Bridge
                  </span>
                </button>
              ))}
              {mainItems.map((item) => (
                <ClipBlock
                  ai={isAiItem(item, assetsById)}
                  duration={duration}
                  item={item}
                  key={item.id}
                  onSelect={onSelectItem}
                  onSelectKey={onSelectKey}
                  selected={selectedItemId === item.id}
                />
              ))}
              {transitions.map((trans) => (
                <div className="ve-transition" key={trans.id} style={{ left: `${trans.leftPct}%` }} title={trans.type}>
                  <Icon.Duplicate size={12} />
                </div>
              ))}
            </div>

            {/* Audio lanes */}
            {audioTracks.map((track, index) => (
              <div className="ve-lane ve-lane-audio" key={track.id}>
                {trackItems(track).map((item) => {
                  const { leftPct, widthPct } = itemGeometry(item, duration);
                  return (
                    <button
                      className={`ve-audio-clip${selectedItemId === item.id ? " selected" : ""}${audioIsAi(track, index) ? " ai" : ""}`}
                      key={item.id}
                      onClick={() => onSelectItem(item)}
                      style={{ left: `${leftPct}%`, width: `${widthPct}%`, "--clip-hue": clipHue(item.id) }}
                      title={item.displayName}
                      type="button"
                    >
                      <svg className="ve-wave" preserveAspectRatio="none" viewBox="0 0 100 40">
                        <path d={waveformPath(item.id)} />
                      </svg>
                      <span className="ve-clip-name">{item.displayName}</span>
                    </button>
                  );
                })}
              </div>
            ))}

            {/* Playhead */}
            <div className="ve-playhead" style={{ left: `${playheadPct}%` }}>
              <span className="ve-playhead-handle" />
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}
