import React from "react";
import { Icon } from "../Icons.jsx";

// The editor's top toolbar (design 2a, epic 12798): app mark + project/timeline
// identity, undo/redo, the centered MM:SS:FF timecode / duration readout, zoom, and the
// filled Export button. Timeline switching/creation lives here (the mock has no separate
// picker) so the existing multi-timeline capability is preserved.
export function EditorToolbar({
  projectName,
  subLabel,
  timelines = [],
  selectedTimelineId,
  onSelectTimeline,
  onNewTimeline,
  onUndo,
  onRedo,
  canUndo,
  canRedo,
  timecode,
  durationTimecode,
  zoomPct,
  onZoomIn,
  onZoomOut,
  onSave,
  saveDisabled,
  onExport,
  exportDisabled,
}) {
  return (
    <div className="ve-toolbar">
      <div className="ve-brand">
        <span className="ve-mark">
          <Icon.Video size={14} />
        </span>
        <div className="ve-brand-text">
          <strong>{projectName}</strong>
          <span className="ve-brand-sub">{subLabel}</span>
        </div>
      </div>

      <div className="ve-toolbar-div" />

      <div className="ve-timeline-switch">
        <select
          aria-label="Timeline"
          className="ve-select ve-timeline-select"
          onChange={(e) => onSelectTimeline?.(e.target.value)}
          value={selectedTimelineId ?? ""}
        >
          <option value="">Select timeline</option>
          {timelines.map((timeline) => (
            <option key={timeline.id} value={timeline.id}>
              {timeline.name}
            </option>
          ))}
        </select>
        <button className="ve-icon-btn" onClick={onNewTimeline} title="New timeline" type="button">
          <Icon.Plus size={15} />
        </button>
      </div>

      <div className="ve-toolbar-div" />

      <div className="ve-btn-group">
        <button className="ve-ghost-btn" disabled={!canUndo} onClick={onUndo} title="Undo" type="button">
          <Icon.ArrowLeft size={15} />
        </button>
        <button className="ve-ghost-btn" disabled={!canRedo} onClick={onRedo} title="Redo" type="button">
          <Icon.ArrowRight size={15} />
        </button>
      </div>

      <div className="ve-timecode">
        <span className="ve-tc-now">{timecode}</span>
        <span className="ve-tc-total">/ {durationTimecode}</span>
      </div>

      <div className="ve-toolbar-right">
        <button className="ve-ghost-btn" onClick={onZoomOut} title="Zoom out" type="button">
          <Icon.Minus size={15} />
        </button>
        <span className="ve-zoom-pct">{zoomPct}</span>
        <button className="ve-ghost-btn" onClick={onZoomIn} title="Zoom in" type="button">
          <Icon.Plus size={15} />
        </button>
        <button className="ve-ghost-btn" disabled={saveDisabled} onClick={onSave} title="Save timeline" type="button">
          <Icon.Save size={15} />
        </button>
        <button className="ve-export" disabled={exportDisabled} onClick={onExport} type="button">
          <Icon.Save size={14} />
          Export
        </button>
      </div>
    </div>
  );
}
