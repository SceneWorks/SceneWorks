import React from "react";
import { Icon } from "../../components/Icons.jsx";
import { StudioGuideButton } from "../../components/StudioFields.jsx";
import { VIDEO_MODE_GROUPS, modeForGroupPick, videoModeGroupFor } from "./videoModeGroups.js";

// Mode picker (studio layout simplification): a segmented group switch (Create / From video /
// People) with the picked group's modes as underline tabs underneath. Prompt guide and Saved
// presets sit right of the switch. Per-mode availability is unchanged: a tab is disabled only
// when `modeTabBlocked` says so and it is not the active tab (the active tab is never disabled,
// so a reduced catalog can't strand the user). A group is disabled only when every one of its
// modes is blocked and it is not the current group.
export function VideoModePicker({
  modeOptions,
  mode,
  setMode,
  modeTabBlocked,
  modeTabBlockedText,
  openGuide,
  onOpenPresets,
}) {
  const labels = new Map(modeOptions);
  const currentGroup = videoModeGroupFor(mode);
  const blocked = (value) => value !== mode && modeTabBlocked(value);
  return (
    <div className="video-mode-picker">
      <div className="video-mode-picker-top">
        <div aria-label="Mode group" className="video-mode-groups" role="group">
          {VIDEO_MODE_GROUPS.map((group) => {
            const active = group.id === currentGroup.id;
            const target = modeForGroupPick(group, mode, blocked);
            return (
              <button
                aria-pressed={active}
                className={active ? "video-mode-group on" : "video-mode-group"}
                disabled={!active && target === null}
                key={group.id}
                onClick={() => {
                  if (target && target !== mode) setMode(target);
                }}
                title={!active && target === null ? modeTabBlockedText : undefined}
                type="button"
              >
                {group.label}
              </button>
            );
          })}
        </div>
        <div className="prompt-hero-links video-mode-picker-links">
          <StudioGuideButton onOpen={openGuide} />
          {onOpenPresets ? (
            <button className="hero-link" onClick={onOpenPresets} type="button">
              <Icon.Folder size={14} /> Saved presets
            </button>
          ) : null}
        </div>
      </div>
      <div aria-label="Video mode" className="studio-mode-tabs" role="tablist">
        {currentGroup.modes.filter((value) => labels.has(value)).map((value) => {
          const active = value === mode;
          const isBlocked = blocked(value);
          return (
            <button
              aria-selected={active}
              className={active ? "studio-mode-tab on" : "studio-mode-tab"}
              disabled={isBlocked}
              key={value}
              onClick={() => setMode(value)}
              role="tab"
              title={isBlocked ? modeTabBlockedText : undefined}
              type="button"
            >
              {labels.get(value)}
            </button>
          );
        })}
      </div>
    </div>
  );
}
