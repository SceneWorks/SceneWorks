// Video Studio mode groups (studio layout simplification). The twelve video modes are grouped
// into three families so the picker reads as one segmented switch plus a short row of underline
// tabs instead of one wrapping row of twelve. Grouping is presentation only: every mode keeps its
// own id, label, availability and gating rules (see `modeTabBlocked` in VideoStudio.jsx).
export const VIDEO_MODE_GROUPS = [
  {
    id: "create",
    label: "Create",
    modes: ["text_to_video", "image_to_video", "first_last_frame", "reference_to_video"],
  },
  {
    id: "video",
    label: "From video",
    modes: [
      "extend_clip",
      "video_bridge",
      "video_to_video",
      "reference_video_to_video",
      "multi_video_to_video",
      "ads2v",
    ],
  },
  {
    id: "people",
    label: "People",
    modes: ["replace_person", "animate_character"],
  },
];

// The group a mode belongs to. The selected mode is the single source of truth for the group, so
// there is no separate persisted "group" state that could disagree with it.
export function videoModeGroupFor(mode) {
  return VIDEO_MODE_GROUPS.find((group) => group.modes.includes(mode)) ?? VIDEO_MODE_GROUPS[0];
}

// The mode to land on when a group is picked: the current mode when it already belongs to the
// group, else the group's first mode that is not blocked. Null when every mode is blocked, so the
// caller leaves the mode alone rather than stranding the user on a disabled tab.
export function modeForGroupPick(group, currentMode, isBlocked = () => false) {
  if (group.modes.includes(currentMode)) {
    return currentMode;
  }
  return group.modes.find((value) => !isBlocked(value)) ?? null;
}

// Settings tab strip (replaces the Advanced disclosure). The open tab is persisted UI state; a
// snapshot written before the strip existed carries `advancedOpen` instead, and an open Advanced
// most closely corresponds to the Sampling tab (seed / steps / guidance / sampler lived there).
export const VIDEO_SETTINGS_TABS = ["model", "loras", "style", "sampling"];
export const DEFAULT_VIDEO_SETTINGS_TAB = "model";

export function initialVideoSettingsTab(saved) {
  if (saved && Object.prototype.hasOwnProperty.call(saved, "settingsTab")) {
    const tab = saved.settingsTab;
    return tab === null || VIDEO_SETTINGS_TABS.includes(tab) ? tab : DEFAULT_VIDEO_SETTINGS_TAB;
  }
  return saved?.advancedOpen === true ? "sampling" : DEFAULT_VIDEO_SETTINGS_TAB;
}
