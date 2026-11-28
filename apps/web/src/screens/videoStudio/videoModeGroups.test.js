import { describe, expect, it } from "vitest";
import {
  VIDEO_MODE_GROUPS,
  initialVideoSettingsTab,
  modeForGroupPick,
  videoModeGroupFor,
} from "./videoModeGroups.js";

const ALL_MODES = [
  "text_to_video",
  "image_to_video",
  "first_last_frame",
  "extend_clip",
  "video_bridge",
  "replace_person",
  "video_to_video",
  "reference_to_video",
  "reference_video_to_video",
  "multi_video_to_video",
  "ads2v",
  "animate_character",
];

describe("Video Studio mode groups", () => {
  it("places every video mode in exactly one group", () => {
    const grouped = VIDEO_MODE_GROUPS.flatMap((group) => group.modes);
    expect([...grouped].sort()).toEqual([...ALL_MODES].sort());
    expect(new Set(grouped).size).toBe(grouped.length);
  });

  it("derives the group from the selected mode", () => {
    expect(videoModeGroupFor("text_to_video").id).toBe("create");
    expect(videoModeGroupFor("reference_to_video").id).toBe("create");
    expect(videoModeGroupFor("ads2v").id).toBe("video");
    expect(videoModeGroupFor("animate_character").id).toBe("people");
  });

  it("keeps the current mode when it is already in the picked group", () => {
    const create = VIDEO_MODE_GROUPS[0];
    expect(modeForGroupPick(create, "first_last_frame")).toBe("first_last_frame");
  });

  it("lands on the group's first available mode, or nothing when every mode is blocked", () => {
    const fromVideo = VIDEO_MODE_GROUPS[1];
    expect(modeForGroupPick(fromVideo, "text_to_video")).toBe("extend_clip");
    const blocked = new Set(["extend_clip", "video_bridge"]);
    expect(modeForGroupPick(fromVideo, "text_to_video", (mode) => blocked.has(mode))).toBe("video_to_video");
    expect(modeForGroupPick(fromVideo, "text_to_video", () => true)).toBeNull();
  });
});

describe("Video Studio settings tab persistence", () => {
  it("defaults to Model & output", () => {
    expect(initialVideoSettingsTab({})).toBe("model");
  });

  it("migrates a pre-tab-strip open Advanced to the Sampling tab", () => {
    expect(initialVideoSettingsTab({ advancedOpen: true })).toBe("sampling");
    expect(initialVideoSettingsTab({ advancedOpen: false })).toBe("model");
  });

  it("restores an explicit tab (including a closed strip) over a stale advancedOpen", () => {
    expect(initialVideoSettingsTab({ advancedOpen: true, settingsTab: "loras" })).toBe("loras");
    expect(initialVideoSettingsTab({ advancedOpen: true, settingsTab: null })).toBeNull();
    expect(initialVideoSettingsTab({ settingsTab: "no-such-tab" })).toBe("model");
  });
});
