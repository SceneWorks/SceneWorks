import { describe, it, expect } from "vitest";
import { timelineEdit, applyTimelineEdit, invertTimelineEdit, rebaseTimeline } from "./timelineEdits.js";
const item = (id, start = 0) => ({ id, displayName: id, timelineStart: start, timelineEnd: start + 4, sourceIn: 0, sourceOut: 4 });
const cut = () => ({ id: "film", revision: 1, tracks: [{ id: "picture", items: [item("A")] }, { id: "sound", gain: 0.5, items: [item("music")] }] });
describe("stable user operations", () => {
  it("undo/redo of an earlier edit preserves later delivered shots and unrelated audio edits", () => {
    const original = cut(); const edited = structuredClone(original);
    edited.tracks[0].items[0].sourceIn = 1;
    const edit = timelineEdit(original, edited);
    const delivered = structuredClone(edited); delivered.revision = 2;
    delivered.tracks[0].items.push(item("B", 4)); delivered.tracks[1].gain = 0.25;
    const undone = applyTimelineEdit(delivered, invertTimelineEdit(edit));
    expect(undone.conflicts).toEqual([]);
    expect(undone.timeline.tracks[0].items.map((i) => i.id)).toEqual(["A", "B"]);
    expect(undone.timeline.tracks[0].items[0].sourceIn).toBe(0);
    expect(undone.timeline.tracks[1].gain).toBe(0.25);
    const redone = applyTimelineEdit(undone.timeline, edit);
    expect(redone.timeline.tracks[0].items[0].sourceIn).toBe(1);
    expect(redone.timeline.tracks[0].items[1].id).toBe("B");
  });
  it("delete undo restores only the deleted item over a newer base", () => {
    const original = cut(), deleted = cut(); deleted.tracks[0].items = [];
    const edit = timelineEdit(original, deleted);
    deleted.tracks[0].items.push(item("B", 4));
    const restored = applyTimelineEdit(deleted, invertTimelineEdit(edit));
    expect(restored.conflicts).toEqual([]);
    expect(restored.timeline.tracks[0].items.map((i) => i.id).sort()).toEqual(["A", "B"]);
  });
  it("disjoint edits rebase and same-item edits expose named choices", () => {
    const original = cut(), working = cut(), stored = cut();
    working.tracks[0].items[0].sourceOut = 3;
    stored.revision = 2; stored.tracks[1].gain = 0.2; stored.tracks[0].items.push(item("B", 4));
    expect(rebaseTimeline(original, working, stored).conflicts).toEqual([]);
    stored.tracks[0].items[0].sourceOut = 2;
    const conflicted = rebaseTimeline(original, working, stored);
    expect(conflicted.conflicts[0].label).toBe("A");
    expect(conflicted.timeline.tracks[0].items[0].sourceOut).toBe(2);
    const local = rebaseTimeline(original, working, stored, { force: true });
    expect(local.timeline.tracks[0].items[0].sourceOut).toBe(3);
    expect(local.timeline.tracks[0].items[1].id).toBe("B");
    expect(local.timeline.tracks[1].gain).toBe(0.2);
  });
});
