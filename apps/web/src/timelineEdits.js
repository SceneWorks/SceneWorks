// Reversible user edits keyed by stable track/item IDs. Never restore a whole old cut:
// asynchronous deliveries and unrelated user edits must survive undo and rebase.
const equal = (a, b) => JSON.stringify(a) === JSON.stringify(b);
const clone = (value) => value === undefined ? undefined : structuredClone(value);
const ignored = new Set(["tracks", "revision", "duration", "updatedAt", "createdAt", "filmAssembly"]);

export function timelineEdit(before, after) {
  const changes = [];
  const add = (path, left, right) => {
    if (!equal(left, right)) changes.push({ path, before: clone(left), after: clone(right) });
  };
  for (const key of new Set([...Object.keys(before ?? {}), ...Object.keys(after ?? {})])) {
    if (!ignored.has(key)) add([key], before?.[key], after?.[key]);
  }
  const priorTracks = new Map((before?.tracks ?? []).map((t) => [t.id, t]));
  const nextTracks = new Map((after?.tracks ?? []).map((t) => [t.id, t]));
  for (const id of new Set([...priorTracks.keys(), ...nextTracks.keys()])) {
    const left = priorTracks.get(id), right = nextTracks.get(id);
    if (!left || !right) { add(["tracks", id], left, right); continue; }
    for (const key of new Set([...Object.keys(left), ...Object.keys(right)])) {
      if (key !== "items") add(["tracks", id, key], left[key], right[key]);
    }
    const prior = new Map((left.items ?? []).map((i) => [i.id, i]));
    const next = new Map((right.items ?? []).map((i) => [i.id, i]));
    for (const itemId of new Set([...prior.keys(), ...next.keys()])) add(["tracks", id, "items", itemId], prior.get(itemId), next.get(itemId));
  }
  const beforeCopies = new Map([...priorTracks].map(([id, track]) => [id, clone(track)]));
  const afterCopies = new Map([...nextTracks].map(([id, track]) => [id, clone(track)]));
  return changes.map((change) => change.path.length > 2 ? { ...change, trackBefore: beforeCopies.get(change.path[1]), trackAfter: afterCopies.get(change.path[1]) } : change);
}

export function invertTimelineEdit(edit) {
  return edit.map(({ path, before, after, trackBefore, trackAfter }) => ({ path, before: after, after: before, trackBefore: trackAfter, trackAfter: trackBefore }));
}

export function applyTimelineEdit(timeline, edit, { force = false } = {}) {
  const result = structuredClone(timeline);
  const conflicts = [];
  for (const change of edit) {
    const { path, before, after } = change;
    let track = result.tracks?.find((t) => t.id === path[1]);
    if (!track && path.length > 2 && force && after !== undefined && change.trackAfter) {
      track = clone(change.trackAfter);
      result.tracks.push(track);
    }
    const list = path.length === 2 ? result.tracks : path.length === 4 ? track?.items : null;
    const index = list?.findIndex((v) => v.id === path.at(-1));
    const current = path.length === 1 ? result[path[0]] : list ? list[index] : track?.[path[2]];
    if (equal(current, after)) continue;
    if (!equal(current, before) || (path.length > 2 && !track)) {
      conflicts.push({ ...change, current: clone(current), label: before?.displayName ?? after?.displayName ?? path.join(" / ") });
      if (!force || (path.length > 2 && !track)) continue;
    }
    if (list) {
      if (after === undefined) { if (index >= 0) list.splice(index, 1); }
      else if (index >= 0) list[index] = clone(after);
      else list.push(clone(after));
    } else {
      const object = path.length === 1 ? result : track;
      const key = path.at(-1);
      if (after === undefined) delete object[key]; else object[key] = clone(after);
    }
  }
  result.duration = Math.max(0, ...(result.tracks ?? []).flatMap((t) => (t.items ?? []).map((i) => Number(i.timelineEnd) || 0)));
  return { timeline: result, conflicts };
}

export function rebaseTimeline(base, working, stored, options) {
  return applyTimelineEdit(stored, timelineEdit(base, working), options);
}
