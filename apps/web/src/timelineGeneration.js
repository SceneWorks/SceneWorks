import { ensureItemVersionFields } from "./timeline.js";

export function applyTimelineGenerationResult(timeline, job) {
  const payload = job.payload ?? {};
  const action = payload.advanced?.timelineAction;
  const context = payload.advanced?.timelineContext ?? {};
  const assetId = job.result?.assetIds?.[0];
  if (!action || !assetId || context.timelineId !== timeline.id) {
    return timeline;
  }
  const resultAsset = job.result?.assets?.[0];
  const displayName = resultAsset?.displayName ?? "Generated clip";
  const createdAt = resultAsset?.createdAt ?? job.updatedAt ?? null;
  if (timeline.timelineGenerationJobs?.[job.id]) return timeline;
  if (action === "replace") {
    const target = timeline.tracks.find((t) => t.id === context.trackId)?.items.find((i) => i.id === context.itemId);
    const measured = Number(resultAsset?.file?.duration ?? resultAsset?.duration);
    if (target && measured > 0 && Number(target.sourceOut) > measured) {
      return { ...timeline, generationTrimConflicts: { ...timeline.generationTrimConflicts, [job.id]: {
        itemId: target.id, trackId: context.trackId, sourceIn: target.sourceIn, sourceOut: target.sourceOut, newDuration: measured,
        choices: measured - Number(target.sourceIn) >= 0.1 ? ["clamp", "reset", "keepCurrent"] : ["reset", "keepCurrent"],
        take: { assetId, displayName, createdAt, jobId: job.id },
      } } };
    }
  }
  const tracks = timeline.tracks.map((track) => {
    if (track.id !== context.trackId) {
      return track;
    }
    if ((track.items ?? []).some((item) => (item.versionHistory ?? []).some((entry) => entry.jobId === job.id))) return track;
    if (action === "bridge") {
      const bridgeItem = ensureItemVersionFields({
        id: `item_job_${job.id.replaceAll(/[^a-zA-Z0-9_]/g, "_")}`,
        trackId: track.id,
        assetId,
        type: "video",
        displayName,
        sourceIn: 0,
        sourceOut: Number(payload.duration) || Math.max(0.1, Number(context.timelineEnd) - Number(context.timelineStart)),
        timelineStart: Number(context.timelineStart),
        timelineEnd: Number(context.timelineEnd),
        speed: 1,
        fit: "fit",
        volume: 1,
        versionAssetIds: [assetId],
        currentVersionAssetId: assetId,
        versionHistory: [{ assetId, createdAt, source: "bridge", jobId: job.id, note: "Generated bridge clip" }],
        transitionIn: { id: `transition_${crypto.randomUUID().replaceAll("-", "")}`, type: "cut", duration: 0 },
        transitionOut: { id: `transition_${crypto.randomUUID().replaceAll("-", "")}`, type: "cut", duration: 0 },
      });
      return { ...track, items: [...track.items, bridgeItem] };
    }
    if (action === "extend") {
      const start = Number(context.timelineStart);
      const duration = Number(payload.duration) || 4;
      const extensionItem = ensureItemVersionFields({
        id: `item_job_${job.id.replaceAll(/[^a-zA-Z0-9_]/g, "_")}`,
        trackId: track.id,
        assetId,
        type: "video",
        displayName,
        sourceIn: 0,
        sourceOut: duration,
        timelineStart: start,
        timelineEnd: start + duration,
        speed: 1,
        fit: "fit",
        volume: 1,
        versionAssetIds: [assetId],
        currentVersionAssetId: assetId,
        versionHistory: [{ assetId, createdAt, source: "extension", jobId: job.id, note: "Generated extension" }],
        transitionIn: { id: `transition_${crypto.randomUUID().replaceAll("-", "")}`, type: "cut", duration: 0 },
        transitionOut: { id: `transition_${crypto.randomUUID().replaceAll("-", "")}`, type: "cut", duration: 0 },
      });
      return { ...track, items: [...track.items, extensionItem] };
    }
    if (action === "replace") {
      return {
        ...track,
        items: track.items.map((item) => {
          if (item.id !== context.itemId) {
            return item;
          }
          const current = ensureItemVersionFields(item);
          return {
            ...current,
            assetId,
            currentVersionAssetId: assetId,
            type: "video",
            displayName,
            versionAssetIds: Array.from(new Set([...current.versionAssetIds, assetId])),
            versionHistory: [
              ...current.versionHistory,
              { assetId, createdAt, source: "replacement", jobId: job.id, note: "Generated replacement" },
            ],
          };
        }),
      };
    }
    return track;
  });
  return { ...timeline, tracks };
}


export function resolveGenerationTrim(saved, jobId, resolution) {
  const conflict = saved.generationTrimConflicts?.[jobId];
  if (!conflict || !conflict.choices.includes(resolution)) return null;
  const next = structuredClone(saved);
  const item = next.tracks.find((track) => track.id === conflict.trackId)?.items.find((clip) => clip.id === conflict.itemId);
  if (!item) throw new Error("The replacement target was deleted. Keep the generated asset in the media bin or restore the clip first.");
  if (resolution !== "keepCurrent") {
    const sourceIn = resolution === "reset" ? 0 : Number(item.sourceIn);
    if (conflict.newDuration - sourceIn < 0.1) throw new Error("The current in-point no longer fits. Choose the full new take.");
    const current = ensureItemVersionFields(item);
    Object.assign(item, current, { assetId: conflict.take.assetId, currentVersionAssetId: conflict.take.assetId,
      sourceIn, sourceOut: conflict.newDuration, timelineEnd: item.timelineStart + (conflict.newDuration - sourceIn) / (item.speed || 1),
      versionAssetIds: Array.from(new Set([...current.versionAssetIds, conflict.take.assetId])),
      versionHistory: [...current.versionHistory, { ...conflict.take, source: "replacement" }],
    });
  }
  delete next.generationTrimConflicts[jobId];
  next.timelineGenerationJobs = { ...next.timelineGenerationJobs, [jobId]: resolution };
  return next;
}
