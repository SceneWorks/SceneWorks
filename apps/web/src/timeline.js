
export function timelineDuration(timeline) {
  return Math.max(0, ...timeline.tracks.flatMap((track) => track.items.map((item) => Number(item.timelineEnd) || 0)));
}

export function itemDuration(item) {
  return Math.max(0.1, Number(item.timelineEnd) - Number(item.timelineStart));
}

export function trackItems(track) {
  return [...track.items].sort((a, b) => a.timelineStart - b.timelineStart);
}

export function sourceTimestampAtPlayhead(item, playheadSeconds) {
  const start = Number(item.timelineStart) || 0;
  const end = Math.max(start, Number(item.timelineEnd) || start);
  const clamped = Math.min(Math.max(Number(playheadSeconds) || start, start), end);
  return (Number(item.sourceIn) || 0) + (clamped - start) * (Number(item.speed) || 1);
}

export function audioPreviewState(item, track, playheadSeconds, trackSoloed = {}) {
  const timelineStart = Number(item.timelineStart) || 0;
  const timelineEnd = Math.max(timelineStart, Number(item.timelineEnd) || timelineStart);
  const playhead = Number(playheadSeconds) || 0;
  const elapsed = Math.max(0, playhead - timelineStart);
  const remaining = Math.max(0, timelineEnd - playhead);
  const fadeInSeconds = Math.max(0, Number(item.fadeInSeconds) || 0);
  const fadeOutSeconds = Math.max(0, Number(item.fadeOutSeconds) || 0);
  const fadeInGain = fadeInSeconds > 0 ? Math.min(1, elapsed / fadeInSeconds) : 1;
  const fadeOutGain = fadeOutSeconds > 0 ? Math.min(1, remaining / fadeOutSeconds) : 1;
  const clipGain = Math.min(2, Math.max(0, Number(item.volume ?? 1)));
  const trackGain = Math.min(4, Math.max(0, Number(track?.gain ?? 1)));
  const anySoloed = Object.values(trackSoloed).some(Boolean);
  const beforePlacement = playhead < timelineStart;
  const afterPlacement = playhead >= timelineEnd;

  return {
    afterPlacement,
    beforePlacement,
    currentTime: sourceTimestampAtPlayhead(item, playhead),
    muted: Boolean(track?.muted) || (anySoloed && !trackSoloed[track?.id]) || beforePlacement || afterPlacement,
    playbackRate: Math.max(0.01, Number(item.speed) || 1),
    // Keep this as the actual mix gain. HTMLMediaElement.volume cannot represent
    // values above one, so the editor routes this value through a Web Audio gain
    // node. Export applies the same clip × track × fade product.
    gain: clipGain * trackGain * fadeInGain * fadeOutGain,
  };
}

export function ensureItemVersionFields(item) {
  const versionAssetIds = Array.from(new Set([...(item.versionAssetIds ?? []), item.assetId].filter(Boolean)));
  return {
    ...item,
    currentVersionAssetId: item.currentVersionAssetId ?? item.assetId,
    versionAssetIds,
    versionHistory:
      item.versionHistory?.length > 0
        ? item.versionHistory
        : [{ assetId: item.assetId, createdAt: null, source: "original", jobId: null, note: null }],
  };
}
