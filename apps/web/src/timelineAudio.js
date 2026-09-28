const number = (value, fallback = 0) => Number.isFinite(Number(value)) && value != null ? Number(value) : fallback;
const clamp = (value, low, high) => Math.min(high, Math.max(low, value));

// Mirrors media_jobs.rs: plan_segments, PictureTiming, and audio_placements.
// Solo is an editor audition control, not a saved/exported mixing policy.
export function timelineAudioPlan(timeline, trackSoloed = {}) {
  const tracks = timeline?.tracks ?? [];
  const pictureTrack = tracks.find((track) => track.id === "track_main" || track.kind === "video");
  const pictures = [...(pictureTrack?.items ?? [])].sort((a, b) => a.timelineStart - b.timelineStart);
  const boundaries = [];
  let cursor = 0, segmentCursor = 0, absorbed = 0, segmentCount = 0;
  for (const item of pictures) {
    const start = number(item.timelineStart);
    if (start > cursor) {
      boundaries.push({ timeline: segmentCursor, absorbed });
      segmentCursor += start - cursor;
      segmentCount += 1;
    }
    if (segmentCount > 0 && item.transitionIn?.type === "crossfade") {
      absorbed += clamp(number(item.transitionIn.duration, 0.5), 0.1, 1.5);
    }
    boundaries.push({ timeline: segmentCursor, absorbed });
    segmentCursor += Math.max(0.1, number(item.timelineEnd) - start);
    cursor = Math.max(cursor, number(item.timelineEnd));
    segmentCount += 1;
  }
  // Audio-only timelines retain ordinary source audition, without requiring picture.
  const duration = pictures.length ? Math.max(0, segmentCursor - absorbed)
    : Math.max(0, ...tracks.flatMap((track) => track.items.map((item) => number(item.timelineEnd))));
  const toPictureTime = (time) => Math.max(0, time - (boundaries.filter((b) => b.timeline <= time + 1e-9).at(-1)?.absorbed ?? 0));
  const toTimelineTime = (time) => time + (boundaries.filter((b) => b.timeline - b.absorbed <= time + 1e-9).at(-1)?.absorbed ?? 0);
  const placements = tracks.flatMap((track) => {
    const excludedBySolo = Object.values(trackSoloed).some(Boolean) && !trackSoloed[track.id];
    if (track.muted || excludedBySolo || (track.kind !== "audio" && track !== pictureTrack)) return [];
    return track.items.flatMap((item) => {
      const generated = track.kind !== "audio";
      if (!item.assetId || (generated && item.generatedAudio !== "include")) return [];
      const start = toPictureTime(Math.max(0, number(item.timelineStart)));
      const span = Math.min(number(item.timelineEnd) - Math.max(0, number(item.timelineStart)), duration - start);
      if (span <= 0) return [];
      const speed = clamp(number(item.speed, 1), 0.1, 8);
      const sourceIn = Math.max(0, number(item.sourceIn));
      const sourceOut = number(item.sourceOut) > sourceIn ? number(item.sourceOut) : sourceIn + span * speed;
      return [{
        key: `${track.id}:${item.id}`, item, track, assetId: item.assetId, generated, start, span, speed, sourceIn, sourceOut,
        gain: clamp(number(track.gain, 1), 0, 4) * clamp(number(item.volume, 1), 0, 2),
        fadeIn: clamp(number(item.fadeInSeconds), 0, span), fadeOut: clamp(number(item.fadeOutSeconds), 0, span),
      }];
    });
  }).sort((a, b) => a.start - b.start || a.track.id.localeCompare(b.track.id) || a.assetId.localeCompare(b.assetId));
  return { duration, placements, pictures, toPictureTime, toTimelineTime };
}

export function timelineAudioState(placement, time) {
  const elapsed = clamp(time - placement.start, 0, placement.span);
  const currentTime = Math.min(placement.sourceOut, placement.sourceIn + elapsed * placement.speed);
  const active = time >= placement.start && time < placement.start + placement.span && currentTime < placement.sourceOut;
  const fadeIn = placement.fadeIn > 0 ? Math.min(1, elapsed / placement.fadeIn) : 1;
  const fadeOut = placement.fadeOut > 0 ? Math.min(1, (placement.span - elapsed) / placement.fadeOut) : 1;
  return { currentTime, active, gain: active ? placement.gain * fadeIn * fadeOut : 0 };
}
