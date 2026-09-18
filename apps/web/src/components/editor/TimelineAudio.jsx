import React, { forwardRef, useCallback, useEffect, useImperativeHandle, useLayoutEffect, useRef } from "react";
import { assetCanRenderAsAudio, assetCanRenderAsVideo, assetUrl } from "../assetMedia.jsx";
import { createTimelineAudioPlayback } from "../../timelineAudioPlayback.js";

function Layer({ placement, asset, playback }) {
  const ref = useCallback((element) => playback.attach(placement.key, element, placement), [playback, placement.key]); // eslint-disable-line react-hooks/exhaustive-deps
  return <audio ref={ref} src={assetUrl(asset)} crossOrigin="anonymous" preload="auto"
    aria-label={`${placement.track.name ?? placement.track.id}: ${placement.item.displayName ?? asset.displayName ?? asset.id}`}
    onError={() => playback.mediaError(placement.key)} />;
}

export const TimelineAudio = forwardRef(function TimelineAudio({ plan, assetsById, time, playing, onError }, ref) {
  const errorRef = useRef(onError);
  errorRef.current = onError;
  const playbackRef = useRef(null);
  if (!playbackRef.current) playbackRef.current = createTimelineAudioPlayback((error) => errorRef.current(error));
  const playback = playbackRef.current;
  const lifetime = useRef(0);
  useEffect(() => {
    lifetime.current += 1;
    return () => {
      const token = ++lifetime.current;
      playback.pause();
      // Strict Mode replays effects without replacing media elements.
      queueMicrotask(() => { if (lifetime.current === token) playback.dispose(); });
    };
  }, [playback]);
  useLayoutEffect(() => playback.update(plan.placements, time, playing), [playback, plan, time, playing]);
  useImperativeHandle(ref, () => playback, [playback]);
  const layers = plan.placements.map((placement) => {
    const asset = assetsById.get(placement.assetId);
    const reason = !asset ? "asset unavailable" : asset.file?.hasAudio === false || (!assetCanRenderAsAudio(asset) && !assetCanRenderAsVideo(asset)) ? "no audio stream" : null;
    return { placement, asset, reason };
  });
  return <>
    <div className="ve-timeline-audio" hidden>
      {layers.filter((layer) => !layer.reason).map(({ placement, asset }) =>
        <Layer key={`${placement.key}:${asset.id}`} placement={placement} asset={asset} playback={playback} />)}
    </div>
    {layers.filter((layer) => layer.reason).map(({ placement, reason }) =>
      <p className="ve-notice" key={placement.key}>Audio preview skips {placement.item.displayName ?? placement.assetId}: {reason}.</p>)}
  </>;
});
