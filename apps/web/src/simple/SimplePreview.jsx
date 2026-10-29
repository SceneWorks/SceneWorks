import React, { useEffect, useMemo, useRef } from "react";
import { Icon } from "../components/Icons.jsx";
import { AssetMedia, assetCanRenderAsAudio, assetCanRenderAsVideo } from "../components/assetMedia.jsx";
import { useAudioTakePlayer } from "../components/audioTakeParts.jsx";
import { audioAssetMetaLine, audioAssetRunGroups } from "../audioTakes.js";
import { useAppContext } from "../context/AppContext.js";
import { usePreviewArrowNavigation } from "../hooks/usePreviewArrowNavigation.js";
import { SimpleAudioViewer } from "./simpleAudioParts.jsx";
import { DownloadButton } from "./studioParts.jsx";

// Full-screen asset preview (design handoff): large media over four actions — "Use as
// reference" (primary), Favorite, Download, Delete (danger).
//
// For an AUDIO asset the stage is the play deck (epic 14361 / sc-14366) rather than a bare
// <audio> element: waveform panel, played band, transport and clock, plus "Send to Video
// Editor" — on desktop at the transport's right edge, on phone in the primary slot that
// "Use as reference" leaves empty (it stays hidden for audio, as before).
//
// Delete routes to `deleteAsset`, which moves the asset to the trash (recoverable in the
// full workspace's Assets → Trashcan). The Simple UI deliberately has no permanent-purge
// path: an irreversible delete is not a control this surface should own.
//
// When opened from a grid, the viewer steps through that grid: prev/next buttons at the stage
// edges, Left/Right arrow keys, and a horizontal swipe on touch screens.

// A swipe must travel this far sideways, and mostly sideways, to count as navigation.
const SWIPE_MIN_PX = 50;
// Native <video> controls sit along the element's bottom edge; a drag there is scrubbing.
const VIDEO_CONTROLS_PX = 56;

export function SimplePreview({
  asset,
  onClose,
  onUseAsReference,
  onToggleFavorite,
  onDelete,
  onSendToVideo,
  audioAutoPlay = false,
  breakpoint = "desktop",
  previousAsset = null,
  nextAsset = null,
  onNavigate,
  position = null,
}) {
  const { audioModels = [] } = useAppContext();
  const audio = assetCanRenderAsAudio(asset);
  const title = assetCanRenderAsVideo(asset) ? "Clip" : audio ? "Audio" : "Image";
  const favorite = Boolean(asset.status?.favorite);
  const phone = breakpoint === "phone";
  const player = useAudioTakePlayer();

  // The clip's own run, reconstructed from its replay record — the same derivation the studios
  // use, so the viewer's mode chip and model name can't disagree with the take grid's.
  const run = useMemo(
    () => (audio ? (audioAssetRunGroups([asset], audioModels)[0] ?? null) : null),
    [audio, asset, audioModels],
  );

  // Stepping to another asset stops the clip — the next one must not play under the last one's
  // audio. Compared against the last shown id (not an effect cleanup) so StrictMode's
  // mount/unmount replay can't cancel an autoplay.
  const shownId = useRef(asset.id);
  const { unload } = player;
  useEffect(() => {
    if (shownId.current !== asset.id) {
      shownId.current = asset.id;
      unload();
    }
  }, [asset.id, unload]);

  // The tile's play button opens the viewer already playing; opening the tile itself just
  // loads the clip. Guarded per asset so a re-render can't restart playback the user paused.
  const started = useRef(null);
  useEffect(() => {
    if (!audio || started.current === asset.id) {
      return;
    }
    started.current = asset.id;
    if (audioAutoPlay) {
      player.toggleTake(asset);
    }
  }, [audio, audioAutoPlay, asset, player]);

  const goPrevious = previousAsset && onNavigate ? () => onNavigate(previousAsset) : null;
  const goNext = nextAsset && onNavigate ? () => onNavigate(nextAsset) : null;
  const rootRef = useRef(null);
  usePreviewArrowNavigation({ rootRef, onPrevious: goPrevious, onNext: goNext });

  // Take focus on open so keys reach the viewer, not the grid tile that opened it; hand it back
  // on close (same contract as the shared Modal).
  useEffect(() => {
    const previousFocus = document.activeElement;
    rootRef.current?.focus();
    return () => {
      if (previousFocus?.isConnected && typeof previousFocus.focus === "function") {
        previousFocus.focus();
      }
    };
  }, []);

  const swipeStart = useRef(null);
  const onTouchStart = (event) => {
    const touch = event.touches[0];
    const target = event.target;
    const onScrubber = target.closest?.('input, [role="slider"]');
    const video = target.closest?.("video");
    const onVideoControls = video && touch.clientY > video.getBoundingClientRect().bottom - VIDEO_CONTROLS_PX;
    swipeStart.current =
      event.touches.length === 1 && !onScrubber && !onVideoControls ? { x: touch.clientX, y: touch.clientY } : null;
  };
  const onTouchEnd = (event) => {
    const start = swipeStart.current;
    swipeStart.current = null;
    const touch = event.changedTouches[0];
    if (!start || !touch) {
      return;
    }
    const dx = touch.clientX - start.x;
    const dy = touch.clientY - start.y;
    if (Math.abs(dx) < SWIPE_MIN_PX || Math.abs(dx) < Math.abs(dy) * 1.5) {
      return;
    }
    // Swipe left reveals the next asset; swipe right goes back.
    (dx < 0 ? goNext : goPrevious)?.();
  };
  const navigable = Boolean(previousAsset || nextAsset);

  return (
    <div aria-label={`${title} preview`} className="su-preview" ref={rootRef} role="dialog" tabIndex={-1}>
      <div className="su-preview-head">
        <button aria-label="Close preview" className="su-preview-close" onClick={onClose} type="button">
          <Icon.Close size={18} />
        </button>
        <strong>{asset.displayName ?? title}</strong>
        {position ? (
          <span className="su-preview-position">
            {position.index} / {position.total}
          </span>
        ) : null}
      </div>
      <div
        className={audio ? "su-preview-stage su-preview-stage--audio" : "su-preview-stage"}
        onTouchCancel={() => (swipeStart.current = null)}
        onTouchEnd={onTouchEnd}
        onTouchStart={onTouchStart}
      >
        {audio ? (
          <SimpleAudioViewer
            asset={asset}
            breakpoint={breakpoint}
            meta={audioAssetMetaLine(asset, run)}
            onSendToVideo={onSendToVideo}
            player={player}
            run={run}
          />
        ) : (
          <AssetMedia asset={asset} />
        )}
        {navigable ? (
          <>
            <button
              aria-label="Previous asset"
              className="su-preview-nav su-preview-nav--previous"
              disabled={!goPrevious}
              onClick={() => goPrevious?.()}
              type="button"
            >
              <Icon.ArrowLeft size={18} />
            </button>
            <button
              aria-label="Next asset"
              className="su-preview-nav su-preview-nav--next"
              disabled={!goNext}
              onClick={() => goNext?.()}
              type="button"
            >
              <Icon.ArrowRight size={18} />
            </button>
          </>
        ) : null}
      </div>
      <div className="su-preview-actions">
        {audio ? (
          phone && onSendToVideo ? (
            <button className="su-preview-primary" onClick={() => onSendToVideo(asset)} type="button">
              <Icon.Editor size={16} />
              Send to Video Editor
            </button>
          ) : null
        ) : (
          <button className="su-preview-primary" onClick={() => onUseAsReference(asset)} type="button">
            <Icon.Image size={16} />
            Use as reference
          </button>
        )}
        <div className="su-preview-row">
          <button onClick={() => onToggleFavorite(asset)} type="button">
            <Icon.Star filled={favorite} size={15} />
            {favorite ? "Favorited" : "Favorite"}
          </button>
          <DownloadButton asset={asset} className="" label="Download" />
          <button className="danger" onClick={() => onDelete(asset)} type="button">
            <Icon.Trash size={15} />
            Delete
          </button>
        </div>
      </div>
    </div>
  );
}
