import React, { useEffect, useMemo, useRef } from "react";
import { Icon } from "../components/Icons.jsx";
import { AssetMedia, assetCanRenderAsAudio, assetCanRenderAsVideo } from "../components/assetMedia.jsx";
import { useAudioTakePlayer } from "../components/audioTakeParts.jsx";
import { audioAssetMetaLine, audioAssetRunGroups } from "../audioTakes.js";
import { useAppContext } from "../context/AppContext.js";
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
export function SimplePreview({
  asset,
  onClose,
  onUseAsReference,
  onToggleFavorite,
  onDelete,
  onSendToVideo,
  audioAutoPlay = false,
  breakpoint = "desktop",
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

  // The tile's play button opens the viewer already playing; opening the tile itself just
  // loads the clip. Guarded so a re-render can't restart playback the user paused.
  const started = useRef(false);
  useEffect(() => {
    if (!audio || started.current) {
      return;
    }
    started.current = true;
    if (audioAutoPlay) {
      player.toggleTake(asset);
    }
  }, [audio, audioAutoPlay, asset, player]);

  return (
    <div aria-label={`${title} preview`} className="su-preview" role="dialog">
      <div className="su-preview-head">
        <button aria-label="Close preview" className="su-preview-close" onClick={onClose} type="button">
          <Icon.Close size={18} />
        </button>
        <strong>{asset.displayName ?? title}</strong>
      </div>
      <div className={audio ? "su-preview-stage su-preview-stage--audio" : "su-preview-stage"}>
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
