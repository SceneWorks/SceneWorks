import React from "react";
import { Icon } from "../components/Icons.jsx";
import { AssetMedia, assetCanRenderAsAudio, assetCanRenderAsVideo } from "../components/assetMedia.jsx";
import { DownloadButton } from "./studioParts.jsx";

// Full-screen asset preview (design handoff): large media over four actions — "Use as
// reference" (primary), Favorite, Download, Delete (danger).
//
// Delete routes to `deleteAsset`, which moves the asset to the trash (recoverable in the
// full workspace's Assets → Trashcan). The Simple UI deliberately has no permanent-purge
// path: an irreversible delete is not a control this surface should own.
export function SimplePreview({ asset, onClose, onUseAsReference, onToggleFavorite, onDelete }) {
  const title = assetCanRenderAsVideo(asset) ? "Clip" : assetCanRenderAsAudio(asset) ? "Audio" : "Image";
  const favorite = Boolean(asset.status?.favorite);
  return (
    <div aria-label={`${title} preview`} className="su-preview" role="dialog">
      <div className="su-preview-head">
        <button aria-label="Close preview" className="su-preview-close" onClick={onClose} type="button">
          <Icon.Close size={18} />
        </button>
        <strong>{asset.displayName ?? title}</strong>
      </div>
      <div className="su-preview-stage">
        <AssetMedia asset={asset} />
      </div>
      <div className="su-preview-actions">
        {assetCanRenderAsAudio(asset) ? null : (
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
