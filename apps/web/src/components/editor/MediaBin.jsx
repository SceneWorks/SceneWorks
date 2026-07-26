import React, { useEffect, useState } from "react";
import { AssetThumbnail, assetCanRenderAsImage } from "../assetMedia.jsx";
import { Icon } from "../Icons.jsx";
import { isAiAsset } from "./editorUtils.js";

const MEDIA_PAGE_SIZE = 60;

// The left media bin (design 2a, epic 12798). Two tabs: "Media" (video + still assets
// you drop onto the timeline) and "Assets" (the full project media library for preview).
// Clicking a Media thumbnail adds it to the timeline; the violet sparkle badge marks
// AI-generated assets (derived from origin/recipe, mirroring the LikenessBadge overlay).
export function MediaBin({ assets = [], onAddToTrack, onPreview }) {
  const [tab, setTab] = useState("media");
  const [visibleCount, setVisibleCount] = useState(MEDIA_PAGE_SIZE);

  const clipAssets = assets.filter(
    (asset) => asset.type === "video" || asset.file?.mimeType?.startsWith("video/") || assetCanRenderAsImage(asset),
  );
  const shown = tab === "media" ? clipAssets : assets;
  const visible = shown.slice(0, visibleCount);

  useEffect(() => {
    setVisibleCount(MEDIA_PAGE_SIZE);
  }, [tab]);

  function handleClick(asset) {
    if (tab === "media") {
      onAddToTrack?.(asset);
    } else {
      onPreview?.(asset);
    }
  }

  return (
    <div className="ve-bin">
      <div className="ve-bin-tabs">
        <button className={`ve-bin-tab${tab === "media" ? " active" : ""}`} onClick={() => setTab("media")} type="button">
          Media
        </button>
        <button className={`ve-bin-tab${tab === "assets" ? " active" : ""}`} onClick={() => setTab("assets")} type="button">
          Assets
        </button>
      </div>
      <div className="ve-bin-grid">
        {shown.length === 0 ? (
          <div className="ve-bin-empty">No media</div>
        ) : (
          visible.map((asset) => (
            <button
              className="ve-bin-item"
              key={asset.id}
              onClick={() => handleClick(asset)}
              title={tab === "media" ? `Add ${asset.displayName} to timeline` : asset.displayName}
              type="button"
            >
              <AssetThumbnail asset={asset} className="ve-bin-thumb" />
              {isAiAsset(asset) ? (
                <span className="ve-ai-badge" aria-label="AI generated">
                  <Icon.Stars size={10} />
                </span>
              ) : null}
              <span className="ve-bin-name">{asset.displayName}</span>
            </button>
          ))
        )}
      </div>
      {shown.length > MEDIA_PAGE_SIZE ? (
        <div className="ve-bin-more">
          <span aria-live="polite">
            Showing {visible.length} of {shown.length}
          </span>
          {visible.length < shown.length ? (
            <button
              onClick={() => setVisibleCount((count) => count + MEDIA_PAGE_SIZE)}
              type="button"
            >
              Show more
            </button>
          ) : null}
        </div>
      ) : null}
    </div>
  );
}
