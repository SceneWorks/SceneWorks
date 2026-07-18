import React, { useState } from "react";
import { AssetMedia, assetCanRenderAsImage } from "../assetMedia.jsx";
import { Icon } from "../Icons.jsx";
import { isAiAsset } from "./editorUtils.js";

// The left media bin (design 2a, epic 12798). Two tabs: "Media" (video + still assets
// you drop onto the timeline) and "Assets" (the full project media library for preview).
// Clicking a Media thumbnail adds it to the timeline; the violet sparkle badge marks
// AI-generated assets (derived from origin/recipe, mirroring the LikenessBadge overlay).
export function MediaBin({ assets = [], onAddToTrack, onPreview }) {
  const [tab, setTab] = useState("media");

  const clipAssets = assets.filter(
    (asset) => asset.type === "video" || asset.file?.mimeType?.startsWith("video/") || assetCanRenderAsImage(asset),
  );
  const shown = tab === "media" ? clipAssets : assets;

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
          shown.slice(0, 60).map((asset) => (
            <button
              className="ve-bin-item"
              key={asset.id}
              onClick={() => handleClick(asset)}
              title={tab === "media" ? `Add ${asset.displayName} to timeline` : asset.displayName}
              type="button"
            >
              <AssetMedia asset={asset} className="ve-bin-thumb" controls={false} />
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
    </div>
  );
}
