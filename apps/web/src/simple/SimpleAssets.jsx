import React, { useMemo, useState } from "react";
import { Icon } from "../components/Icons.jsx";
import { AssetThumbnail, assetCanRenderAsAudio, assetCanRenderAsVideo } from "../components/assetMedia.jsx";
import { useAppContext } from "../context/AppContext.js";
import { isLibraryAsset } from "../constants.js";
import { foldUpscaledAssetVariants } from "../assetVariants.js";
import { assetMatchesSearch } from "../screens/LibraryScreen.jsx";
import { assetGridColumns } from "./breakpoint.js";
import { useSimpleUi } from "./SimpleUiContext.js";

// Simple Assets (design handoff): search + type pills + a responsive thumbnail grid.
// Reuses the Library screen's own hygiene rules so the two views can't disagree about
// what belongs in the library: the `isLibraryAsset` origin allow-list, the trashed /
// rejected exclusions, the upscale-variant fold, and the shared free-text matcher.

const FILTERS = [
  { id: "all", label: "All" },
  { id: "image", label: "Images" },
  { id: "video", label: "Clips" },
  { id: "audio", label: "Audio" },
];

export function SimpleAssets() {
  const { assets = [] } = useAppContext();
  const { breakpoint, openPreview } = useSimpleUi();
  const [filter, setFilter] = useState("all");
  const [query, setQuery] = useState("");

  const visible = useMemo(() => {
    const needle = query.trim().toLowerCase();
    const library = assets.filter(isLibraryAsset);
    return foldUpscaledAssetVariants(
      library.filter((asset) => {
        if (asset.status?.trashed || asset.status?.rejected) {
          return false;
        }
        if (filter !== "all" && !matchesFilter(asset, filter)) {
          return false;
        }
        return assetMatchesSearch(asset, needle);
      }),
    );
  }, [assets, filter, query]);

  return (
    <div className="su-screen su-screen--tight">
      <div className="su-search">
        <Icon.Search size={16} />
        <input
          aria-label="Search assets"
          onChange={(event) => setQuery(event.target.value)}
          placeholder="Search assets…"
          type="search"
          value={query}
        />
      </div>

      <div className="su-filter-pills su-scroll" role="radiogroup" aria-label="Asset type">
        {FILTERS.map((entry) => (
          <button
            aria-checked={filter === entry.id}
            className={filter === entry.id ? "su-filter-pill active" : "su-filter-pill"}
            key={entry.id}
            onClick={() => setFilter(entry.id)}
            role="radio"
            type="button"
          >
            {entry.label}
          </button>
        ))}
      </div>

      {visible.length ? (
        <div
          className="su-asset-grid"
          style={{ "--su-asset-cols": assetGridColumns(breakpoint) }}
        >
          {visible.map((asset) => (
            <button
              className="su-asset"
              key={asset.id}
              onClick={() => openPreview(asset)}
              title={asset.displayName ?? asset.id}
              type="button"
            >
              {assetCanRenderAsAudio(asset) ? (
                <span className="su-asset-placeholder">
                  <Icon.Audio size={20} />
                </span>
              ) : (
                <AssetThumbnail asset={asset} />
              )}
              <span aria-hidden="true" className="su-asset-kind">
                {assetCanRenderAsVideo(asset) ? (
                  <Icon.Play size={11} />
                ) : assetCanRenderAsAudio(asset) ? (
                  <Icon.Audio size={11} />
                ) : (
                  <Icon.Image size={11} />
                )}
              </span>
              {asset.status?.favorite ? (
                <span aria-hidden="true" className="su-asset-fav">
                  <Icon.Star filled size={14} />
                </span>
              ) : null}
            </button>
          ))}
        </div>
      ) : (
        <p className="su-empty">
          {query.trim() || filter !== "all"
            ? "Nothing matches that filter yet."
            : "Nothing here yet — generate something in a studio and it lands in Assets."}
        </p>
      )}
    </div>
  );
}

// "Clips" covers both generated videos and uploaded footage, mirroring how the Library
// screen counts them (video + upload) rather than a bare `type === "video"` test.
function matchesFilter(asset, filter) {
  if (filter === "video") {
    return assetCanRenderAsVideo(asset);
  }
  if (filter === "audio") {
    return assetCanRenderAsAudio(asset);
  }
  return asset.type === "image";
}
