import React from "react";
import { API_BASE_URL } from "../api.js";

export function assetUrl(asset) {
  if (asset?.url) {
    return API_BASE_URL + asset.url;
  }
  if (asset?.projectId && asset?.file?.path) {
    const normalizedPath = String(asset.file.path).replaceAll("\\", "/");
    return `${API_BASE_URL}/api/v1/projects/${asset.projectId}/files/${normalizedPath}`;
  }
  return "";
}

export function assetCanRenderAsImage(asset) {
  return asset?.type === "image" || asset?.file?.mimeType?.startsWith("image/");
}

export function assetCanRenderAsVideo(asset) {
  return asset?.type === "video" || asset?.file?.mimeType?.startsWith("video/");
}

// Generated videos get a sibling `<name>.poster.jpg` (the worker extracts frame 0).
// WKWebView won't paint a <video>'s own first frame as a poster, so the UI shows
// this real image instead — as the thumbnail and as the player's poster attribute.
export function posterUrl(asset) {
  const src = assetUrl(asset);
  if (!src || !assetCanRenderAsVideo(asset)) {
    return "";
  }
  return src.replace(/\.\w+$/, ".poster.jpg");
}

export function AssetThumbnail({ asset, className = "" }) {
  if (!asset) {
    return null;
  }
  const src = assetUrl(asset);
  if (!src) {
    return <span className={className}>{asset.type ?? "asset"}</span>;
  }
  if (assetCanRenderAsVideo(asset)) {
    return <VideoPoster asset={asset} className={className} />;
  }
  if (assetCanRenderAsImage(asset)) {
    return <img alt="" className={className} src={src} />;
  }
  return <span className={className}>{asset.type ?? "asset"}</span>;
}

function VideoPoster({ asset, className }) {
  const [failed, setFailed] = React.useState(false);
  const poster = posterUrl(asset);
  if (!poster || failed) {
    return <span className={className}>{asset.type ?? "video"}</span>;
  }
  return <img alt="" className={className} src={poster} onError={() => setFailed(true)} />;
}

export const AssetMedia = React.forwardRef(function AssetMedia({ asset, className = "", controls = true, ...mediaProps }, ref) {
  if (!asset) {
    return null;
  }
  const src = assetUrl(asset);
  if (!src) {
    return <span className={className}>{asset.type ?? "asset"}</span>;
  }
  if (assetCanRenderAsVideo(asset)) {
    return (
      <video
        className={className}
        controls={controls}
        muted
        playsInline
        poster={posterUrl(asset)}
        preload="metadata"
        ref={ref}
        src={src}
        {...mediaProps}
      />
    );
  }
  if (assetCanRenderAsImage(asset)) {
    return <img alt="" className={className} ref={ref} src={src} />;
  }
  return <span className={className}>{asset.type}</span>;
});
