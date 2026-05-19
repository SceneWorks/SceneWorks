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

export function AssetMedia({ asset, className = "" }) {
  if (!asset) {
    return null;
  }
  const src = assetUrl(asset);
  if (asset.file?.mimeType?.startsWith("video/")) {
    return <video className={className} controls muted playsInline src={src} />;
  }
  if (assetCanRenderAsImage(asset)) {
    return <img alt="" className={className} src={src} />;
  }
  return <span className={className}>{asset.type}</span>;
}
