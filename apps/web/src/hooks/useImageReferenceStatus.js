import { useEffect, useRef, useState } from "react";
import { mediaResponseWithTicketRetry } from "../api/media.js";
import { assetUrl } from "../components/assetMedia.jsx";
import { loadImage } from "../imageDimensions.js";
import { useMediaTicket } from "./useMediaTicket.js";

// Keep the last known name after deletion/filtering so an unavailable selection
// still explains which image was guiding the render. Never substitute another image.
export function useImageReferenceStatus({ enabled, assetId, projectId, assets }) {
  const remembered = useRef(null);
  const [check, setCheck] = useState(null);
  const [revision, setRevision] = useState(0);
  const ticket = useMediaTicket();
  const asset = assets.find((item) => item.id === assetId && item.projectId === projectId);
  const details = asset ? {
    assetId, projectId,
    name: asset.displayName || asset.name || asset.file?.originalName || asset.file?.path?.split(/[\\/]/).pop() || assetId,
    filePath: asset.file?.path || "",
  } : remembered.current?.assetId === assetId && remembered.current?.projectId === projectId ? remembered.current : null;
  useEffect(() => {
    if (asset) remembered.current = details;
    else if (!assetId) remembered.current = null;
  });
  const active = enabled && Boolean(assetId);
  const eligible = asset && (asset.type === "image" || asset.type === "frame") && !asset.status?.trashed && !asset.status?.rejected;
  const url = eligible ? assetUrl(asset) : "";
  const key = JSON.stringify([projectId, assetId, url, ticket, revision]);
  useEffect(() => {
    if (!active || !url) return undefined;
    const controller = new AbortController();
    const timer = setTimeout(() => controller.abort(), 15000);
    let current = true;
    // Check the original file, not a cached thumbnail that may outlive it. HEAD
    // works for worker-supported formats the browser cannot decode (e.g. EXR).
    mediaResponseWithTicketRetry(() => assetUrl(asset), { method: "HEAD", signal: controller.signal })
      .then(async (response) => {
        if (!response.ok) throw new Error("Reference file unavailable");
        // TIFF/HEIC/EXR can be valid worker inputs without browser codec support.
        const browserDecodable = /^image\/(png|jpeg|webp|gif|bmp|avif)$/.test(asset.file?.mimeType || "") ||
          /\.(png|jpe?g|webp|gif|bmp|avif)(?:$|[?#])/i.test(asset.file?.path || asset.url || "");
        if (browserDecodable) {
          const image = await loadImage(assetUrl(asset), { signal: controller.signal });
          if (!image.naturalWidth || !image.naturalHeight) throw new Error("Reference image is empty");
        }
        if (current) setCheck({ key, available: true });
      })
      .catch(() => { if (current) setCheck({ key, available: false }); })
      .finally(() => clearTimeout(timer));
    return () => { current = false; clearTimeout(timer); controller.abort(); };
    // Asset list refreshes must not invalidate a completed check of the same file.
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [active, key, url]);
  const label = details?.name || "Previously selected image";
  const description = details?.filePath ? `${label} (${details.filePath})` : label;
  let error = null;
  if (active && (!eligible || !url)) {
    error = `Image reference is on, but “${description}” is no longer available in this project. Remove it or select another reference image.`;
  } else if (active && check?.key === key && !check.available) {
    error = `Image reference is on, but the original file for “${description}” could not be read. Restore the file, retry the check, or remove or replace the reference image.`;
  }
  return { active, details, label, error, pending: active && eligible && Boolean(url) && check?.key !== key, retry: () => setRevision((value) => value + 1) };
}
