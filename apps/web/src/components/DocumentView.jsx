import React from "react";
import { assetUrl } from "./assetMedia.jsx";

// Renders an interleaved document: ordered text runs + generated images.
// Image segments resolve through the asset catalog when available (so they pick up
// the normalized `url`), falling back to building the file URL from projectId + path
// (used by the Library reader, which only has the document asset itself).
export function DocumentView({ segments, assets = [], projectId }) {
  if (!Array.isArray(segments) || !segments.length) {
    return <p className="empty-panel">This document has no content.</p>;
  }
  return (
    <article className="document-view" aria-label="Generated document">
      {segments.map((segment, index) => {
        if (segment.type === "text") {
          return (
            <p className="document-text" key={`segment-${index}`}>
              {segment.text}
            </p>
          );
        }
        const asset = assets.find((item) => item.id === segment.assetId);
        const src = assetUrl(asset ?? { projectId, file: { path: segment.path } });
        return src ? <img alt="" className="document-image" key={`segment-${index}`} src={src} /> : null;
      })}
    </article>
  );
}
