import React from "react";
import { API_BASE_URL, withMediaTicket } from "../api.js";

// Ticket-free URL builder; internal so every exported producer appends the media
// ticket (sc-8810) exactly once, and posterUrl can swap the file extension before
// the query string exists.
function bareAssetUrl(asset) {
  // Inline payloads (the live denoise preview frame, sc-16905) are complete URLs already:
  // no API prefix, and downstream no ticket or thumbnail params — appending either would
  // corrupt the base64 body.
  if (asset?.url?.startsWith("data:")) {
    return asset.url;
  }
  if (asset?.url) {
    return API_BASE_URL + asset.url;
  }
  if (asset?.projectId && asset?.file?.path) {
    const normalizedPath = String(asset.file.path)
      .replaceAll("\\", "/")
      .split("/")
      .filter(Boolean)
      .map((segment) => encodeURIComponent(segment))
      .join("/");
    return `${API_BASE_URL}/api/v1/projects/${asset.projectId}/files/${normalizedPath}`;
  }
  return "";
}

// Displayable/fetchable URL for an asset. In remote-auth mode this carries the
// media ticket so element-driven requests (<img>/<video>/<a download>) — which
// cannot send the token header — still authenticate (sc-8810).
export function assetUrl(asset) {
  const bare = bareAssetUrl(asset);
  return bare.startsWith("data:") ? bare : withMediaTicket(bare);
}

// Grid thumbnails use one bounded, server-cached representation. The API
// backfills this derivative on first request for old assets and caches it under
// data/cache/media-thumbnails; a missing/corrupt source fails the image request
// and the existing MissingMedia fallback is shown. Full preview, playback and
// download continue to use assetUrl()/posterUrl() and therefore the original.
export const ASSET_THUMBNAIL_SIZE = 384;

function withThumbnailRequest(url) {
  if (!url) {
    return "";
  }
  if (url.startsWith("data:")) {
    return url;
  }
  const thumbnail = new URL(url);
  thumbnail.searchParams.set("thumbnail", String(ASSET_THUMBNAIL_SIZE));
  return withMediaTicket(thumbnail.toString());
}

export function assetCanRenderAsImage(asset) {
  return asset?.type === "image" || asset?.file?.mimeType?.startsWith("image/");
}

export function assetCanRenderAsVideo(asset) {
  return asset?.type === "video" || asset?.file?.mimeType?.startsWith("video/");
}

// Scene-linear HDR stills (OpenEXR, sc-18790). These ARE images — the mime starts with `image/`,
// so assetCanRenderAsImage is true and every picker/grid treats them as stills — but no browser
// decodes OpenEXR, so an <img> pointed at the stored file paints nothing.
//
// The stored bytes stay the deliverable (download must hand back the original float frame), so the
// UI shows the server's tone-mapped PNG derivative instead and labels it as a proxy. That is why
// this predicate exists separately from assetCanRenderAsImage: the two answer different questions —
// "is this a still?" versus "can the browser paint the stored bytes?".
export function assetIsHdrSource(asset) {
  if (asset?.file?.mimeType === "image/x-exr") {
    return true;
  }
  const path = asset?.file?.path ?? asset?.url ?? "";
  return typeof path === "string" && path.toLowerCase().endsWith(".exr");
}

// URL to PAINT an asset with. Identical to assetUrl() for everything the browser can decode; for an
// HDR source it is the server-rendered derivative, since the original would render as a broken
// image. Never use this for downloads — assetUrl() is the original bytes, and for an EXR that
// distinction is the whole point.
export function assetDisplayUrl(asset) {
  return assetIsHdrSource(asset) ? thumbnailUrl(asset) : assetUrl(asset);
}

// Audio outputs (SceneWorks Audio Studio, epic 13400 A5). A `type:"audio"` asset
// (or any file whose mimeType is audio/*) is playable via an <audio> element —
// there is no poster/thumbnail frame, so the shared results zone renders a
// transport instead (WorkerProgressCard audio-player variant, sc-13405).
export function assetCanRenderAsAudio(asset) {
  return asset?.type === "audio" || asset?.file?.mimeType?.startsWith("audio/");
}

// Suppress the native WKWebView context menu on grid thumbnails (sc-8731). Right-
// clicking a thumbnail image in a Tauri webview otherwise pops the OS "Download
// Image / Copy Image / Share" menu; thumbnails have no custom menu, so we just
// swallow the default. Only preventDefault the contextmenu event — never
// stopPropagation of clicks — so left-click selection / open-preview stay intact.
// Applied at the shared AssetThumbnail seam so every grid that renders it (Queue's
// WorkerProgressCard, pickers, studios) inherits the suppression from one place.
// The Library grid renders AssetMedia directly (assetPanels.jsx), so AssetGrid
// imports this and attaches it at the tile-cell level. The full-size
// FullscreenPreview renderer is intentionally left alone: it gets its own custom
// right-click menu in sc-8729.
export function suppressThumbnailContextMenu(event) {
  event.preventDefault();
}

// Generated videos get a sibling `<name>.poster.jpg` (the worker extracts frame 0).
// WKWebView won't paint a <video>'s own first frame as a poster, so the UI shows
// this real image instead — as the thumbnail and as the player's poster attribute.
//
// The server attaches `posterUrl` to a normalized asset ONLY when that poster file
// actually exists on disk (sc-10468). So for a persisted (normalized) asset — which
// always carries a server-built `url` — the ABSENCE of `posterUrl` means the video
// has no poster, and we must NOT probe `<name>.poster.jpg` (that 404'd on every
// render, spamming the log). We only fall back to deriving the poster path for
// transient/live-job assets the server hasn't normalized yet (no `url`), preserving
// the in-progress poster behavior in the studios.
function barePosterUrl(asset) {
  if (!assetCanRenderAsVideo(asset)) {
    return "";
  }
  if (asset?.posterUrl) {
    return API_BASE_URL + asset.posterUrl;
  }
  if (asset?.url) {
    // Normalized asset with no server-advertised poster ⇒ none exists; don't probe.
    return "";
  }
  const src = bareAssetUrl(asset);
  return src ? src.replace(/\.\w+$/, ".poster.jpg") : "";
}

export function posterUrl(asset) {
  return withMediaTicket(barePosterUrl(asset));
}

export function thumbnailUrl(asset) {
  const source = assetCanRenderAsVideo(asset) ? barePosterUrl(asset) : bareAssetUrl(asset);
  return withThumbnailRequest(source);
}

// Placeholder shown when an asset's underlying file can't be loaded — e.g. it
// was purged from disk after the job ran, so the URL now 404s. Replaces the
// browser's broken-image glyph with a clear "deleted" marker (a red X) so queue
// thumbnails for purged outputs read as removed rather than broken.
export function MissingMedia({ className = "" }) {
  return (
    <span
      aria-label="Deleted asset"
      className={`asset-thumb-missing ${className}`.trim()}
      onContextMenu={suppressThumbnailContextMenu}
      role="img"
      title="Deleted"
    >
      <svg viewBox="0 0 24 24" aria-hidden="true" focusable="false">
        <path d="M6 6l12 12M18 6L6 18" />
      </svg>
    </span>
  );
}

// Image thumbnail that falls back to the deleted-asset placeholder once the
// source fails to load (the file is gone), rather than leaving a broken image.
function ImageThumb({ src, className }) {
  const [failed, setFailed] = React.useState(false);
  // sc-9063: a load failure is per-URL, not per-asset. When the src changes —
  // e.g. a media ticket finally arrives after a mint failure degraded thumbnails
  // to the placeholder — retry the load instead of sticking on the marker.
  React.useEffect(() => {
    setFailed(false);
  }, [src]);
  if (failed) {
    return <MissingMedia className={className} />;
  }
  return (
    <img
      alt=""
      className={className}
      decoding="async"
      loading="lazy"
      onContextMenu={suppressThumbnailContextMenu}
      onError={() => setFailed(true)}
      src={src}
    />
  );
}

export function AssetThumbnail({ asset, className = "" }) {
  if (!asset) {
    return null;
  }
  const src = thumbnailUrl(asset);
  if (!src) {
    return <span className={className} onContextMenu={suppressThumbnailContextMenu}>{asset.type ?? "asset"}</span>;
  }
  if (assetCanRenderAsVideo(asset)) {
    return <VideoPoster asset={asset} className={className} />;
  }
  if (assetCanRenderAsImage(asset)) {
    return <ImageThumb src={src} className={className} />;
  }
  return <span className={className}>{asset.type ?? "asset"}</span>;
}

function VideoPoster({ asset, className }) {
  const [failed, setFailed] = React.useState(false);
  const poster = thumbnailUrl(asset);
  // Same per-URL retry as ImageThumb (sc-9063): a new poster URL (fresh media
  // ticket) clears a stale failure so the placeholder isn't permanent.
  React.useEffect(() => {
    setFailed(false);
  }, [poster]);
  if (!poster) {
    return <span className={className} onContextMenu={suppressThumbnailContextMenu}>{asset.type ?? "video"}</span>;
  }
  if (failed) {
    return <MissingMedia className={className} />;
  }
  return (
    <img
      alt=""
      className={className}
      decoding="async"
      loading="lazy"
      onContextMenu={suppressThumbnailContextMenu}
      onError={() => setFailed(true)}
      src={poster}
    />
  );
}

// `muted` is a PROP, defaulting to false (sc-17161). It used to be hard-set on every <video> this
// component rendered, and no call site overrode it, so every clip in the app played silent — the
// asset detail, the preview modal, the queue card, the A/B compare. That was survivable while
// every video model produced a video-only mp4; MiniMax-H3 is a joint audio+video family whose
// mp4 carries an AAC track (`video_jobs::mod.rs` muxes `-c:a aac` whenever the pipeline returns
// audio), and a silent player is indistinguishable from a model that generated no sound at all.
// Surfaces that drive playback from SCRIPT rather than from a user gesture still pass `muted`
// explicitly — autoplay policy blocks an unmuted programmatic play() — which is why this is a
// default rather than a removal.
export const AssetMedia = React.forwardRef(function AssetMedia({ asset, className = "", controls = true, muted = false, ...mediaProps }, ref) {
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
        muted={muted}
        playsInline
        poster={posterUrl(asset)}
        preload="metadata"
        ref={ref}
        src={src}
        {...mediaProps}
      />
    );
  }
  if (assetCanRenderAsAudio(asset)) {
    // Audio has no poster/first-frame; render a plain <audio> so the results zone
    // (and the WorkerProgressCard audio-player transport, which drives this via a
    // ref with controls off) can play the clip. Mirrors the <video> branch above
    // for src resolution, controls, preload and ref/prop passthrough.
    return (
      <audio
        className={className}
        controls={controls}
        muted={muted}
        preload="metadata"
        ref={ref}
        src={src}
        {...mediaProps}
      />
    );
  }
  if (assetCanRenderAsImage(asset)) {
    // An HDR source paints the server's tone-mapped derivative — no browser decodes OpenEXR, so
    // `src` (the original float frame) would render as a broken image. The title says so, because
    // a silently tone-mapped preview of a grading asset is a claim about the pixels that is not
    // true. Download still hands back the original: it goes through assetUrl(), not this.
    const hdr = assetIsHdrSource(asset);
    return (
      <img
        alt=""
        className={className}
        ref={ref}
        src={hdr ? assetDisplayUrl(asset) : src}
        title={hdr ? "HDR source (OpenEXR) — preview is a tone-mapped proxy; download is the original scene-linear frame" : undefined}
      />
    );
  }
  return <span className={className}>{asset.type}</span>;
});
