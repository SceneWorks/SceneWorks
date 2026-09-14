// Soundtrack indicator (epic 17137, sc-19577). Surfaces `asset.file.hasAudio` — the worker's
// MEASUREMENT of what it actually muxed into the mp4 (`EncodedClip::has_audio`, written from
// `decoded.audio.is_some()`, the same predicate `encode_inner` branches on to run the AAC mux).
//
// WHY IT EXISTS. sc-17161 made generated video play with sound (`AssetMedia`'s `muted` prop now
// defaults to false), but nothing on an asset said a soundtrack was there — so the half of its
// acceptance criterion reading "the user can tell the audio is generated (not silent)" could only be
// satisfied by pressing play on every clip. MiniMax-H3 is the app's first family that renders video
// and synchronized stereo audio jointly, and it is exactly the family for which "did this one come
// out with sound?" is the question.
//
// ⚠️ NEVER KEY THIS ON A MODEL ID OR FAMILY. The same MiniMax-H3 checkpoint produces clips with and
// without a soundtrack, so a per-family lookup would mark silent renders as having audio — a
// declaration with no evidence behind it, which is the defect class this epic keeps hitting. The
// recorded field is the only input.
//
// Three render states, mirroring LikenessBadge:
//   - true    -> the "Sound" chip
//   - false   -> nothing. Measured and silent; the badge answers "is there audio", and there is not.
//                The asset detail's metadata row is where false is spelled out, because it has the
//                room to distinguish it from unknown.
//   - absent  -> nothing. Every asset generated before sc-19577 is here, and treating those as
//                silent would be a fresh false claim about files on disk.
import React from "react";

// Read the recorded soundtrack measurement off an asset.
//
// Returns `true` / `false` when the worker measured it, and `null` when the asset predates the
// measurement or is not a generated clip. A non-boolean value is `null` too, not coerced: a
// truthy string would otherwise light the badge on garbage.
export function assetHasAudioTrack(asset) {
  const value = asset?.file?.hasAudio;
  return typeof value === "boolean" ? value : null;
}

export function AudioTrackBadge({ asset = null, hasAudio = undefined }) {
  const value = hasAudio === undefined ? assetHasAudioTrack(asset) : hasAudio;
  if (value !== true) {
    return null;
  }
  return (
    <span
      className="audio-track-badge"
      data-has-audio="true"
      title="This clip was rendered with a synchronized soundtrack."
      aria-label="Has audio"
    >
      <span className="audio-track-badge__glyph" aria-hidden="true">
        ♪
      </span>
      <span className="audio-track-badge__label">Sound</span>
    </span>
  );
}

export default AudioTrackBadge;
