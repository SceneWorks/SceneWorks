import { useEffect, useState } from "react";
import { assetCanRenderAsAudio, assetUrl } from "./components/assetMedia.jsx";

// Waveform peaks for the Audio Studio redesign (epic 14361 / sc-14362).
//
// Every audio surface in the redesign — the take card, the play-deck stage, the Simple
// take grid and the Simple Assets tile/viewer — draws the clip as a bar waveform. The
// peaks are DERIVED FROM THE RENDERED AUDIO, client-side: the clip is fetched once and
// decoded through the Web Audio API, then downsampled to a fixed-resolution peak array
// that every surface shares. (The design bundle ships a `waveforms.json` of synthetic
// paths; those are mock stand-ins and deliberately do NOT ship.)
//
// Degradation is total and silent: no AudioContext (jsdom, an old webview), no fetch, a
// 404 on a purged clip or an undecodable codec all resolve to `null`, and the renderer
// draws a flat idle bar row. A failure is remembered so a broken asset is never re-fetched
// on every re-render.

// Peak resolution the decoder caches at. Each surface resamples DOWN from this to its own
// bar count, so one decode serves the 22-bar Assets tile and the 96-bar deck stage alike.
export const PEAK_RESOLUTION = 160;

// Downsample raw PCM samples to `bars` normalized 0..1 peaks — the max absolute amplitude
// in each window, scaled so the loudest bar is 1. A silent clip yields all-zero peaks
// (drawn as the minimum bar), never NaN.
export function peaksFromSamples(samples, bars = PEAK_RESOLUTION) {
  const count = Math.max(1, Math.floor(bars));
  const out = new Array(count).fill(0);
  const total = samples?.length ?? 0;
  if (total === 0) {
    return out;
  }
  const window = total / count;
  let loudest = 0;
  for (let bar = 0; bar < count; bar += 1) {
    const start = Math.floor(bar * window);
    const end = Math.min(total, Math.max(start + 1, Math.floor((bar + 1) * window)));
    let peak = 0;
    for (let index = start; index < end; index += 1) {
      const value = Math.abs(samples[index]);
      if (value > peak) {
        peak = value;
      }
    }
    out[bar] = peak;
    if (peak > loudest) {
      loudest = peak;
    }
  }
  if (loudest > 0) {
    for (let bar = 0; bar < count; bar += 1) {
      out[bar] /= loudest;
    }
  }
  return out;
}

// Resample a cached peak array to `bars` entries (max within each window), so one decode
// drives every bar count in the UI.
export function resamplePeaks(peaks, bars) {
  const source = Array.isArray(peaks) ? peaks : [];
  const count = Math.max(1, Math.floor(bars));
  if (source.length === 0) {
    return new Array(count).fill(0);
  }
  if (source.length === count) {
    return source;
  }
  const out = new Array(count).fill(0);
  const window = source.length / count;
  for (let bar = 0; bar < count; bar += 1) {
    const start = Math.floor(bar * window);
    const end = Math.min(source.length, Math.max(start + 1, Math.floor((bar + 1) * window)));
    let peak = 0;
    for (let index = start; index < end; index += 1) {
      if (source[index] > peak) {
        peak = source[index];
      }
    }
    out[bar] = peak;
  }
  return out;
}

// The intrinsic width of a waveform drawn with `bars` bars at `pitch` spacing. The SVG is
// stretched to its cell with preserveAspectRatio="none", so this is a viewBox unit, not a
// CSS pixel count.
export function waveformWidth(bars, pitch = 5, inset = 3) {
  return inset * 2 + Math.max(0, Math.floor(bars) - 1) * pitch;
}

// ONE `<path>` `d` of vertical strokes — the design's bar waveform, drawn as a single path
// with round caps rather than one element per bar. Bars are centred on the mid-line and
// floored at `minBar` so silence still reads as a waveform rather than a gap.
export function waveformPath(peaks, options = {}) {
  const { bars = PEAK_RESOLUTION, height = 56, pitch = 5, inset = 3, minBar = 0.06 } = options;
  const values = resamplePeaks(peaks, bars);
  const mid = height / 2;
  const room = Math.max(1, mid - inset);
  let path = "";
  for (let bar = 0; bar < values.length; bar += 1) {
    const magnitude = Math.max(minBar, Math.min(1, Number(values[bar]) || 0));
    const half = magnitude * room;
    const x = inset + bar * pitch;
    path += `M${x} ${(mid - half).toFixed(1)}V${(mid + half).toFixed(1)}`;
  }
  return path;
}

// A flat idle bar row — what a surface draws before the decode lands, and permanently when
// there is nothing to decode against (no AudioContext, a purged clip, an odd codec).
export function idlePeaks(bars = PEAK_RESOLUTION) {
  return new Array(Math.max(1, Math.floor(bars))).fill(0);
}

// Decoded peaks, keyed by asset id: one decode per clip per session, shared by every
// surface. `failedAssetIds` is the never-retry set — a clip whose file is gone would
// otherwise re-fetch on every render.
const peakCache = new Map();
const inflightDecodes = new Map();
const failedAssetIds = new Set();

// Test seam: drop the module-level caches so a suite can exercise a fresh decode.
export function resetAudioPeakCache() {
  peakCache.clear();
  inflightDecodes.clear();
  failedAssetIds.clear();
}

function audioContextCtor() {
  if (typeof window === "undefined") {
    return null;
  }
  return window.AudioContext ?? window.webkitAudioContext ?? null;
}

async function decodeAssetPeaks(asset) {
  const Ctx = audioContextCtor();
  const url = assetUrl(asset);
  if (!Ctx || !url || typeof fetch !== "function") {
    return null;
  }
  const response = await fetch(url);
  if (!response.ok) {
    return null;
  }
  const bytes = await response.arrayBuffer();
  const context = new Ctx();
  try {
    const decoded = await context.decodeAudioData(bytes);
    return peaksFromSamples(decoded.getChannelData(0), PEAK_RESOLUTION);
  } finally {
    // The context exists only to decode; closing it releases the (limited) hardware slot.
    context.close?.();
  }
}

// Peaks for an audio asset, or `null` until they land / forever when they can't. Safe to
// call for any asset — a non-audio one never decodes.
export function useAudioPeaks(asset) {
  const assetId = asset?.id ?? null;
  const playable = Boolean(asset) && assetCanRenderAsAudio(asset);
  const [peaks, setPeaks] = useState(() => (assetId ? (peakCache.get(assetId) ?? null) : null));

  useEffect(() => {
    if (!playable || !assetId) {
      setPeaks(null);
      return undefined;
    }
    const cached = peakCache.get(assetId);
    if (cached) {
      setPeaks(cached);
      return undefined;
    }
    setPeaks(null);
    if (failedAssetIds.has(assetId)) {
      return undefined;
    }
    let live = true;
    let pending = inflightDecodes.get(assetId);
    if (!pending) {
      // A second mount of the same clip joins this promise rather than starting a competing
      // fetch. The decode is deliberately NOT aborted on unmount: it is shared, so cancelling
      // it for one subscriber would poison the result for the others, and the payload is a
      // single short clip that lands in the cache for the rest of the session either way.
      pending = decodeAssetPeaks(asset).then(
        (result) => {
          inflightDecodes.delete(assetId);
          if (result) {
            peakCache.set(assetId, result);
          } else {
            failedAssetIds.add(assetId);
          }
          return result;
        },
        () => {
          inflightDecodes.delete(assetId);
          failedAssetIds.add(assetId);
          return null;
        },
      );
      inflightDecodes.set(assetId, pending);
    }
    pending.then((result) => {
      if (live && result) {
        setPeaks(result);
      }
    });
    return () => {
      live = false;
    };
  }, [assetId, playable, asset]);

  return peaks;
}
