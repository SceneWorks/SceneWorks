import React, { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { Icon } from "./Icons.jsx";
import { AssetMedia, assetUrl } from "./assetMedia.jsx";
import { audioAssetDurationSeconds } from "./WorkerProgressCard.jsx";
import { idlePeaks, useAudioPeaks, waveformPath, waveformWidth } from "../audioWaveform.js";
import { formatClock } from "../audioTakes.js";

// Shared audio-take primitives for the Audio Studio redesign (epic 14361).
//
// Two pieces, used verbatim by the Advanced studio, the Simple studio and the Simple
// Assets viewer so the three surfaces can't drift:
//   • AudioWaveform — the bar waveform, its played band and its playhead (optionally
//     seekable), drawn from peaks decoded off the real clip (audioWaveform.js).
//   • useAudioTakePlayer — the transport. ONE <audio> element per surface, ref-driven with
//     native controls off, exactly the way WorkerProgressCard's AudioTransport already
//     works: `onLoadedMetadata` corrects the declared duration and `timeupdate` — not a
//     timer — drives `currentTime`. Loading a different take pauses the previous one, so
//     only one clip ever plays at a time.

// Bar counts per surface size. Fewer bars on a small cell keeps the pitch readable rather
// than smearing 160 hairlines into a block.
export const WAVEFORM_BARS = Object.freeze({ tile: 26, card: 44, deck: 96, stage: 120 });

// Glyph size per button size, so the triangle stays optically centred at 22px and at 56px.
const PLAY_GLYPH_SIZE = Object.freeze({ xs: 11, sm: 13, md: 15, lg: 19, xl: 22 });

/**
 * The clip's waveform: a single round-capped bar path, the played band and the playhead.
 * Stretches to its cell (`preserveAspectRatio="none"`), so the caller owns the box.
 * Passing `onSeek` makes the surface clickable — x-offset maps to a 0..1 fraction.
 */
export function AudioWaveform({
  asset,
  bars = WAVEFORM_BARS.card,
  height = 56,
  progress = 0,
  onSeek = null,
  className = "",
  label = "Waveform",
}) {
  const decoded = useAudioPeaks(asset);
  const peaks = decoded ?? idlePeaks(bars);
  const path = useMemo(() => waveformPath(peaks, { bars, height }), [peaks, bars, height]);
  const width = waveformWidth(bars);
  const played = Math.max(0, Math.min(1, Number(progress) || 0));

  const seek = (event) => {
    if (!onSeek) {
      return;
    }
    const box = event.currentTarget.getBoundingClientRect();
    if (!box.width) {
      return;
    }
    onSeek(Math.max(0, Math.min(1, (event.clientX - box.left) / box.width)));
  };

  const classes = ["audio-wave", decoded ? "" : "audio-wave--idle", className].filter(Boolean).join(" ");
  const body = (
    <>
      <svg
        aria-hidden="true"
        className="audio-wave__svg"
        focusable="false"
        preserveAspectRatio="none"
        viewBox={`0 0 ${width} ${height}`}
      >
        <path d={path} />
      </svg>
      {played > 0 ? (
        <span aria-hidden="true" className="audio-wave__played" style={{ width: `${played * 100}%` }} />
      ) : null}
    </>
  );
  if (!onSeek) {
    return (
      <span aria-label={label} className={classes} role="img">
        {body}
      </span>
    );
  }
  return (
    <button aria-label={`Seek ${label.toLowerCase()}`} className={classes} onClick={seek} type="button">
      {body}
    </button>
  );
}

/**
 * The now-playing transport shared by every audio surface.
 *
 * Returns the loaded take's id (null = nothing loaded, so no deck), live playback state and
 * the actions the deck / cards call. `mediaProps` + `audioRef` are spread onto ONE
 * <AssetMedia> the surface renders (see LoadedAudioElement below).
 */
export function useAudioTakePlayer() {
  const audioRef = useRef(null);
  const [loadedTakeId, setLoadedTakeId] = useState(null);
  const [isPlaying, setIsPlaying] = useState(false);
  const [currentTime, setCurrentTime] = useState(0);
  const [duration, setDuration] = useState(0);
  const [loop, setLoop] = useState(false);
  // Set when a NEW take is loaded, so playback starts once the element has swapped src —
  // calling play() before React commits the new source would play the old clip.
  const autoPlay = useRef(false);

  const play = useCallback(() => {
    const el = audioRef.current;
    // play() may reject (autoplay policy) or be unimplemented (jsdom); the onPlay/onPause
    // events keep state authoritative in a real browser.
    try {
      const played = el?.play?.();
      if (played && typeof played.catch === "function") {
        played.catch(() => setIsPlaying(false));
      }
    } catch {
      /* ignore — state falls back to the media events */
    }
    setIsPlaying(true);
  }, []);

  const pause = useCallback(() => {
    audioRef.current?.pause?.();
    setIsPlaying(false);
  }, []);

  useEffect(() => {
    if (!loadedTakeId || !autoPlay.current) {
      return;
    }
    autoPlay.current = false;
    play();
  }, [loadedTakeId, play]);

  // Play a take. The same take toggles pause/resume; a different one loads from 0:00 and
  // plays, which pauses the previous clip (there is only ever one element).
  const toggleTake = useCallback(
    (asset) => {
      if (!asset?.id) {
        return;
      }
      if (asset.id === loadedTakeId) {
        if (isPlaying) {
          pause();
        } else {
          play();
        }
        return;
      }
      pause();
      autoPlay.current = true;
      setCurrentTime(0);
      setDuration(audioAssetDurationSeconds(asset));
      setLoadedTakeId(asset.id);
    },
    [loadedTakeId, isPlaying, pause, play],
  );

  const unload = useCallback(() => {
    pause();
    autoPlay.current = false;
    setLoadedTakeId(null);
    setCurrentTime(0);
    setDuration(0);
  }, [pause]);

  const seekTo = useCallback((seconds) => {
    const el = audioRef.current;
    const next = Math.max(0, Number(seconds) || 0);
    setCurrentTime(next);
    if (el) {
      el.currentTime = next;
    }
  }, []);

  const seekFraction = useCallback(
    (fraction) => {
      if (duration > 0) {
        seekTo(fraction * duration);
      }
    },
    [duration, seekTo],
  );

  const skip = useCallback(
    (delta) => {
      const limit = duration > 0 ? duration : currentTime;
      seekTo(Math.min(limit, currentTime + delta));
    },
    [currentTime, duration, seekTo],
  );

  const mediaProps = useMemo(
    () => ({
      controls: false,
      loop,
      onLoadedMetadata: (event) => {
        const real = Number(event.currentTarget?.duration);
        if (Number.isFinite(real) && real > 0) {
          setDuration(real);
        }
      },
      onTimeUpdate: (event) => setCurrentTime(Number(event.currentTarget?.currentTime) || 0),
      onPlay: () => setIsPlaying(true),
      onPause: () => setIsPlaying(false),
      onEnded: () => {
        setIsPlaying(false);
        setCurrentTime(0);
      },
    }),
    [loop],
  );

  const progress = duration > 0 ? Math.min(1, currentTime / duration) : 0;

  return {
    audioRef,
    mediaProps,
    loadedTakeId,
    isPlaying,
    currentTime,
    duration,
    progress,
    loop,
    toggleLoop: () => setLoop((value) => !value),
    toggleTake,
    unload,
    seekFraction,
    skip,
  };
}

/**
 * The single <audio> element a surface renders for its loaded take. Native controls are
 * off — the deck's own transport drives it through the player's ref.
 */
export function LoadedAudioElement({ asset, player, className = "audio-deck__el" }) {
  if (!asset || !assetUrl(asset)) {
    return null;
  }
  return (
    <AssetMedia
      asset={asset}
      className={className}
      ref={player.audioRef}
      {...player.mediaProps}
    />
  );
}

/**
 * Round accent play/pause button — the take card's 32px overlay, the deck's 46px hero and
 * the Simple/Assets sizes are all this button with a size modifier.
 */
export function AudioPlayButton({ playing, onClick, size = "md", label = null, className = "" }) {
  const classes = ["audio-play-btn", `audio-play-btn--${size}`, className].filter(Boolean).join(" ");
  const glyph = PLAY_GLYPH_SIZE[size] ?? PLAY_GLYPH_SIZE.md;
  return (
    <button
      aria-label={label ?? (playing ? "Pause" : "Play")}
      aria-pressed={playing}
      className={classes}
      onClick={onClick}
      type="button"
    >
      {playing ? <Icon.Pause size={glyph} /> : <Icon.Play size={glyph} />}
    </button>
  );
}

/** The mono `m:ss / m:ss` clock every transport shows. */
export function AudioClock({ current, total, className = "audio-clock" }) {
  return (
    <span className={className}>
      <strong>{formatClock(current)}</strong>
      <span aria-hidden="true"> / </span>
      <span className="audio-clock__total">{formatClock(total)}</span>
    </span>
  );
}

/**
 * Download control shared by the deck and the take card. A real anchor click (the media URL
 * already carries the auth ticket in remote-auth mode, sc-8810) rather than a fetch, so the
 * browser owns the save dialog — same mechanism the Simple UI's DownloadButton uses.
 */
export function AudioDownloadButton({ asset, className = "", label = null, iconSize = 14 }) {
  const anchorRef = useRef(null);
  return (
    <>
      <a
        aria-hidden="true"
        download={asset?.displayName ?? ""}
        href={assetUrl(asset)}
        ref={anchorRef}
        style={{ display: "none" }}
        tabIndex={-1}
      >
        download
      </a>
      <button
        aria-label="Download"
        className={className}
        onClick={() => anchorRef.current?.click()}
        title="Download"
        type="button"
      >
        <Icon.Download size={iconSize} />
        {label}
      </button>
    </>
  );
}
