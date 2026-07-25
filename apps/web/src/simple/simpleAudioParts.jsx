import React from "react";
import { Icon } from "../components/Icons.jsx";
import { audioAssetDurationSeconds } from "../components/WorkerProgressCard.jsx";
import {
  AudioClock,
  AudioDownloadButton,
  AudioPlayButton,
  AudioWaveform,
  LoadedAudioElement,
  WAVEFORM_BARS,
} from "../components/audioTakeParts.jsx";
import { audioTakeTitle, formatClock } from "../audioTakes.js";

// Simple UI audio parts (epic 14361 / sc-14365, sc-14366) — the take grid, the deck and the
// Assets viewer stage, in Simple's own vocabulary.
//
// Same behavior as the Advanced surfaces (one <audio>, one loaded take, played band + clock
// off `timeupdate`), sized off the MEASURED container band rather than a media query, so the
// screens stay correct when the shell is embedded in a narrow frame.

// Stage waveform height and play-button size per band — the design's three steps.
const DECK_WAVE_HEIGHT = Object.freeze({ desktop: 84, tablet: 72, phone: 58 });
const VIEWER_WAVE_HEIGHT = Object.freeze({ desktop: 150, tablet: 132, phone: 104 });

/** Take-grid column count per band (3 desktop / 2 tablet / 1 phone). */
export function takeGridColumns(breakpoint) {
  if (breakpoint === "desktop") {
    return 3;
  }
  return breakpoint === "tablet" ? 2 : 1;
}

/** One take in the Simple results grid. */
export function SimpleTakeCard({ run, asset, index, loaded, playing, progress, onToggle, phone }) {
  const duration = audioAssetDurationSeconds(asset);
  return (
    <div className={loaded ? "su-take is-loaded" : "su-take"} data-testid="su-take-card">
      <div className="su-take__wave">
        <AudioWaveform
          asset={asset}
          bars={WAVEFORM_BARS.card}
          height={88}
          label={`Take ${index + 1}`}
          progress={loaded ? progress : 0}
        />
        <AudioPlayButton
          className="su-take__play"
          label={`${loaded && playing ? "Pause" : "Play"} take ${index + 1}`}
          onClick={() => onToggle(asset)}
          playing={loaded && playing}
          size={phone ? "lg" : "sm"}
        />
        <span className="su-take__duration">{formatClock(duration)}</span>
      </div>
      <strong className="su-take__title">
        Take {index + 1} · {audioTakeTitle(run?.job, asset)}
      </strong>
      <span className="su-take__meta">
        {run?.modeLabel} · {run?.modelName}
      </span>
    </div>
  );
}

/**
 * The Simple studio's play deck: the same head / stage / transport stack as the Advanced
 * deck at reduced sizes. On phone the Download / Run again pair moves out of the transport
 * row into full-width buttons under the clock, so the no-wrap row can't overflow.
 */
export function SimpleAudioDeck({ run, asset, takeIndex, player, breakpoint, onRunAgain }) {
  const phone = breakpoint === "phone";
  const duration = player.duration > 0 ? player.duration : audioAssetDurationSeconds(asset);
  const meta = [run?.modelName, ...(run?.chips ?? [])].filter(Boolean).join(" · ");
  const actions = (
    <>
      <AudioDownloadButton
        asset={asset}
        className={phone ? "su-deck__wide-action" : "su-deck__action"}
        iconSize={15}
        label="Download"
      />
      {onRunAgain && run?.job ? (
        <button
          className={phone ? "su-deck__wide-action" : "su-deck__action"}
          onClick={() => onRunAgain(run.job)}
          type="button"
        >
          <Icon.Refresh size={15} />
          Run again
        </button>
      ) : null}
    </>
  );
  return (
    <div className="su-deck" data-testid="su-audio-deck">
      <LoadedAudioElement asset={asset} player={player} className="su-deck__el" />
      <div className="su-deck__head">
        <span className={`audio-mode-chip audio-mode-chip--${run?.mode ?? "speech"}`}>
          {run?.modeLabel ?? "Audio"}
          {takeIndex >= 0 ? <span className="audio-mode-chip__extra">Take {takeIndex + 1}</span> : null}
        </span>
        <strong className="su-deck__title">{audioTakeTitle(run?.job, asset)}</strong>
        <button aria-label="Close player" className="su-deck__close" onClick={player.unload} type="button">
          <Icon.Close size={16} />
        </button>
      </div>
      <div className="su-deck__stage">
        <AudioWaveform
          asset={asset}
          bars={WAVEFORM_BARS.deck}
          height={DECK_WAVE_HEIGHT[breakpoint] ?? DECK_WAVE_HEIGHT.desktop}
          label="Clip"
          onSeek={player.seekFraction}
          progress={player.progress}
        />
      </div>
      <div className="su-deck__transport">
        {/* 44px desktop/tablet, 48px phone — the phone step is a touch-target bump, not a
            different control, so it rides a modifier rather than a second size. */}
        <AudioPlayButton
          className={phone ? "su-deck__play su-deck__play--phone" : "su-deck__play"}
          onClick={() => player.toggleTake(asset)}
          playing={player.isPlaying}
          size="lg"
        />
        <AudioClock current={player.currentTime} total={duration} />
        {phone ? null : <span className="su-deck__meta">{meta}</span>}
        {phone ? null : <div className="su-deck__actions">{actions}</div>}
      </div>
      {phone ? (
        <>
          <span className="su-deck__meta su-deck__meta--phone">{meta}</span>
          <div className="su-preview-row su-deck__actions--phone">{actions}</div>
        </>
      ) : null}
    </div>
  );
}

/**
 * The Simple Assets viewer stage for an audio asset — the deck at full size, replacing the
 * bare <audio> the overlay used to render. The viewer's own Favorite / Download / Delete row
 * stays in SimplePreview; only the stage lives here.
 */
export function SimpleAudioViewer({ asset, player, breakpoint, meta, run, onSendToVideo }) {
  const phone = breakpoint === "phone";
  const duration = player.duration > 0 ? player.duration : audioAssetDurationSeconds(asset);
  const loaded = player.loadedTakeId === asset.id;
  return (
    <div className="su-audio-viewer" data-testid="su-audio-viewer">
      <LoadedAudioElement asset={asset} player={player} className="su-deck__el" />
      {run || meta ? (
        <div className="su-audio-viewer__meta">
          {run ? (
            <span className={`audio-mode-chip audio-mode-chip--${run.mode}`}>{run.modeLabel}</span>
          ) : null}
          {meta ? <span className="su-audio-viewer__meta-text">{meta}</span> : null}
        </div>
      ) : null}
      <div className="su-audio-viewer__panel">
        <AudioWaveform
          asset={asset}
          bars={WAVEFORM_BARS.stage}
          height={VIEWER_WAVE_HEIGHT[breakpoint] ?? VIEWER_WAVE_HEIGHT.desktop}
          label="Clip"
          onSeek={player.seekFraction}
          progress={loaded ? player.progress : 0}
        />
      </div>
      <div className="su-audio-viewer__transport">
        <AudioPlayButton
          onClick={() => player.toggleTake(asset)}
          playing={loaded && player.isPlaying}
          size="xl"
        />
        <AudioClock className="audio-clock su-audio-viewer__clock" current={loaded ? player.currentTime : 0} total={duration} />
        {!phone && onSendToVideo ? (
          <button className="secondary-action su-audio-viewer__send" onClick={() => onSendToVideo(asset)} type="button">
            <Icon.Editor size={15} />
            Send to Video Editor
          </button>
        ) : null}
      </div>
    </div>
  );
}

/**
 * The Assets grid cell for an audio asset: a waveform over a bottom bar carrying the play
 * button, the display name and the duration — inside the same square cell images and clips
 * use, so the grid keeps its rhythm.
 *
 * The cell carries no transport of its own: in Assets the DECK IS THE VIEWER, so the play
 * button opens the viewer already playing (`onToggle`) while the cell body opens it paused
 * (`onOpen`). `loaded` just rings the clip the viewer last held.
 */
export function SimpleAudioTile({ asset, loaded, onOpen, onToggle, phone }) {
  const duration = audioAssetDurationSeconds(asset);
  const name = asset.displayName ?? asset.id;
  return (
    <div className={loaded ? "su-asset su-asset--audio is-loaded" : "su-asset su-asset--audio"}>
      <button
        className="su-asset-open"
        onClick={() => onOpen(asset)}
        title={name}
        type="button"
      >
        <span className="su-asset-open__label">{name}</span>
      </button>
      <span className="su-asset-audio__wave">
        <AudioWaveform asset={asset} bars={WAVEFORM_BARS.tile} height={phone ? 38 : 42} label={name} />
      </span>
      <span className="su-asset-audio__bar">
        <AudioPlayButton label={`Play ${name}`} onClick={() => onToggle(asset)} playing={false} size="xs" />
        {phone ? null : <span className="su-asset-audio__name">{name}</span>}
        <span className="su-asset-audio__time">{formatClock(duration)}</span>
      </span>
      <span aria-hidden="true" className="su-asset-kind">
        <Icon.Audio size={11} />
      </span>
      {asset.status?.favorite ? (
        <span aria-hidden="true" className="su-asset-fav">
          <Icon.Star filled size={14} />
        </span>
      ) : null}
    </div>
  );
}
