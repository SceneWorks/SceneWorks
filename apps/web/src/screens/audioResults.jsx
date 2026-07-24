import React, { useMemo } from "react";
import { Icon } from "../components/Icons.jsx";
import { WorkerProgressCard, audioAssetDurationSeconds } from "../components/WorkerProgressCard.jsx";
import {
  AudioClock,
  AudioDownloadButton,
  AudioPlayButton,
  AudioWaveform,
  LoadedAudioElement,
  WAVEFORM_BARS,
} from "../components/audioTakeParts.jsx";
import {
  audioAssetRunGroups,
  audioExpectedTakes,
  audioRunGroups,
  audioTakeTitle,
  formatClock,
  formatRelativeTime,
} from "../audioTakes.js";
import { terminalStatuses } from "../jobTypes.js";
// `job.progress` is a 0..1 fraction (the same value WorkerProgressCard renders); `percent`
// is the one place that conversion lives.
import { percent } from "../formatting.js";

// Audio Studio results zone (epic 14361 / sc-14364) — the "Latest audio" surface.
//
// It replaces a stack of queue-worker cards with a CREATIVE surface: a slim in-flight strip
// pinned at the top, run-grouped waveform take cards in the same .review-grid geometry the
// Image and Video studios use, and a now-playing play deck docked above the runs while a
// take is loaded. The FULL WorkerProgressCard — job id, GPU meters, attempt counter,
// hardware pills — deliberately stays in the Queue screen; the only place it still appears
// here is a FAILED run, which keeps its existing card treatment (and its Retry actions).
//
// Everything is derived: a run IS a job in the audio local-job lane, its takes are the
// assets that job produced, and the header chips are read off the job's own payload
// (audioTakes.js). No new API surface, no stored grouping.

// The head's count clause. Written as the design words it, from the takes actually resolved.
function clipCountLabel(count) {
  return `last ${count} ${count === 1 ? "clip" : "clips"} in this project`;
}

function ModeChip({ mode, label, extra = null, className = "" }) {
  return (
    <span className={`audio-mode-chip audio-mode-chip--${mode} ${className}`.trim()}>
      {label}
      {extra ? <span className="audio-mode-chip__extra">{extra}</span> : null}
    </span>
  );
}

/**
 * One running audio run, pinned above the run groups. A trimmed progress row — status chip,
 * title, message, a 6px track and Cancel / Queue — rather than the full worker card.
 */
export function AudioInflightStrip({ run, onCancel, onOpenQueue }) {
  const { job } = run;
  const takes = audioExpectedTakes(job);
  const numeric = Number(job.progress);
  const hasProgress = Number.isFinite(numeric) && numeric > 0;
  const title = [run.modeLabel, run.modelName, takes ? `${takes} takes` : null]
    .filter(Boolean)
    .join(" · ");
  const message = job.message ?? "";
  return (
    <div className="audio-inflight" data-testid="audio-inflight-strip">
      <span className="audio-inflight__status">{job.cancelRequested ? "Canceling" : "Rendering"}</span>
      <div className="audio-inflight__body">
        <div className="audio-inflight__titles">
          <strong className="audio-inflight__title">{title}</strong>
          {message ? <span className="audio-inflight__message">{message}</span> : null}
        </div>
        <div className="progress-track audio-inflight__track" aria-label="Run progress">
          <span style={{ width: hasProgress ? percent(Math.min(1, numeric)) : "0%" }} />
        </div>
      </div>
      <div className="audio-inflight__actions">
        {onCancel ? (
          <button
            className="secondary-action"
            disabled={job.cancelRequested}
            onClick={() => onCancel(job)}
            type="button"
          >
            {job.cancelRequested ? "Canceling…" : "Cancel"}
          </button>
        ) : null}
        {onOpenQueue ? (
          <button className="audio-link" onClick={() => onOpenQueue(job)} type="button">
            Queue
          </button>
        ) : null}
      </div>
    </div>
  );
}

/** A single take: waveform thumbnail, prompt excerpt, meta and the four asset actions. */
export function AudioTakeCard({
  run,
  asset,
  index,
  loaded,
  playing,
  progress,
  onToggle,
  onFavorite,
  onSendToVideo,
  onDiscard,
}) {
  const duration = audioAssetDurationSeconds(asset);
  const favorite = Boolean(asset.status?.favorite);
  return (
    <article className={loaded ? "audio-take is-loaded" : "audio-take"} data-testid="audio-take-card">
      <div className="audio-take__wave">
        <AudioWaveform
          asset={asset}
          bars={WAVEFORM_BARS.card}
          height={118}
          label={`Take ${index + 1}`}
          progress={loaded ? progress : 0}
        />
        <span className="audio-take__badge">Take {index + 1}</span>
        <AudioPlayButton
          className="audio-take__play"
          label={`${playing && loaded ? "Pause" : "Play"} take ${index + 1}`}
          onClick={() => onToggle(asset)}
          playing={loaded && playing}
          size="md"
        />
        <span className="audio-take__duration">{formatClock(duration)}</span>
      </div>
      <p className="audio-take__prompt">{audioTakeTitle(run.job, asset)}</p>
      <div className="audio-take__meta">
        <span>
          {run.modeLabel} · {run.modelName}
        </span>
        <span className="audio-take__meta-duration">{formatClock(duration)}</span>
      </div>
      <div className="audio-take__actions">
        <button
          aria-label={favorite ? "Remove from favorites" : "Add to favorites"}
          aria-pressed={favorite}
          className={favorite ? "audio-icon-btn is-on" : "audio-icon-btn"}
          onClick={() => onFavorite?.(asset)}
          title="Favorite"
          type="button"
        >
          <Icon.Star filled={favorite} size={14} />
        </button>
        <AudioDownloadButton asset={asset} className="audio-icon-btn" />
        <button
          aria-label="Send to Video Editor"
          className="audio-icon-btn"
          onClick={() => onSendToVideo?.(asset)}
          title="Send to Video Editor"
          type="button"
        >
          <Icon.Editor size={14} />
        </button>
        <button
          aria-label="Discard take"
          className="audio-icon-btn audio-icon-btn--danger"
          onClick={() => onDiscard?.(asset)}
          title="Discard"
          type="button"
        >
          <Icon.Trash size={14} />
        </button>
      </div>
    </article>
  );
}

/** One generation run: the header (mode / model / settings chips / time / Run again) + its takes. */
export function AudioRunGroup({
  run,
  loadedTakeId,
  playing,
  progress,
  onToggle,
  onRunAgain,
  onFavorite,
  onSendToVideo,
  onDiscard,
}) {
  return (
    <section className="audio-run" data-testid="audio-run-group">
      <header className="audio-run__head">
        <ModeChip label={run.modeLabel} mode={run.mode} />
        <strong className="audio-run__model">{run.modelName}</strong>
        <div className="audio-run__chips">
          {run.chips.map((chip) => (
            <span className="audio-run__chip" key={chip}>
              {chip}
            </span>
          ))}
        </div>
        <span aria-hidden="true" className="audio-run__rule" />
        {run.createdAt ? <span className="audio-run__time">{formatRelativeTime(run.createdAt)}</span> : null}
        {onRunAgain && run.replayable ? (
          <button className="audio-link" onClick={() => onRunAgain(run.job)} type="button">
            <Icon.Refresh size={13} />
            Run again
          </button>
        ) : null}
      </header>
      <div className="audio-take-grid">
        {run.takes.map((asset, index) => (
          <AudioTakeCard
            asset={asset}
            index={index}
            key={asset.id}
            loaded={asset.id === loadedTakeId}
            onDiscard={onDiscard}
            onFavorite={onFavorite}
            onSendToVideo={onSendToVideo}
            onToggle={onToggle}
            playing={playing}
            progress={progress}
            run={run}
          />
        ))}
      </div>
    </section>
  );
}

/**
 * The now-playing deck: head + waveform stage + transport. Rendered ONLY while a take is
 * loaded, docked directly above the first run group. Drives the one real <audio> element.
 */
export function AudioPlayDeck({ run, asset, takeIndex, player, onRunAgain, onSendToVideo }) {
  const duration = player.duration > 0 ? player.duration : audioAssetDurationSeconds(asset);
  const meta = [run?.modelName, ...(run?.chips ?? [])].filter(Boolean).join(" · ");
  return (
    <section className="audio-deck" data-testid="audio-play-deck">
      <LoadedAudioElement asset={asset} player={player} />
      <header className="audio-deck__head">
        <ModeChip
          extra={takeIndex >= 0 ? `Take ${takeIndex + 1}` : null}
          label={run?.modeLabel ?? "Audio"}
          mode={run?.mode ?? "speech"}
        />
        <strong className="audio-deck__title" title={audioTakeTitle(run?.job, asset)}>
          {audioTakeTitle(run?.job, asset)}
        </strong>
        <div className="audio-deck__head-actions">
          <AudioDownloadButton asset={asset} className="secondary-action" iconSize={15} label="Download" />
          {onSendToVideo ? (
            <button className="secondary-action" onClick={() => onSendToVideo(asset)} type="button">
              <Icon.Editor size={15} />
              Send to Video Editor
            </button>
          ) : null}
          {onRunAgain && run?.job && run?.replayable ? (
            <button className="secondary-action audio-deck__again" onClick={() => onRunAgain(run.job)} type="button">
              <Icon.Refresh size={15} />
              Run again
            </button>
          ) : null}
          <button
            aria-label="Close player"
            className="secondary-action audio-deck__close"
            onClick={player.unload}
            title="Close player"
            type="button"
          >
            <Icon.Close size={15} />
          </button>
        </div>
      </header>

      <div className="audio-deck__stage">
        <AudioWaveform
          asset={asset}
          bars={WAVEFORM_BARS.deck}
          height={140}
          label="Clip"
          onSeek={player.seekFraction}
          progress={player.progress}
        />
        <div className="audio-deck__ticks">
          <span>0:00</span>
          <span>{formatClock(duration)}</span>
        </div>
      </div>

      <div className="audio-deck__transport">
        <button
          aria-label="Back 5 seconds"
          className="audio-round-btn"
          onClick={() => player.skip(-5)}
          type="button"
        >
          <Icon.ArrowLeft size={15} />
        </button>
        <AudioPlayButton onClick={() => player.toggleTake(asset)} playing={player.isPlaying} size="xl" />
        <button
          aria-label="Forward 5 seconds"
          className="audio-round-btn"
          onClick={() => player.skip(5)}
          type="button"
        >
          <Icon.ArrowRight size={15} />
        </button>
        <button
          aria-label="Loop"
          aria-pressed={player.loop}
          className={player.loop ? "audio-round-btn is-on" : "audio-round-btn"}
          onClick={player.toggleLoop}
          type="button"
        >
          <Icon.Refresh size={15} />
        </button>
        <AudioClock current={player.currentTime} total={duration} />
        {meta ? <span className="audio-deck__meta">{meta}</span> : null}
      </div>
    </section>
  );
}

/**
 * The whole "Latest audio" zone.
 *
 * `player` is the surface's single useAudioTakePlayer instance — the deck and every take
 * card read their loaded/playing state from it, so only one clip plays at a time.
 */
export function AudioResults({
  jobs,
  assets,
  recentAssets = [],
  models,
  player,
  streamingBadge = null,
  onCancelJob,
  onOpenQueue,
  onSeeAll,
  onRunAgain,
  onFavorite,
  onSendToVideo,
  onDiscard,
}) {
  // This session's runs (the local-job lane) own the takes they produced; the project's
  // recent clips fill in behind them, reconstructed from each asset's own replay record, so
  // the surface is populated on a fresh launch rather than empty until you generate.
  const runs = useMemo(() => {
    const jobRuns = audioRunGroups(jobs, assets, models);
    const covered = new Set(jobRuns.flatMap((run) => run.takes.map((asset) => asset.id)));
    return [...jobRuns, ...audioAssetRunGroups(recentAssets, models, covered)];
  }, [jobs, assets, recentAssets, models]);

  const inflight = runs.filter((run) => run.running);
  // A failed run has no takes to show, so it keeps the full worker card — the surface where
  // the error, the attempt count and Retry live.
  const failed = runs.filter((run) => !run.running && run.takes.length === 0 && terminalStatuses.has(run.job.status));
  const completed = runs.filter((run) => !run.running && run.takes.length > 0);
  const takeCount = completed.reduce((total, run) => total + run.takes.length, 0);

  const loaded = useMemo(() => {
    if (!player.loadedTakeId) {
      return null;
    }
    for (const run of runs) {
      const index = run.takes.findIndex((asset) => asset.id === player.loadedTakeId);
      if (index >= 0) {
        return { run, asset: run.takes[index], index };
      }
    }
    return null;
  }, [runs, player.loadedTakeId]);

  return (
    <section className="review-panel audio-results">
      <div className="review-panel-head">
        <div className="review-panel-head-title">
          <h2>Latest audio</h2>
          {takeCount ? <span className="audio-results__count">{clipCountLabel(takeCount)}</span> : null}
          {streamingBadge}
        </div>
        {onSeeAll ? (
          <button className="audio-link audio-results__all" onClick={onSeeAll} type="button">
            See all in Assets
            <Icon.ArrowRight size={14} />
          </button>
        ) : null}
      </div>

      {inflight.map((run) => (
        <AudioInflightStrip key={run.id} onCancel={onCancelJob} onOpenQueue={onOpenQueue} run={run} />
      ))}

      {failed.map((run) => (
        <WorkerProgressCard job={run.job} key={run.id} onCancel={onCancelJob} onOpenQueue={onOpenQueue} />
      ))}

      {loaded ? (
        <AudioPlayDeck
          asset={loaded.asset}
          onRunAgain={onRunAgain}
          onSendToVideo={onSendToVideo}
          player={player}
          run={loaded.run}
          takeIndex={loaded.index}
        />
      ) : null}

      {completed.map((run) => (
        <AudioRunGroup
          key={run.id}
          loadedTakeId={player.loadedTakeId}
          onDiscard={onDiscard}
          onFavorite={onFavorite}
          onRunAgain={onRunAgain}
          onSendToVideo={onSendToVideo}
          onToggle={player.toggleTake}
          playing={player.isPlaying}
          progress={player.progress}
          run={run}
        />
      ))}

      {runs.length === 0 ? <div className="empty-panel">No audio yet</div> : null}
    </section>
  );
}
