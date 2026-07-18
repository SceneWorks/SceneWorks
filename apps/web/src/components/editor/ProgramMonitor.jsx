import React from "react";
import { AssetMedia, assetCanRenderAsVideo } from "../assetMedia.jsx";
import { Icon } from "../Icons.jsx";

// The center program monitor (design 2a, epic 12798): the 16:9/9:16/1:1 preview of the
// selected clip with an overlaid label + AI pill + resolution readout, and the transport
// row (prev / play-pause / next). Playback is driven by the screen's existing effect via
// the forwarded video ref; the play/pause events flip the screen's isPlaying state.
export function ProgramMonitor({
  selectedAsset,
  aspectClass,
  clipLabel,
  isAi,
  resolutionLabel,
  isPlaying,
  onTogglePlay,
  onPrev,
  onNext,
  previewVideoRef,
  onPlay,
  onPause,
  onEnded,
}) {
  const canPlay = assetCanRenderAsVideo(selectedAsset);

  return (
    <div className="ve-monitor">
      <div className="ve-monitor-stage">
        <div className={`ve-program ${aspectClass}`}>
          {selectedAsset ? (
            <AssetMedia
              asset={selectedAsset}
              className="ve-program-media"
              controls={false}
              onEnded={onEnded}
              onPause={onPause}
              onPlay={onPlay}
              ref={previewVideoRef}
            />
          ) : (
            <span className="ve-program-empty">Select a timeline item</span>
          )}
          {clipLabel ? <span className="ve-program-label">{clipLabel}</span> : null}
          {isAi ? (
            <span className="ve-program-ai">
              <Icon.Stars size={11} />
              AI clip
            </span>
          ) : null}
          {resolutionLabel ? <span className="ve-program-res">{resolutionLabel}</span> : null}
        </div>
      </div>
      <div className="ve-transport">
        <button className="ve-transport-btn" onClick={onPrev} title="Previous edit" type="button">
          <Icon.ArrowLeft size={16} />
        </button>
        <button
          className="ve-play"
          disabled={!canPlay}
          onClick={onTogglePlay}
          title={isPlaying ? "Pause" : "Play"}
          type="button"
        >
          {isPlaying ? <Icon.Pause size={18} /> : <Icon.Play size={18} />}
        </button>
        <button className="ve-transport-btn" onClick={onNext} title="Next edit" type="button">
          <Icon.ArrowRight size={16} />
        </button>
      </div>
    </div>
  );
}
