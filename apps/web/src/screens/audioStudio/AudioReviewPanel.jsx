import React from "react";
import { AudioResults } from "../audioResults.jsx";

export function AudioReviewPanel({
  assets,
  audioLocalJobs,
  audioModels,
  recentAudioAssets,
  onCancelJob,
  deleteAsset,
  updateAssetStatus,
  onOpenQueue,
  runAgain,
  setActiveView,
  sendAssetToVideo,
  player,
  showStreaming
}) {
  return (
<div className="studio-results">
            <AudioResults
              assets={assets}
              jobs={audioLocalJobs}
              models={audioModels}
              recentAssets={recentAudioAssets}
              onCancelJob={onCancelJob}
              onDiscard={(asset) => deleteAsset?.(asset)}
              onFavorite={(asset) =>
                updateAssetStatus?.(asset, { favorite: !asset.status?.favorite })
              }
              onOpenQueue={onOpenQueue}
              onRunAgain={runAgain}
              onSeeAll={() => setActiveView("Library")}
              onSendToVideo={(asset) => sendAssetToVideo?.(asset)}
              player={player}
              streamingBadge={
                /* Streaming reveal (sc-13675): shown ONLY when the selected model advertises
                   audio.supportsStreaming — the clip renders incrementally, so the in-flight strip
                   advances THROUGH the stream as chunks arrive and the first audio is produced
                   before the full clip finishes. Capability-driven, never a hardcoded id. */
                showStreaming ? (
                  <span
                    className="streaming-badge"
                    data-testid="audio-streaming-badge"
                    title="This model streams audio incrementally — the first audio arrives before the full clip finishes rendering."
                  >
                    Streams incrementally
                  </span>
                ) : null
              }
            />
          </div>
  );
}
