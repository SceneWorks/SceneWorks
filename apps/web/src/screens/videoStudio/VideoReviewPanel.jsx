import { jobVideoResultAssets } from "./videoTierHelpers.js";
import React from "react";
import { AssetCard } from "../../components/assetPanels.jsx";
import { AssetMedia } from "../../components/assetMedia.jsx";
import { WorkerProgressCard } from "../../components/WorkerProgressCard.jsx";

export function VideoReviewPanel({
  localJobs,
  assets,
  onPreview,
  onCancelJob,
  onOpenQueue,
  latestAssets,
  deleteAsset,
  purgeAsset,
  updateAssetStatus,
  comparisonAsset,
  comparisonSource,
  comparisonMode,
  setComparisonMode,
  abSide,
  setAbSide
}) {
  return (
<div className="studio-results">
          <section className="review-panel">
            <div className="review-panel-head">
              <h2>Latest batch</h2>
              <span className="kbd-hint">
                <kbd>⌘</kbd>
                <kbd>↵</kbd>
                to render
              </span>
            </div>
            {localJobs.length ? (
              <div className="worker-progress-card-stack local-job-stack">
                {localJobs.map((job) => {
                  const jobAssets = jobVideoResultAssets(job, assets);
                  return (
                    <WorkerProgressCard
                      key={job.id}
                      job={job}
                      thumbnailsVariant="video-player"
                      thumbnailAssets={jobAssets}
                      onThumbnailClick={(asset) => onPreview(asset, jobAssets)}
                      onCancel={onCancelJob}
                      onOpenQueue={onOpenQueue}
                    />
                  );
                })}
              </div>
            ) : null}
            {latestAssets.length ? (
              <div className="recent-assets">
                {localJobs.length ? <h3 className="recent-assets__title">Recent Assets</h3> : null}
                <div className="review-grid">
                  {latestAssets.map((asset) => (
                    <AssetCard
                      asset={asset}
                      deleteAsset={deleteAsset}
                      key={asset.id}
                      onPreview={(previewed) => onPreview(previewed, latestAssets)}
                      purgeAsset={purgeAsset}
                      updateAssetStatus={updateAssetStatus}
                    />
                  ))}
                </div>
              </div>
            ) : localJobs.length ? null : (
              <div className="empty-panel">No fresh clip batch</div>
            )}
          </section>

          {/* Replace-person A/B / side-by-side review (video-specific, no Image Studio
              equivalent) — surfaces the latest replacement clip against its source. */}
          {comparisonAsset?.recipe?.mode === "replace_person" && comparisonSource ? (
            <div className="comparison-panel">
              <div className="comparison-toolbar">
                <div className="segmented-control compact-segment" aria-label="Comparison mode">
                  <button className={comparisonMode === "side_by_side" ? "active" : ""} onClick={() => setComparisonMode("side_by_side")} type="button">
                    Side by Side
                  </button>
                  <button className={comparisonMode === "ab" ? "active" : ""} onClick={() => setComparisonMode("ab")} type="button">
                    A/B
                  </button>
                </div>
                {comparisonMode === "ab" ? (
                  <div className="segmented-control compact-segment" aria-label="A/B source">
                    <button className={abSide === "original" ? "active" : ""} onClick={() => setAbSide("original")} type="button">
                      A
                    </button>
                    <button className={abSide === "replacement" ? "active" : ""} onClick={() => setAbSide("replacement")} type="button">
                      B
                    </button>
                  </div>
                ) : null}
              </div>
              {comparisonMode === "side_by_side" ? (
                <div className="comparison-grid">
                  <div>
                    <p className="eyebrow">Original</p>
                    <AssetMedia asset={comparisonSource} />
                  </div>
                  <div>
                    <p className="eyebrow">Replacement</p>
                    <AssetMedia asset={comparisonAsset} />
                  </div>
                </div>
              ) : (
                <div className="comparison-single">
                  <p className="eyebrow">{abSide === "original" ? "A Original" : "B Replacement"}</p>
                  <AssetMedia asset={abSide === "original" ? comparisonSource : comparisonAsset} />
                </div>
              )}
            </div>
          ) : null}
        </div>
  );
}
