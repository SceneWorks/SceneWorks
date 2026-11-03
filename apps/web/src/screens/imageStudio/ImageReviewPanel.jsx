import React from "react";
import { AssetCard } from "../../components/assetPanels.jsx";
import { WorkerProgressCard } from "../../components/WorkerProgressCard.jsx";

export function ImageReviewPanel({
  localJobGroups,
  onPreview,
  onCancelJob,
  onOpenQueue,
  latestAssets,
  deleteAsset,
  purgeAsset,
  updateAssetStatus
}) {
  return (
<section className="review-panel">
            <div className="review-panel-head">
              <h2>Latest batch</h2>
              <span className="kbd-hint">
                <kbd>⌘</kbd>
                <kbd>↵</kbd>
                to generate
              </span>
            </div>
            {localJobGroups.length ? (
              <div className="worker-progress-card-stack local-job-stack">
                {localJobGroups.map(({ job, completedAssets, expectedCount }) => (
                  <WorkerProgressCard
                    key={job.id}
                    job={job}
                    thumbnailsVariant="image-grid"
                    thumbnailAssets={completedAssets}
                    expectedThumbnailCount={expectedCount}
                    onThumbnailClick={(asset) => onPreview(asset, completedAssets)}
                    onCancel={onCancelJob}
                    onOpenQueue={onOpenQueue}
                  />
                ))}
              </div>
            ) : null}
            {latestAssets.length ? (
              <div className="recent-assets">
                {localJobGroups.length ? <h3 className="recent-assets__title">Recent Assets</h3> : null}
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
            ) : localJobGroups.length ? null : (
              <div className="empty-panel">No fresh image batch</div>
            )}
          </section>
  );
}
