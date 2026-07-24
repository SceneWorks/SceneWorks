export function hasVisibleLocalFailureForView(activeView, localJobIds, job) {
  if (activeView === "Image" && localJobIds.image.includes(job.id)) {
    return true;
  }
  if (activeView === "Video" && localJobIds.video.includes(job.id)) {
    return true;
  }
  if (activeView === "Audio" && localJobIds.audio.includes(job.id)) {
    return true;
  }
  if (activeView === "Document" && localJobIds.document.includes(job.id)) {
    return true;
  }
  return activeView === "Models" && job.type === "model_download";
}

export function reconcileSelectedAssetId(items, currentId) {
  if (currentId && items.some((asset) => asset.id === currentId)) {
    return currentId;
  }
  const defaultAsset =
    items.find((asset) => !asset.status?.trashed && !asset.status?.rejected) ?? items[0] ?? null;
  return defaultAsset?.id ?? null;
}

export function isCurrentAssetRefresh(activeProjectId, requestedProjectId) {
  return activeProjectId === requestedProjectId;
}
