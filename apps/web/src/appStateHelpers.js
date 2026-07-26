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

export function isCurrentProjectRequest(activeProjectId, requestedProjectId) {
  return activeProjectId === requestedProjectId;
}

export function reconcileActiveProject(projects, current) {
  if (!current) return projects[0] ?? null;
  return projects.find((project) => project.id === current.id) ?? projects[0] ?? null;
}

export function applySuccessfulProjectRefresh(result, setProjects, setActiveProject) {
  if (result.error) return false;
  const projects = result.value;
  setProjects(projects);
  setActiveProject((current) => reconcileActiveProject(projects, current));
  return true;
}
