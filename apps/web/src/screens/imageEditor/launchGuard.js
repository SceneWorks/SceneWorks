

export function createEditorLaunchGuard() {
  let activeId = null;
  return {
    async consume({ launch, isCurrent, confirmDiscard, openAsset, clearLaunch }) {
      if (!launch?.assetId || activeId === launch.id) return false;
      const launchId = launch.id;
      activeId = launchId;
      try {
        if (!(await confirmDiscard()) || activeId !== launchId || !isCurrent(launchId)) return false;
        await openAsset(launch.assetId);
        if (activeId !== launchId || !isCurrent(launchId)) return false;
        clearLaunch?.();
        return true;
      } finally {
        if (activeId === launchId) activeId = null;
      }
    },
  };
}
