// Keeps only the newest smart-select mask asset load eligible to commit. Starting a new
// selection invalidates older job completions and aborts an asset fetch already in flight.
export function createLatestMaskAssetLoader({ fetchImpl = fetch, assetUrlFor, blobToImage }) {
  let generation = 0;
  let controller = null;

  function invalidate() {
    generation += 1;
    controller?.abort();
    controller = null;
    return generation;
  }

  async function load(asset, requestGeneration, onLoad) {
    if (requestGeneration !== generation) return false;
    controller?.abort();
    const activeController = new AbortController();
    controller = activeController;
    try {
      const response = await fetchImpl(assetUrlFor(asset), { signal: activeController.signal });
      if (!response.ok) throw new Error(`Failed to load mask (${response.status})`);
      const decoded = await blobToImage(await response.blob());
      if (requestGeneration !== generation || activeController.signal.aborted) {
        URL.revokeObjectURL(decoded.objectUrl);
        return false;
      }
      onLoad(decoded.image);
      URL.revokeObjectURL(decoded.objectUrl);
      return true;
    } catch (error) {
      // A superseded request is expected cancellation, not an editor error.
      if (requestGeneration !== generation || activeController.signal.aborted) return false;
      throw error;
    } finally {
      if (controller === activeController) controller = null;
    }
  }

  return { invalidate, load };
}
