import { assetsApi } from "../api/assetsResources.js";
import { useCallback, useLayoutEffect } from "react";
import { appConfirm } from "../appConfirm.jsx";

// F-189: extracted from App along its existing concern boundary.
export function useAssetActions({
  token, setAssets, setError, refreshCharactersRef, setSelectedAssetId, previewNavigation, previewDirectionRef, setPreviewAsset, closePreview, purgeAssetRef, activeProject, activeProjectRef
}) {
  const updateAssetStatus = useCallback(
    async (asset, changes) => {
      try {
        const updated = await assetsApi.status(asset.projectId, asset.id, token, {
          method: "PATCH",
          body: JSON.stringify(changes),
        });
        setAssets((items) => items.map((item) => (item.id === updated.id ? updated : item)));
        setError("");
      } catch (err) {
        setError(err.message);
      }
    },
    [setAssets, setError, token],
  );

  const updateAssetTags = useCallback(
    async (asset, tags) => {
      try {
        const updated = await assetsApi.tags(asset.projectId, asset.id, token, {
          method: "PATCH",
          body: JSON.stringify({ tags }),
        });
        setAssets((items) => items.map((item) => (item.id === updated.id ? updated : item)));
        setError("");
      } catch (err) {
        setError(err.message);
      }
    },
    [setAssets, setError, token],
  );

  // A move endpoint returns the whole moved upscale-fold group (sc-10205) as an
  // array, requested asset first — merge every member into state or the hidden
  // fold-mate lingers with a stale origin until the next full refresh.
  const mergeMovedAssets = useCallback((moved) => {
    const list = Array.isArray(moved) ? moved : [moved];
    const byId = new Map(list.map((item) => [item.id, item]));
    setAssets((items) => items.map((item) => byId.get(item.id) ?? item));
    return list[0];
  }, [setAssets]);

  // Promote a character asset into the Main Asset Library (sc-8341): a true move —
  // the backend flips origin + detaches the character, so refresh characters too.
  const moveAssetToLibrary = useCallback(
    async (asset) => {
      try {
        const moved = await assetsApi.moveToLibrary(asset.projectId, asset.id, token, {
          method: "POST",
        });
        const updated = mergeMovedAssets(moved);
        refreshCharactersRef.current?.(asset.projectId);
        setError("");
        return updated;
      } catch (err) {
        setError(err.message);
        throw err;
      }
    },
    [token, mergeMovedAssets, refreshCharactersRef, setError],
  );

  // Move an asset into a character's assets (sc-10200): the true-move twin of
  // moveAssetToLibrary, NOT a link — the backend flips origin to character_studio
  // (so the asset leaves the Library) and re-anchors the character association
  // without touching the curated references[] ("Approved set"). Refresh characters
  // so any curated reference the move detached drops out of their panels too.
  const moveAssetToCharacter = useCallback(
    async (asset, characterId) => {
      try {
        const moved = await assetsApi.moveToCharacter(asset.projectId, asset.id, token, {
          method: "POST",
          body: JSON.stringify({ characterId }),
        });
        const updated = mergeMovedAssets(moved);
        refreshCharactersRef.current?.(asset.projectId);
        setError("");
        return updated;
      } catch (err) {
        setError(err.message);
        throw err;
      }
    },
    [token, mergeMovedAssets, refreshCharactersRef, setError],
  );

  const deleteAsset = useCallback(
    async (asset) => {
      try {
        await assetsApi.item(asset.projectId, asset.id, token, { method: "DELETE" });
        setAssets((items) =>
          items.map((item) =>
            item.id === asset.id ? { ...item, status: { ...item.status, trashed: true } } : item,
          ),
        );
        setError("");
      } catch (err) {
        setError(err.message);
      }
    },
    [setAssets, setError, token],
  );

  const purgeAsset = useCallback(
    async (asset) => {
      try {
        let result = await assetsApi.purge(asset.projectId, asset.id, token, { method: "DELETE" });
        // The purge first tries the OS trash (recoverable). If that fails nothing was
        // removed; confirm before falling back to a permanent delete. Routed through the
        // desktop-safe appConfirm (sc-12068) — window.confirm no-ops in the Tauri WebView.
        if (result?.status === "trash_unavailable") {
          const proceed = await appConfirm({
            title: "Move to trash failed",
            message: "Cannot move to trash. Continue to permanently delete.",
            confirmLabel: "Delete permanently",
            cancelLabel: "Cancel",
            tone: "danger",
          });
          if (!proceed) {
            setError("");
            return;
          }
          result = await assetsApi.purgePermanently(asset.projectId, asset.id, token, { method: "DELETE" });
        }
        setAssets((items) => items.filter((item) => item.id !== asset.id));
        setSelectedAssetId((current) => (current === asset.id ? null : current));
        setError("");
      } catch (err) {
        setError(err.message);
      }
    },
    [setAssets, setError, setSelectedAssetId, token],
  );

  const deletePreviewAsset = useCallback(
    async (asset) => {
      const { previous, next } = previewNavigation;
      const target =
        previewDirectionRef.current === "previous" ? previous ?? next : next ?? previous;
      await deleteAsset(asset);
      if (target) {
        setPreviewAsset(target);
      } else {
        closePreview();
      }
    },
    [closePreview, deleteAsset, previewDirectionRef, previewNavigation, setPreviewAsset],
  );

  const previewAssetInDirection = useCallback((asset, direction) => {
    if (direction) {
      previewDirectionRef.current = direction;
    }
    setPreviewAsset(asset);
  }, [previewDirectionRef, setPreviewAsset]);

  const purgePreviewAsset = useCallback(
    async (asset) => {
      await purgeAsset(asset);
      closePreview();
    },
    [closePreview, purgeAsset],
  );

  // Keep the ref current for the App-level scratch-op survivor (sc-8850), which purges
  // from the SSE/sweep path without re-subscribing. Published from a post-commit effect
  // rather than the render body (sc-9641, following sc-8940): render-body ref mutation is
  // unsafe because a discarded concurrent/StrictMode render could leave the ref pointing
  // at an uncommitted closure. useLayoutEffect (no dep array) mirrors the sc-8940 refresh
  // block above so the ref always holds the newest *committed* purgeAsset, and flushes
  // before any passive effect on the same commit. `purgeAsset` is a useCallback([token]),
  // so this only rewrites when the token changes. It lives here (not in the sc-8940 block)
  // because purgeAsset is defined after it; the sole read site (the scratch registry's
  // purge callback) fires from the SSE/sweep path — post-commit — never during render.
  useLayoutEffect(() => {
    purgeAssetRef.current = purgeAsset;
  });

  const importAsset = useCallback(
    async (file, options = {}) => {
      if (!activeProject || !file) {
        const error = new Error("Create or open a project first.");
        if (options.throwOnError) {
          throw error;
        }
        setError(error.message);
        return;
      }
      const projectId = activeProject.id;
      const isCurrent = () => !options.signal?.aborted && activeProjectRef.current?.id === projectId;
      const body = new FormData();
      body.append("file", file);
      // Optional lineage for derived imports (Image Editor Save, sc-2434): link the
      // new asset to the source it was opened from + record the edit-chain provenance.
      if (options.sourceAssetId) body.append("sourceAssetId", options.sourceAssetId);
      if (options.provenance) body.append("provenance", JSON.stringify(options.provenance));
      try {
        const imported = await assetsApi.list(projectId, token, {
          method: "POST",
          body,
          signal: options.signal,
        });
        if (!isCurrent()) return null;
        setAssets((items) => [imported, ...items.filter((item) => item.id !== imported.id)]);
        // Field-scoped pickers select the imported asset in their own local value. Let them opt
        // out of changing the app-wide Library selection, which studios also observe as a request
        // to replace their PRIMARY source (for example First frame / Left clip).
        if (options.select !== false) {
          setSelectedAssetId(imported.id);
        }
        setError("");
        return imported;
      } catch (err) {
        if (!isCurrent()) return null;
        if (options.throwOnError) {
          throw err;
        }
        setError(err.message);
        return null;
      }
    },
    [activeProject, activeProjectRef, setAssets, setError, setSelectedAssetId, token],
  );
  return { updateAssetStatus, updateAssetTags, moveAssetToLibrary, moveAssetToCharacter, deleteAsset, purgeAsset, deletePreviewAsset, previewAssetInDirection, purgePreviewAsset, importAsset };
}
