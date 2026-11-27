import { useEffect, useMemo, useRef } from "react";
import { assetsApi } from "../api/assetsResources.js";
import { isSummaryAsset } from "../assetCatalog.js";

// Upgrade the catalog's OPEN assets (the selected one, the previewed one) from their summary
// listing to the full envelope, so the detail panel and fullscreen viewer read the complete
// recipe, generation set and adapter settings. One request per summary object: a newer summary
// of the same asset (it changed) is fetched again, and the upgrade only lands while the catalog
// still holds the summary it was fetched for, so a stale response never overwrites newer data.
export function useFullAssetHydration({ assets, assetIds, token, setAssets, activeProjectRef }) {
  const requestedRef = useRef(new WeakSet());
  const idsKey = assetIds.filter(Boolean).join("\n");
  const targets = useMemo(() => {
    const ids = new Set(idsKey ? idsKey.split("\n") : []);
    return ids.size ? assets.filter((asset) => ids.has(asset.id) && isSummaryAsset(asset)) : [];
  }, [assets, idsKey]);
  useEffect(() => {
    for (const summary of targets) {
      if (requestedRef.current.has(summary)) continue;
      requestedRef.current.add(summary);
      assetsApi.item(summary.projectId, summary.id, token).then((full) => {
        if (!full?.id || activeProjectRef.current?.id !== summary.projectId) return;
        setAssets((items) => items.map((item) => (item === summary ? full : item)));
      }).catch(() => {
        // The summary keeps rendering; replay paths load the full asset themselves and surface
        // their own errors (ensureFullAsset).
      });
    }
  }, [activeProjectRef, setAssets, targets, token]);
}
