import { useEffect, useMemo, useState } from "react";
import { assetsApi } from "../api/assetsResources.js";

// Resolve one asset by id from the catalog, fetching it on its own when the catalog does not hold
// it. The Asset Library routes load the server's `scope=library` slice, so a link out of that
// slice — an edit whose source is a Character Studio image — is not in `assets` there even though
// the asset exists. A missing asset (404, purged) resolves to null, as it did when the lookup was
// catalog-only. The catalog copy always wins once it is present, so a widened catalog or a newer
// listing supersedes the one-off fetch.
export function useCatalogAssetById({ assets, assetId, projectId, token }) {
  const inCatalog = useMemo(
    () => (assetId ? assets.find((candidate) => candidate.id === assetId) ?? null : null),
    [assets, assetId],
  );
  const [fetched, setFetched] = useState(null);
  const needsFetch = Boolean(assetId && projectId && !inCatalog);
  useEffect(() => {
    if (!needsFetch) return undefined;
    const controller = new AbortController();
    assetsApi.item(projectId, assetId, token, { signal: controller.signal })
      .then((asset) => {
        if (!controller.signal.aborted && asset?.id === assetId) setFetched({ projectId, asset });
      })
      .catch((error) => {
        // A definitive not-found (purged since an earlier lookup) drops the cached copy, so a
        // reopened preview never compares against a deleted source. A transient failure keeps
        // whatever was last known, as the catalog-only lookup kept its last listing.
        if (controller.signal.aborted || error?.status !== 404) return;
        setFetched((current) =>
          current?.projectId === projectId && current.asset.id === assetId ? null : current,
        );
      });
    return () => controller.abort();
  }, [assetId, needsFetch, projectId, token]);
  if (inCatalog) return inCatalog;
  return fetched?.projectId === projectId && fetched.asset.id === assetId ? fetched.asset : null;
}
