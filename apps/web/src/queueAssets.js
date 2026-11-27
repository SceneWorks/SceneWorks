import { assetsApi } from "./api/assetsResources.js";

// Queue needs job outputs, not every source/reference in the project's library.
// Reuse known output records and fetch only newly reported IDs, four at a time.
export async function loadQueueAssets(projectId, token, { signal } = {}, jobs = [], knownAssets = []) {
  const known = new Map(knownAssets.map((asset) => [asset.id, asset]));
  const wanted = new Set();
  const owners = new Map();
  const legacySets = new Map();
  for (const job of jobs) {
    const owner = job.projectId ?? projectId;
    if (!owner) continue;
    for (const asset of job.result?.assets ?? []) {
      // Job snapshots predate library edits (trash, ratings, etc.). Acquire the
      // current record even when the job embeds an older copy of that output.
      wanted.add(asset.id);
      owners.set(asset.id, owner);
    }
    for (const id of job.result?.assetIds ?? []) {
      wanted.add(id);
      owners.set(id, owner);
    }
    if (job.result?.generationSetId && !job.result.assetIds?.length && !job.result.assets?.length) {
      const sets = legacySets.get(owner) ?? new Set();
      sets.add(job.result.generationSetId);
      legacySets.set(owner, sets);
    }
  }
  // Old saved jobs can identify outputs only by generation set. Resolve that
  // compatibility shape transiently; never retain the unrelated library rows.
  for (const [owner, sets] of legacySets) {
    const catalog = await assetsApi.listSummaryIncludingHidden(owner, token, { signal });
    for (const asset of catalog) {
      if (!sets.has(asset.generationSetId)) continue;
      known.set(asset.id, asset);
      wanted.add(asset.id);
    }
  }
  const missing = [...wanted].filter((id) => !known.has(id));
  for (let offset = 0; offset < missing.length; offset += 4) {
    if (signal?.aborted) throw new DOMException("Aborted", "AbortError");
    const results = await Promise.all(missing.slice(offset, offset + 4).map(async (id) => {
      try { return await assetsApi.item(owners.get(id), id, token, { signal }); }
      catch (error) { if (error.status === 404) return null; throw error; }
    }));
    for (const asset of results) if (asset?.id) known.set(asset.id, asset);
  }
  return [...wanted].map((id) => known.get(id)).filter(Boolean);
}

export function queueAssetRequestKey(jobs) {
  return jobs.flatMap((job) => {
    const ids = job.result?.assetIds?.length ? job.result.assetIds : job.result?.assets?.map((asset) => asset.id) ?? [];
    return ids.length ? ids.map((id) => `${job.projectId}:${id}`)
      : job.result?.generationSetId ? [`${job.projectId}:set:${job.result.generationSetId}`] : [];
  }).join("\n");
}
