export function titleForAsset(asset) {
  return asset?.displayName ?? asset?.title ?? asset?.name ?? asset?.id ?? "Untitled asset";
}
