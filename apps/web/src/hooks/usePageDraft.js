import { useEffect, useState } from "react";

// Small form snapshots live in session storage rather than mounted pages. Media,
// catalogs, editor documents and undo history must never enter these snapshots.
const PREFIX = "sceneworks-page-draft:";
export function usePageDraft(page, workspaceId) {
  const key = `${PREFIX}${page}:${workspaceId ?? "default"}`;
  const [saved] = useState(() => {
    try {
      const value = JSON.parse(sessionStorage.getItem(key) ?? "{}");
      return value && typeof value === "object" && !Array.isArray(value) ? value : {};
    }
    catch { return {}; }
  });
  return saved;
}

export function usePageDraftWriter(page, workspaceId, settings) {
  const key = `${PREFIX}${page}:${workspaceId ?? "default"}`;
  const serialized = JSON.stringify(settings);
  useEffect(() => {
    try { sessionStorage.setItem(key, serialized); } catch { /* Storage can be unavailable. */ }
  }, [key, serialized]);
}
