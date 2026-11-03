import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { yue2ProjectRuns } from "../../yue2Lab.js";
import { listYue2Comparisons, listYue2ScoreVersions } from "../../api/yue2.js";

// Extracted from Yue2SongLab along its existing concern boundary.
export function useYue2History({
  projectId,
  token,
  enabled,
  jobs
}) {
  // Score versions and comparisons (server records).
  const [versions, setVersions] = useState([]);

  const [versionsError, setVersionsError] = useState(null);

  const [regenerationNotice, setRegenerationNotice] = useState("");

  const [comparisons, setComparisons] = useState([]);

  const [comparisonsError, setComparisonsError] = useState(null);

  const historyRequests = useRef({ versions: 0, comparisons: 0 });

  useEffect(() => () => { historyRequests.current.versions += 1; historyRequests.current.comparisons += 1; }, [projectId, token]);

  const reloadVersions = useCallback(async () => {
    if (!projectId) return;
    const request = ++historyRequests.current.versions;
    try {
      const listing = await listYue2ScoreVersions(projectId, token);
      if (historyRequests.current.versions !== request) return;
      setVersions([...(listing?.items ?? [])].sort((a, b) => Date.parse(b.createdAt ?? 0) - Date.parse(a.createdAt ?? 0)));
      setRegenerationNotice(listing?.renderNotice ?? "");
      setVersionsError(
        Array.isArray(listing?.unreadable) && listing.unreadable.length
          ? `Some score versions could not be read: ${listing.unreadable.join(", ")}`
          : null,
      );
    } catch (error) {
      if (historyRequests.current.versions === request) setVersionsError(error);
    }
  }, [projectId, token]);

  const reloadComparisons = useCallback(async () => {
    if (!projectId) return;
    const request = ++historyRequests.current.comparisons;
    try {
      const listing = await listYue2Comparisons(projectId, token);
      if (historyRequests.current.comparisons !== request) return;
      setComparisons([...(listing?.items ?? [])].sort((a, b) => Date.parse(b.createdAt ?? 0) - Date.parse(a.createdAt ?? 0)));
      setComparisonsError(
        Array.isArray(listing?.unreadable) && listing.unreadable.length
          ? `Some comparisons could not be read: ${listing.unreadable.join(", ")}`
          : null,
      );
    } catch (error) {
      if (historyRequests.current.comparisons === request) setComparisonsError(error);
    }
  }, [projectId, token]);

  useEffect(() => {
    if (enabled) {
      reloadVersions();
      reloadComparisons();
    }
  }, [enabled, reloadVersions, reloadComparisons]);

  // Runs: the project's YuE2 jobs from the live job list, plus any this lab just submitted that
  // the list has not caught up with (the server's copy wins once it arrives).
  const [submitted, setSubmitted] = useState([]);

  const runs = useMemo(() => {
    const live = yue2ProjectRuns(jobs, projectId);
    const known = new Set(live.map((job) => job.id));
    return [...yue2ProjectRuns(submitted.filter((job) => !known.has(job.id)), projectId), ...live];
  }, [jobs, submitted, projectId]);

  // A finished plan becomes a score version server-side; refresh the history when one lands.
  const finishedKey = runs.filter((job) => job.status === "completed").map((job) => job.id).join(",");

  useEffect(() => {
    if (enabled && finishedKey) reloadVersions();
  }, [enabled, finishedKey, reloadVersions]);
  return { versions, versionsError, regenerationNotice, comparisons, comparisonsError, reloadVersions, reloadComparisons, setSubmitted, runs, finishedKey };
}
