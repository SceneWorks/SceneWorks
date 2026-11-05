import { useCallback, useState } from "react";
import { stripYue2ExportHeader } from "../../yue2Lab.js";
import { getYue2ScoreVersion, inspectYue2Score } from "../../api/yue2.js";

// Extracted from Yue2SongLab along its existing concern boundary.
export function useYue2Inspection({
  token,
  projectId
}) {
  // Score inspection (supplied ABC / pasted cover score / chosen cover version).
  const [preview, setPreview] = useState({ key: "", inspection: null, error: null });

  async function inspect(key, abc) {
    setPreview({ key, inspection: null, error: null });
    try {
      // A lab-exported score opens with the `% SceneWorks YuE2 export:` licence header, which the
      // native dialect (it requires `X:1` first) refuses — strip it, as a submission does.
      const inspection = await inspectYue2Score(stripYue2ExportHeader(abc), token);
      setPreview({ key, inspection, error: null });
    } catch (error) {
      setPreview({ key, inspection: null, error });
    }
  }

  async function inspectCoverVersion(versionId) {
    setPreview({ key: "coverVersion", inspection: null, error: null });
    try {
      const detail = await getYue2ScoreVersion(projectId, versionId, token);
      const inspection = await inspectYue2Score(detail?.version?.score?.abc ?? "", token);
      setPreview({ key: "coverVersion", inspection, error: null });
    } catch (error) {
      setPreview({ key: "coverVersion", inspection: null, error });
    }
  }

  const fetchVersionAbc = useCallback(
    async (versionId) => (await getYue2ScoreVersion(projectId, versionId, token))?.version?.score?.abc ?? null,
    [projectId, token],
  );
  return { preview, inspect, inspectCoverVersion, fetchVersionAbc };
}
