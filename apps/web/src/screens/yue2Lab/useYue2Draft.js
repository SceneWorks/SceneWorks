import { useCallback, useMemo, useState } from "react";
import { loadStudioSettings, sessionOnlyFields, useStudioSettingsWriter } from "../../hooks/useStudioSettings.js";
import { restoreYue2Settings } from "../../yue2Lab.js";

// Extracted from Yue2SongLab along its existing concern boundary.
export function useYue2Draft({
  projectId,
  preferencesHydrated,
  model
}) {
  const saved = useMemo(() => restoreYue2Settings(loadStudioSettings("yue2lab", projectId)), [projectId]);

  const [settings, setSettings] = useState(saved);

  const update = useCallback((patch) => setSettings((current) => ({ ...current, ...patch })), []);

  useStudioSettingsWriter("yue2lab", projectId, settings, Boolean(preferencesHydrated) && Boolean(model));

  // Texts and drafts the durable copy leaves out (over their cap, or dropped by the snapshot
  // budget) survive only this session; each surface says so beside the text.
  const sessionOnly = useMemo(() => sessionOnlyFields("yue2lab", settings), [settings]);
  return { settings, setSettings, update, sessionOnly };
}
