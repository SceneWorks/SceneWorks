import { useCallback, useEffect, useMemo, useState } from "react";
import { appearanceProfile, appearanceWarnings, emptyAppearance, hasAppearance, normalizeAppearance, pruneAppearance, resolveAppearance } from "../data/appearanceCatalog.js";

export function useAppearance({ model, initial, enabled = true, prompt = "" }) {
  const [stored, setStored] = useState(() => normalizeAppearance(initial));
  const [notice, setNotice] = useState("");
  const setSelection = useCallback((value) => { setStored(normalizeAppearance(value)); setNotice(""); }, []);
  const modelId = model?.id;
  const effectiveModel = enabled ? model : null;
  const { selection, removed } = useMemo(() => pruneAppearance(stored, effectiveModel), [stored, effectiveModel]);
  const removedKey = removed.join(", ");
  useEffect(() => setNotice(""), [modelId]);
  useEffect(() => {
    if (!modelId || !removedKey) return;
    setStored(selection);
    setNotice(`Cleared recipes unavailable for this model: ${removedKey.replaceAll(/(?:camera|film|look|lighting)-/g, "").replaceAll("-", " ")}.`);
  }, [modelId, removedKey, selection]);
  const resolved = useMemo(() => resolveAppearance(selection, effectiveModel), [selection, effectiveModel]);
  return {
    selection: model ? selection : emptyAppearance(),
    setSelection,
    resolved,
    active: hasAppearance(selection),
    available: Boolean(enabled && appearanceProfile(model)),
    model: effectiveModel,
    notice,
    warnings: appearanceWarnings(resolved, prompt),
  };
}
