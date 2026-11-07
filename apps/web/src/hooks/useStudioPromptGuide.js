import { useCallback, useState } from "react";

export function useStudioPromptGuide(model, fallbackGuide) {
  const [guideOpen, setGuideOpen] = useState(false);
  const openGuide = useCallback(() => setGuideOpen(true), []);
  const closeGuide = useCallback(() => setGuideOpen(false), []);
  // This is render-derived: an open guide follows the current render's model.
  const promptGuide = model?.ui?.promptGuide ?? fallbackGuide;
  return { guideOpen, openGuide, closeGuide, promptGuide };
}
