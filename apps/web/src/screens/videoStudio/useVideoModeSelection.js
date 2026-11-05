import { useEffect } from "react";
import { videoModelServesMode } from "../../modelEligibility.js";

// Extracted from VideoStudio along its existing concern boundary.
export function useVideoModeSelection({
  selectedModel,
  mode,
  macCapabilities,
  modelsForMode,
  model,
  setModel,
  videoModels
}) {
  // Models are gated on the selected tab (sc-5716): when the active mode isn't served by the current
  // model, snap to the first model that serves it so the user can always leave a mode. Generalizes
  // the old per-mode snaps (replace_person → first replace-capable model; animate_character →
  // scail2_14b) to every mode, including the Bernini editing/reference modes. A no-op when the
  // current model already serves the mode (e.g. an LTX image_to_video → text_to_video switch) or
  // when no model serves it (a reduced catalog) — there's nothing to snap to.
  useEffect(() => {
    if (videoModelServesMode(selectedModel, mode, macCapabilities)) {
      return;
    }
    const fallback = modelsForMode(mode)[0];
    if (fallback && fallback.id !== model) {
      setModel(fallback.id);
    }
    // videoModelServesMode / modelsForMode close over videoModels + macCapabilities, captured below.
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [mode, model, selectedModel, videoModels, macCapabilities]);
}
