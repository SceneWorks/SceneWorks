import { isMossImage, mossEditDefaults } from "../../mossImage.js";
import { useEffect } from "react";

// Extracted from ImageStudio along its existing concern boundary.
export function useMossImageTransition({
  model,
  mossEditTransition,
  mossEdit,
  mossImage,
  resolution,
  stepsOverride,
  guidanceOverride,
  schedulerShift,
  negativePrompt,
  widthOverride,
  heightOverride,
  setResolution,
  setStepsOverride,
  setGuidanceOverride,
  setSchedulerShift,
  setWidthOverride,
  setHeightOverride,
  setMossImageEdit,
  setMossImage,
  setNegativePrompt
}) {
  useEffect(() => {
    const transition = mossEditTransition.current;
    if (transition.active === mossEdit) return;
    transition.active = mossEdit;
    if (transition.replay) { transition.replay = false; return; }
    if (mossEdit) {
      transition.ordinary = { mossImage, resolution, stepsOverride, guidanceOverride, schedulerShift, negativePrompt, widthOverride, heightOverride };
      setResolution("512x512"); setStepsOverride(30); setGuidanceOverride(4); setSchedulerShift(5);
      setWidthOverride(""); setHeightOverride("");
      setMossImageEdit(mossEditDefaults());
    } else if (transition.ordinary && isMossImage(model)) {
      const ordinary = transition.ordinary;
      setMossImage(ordinary.mossImage); setResolution(ordinary.resolution);
      setStepsOverride(ordinary.stepsOverride); setGuidanceOverride(ordinary.guidanceOverride);
      setSchedulerShift(ordinary.schedulerShift); setNegativePrompt(ordinary.negativePrompt);
      setWidthOverride(ordinary.widthOverride); setHeightOverride(ordinary.heightOverride);
      transition.ordinary = null;
    }
    // Only mode-boundary transitions seed or restore defaults. Editing knobs never resets them.
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [mossEdit]);
}
