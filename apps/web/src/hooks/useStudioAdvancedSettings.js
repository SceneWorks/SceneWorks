import { useCallback, useState } from "react";

// Shared sticky authoring state. Defaults are explicit lane inputs: MOSS image
// edit starts at shift 5, steps 30, guidance 4; ordinary Image leaves them unset;
// Video starts at shift 3. Model/backend reconciliation stays with each owner.
export function useStudioAdvancedSettings(saved, { schedulerShift: shift = null, steps = "", guidanceScale = "" } = {}) {
  const [advancedOpen, setAdvancedOpen] = useState(saved.advancedOpen ?? false);
  const [seed, setSeed] = useState(saved.seed ?? "");
  const [negativePrompt, setNegativePrompt] = useState(saved.negativePrompt ?? "");
  const [sampler, setSampler] = useState(saved.sampler ?? "default");
  const [scheduler, setScheduler] = useState(saved.scheduler ?? "default");
  const [schedulerShift, setSchedulerShift] = useState(saved.schedulerShift ?? shift);
  const [stepsOverride, setStepsOverride] = useState(saved.steps ?? steps);
  const [guidanceOverride, setGuidanceOverride] = useState(saved.guidanceScale ?? guidanceScale);
  const toggleAdvanced = useCallback(() => setAdvancedOpen((value) => !value), []);
  return { advancedOpen, setAdvancedOpen, toggleAdvanced, seed, setSeed, negativePrompt, setNegativePrompt,
    sampler, setSampler, scheduler, setScheduler, schedulerShift, setSchedulerShift,
    stepsOverride, setStepsOverride, guidanceOverride, setGuidanceOverride };
}
