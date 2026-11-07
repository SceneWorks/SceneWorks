import { useEffect, useMemo } from "react";
import {
  samplerDefaultFromModel, samplerOptionsFromModel,
  schedulerDefaultFromModel, schedulerOptionsFromModel,
  stepsDefaultFromModel, stepsMenuFromModel,
} from "../samplerOptions.js";
import { preferredOption } from "../studioDefaults.js";

// Existing surfaces intentionally reconcile at different times: Image/Video snap
// invalid choices, Character reseeds on a model change, the editor clamps on that
// change only, and a preset preserves its blank "no default" authoring value.
export function useSamplerControls(model, backend, {
  sampler, scheduler, setSampler, setScheduler, setSchedulerShift,
  reconcile = "invalid", emptyMenus = false, emptyValue = undefined,
} = {}) {
  const samplerOptions = useMemo(() => !model && emptyMenus ? [] : samplerOptionsFromModel(model, backend), [model, backend, emptyMenus]);
  const schedulerOptions = useMemo(() => !model && emptyMenus ? [] : schedulerOptionsFromModel(model, backend), [model, backend, emptyMenus]);
  const samplerDefault = preferredOption(samplerDefaultFromModel(model), samplerOptions, [], { empty: emptyValue });
  const schedulerDefault = preferredOption(schedulerDefaultFromModel(model), schedulerOptions, [], { empty: emptyValue });
  // Wait for catalog resolution so restored choices survive the loading window.
  useEffect(() => {
    if (!model || reconcile !== "invalid") return;
    if (!samplerOptions.includes(sampler)) setSampler(samplerDefault);
    if (!schedulerOptions.includes(scheduler)) setScheduler(schedulerDefault);
  }, [model, reconcile, samplerOptions, schedulerOptions, sampler, scheduler, setSampler, setScheduler, samplerDefault, schedulerDefault]);
  useEffect(() => {
    if (reconcile !== "model-reset" && reconcile !== "model-invalid") return;
    if (!model && reconcile === "model-invalid") return;
    if (reconcile === "model-reset" || !samplerOptions.includes(sampler)) {
      setSampler(samplerDefaultFromModel(model));
    }
    if (reconcile === "model-reset" || !schedulerOptions.includes(scheduler)) {
      setScheduler(schedulerDefaultFromModel(model));
      setSchedulerShift?.("");
    }
  // These two policies deliberately use the committed model-change snapshot,
  // preserving the owners' reset semantics rather than resetting on each edit.
  // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [model?.id]);
  return {
    samplerOptions, schedulerOptions, samplerDefault, schedulerDefault,
    showSamplerPicker: samplerOptions.length > 1,
    showSchedulerPicker: schedulerOptions.length > 1,
    stepsDefault: stepsDefaultFromModel(model),
    stepsMenu: stepsMenuFromModel(model),
  };
}
