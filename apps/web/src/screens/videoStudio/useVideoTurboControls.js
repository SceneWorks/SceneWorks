import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { defaultTurboVariant, modelIsMinimaxH3, selectedTurboVariant, turboVariantsForModel } from "../../minimaxH3Turbo.js";
import { loraModelForMode } from "../../presetUtils.js";

// Extracted from VideoStudio along its existing concern boundary.
export function useVideoTurboControls({
  saved,
  selectedModel,
  compatibleLoras,
  selectedLoraIds,
  model,
  mode,
  setSelectedLoraIds
}) {
  // A hybrid MiniMax-H3 import serves both partitions but a render runs one: the partition this
  // mode runs is the one whose turbo adapter is offered and defaulted (`loraModelForMode`).
  const loraModel = useMemo(() => loraModelForMode(selectedModel, mode), [selectedModel, mode]);
  // Which MiniMax-H3 models have already had the default-on turbo variant applied (sc-18727).
  //
  // Turbo IS a LoRA selection — the control and the generic LoRA picker write the same
  // `selectedLoraIds`, which is the only way the two entry points can't disagree (sc-18727's "route
  // both through one resolver"). That makes "default on" a ONE-SHOT seed rather than a default
  // value: without this marker, re-selecting the default every time the studio saw an empty
  // selection would silently undo a deliberate "Off", and undo it identically whether the user
  // turned it off in the turbo control or deselected the adapter in the picker.
  //
  // Persisted in the studio snapshot (mirrored to server ui-preferences, so it survives a desktop
  // relaunch — a localStorage-only marker would re-seed on every launch and re-defeat "Off").
  // Per-model because the two partitions take different adapters.
  const [turboSeededModels, setTurboSeededModels] = useState(saved.turboSeededModels ?? []);

  // ── MiniMax-H3 turbo (sc-18726 / sc-18727) ────────────────────────────────────────────────
  //
  // Every value below is derived from the LoRA catalog and the live selection; there is no separate
  // turbo state to keep in sync. `turboVariants` is already narrowed to installed + compatible by
  // `compatibleLoras`, so the control can only ever offer an adapter that would actually enqueue.
  const turboVariants = useMemo(
    () => turboVariantsForModel(loraModel, compatibleLoras),
    [loraModel, compatibleLoras],
  );

  const activeTurboVariant = selectedTurboVariant(turboVariants, selectedLoraIds);

  // Shown whenever the model is MiniMax-H3, even with nothing installed: an absent control would
  // answer "why is this render two and a half hours?" with silence. With no variant installed the
  // control renders the reason and the Model Manager pointer instead of an empty menu.
  const showTurbo = modelIsMinimaxH3(selectedModel);

  // Default-on seed. Runs once per model (see `turboSeededModels`) and only once the LoRA catalog
  // has actually resolved — seeding against an empty `turboVariants` during the restart-restore
  // window would mark the model seeded and leave turbo permanently off, the same sc-11962 trap the
  // preset/LoRA prunes above guard.
  useEffect(() => {
    if (!showTurbo || !turboVariants.length) return;
    if (turboSeededModels.includes(model)) return;
    setTurboSeededModels((seeded) => (seeded.includes(model) ? seeded : [...seeded, model]));
    // A variant already selected — a replayed recipe, a restored snapshot, a preset — is the
    // caller's choice and the seed must not stack a SECOND accelerator on top of it. Two adapters
    // asking for different schedules is refused by the worker, so seeding blind here would turn
    // "replay this render" into a hard enqueue failure.
    if (selectedTurboVariant(turboVariants, selectedLoraIds)) return;
    const preferred = defaultTurboVariant(loraModel, turboVariants);
    if (!preferred) return;
    setSelectedLoraIds((ids) => (ids.includes(preferred.id) ? ids : [...ids, preferred.id]));
  }, [
    showTurbo,
    turboVariants,
    turboSeededModels,
    model,
    loraModel,
    selectedLoraIds,
    setSelectedLoraIds,
  ]);

  // Partition switch on the SAME model (a hybrid import moving into or out of reference mode).
  // The shared prune drops the adapter the new partition cannot load, and the one-shot seed above
  // has already run for this model — so without this a user who had turbo on would silently fall
  // back to the 50-step base schedule. Turbo stays on: the old partition's adapter is replaced by
  // this partition's default. Off stays off — nothing is added unless a turbo adapter was selected.
  // The update drops every turbo id of either partition rather than one specific id, so it composes
  // with the prune's own queued update in either order.
  const previousTurbo = useRef({ modelId: selectedModel?.id, partitions: loraModel?.adapterPartitionModels, variantIds: [] });
  useEffect(() => {
    const previous = previousTurbo.current;
    const variantIds = turboVariants.map((variant) => variant.id);
    const partitions = loraModel?.adapterPartitionModels;
    previousTurbo.current = { modelId: selectedModel?.id, partitions, variantIds };
    if (previous.modelId !== selectedModel?.id || String(previous.partitions) === String(partitions)) return;
    const hadStaleTurbo = selectedLoraIds.some(
      (id) => previous.variantIds.includes(id) && !variantIds.includes(id),
    );
    if (!hadStaleTurbo) return;
    const preferred = defaultTurboVariant(loraModel, turboVariants);
    if (!preferred) return;
    const turboIds = new Set([...previous.variantIds, ...variantIds]);
    setSelectedLoraIds((ids) => [...ids.filter((id) => !turboIds.has(id)), preferred.id]);
  }, [selectedModel, loraModel, turboVariants, selectedLoraIds, setSelectedLoraIds]);

  // Selecting a variant REPLACES any other turbo adapter rather than stacking: two accelerators ask
  // for two different schedules and the worker refuses the pair by name
  // (`resolve_turbo_recipe`), so letting the control build that payload would offer a selection that
  // can only fail. Plain (non-accelerator) LoRAs are untouched — a style LoRA rides alongside turbo.
  const selectTurboVariant = useCallback(
    (id) => {
      const turboIds = new Set(turboVariants.map((variant) => variant.id));
      setSelectedLoraIds((ids) => {
        const withoutTurbo = ids.filter((existing) => !turboIds.has(existing));
        return id ? [...withoutTurbo, id] : withoutTurbo;
      });
    },
    [turboVariants, setSelectedLoraIds],
  );
  return { turboSeededModels, turboVariants, activeTurboVariant, showTurbo, selectTurboVariant };
}
