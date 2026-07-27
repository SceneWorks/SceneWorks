import { useCallback, useEffect, useMemo, useState } from "react";
import {
  MAX_USER_JOB_LORAS,
  loraHasResolvableFamily,
  loraMatchesModel,
  loraWeight,
  serializeLora,
} from "../presetUtils.js";
import { terminalStatuses } from "../jobTypes.js";

// LoRA selection for the Simple studios (epic 15404).
//
// This deliberately MIRRORS `useGenerationStudio`'s LoRA bundle rather than inventing a
// second contract: same eligibility predicates, same per-user cap, same prune guard, same
// weight fallback. Simple shows fewer knobs — no "Show incompatible", no scope/family
// pickers — but a Simple run's `loras` array is the array the advanced studio would send
// for the same picks.

/** `scope · family` (or just the scope) — the same meta line `LoraPickerSection` renders. */
export function loraMeta(lora) {
  const scope = lora.scope ?? "global";
  return lora.family ? `${scope} · ${lora.family}` : scope;
}

/**
 * @param {object} input
 * @param {object[]} input.loras - the live LoRA catalog from `useAppContext()`.
 * @param {object|null} input.selectedModel - the studio's resolved catalog model.
 * @param {object[]} [input.jobs] - the live job feed, for pending `lora_import` runs.
 * @param {string|null} [input.excludeLoraId] - a LoRA the studio applies itself and must not
 *   also offer in the picker (Simple's edit mode auto-applies Krea's managed `image_edit`
 *   LoRA — the advanced studio hides it from its picker the same way).
 */
export function useSimpleLoras({ loras = [], selectedModel, jobs = [], excludeLoraId = null }) {
  const [selectedLoraIds, setSelectedLoraIds] = useState([]);
  const [weightOverrides, setWeightOverrides] = useState({});
  // Only the IDS of imports started from this studio's sheet. Status, progress and the
  // detected family are read LIVE off the job feed on every render — the pending slot must
  // not carry a second, drifting copy of what the Queue already knows.
  const [importJobIds, setImportJobIds] = useState([]);

  const compatibleLoras = useMemo(
    () =>
      loras.filter((lora) => {
        if (lora.presetManaged || lora.installState === "missing") {
          return false;
        }
        if (excludeLoraId && lora.id === excludeLoraId) {
          return false;
        }
        // Fails CLOSED on an unknown family (sc-10509): the API's generate-time gate 400s on
        // an empty family, so a family-less LoRA would be a dead-end selection. Simple has no
        // "Show incompatible" escape hatch, so this is unconditional here.
        if (!loraHasResolvableFamily(lora)) {
          return false;
        }
        return loraMatchesModel(lora, selectedModel);
      }),
    [loras, selectedModel, excludeLoraId],
  );
  const compatibleLoraKey = useMemo(() => compatibleLoras.map((lora) => lora.id).join("|"), [compatibleLoras]);

  const selectedLoras = useMemo(
    () => selectedLoraIds.map((id) => compatibleLoras.find((lora) => lora.id === id)).filter(Boolean),
    [selectedLoraIds, compatibleLoras],
  );
  const availableLoras = useMemo(
    () => compatibleLoras.filter((lora) => !selectedLoraIds.includes(lora.id)),
    [compatibleLoras, selectedLoraIds],
  );
  const userSelectedLoraCount = selectedLoras.filter((lora) => lora.scope !== "builtin").length;
  const atLimit = userSelectedLoraCount >= MAX_USER_JOB_LORAS;

  const hasPendingCompatibleLoras =
    Boolean(selectedModel) &&
    loras.some((lora) => lora.installState === "missing" && loraMatchesModel(lora, selectedModel));
  // Verbatim from `LoraPickerSection` minus its "Show incompatible" branch, which Simple
  // doesn't expose.
  const loraEmptyMessage = !selectedModel
    ? "No model selected"
    : hasPendingCompatibleLoras
      ? "No installed compatible LoRAs. Imports appear after the Queue completes."
      : `No installed LoRAs match ${selectedModel.name ?? selectedModel.id}.`;

  // Drop selections that fall out of the compatible set on a model change. Guarded on loaded
  // inputs exactly as sc-11962 guards the advanced prune: `compatibleLoras` is empty both when
  // the catalog hasn't resolved AND when the model hasn't, and running the filter during that
  // window would strip selections permanently.
  useEffect(() => {
    if (!loras.length || !selectedModel) {
      return;
    }
    setSelectedLoraIds((ids) => {
      const kept = ids.filter((id) => compatibleLoras.some((lora) => lora.id === id));
      return kept.length === ids.length ? ids : kept;
    });
  }, [loras.length, selectedModel, compatibleLoraKey, compatibleLoras]);

  const effectiveLoraWeight = useCallback(
    (lora) => {
      const override = weightOverrides[lora.id];
      return Number.isFinite(override) ? override : loraWeight(lora);
    },
    [weightOverrides],
  );

  const addLora = useCallback((lora) => {
    setSelectedLoraIds((ids) => (ids.includes(lora.id) ? ids : [...ids, lora.id]));
  }, []);

  const removeLora = useCallback((lora) => {
    setSelectedLoraIds((ids) => ids.filter((id) => id !== lora.id));
    // The override goes with it, so re-adding the LoRA starts from its own default again
    // rather than silently inheriting a weight the user set for a selection they removed.
    setWeightOverrides((current) => {
      if (!(lora.id in current)) {
        return current;
      }
      const next = { ...current };
      delete next[lora.id];
      return next;
    });
  }, []);

  const setLoraWeight = useCallback((id, value) => {
    setWeightOverrides((current) => ({ ...current, [id]: value }));
  }, []);

  const noteImportJob = useCallback((job) => {
    if (job?.id) {
      setImportJobIds((ids) => (ids.includes(job.id) ? ids : [...ids, job.id]));
    }
  }, []);

  // Resolved against the live feed each render, so a slot disappears the moment the import
  // reaches a terminal status — no local progress model, no polling.
  const pendingImports = useMemo(
    () =>
      importJobIds
        .map((id) => jobs.find((job) => job.id === id))
        .filter((job) => Boolean(job) && !terminalStatuses.has(job.status)),
    [importJobIds, jobs],
  );

  // The outgoing payload entries — the same `serializeLora` shape the advanced studios send.
  const serializedLoras = useMemo(
    () => selectedLoras.map((lora) => serializeLora(lora, { weight: effectiveLoraWeight(lora) })),
    [selectedLoras, effectiveLoraWeight],
  );

  return {
    selectedLoraIds,
    selectedLoras,
    compatibleLoras,
    availableLoras,
    userSelectedLoraCount,
    atLimit,
    loraEmptyMessage,
    effectiveLoraWeight,
    addLora,
    removeLora,
    setLoraWeight,
    noteImportJob,
    pendingImports,
    serializedLoras,
  };
}
