import { useConsentRequestOwner } from "../hooks/useConsentRequestOwner.js";
import { downloadSizeText } from "../formatting.js";
import React, { useMemo, useState } from "react";
import { CheckpointImportPanel } from "../components/CheckpointImportPanel.jsx";
import { Icon } from "../components/Icons.jsx";
import { ModelLicenseSummary } from "../components/ModelLicenseSummary.jsx";
import {
  hasImportedCheckpointSurface,
  importedSourceSummary,
} from "../components/ImportedCheckpointSurface.jsx";
import { licenseComponentForModel } from "../data/bundledLicenses.js";
import { useAppContext } from "../context/AppContext.js";
import { terminalStatuses } from "../constants.js";
import { useHostMemory } from "../hooks/useHostMemory.js";
import { hostMemoryGbForBackend } from "../hostMemory.js";
import {
  borrowedLicenseAcknowledgmentBlocked,
  licenseAcknowledgmentBlocked,
} from "../licenseAcknowledgment.js";
import { modelLoraFamilies } from "../presetUtils.js";
import {
  endpointEstimate,
  installedTiers,
  memoryLane,
  nativeTierEligibility,
  tierHostEligible,
} from "../quantTier.js";
import { useSimpleUi } from "./SimpleUiContext.js";

// Simple Model Manager (design handoff): a tabbed catalog (Image / Video / Utility /
// LoRAs) of one-line rows — glyph, name, "size · needs N GB", and a right-aligned
// Manage (installed) or Download (missing) action.
//
// Adding a model is the ONE exception to that handoff (epic 20398, sc-20650). Choosing between
// "Use existing model library" and "Add to SceneWorks" is the first decision a new user makes about
// their models, and a Simple shell that could only send them to the advanced screen for it would
// make the reduced shell unusable for exactly the person it exists for. It renders the SAME panel
// the advanced screen does — one experience, not a reduced second copy — collapsed by default so
// the catalog stays the page's subject.
//
// Everything else: "Manage" hands off to the full Models screen (openInAdvanced — which switches the SHELL
// as well as the view, since pointing the workspace at a screen while Simple is rendering
// would be inert): per-tier downloads, conversion, repair, import and delete are exactly
// the surface Simple is meant to hide, and duplicating a reduced version of them here would
// be a second place to get destructive actions wrong. Download enqueues the model's default
// tier through the same createModelDownloadJob the advanced screen uses.

const EMPTY_CATALOG = [];
const TABS = [
  { id: "image", label: "Image" },
  { id: "video", label: "Video" },
  { id: "utility", label: "Utility" },
  { id: "loras", label: "LoRAs" },
];

export function SimpleModelManager() {
  const {
    models = EMPTY_CATALOG,
    loras = EMPTY_CATALOG,
    jobs = [],
    token = "",
    createModelDownloadJob,
    createLoraDownloadJob,
    createModelImportJob,
    jobAction,
    macCapabilities,
    refreshData,
    visibleWorkers = [],
  } = useAppContext();
  const { toast, openInAdvanced } = useSimpleUi();
  const hostMemory = useHostMemory();
  const [tab, setTab] = useState("image");
  const captureConsentOwner = useConsentRequestOwner(token, models, loras, tab);
  // Which lane's memory evidence the rows quote. Same derivation as ImageStudio / SimpleImageStudio —
  // the memory numbers are per-backend and must not be crossed (see `needsLabel`).
  const backend = macCapabilities?.macGatingActive ? "mlx" : "candle";
  const hostMemoryGb = hostMemoryGbForBackend(hostMemory, backend);
  const tierOptions = useMemo(() => ({ backend, hostMemoryGb }), [backend, hostMemoryGb]);

  const rows = useMemo(() => {
    if (tab === "loras") {
      return loras.map((lora) => ({
        id: lora.id,
        name: lora.name ?? lora.id,
        meta: [downloadSizeText(lora), lora.baseModel ?? lora.family].filter(Boolean).join(" · "),
        manageOnly: lora.installState === "installed",
        entry: lora,
        kind: "lora",
      }));
    }
    // Audio models ride the Utility tab alongside upscalers/refiners: the design's four
    // tabs predate the Audio Studio landing, and a model the user can't find is worse
    // than one on a slightly broad tab.
    const wanted = tab === "utility" ? (type) => type !== "image" && type !== "video" : (type) => type === tab;
    return models
      .filter((model) => wanted(model.type))
      .map((model) => {
        // Match the generation picker's per-model native format contract. In particular,
        // generic nvfp4 workers cannot supply FLUX.3 packed-provider measurements.
        const options = { ...tierOptions, ...nativeTierEligibility(model, visibleWorkers) };
        const needs = needsLabel(model, options);
        const endpoint = memoryLane(model, options.backend)?.endpointEstimate;
        return {
          id: model.id,
          name: model.name ?? model.id,
          meta: [downloadSizeText(model), needs, importedCheckpointLabel(model)]
            .filter(Boolean)
            .join(" · "),
          memoryTitle: endpoint && needs?.startsWith(endpoint.label) ? endpoint.description : undefined,
          manageOnly: model.installState === "installed" || model.platformCleanupOnly === true,
          entry: model,
          kind: "model",
        };
      });
  }, [tab, models, loras, tierOptions, visibleWorkers]);

  const activeDownloads = useMemo(
    () =>
      new Set(
        jobs
          .filter(
            (job) =>
              (job.type === "model_download" || job.type === "lora_download") &&
              !terminalStatuses.has(job.status),
          )
          .map((job) => job.payload?.modelId ?? job.payload?.loraId)
          .filter(Boolean),
      ),
    [jobs],
  );

  async function download(row) {
    const ownsRequest = captureConsentOwner(token);
    if (!ownsRequest()) return;
    const enqueue = row.kind === "lora" ? createLoraDownloadJob : createModelDownloadJob;
    if (typeof enqueue !== "function") {
      return;
    }
    // Licence gate (sc-17227). Simple renders no licence UI — by design, it is the reduced shell —
    // so a model whose licence must be accepted cannot be accepted HERE. `createModelDownloadJob`
    // refuses it either way; hand off to the Models screen, which owns the notice and the
    // checkbox, instead of toasting a refusal the user has no way to act on.
    //
    // A LoRA row takes the same handoff, on the BORROWED acknowledgment (sc-17227). Its gating
    // model is stamped onto the row by the server (`annotate_license_acknowledgment_sources`) and
    // the ack lives in the same store, so accepting once on that model's card clears the LoRA too.
    // Without this branch the refusal is a dead end HERE specifically: `createLoraDownloadJob`'s
    // own gate returns null and writes the real reason to `loraError`, which Simple renders
    // nowhere, so the user saw only "Could not start the … download" with no way to act on it.
    try {
      const blocked = row.kind === "lora"
        ? await borrowedLicenseAcknowledgmentBlocked(row.entry, token)
        : await licenseAcknowledgmentBlocked(row.entry, token);
      if (!ownsRequest()) return;
      if (blocked) {
        toast(`${row.name} needs its license accepted first — opening Models.`);
        openInAdvanced("Models");
        return;
      }
    } catch (error) {
      if (!ownsRequest()) return;
      toast(`Could not check license acknowledgment: ${error.message}`);
      return;
    }
    const job = await enqueue(row.entry);
    if (!ownsRequest()) return;
    toast(job ? `Downloading ${row.name}…` : `Could not start the ${row.name} download`);
  }

  const modelImportJobs = useMemo(() => jobs.filter((job) => job.type === "model_import"), [jobs]);

  // The SAME family list the advanced screen's import form offers, derived the same way: from each
  // model's LoRA-compatibility set rather than its `family` identity, because that is what the
  // import validator keys off. Without it Simple's Family select was permanently disabled — a
  // reduced copy of the panel, which is exactly what mounting the shared component is meant to
  // prevent.
  const families = useMemo(
    () => Array.from(new Set(models.flatMap((model) => modelLoraFamilies(model)).filter(Boolean))).sort(),
    [models],
  );

  return (
    <div className="su-screen su-screen--tight">
      <CheckpointImportPanel
        compact
        completedJobs={modelImportJobs.filter((job) => job.status === "completed")}
        families={families}
        headingId="simple-checkpoint-import-heading"
        macCapabilities={macCapabilities}
        models={models}
        onCancelJob={jobAction ? (job) => jobAction(job, "cancel") : undefined}
        onImportModel={createModelImportJob}
        onOpenQueue={() => openInAdvanced("Queue")}
        onOpenSettings={() => openInAdvanced("Settings")}
        // Simple renders no linked-status badge of its own, but its rows ARE the catalog: a relink
        // or forget inside the panel changes which models exist and whether they are usable, so the
        // catalog has to be re-read rather than left showing the pre-relink answer.
        onRefreshCatalog={refreshData}
        onRetryJob={jobAction ? (job, payload) => jobAction(job, "retry", { body: payload ?? {} }) : undefined}
        pendingJobs={modelImportJobs.filter((job) => !terminalStatuses.has(job.status))}
        token={token}
      />
      <div className="su-tabs su-scroll" role="tablist">
        {TABS.map((entry) => (
          <button
            aria-selected={tab === entry.id}
            className={tab === entry.id ? "su-tab active" : "su-tab"}
            key={entry.id}
            onClick={() => setTab(entry.id)}
            role="tab"
            type="button"
          >
            {entry.label}
          </button>
        ))}
      </div>

      {rows.length ? (
        rows.map((row) => {
          const downloading = activeDownloads.has(row.id);
          // sc-24108: Simple takes no acknowledgment — `download()` hands a gated model off to the
          // Models screen — but "renders no licence UI" was never meant to mean the RESTRICTIONS
          // are invisible here. A research-only model listed beside Apache-2.0 ones with nothing
          // to tell them apart is the gap this closes. Read-only: name, link, collapsed notice.
          const licenseTerms =
            row.kind === "model" && (row.entry?.licenseNotice || row.entry?.licenseUrl) ? (
              <ModelLicenseSummary
                className="su-license-note"
                licenseName={licenseComponentForModel(row.id)?.license}
                licenseNotice={row.entry.licenseNotice}
                licenseUrl={row.entry.licenseUrl}
                nonCommercial={row.entry.nonCommercial === true}
              />
            ) : null;
          return (
            <React.Fragment key={row.id}>
            <div className="su-row">
              <span aria-hidden="true" className="su-row-glyph">
                <Icon.Model size={20} />
              </span>
              <span className="su-row-text">
                <span className="su-row-title">{row.name}</span>
                <span className="su-row-meta" title={row.memoryTitle}>{row.meta || "—"}</span>
              </span>
              {row.manageOnly ? (
                <button
                  className="su-row-action"
                  onClick={() => openInAdvanced("Models")}
                  title="Opens the full Models screen (tiers, convert, repair, delete)"
                  type="button"
                >
                  Manage
                </button>
              ) : (
                <button
                  className="su-row-action primary"
                  disabled={downloading}
                  onClick={() => download(row)}
                  type="button"
                >
                  {downloading ? "Downloading…" : "Download"}
                </button>
              )}
            </div>
            {licenseTerms}
            </React.Fragment>
          );
        })
      ) : (
        <p className="su-empty">No {TABS.find((entry) => entry.id === tab)?.label.toLowerCase()} entries in the catalog.</p>
      )}

      {hostMemory ? (
        <p className="su-empty">
          This machine reports {hostMemory.gb.toFixed(0)} GB of {hostMemory.kind} memory.
        </p>
      ) : null}
    </div>
  );
}

// The imported-checkpoint line in a row's meta (sc-26687): which source variant was imported and
// how much of its component assembly is present here. Simple deliberately keeps to one-line rows
// and hands repair to the advanced Models screen through "Manage", so this states the state and the
// full card offers the fix — the same split the memory-number comment above describes.
function importedCheckpointLabel(model) {
  if (!hasImportedCheckpointSurface(model)) {
    return null;
  }
  const summary = importedSourceSummary(model);
  const components = model?.componentReadiness?.components;
  if (!Array.isArray(components) || components.length === 0) {
    return summary || null;
  }
  const ready = components.filter((component) => component.present === true).length;
  return `${summary ? `${summary} · ` : ""}${ready}/${components.length} components ready`;
}

// The catalog's own human-readable download size; `~` marks an estimate, exactly as the
// advanced Models screen labels it.

// The model's memory requirement on the ACTIVE lane, PREFERRING the per-tier measured peak over the
// blanket per-lane `minMemoryGb` (sc-15400): showing the blanket unconditionally over-warns every
// variant-matrix model whose user only wants a lighter tier. The advanced Models screen defers to its
// per-tier download panel; Simple has no such panel, so it shows the number for the tiers in play.
//
// Every number is API-served (`memoryGuidance.<lane>`, `sceneworks_core::memory_guidance`): each tier
// row already carries its host size in its own lane's terms (MLX headroom fraction vs Candle allocator
// slack), its lane, whether it is that tier's own evidence or an estimate/proxy, and its declared
// floor. This function only selects the rows of the tiers this install has and composes wording:
//   1. Tiers installed ⇒ `needs N GB` over exactly those tiers. With every installed tier carrying
//      its own calibrated evidence, the max of their host sizes. Otherwise the blanket is a FLOOR
//      (the default tier's peak plus headroom can sit below a heavier installed tier): the max of the
//      host sizes and the tiers' floors, or nothing when an installed tier has no number and no floor.
//   2. Not installed, and some eligible declared tier has its own measured number ⇒ `from N GB`, the
//      ENTRY cost (the client cannot know which tier Download will fetch). An uncalibrated lane
//      floors it at the entry floor.
//   3. Otherwise the lane's entry floor (the lightest installable tier's declared floor).
// EXPORTED for the catalog-driven guidance tests.
export function needsLabel(model, tierOptions) {
  const lane = memoryLane(model, tierOptions?.backend);
  if (lane === null) {
    return null;
  }
  const endpoint = lane.endpointEstimate
    ? endpointEstimate(model, lane.endpointEstimate.tier, tierOptions?.backend)
    : null;
  const installedVariants = model?.variants?.filter((variant) => variant.installState === "installed") ?? [];
  if (endpoint && model.variants?.some((variant) => variant.variant === endpoint.tier)
      && ((installedVariants.length === 0 && model.installState !== "installed")
        || (installedVariants.length === 1 && installedVariants[0].variant === endpoint.tier))) {
    return endpoint.label + (Number.isFinite(tierOptions?.hostMemoryGb)
      && tierOptions.hostMemoryGb >= endpoint.optimizedHostGb ? " · can fit with optimizations" : "");
  }
  const installed = installedFloorHostGb(model, lane, tierOptions);
  if (installed !== null) {
    return `needs ${installed} GB`;
  }
  if (model?.installState !== "installed") {
    const entry = declaredFloorHostGb(lane, tierOptions);
    if (entry !== null) {
      return `from ${entry} GB`;
    }
  }
  const floor = typeof lane.entryFloorGb === "number" ? lane.entryFloorGb : null;
  return floor === null ? null : `needs ${floor} GB`;
}

const servedNumber = (value) => (typeof value === "number" ? value : null);
const maxOf = (a, b) => (a === null ? b : b === null ? a : Math.max(a, b));

// Host GB over the INSTALLED tiers (host-eligible per `tierOptions`), or null.
function installedFloorHostGb(model, lane, tierOptions) {
  const tiers = installedTiers(model, tierOptions);
  if (tiers.length === 0) {
    return null;
  }
  let ceiling = null;
  let floor = null;
  let complete = true;
  let estimated = false;
  for (const tier of tiers) {
    const row = lane.tiers?.[tier];
    const hostGb = servedNumber(row?.installedHostGb);
    if (hostGb === null) {
      complete = false;
    } else {
      ceiling = maxOf(ceiling, hostGb);
      estimated = estimated || row.installedEstimated === true;
    }
    floor = maxOf(floor, servedNumber(row?.floorGb) ?? servedNumber(lane.blanketGb));
  }
  if (complete && !estimated && lane.uncalibrated !== true) {
    return ceiling;
  }
  if (!complete && floor === null) {
    return null;
  }
  return maxOf(ceiling, floor);
}

// The ENTRY host GB for a model that is not installed: the cheapest host-eligible tier's own measured
// number, floored at the entry floor on an uncalibrated lane. Null when no tier is measured.
function declaredFloorHostGb(lane, tierOptions) {
  let cheapest = null;
  for (const [tier, row] of Object.entries(lane.tiers ?? {})) {
    const hostGb = servedNumber(row?.measuredHostGb);
    if (hostGb !== null && tierHostEligible(tier, tierOptions) && (cheapest === null || hostGb < cheapest)) {
      cheapest = hostGb;
    }
  }
  if (cheapest === null || lane.uncalibrated !== true) {
    return cheapest;
  }
  return maxOf(cheapest, servedNumber(lane.entryFloorGb));
}
