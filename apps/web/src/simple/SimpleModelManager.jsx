import React, { useMemo, useState } from "react";
import { Icon } from "../components/Icons.jsx";
import { useAppContext } from "../context/AppContext.js";
import { terminalStatuses } from "../constants.js";
import { useHostMemory } from "../hooks/useHostMemory.js";
import { blanketFloorGb, declaredFloorHostGb, installedFloorHostGb } from "../tierSuggestion.js";
import { workerAdvertises } from "./simpleJobs.js";
import { useSimpleUi } from "./SimpleUiContext.js";

// Simple Model Manager (design handoff): a tabbed catalog (Image / Video / Utility /
// LoRAs) of one-line rows — glyph, name, "size · needs N GB", and a right-aligned
// Manage (installed) or Download (missing) action.
//
// "Manage" hands off to the full Models screen (openInAdvanced — which switches the SHELL
// as well as the view, since pointing the workspace at a screen while Simple is rendering
// would be inert): per-tier downloads, conversion, repair, import and delete are exactly
// the surface Simple is meant to hide, and duplicating a reduced version of them here would
// be a second place to get destructive actions wrong. Download enqueues the model's default
// tier through the same createModelDownloadJob the advanced screen uses.

const TABS = [
  { id: "image", label: "Image" },
  { id: "video", label: "Video" },
  { id: "utility", label: "Utility" },
  { id: "loras", label: "LoRAs" },
];

export function SimpleModelManager() {
  const {
    models = [],
    loras = [],
    jobs = [],
    createModelDownloadJob,
    createLoraDownloadJob,
    macCapabilities,
    visibleWorkers = [],
  } = useAppContext();
  const { toast, openInAdvanced } = useSimpleUi();
  const hostMemory = useHostMemory();
  const [tab, setTab] = useState("image");
  // Which lane's memory evidence the rows quote. Same derivation as ImageStudio / SimpleImageStudio —
  // the memory numbers are per-backend and must not be crossed (see `needsLabel`).
  const backend = macCapabilities?.macGatingActive ? "mlx" : "candle";
  // The candle-only tiers' host-eligibility gates, derived exactly as SimpleImageStudio derives them for
  // tier RESOLUTION (sc-9300 / sc-11042). The memory floor has to agree with the picker: a host where no
  // live worker advertises `int8_convrot` will never be offered that tier, so its measured row must not
  // set this row's number. Defaulting them to true — as `installedTiers` does for callers that pass
  // nothing — would size an ineligible host against a tier it cannot run.
  const tierOptions = useMemo(
    () => ({
      backend,
      convRotEligible: workerAdvertises(visibleWorkers, "int8_convrot"),
      nvfp4Eligible: workerAdvertises(visibleWorkers, "nvfp4"),
    }),
    [backend, visibleWorkers],
  );

  const rows = useMemo(() => {
    if (tab === "loras") {
      return loras.map((lora) => ({
        id: lora.id,
        name: lora.name ?? lora.id,
        meta: [sizeLabel(lora), lora.baseModel ?? lora.family].filter(Boolean).join(" · "),
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
      .map((model) => ({
        id: model.id,
        name: model.name ?? model.id,
        meta: [sizeLabel(model), needsLabel(model, tierOptions)].filter(Boolean).join(" · "),
        manageOnly: model.installState === "installed" || model.platformCleanupOnly === true,
        entry: model,
        kind: "model",
      }));
  }, [tab, models, loras, tierOptions]);

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
    const enqueue = row.kind === "lora" ? createLoraDownloadJob : createModelDownloadJob;
    if (typeof enqueue !== "function") {
      return;
    }
    const job = await enqueue(row.entry);
    toast(job ? `Downloading ${row.name}…` : `Could not start the ${row.name} download`);
  }

  return (
    <div className="su-screen su-screen--tight">
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
          return (
            <div className="su-row" key={row.id}>
              <span aria-hidden="true" className="su-row-glyph">
                <Icon.Model size={20} />
              </span>
              <span className="su-row-text">
                <span className="su-row-title">{row.name}</span>
                <span className="su-row-meta">{row.meta || "—"}</span>
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

// The catalog's own human-readable download size; `~` marks an estimate, exactly as the
// advanced Models screen labels it.
function sizeLabel(entry) {
  if (!entry?.downloadSizeLabel) {
    return null;
  }
  return entry.downloadSizeEstimated ? `~${entry.downloadSizeLabel}` : entry.downloadSizeLabel;
}

// The model's memory requirement, PREFERRING the per-tier measured peak over the blanket per-lane
// `minMemoryGb` (sc-15400). Showing the blanket unconditionally over-warns every variant-matrix model
// whose user only wants a lighter tier (z_image declares 48 while its measured q4 tier peaks at 19.42
// GiB; krea_realtime_14b declares 64 against a 27.90 GiB q4). The advanced Models screen dodges this by
// SUPPRESSING the badge for matrix models and deferring to its per-tier download panel
// (ModelManagerScreen's `hasTierMatrix`); Simple has no such panel — "Manage" hands off to that screen —
// so instead of hiding the number it shows the one that applies to the tier actually in play.
//
// THREE THINGS THE NUMBER MUST GET RIGHT, all one-directional (getting them wrong UNDER-states a
// requirement, and the user downloads a model that then OOMs). The arithmetic for all three lives in
// `tierSuggestion.installedFloorHostGb` / `declaredFloorHostGb`; this function composes wording only.
//
// A. HEADROOM, PER LANE. A declared peak is a RAW high-water mark, and the two lanes convert it to a host
//    size differently: MLX budgets `peak <= host × MEMORY_HEADROOM_FRACTION` (`tierFits`, the resolution
//    gate), while candle's own gate admits at `peak + 2` (`vram_gate.rs` `HEADROOM_GB`). Quoting the raw
//    peak under-states on both; quoting the MLX form on candle OVER-states against the gate that actually
//    rejects (`flux2_dev` bf16: 143 vs the 130 the gate requires).
//
// B. LANE. `footprint.peakMemoryBytes` is an Apple unified-memory measurement and `candle.vramGbByTier`
//    is a discrete-VRAM one; they are not interchangeable in either direction (see the lane note in
//    tierSuggestion.js). So the row reads the ACTIVE backend's evidence only. When the candle lane has
//    none, the row says nothing rather than borrowing the MLX figure — `qwen_image` declares mlx 50
//    against candle 56, so the MLX number is not even reliably the conservative one.
//
// C. THE BLANKET IS A FLOOR, NOT A CEILING. `minMemoryGb` is the DEFAULT (lightest) tier's peak plus
//    headroom — the schema and `vram_gate.rs:180` both say so — so it can sit BELOW a heavier installed
//    tier's requirement, and falling back to it alone under-stated four real install-sets (`flux_dev` /
//    `flux_schnell` read 24 against a measured q8 of 31.8; `krea_2_turbo` read 32 against a measured
//    bf16 of 47.2). Whenever the per-tier evidence is incomplete or the lane flags it estimated, the
//    displayed number is the MAX of the two bases, never the blanket alone.
//
// Three cases, in order:
//   1. The model has tiers installed ⇒ `needs N GB`, over exactly those tiers (case 1/2/3 of
//      `installedFloorHostGb`). This is what is on disk, so it is a statement about this install.
//   2. Not installed, and some declared tier has a number ⇒ `from N GB`. The client cannot know which
//      tier Download will fetch (the catalog omits the manifest `default` flag, and 3 of the 53 matrix
//      models default to a heavier tier than the lightest), so the honest statement is the ENTRY cost —
//      which is also what a "can I use this model at all" decision needs. Deliberately worded as a
//      floor, never as a specific tier's requirement.
//   3. No per-tier evidence at all ⇒ this lane's blanket floor, exactly as before. Covers every
//      single-variant model and every matrix model epic 15448 has not measured yet.
//
// EXPORTED for memoryFloorCatalogParity.test.js, which drives THIS function against every
// (model, OS, installed-subset) the real manifest can produce. That test exists because the previous
// round's suite asserted the helpers individually and every composed-label case used a hand-written
// literal — so a defect that lived only in how the three cases compose (the blanket fallback) had no
// test that could see it.
export function needsLabel(model, tierOptions) {
  const installed = installedFloorHostGb(model, tierOptions);
  if (installed !== null) {
    return `needs ${installed} GB`;
  }
  if (model?.installState !== "installed") {
    const entry = declaredFloorHostGb(model, tierOptions);
    if (entry !== null) {
      return `from ${entry} GB`;
    }
  }
  const floor = blanketFloorGb(model, tierOptions?.backend);
  return floor === null ? null : `needs ${floor} GB`;
}
