import React, { useMemo, useState } from "react";
import { Icon } from "../components/Icons.jsx";
import { useAppContext } from "../context/AppContext.js";
import { terminalStatuses } from "../constants.js";
import { useUnifiedMemoryGb } from "../hooks/useUnifiedMemoryGb.js";
import { cheapestDeclaredTierPeakGb, installedTierPeakGb } from "../tierSuggestion.js";
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
  const { models = [], loras = [], jobs = [], createModelDownloadJob, createLoraDownloadJob } =
    useAppContext();
  const { toast, openInAdvanced } = useSimpleUi();
  const unifiedMemoryGb = useUnifiedMemoryGb();
  const [tab, setTab] = useState("image");

  const rows = useMemo(() => {
    if (tab === "loras") {
      return loras.map((lora) => ({
        id: lora.id,
        name: lora.name ?? lora.id,
        meta: [sizeLabel(lora), lora.baseModel ?? lora.family].filter(Boolean).join(" · "),
        installed: lora.installState === "installed",
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
        meta: [sizeLabel(model), needsLabel(model)].filter(Boolean).join(" · "),
        installed: model.installState === "installed",
        entry: model,
        kind: "model",
      }));
  }, [tab, models, loras]);

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
              {row.installed ? (
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

      {unifiedMemoryGb ? (
        <p className="su-empty">This machine reports {unifiedMemoryGb.toFixed(0)} GB of usable memory.</p>
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

// The model's memory requirement, PREFERRING the per-tier measured footprint over the blanket
// `mlx.minMemoryGb` (sc-15400). That blanket integer is a single per-model value, so it must admit the
// model's heaviest INSTALLABLE tier — showing it unconditionally over-warns every variant-matrix model
// whose user only wants a lighter tier (z_image declares 48 while its measured q4 tier peaks at 19.42
// GiB; krea_realtime_14b declares 64 against a 27.90 GiB q4). The advanced Models screen dodges this by
// SUPPRESSING the badge for matrix models and deferring to its per-tier download panel
// (ModelManagerScreen's `hasTierMatrix`); Simple has no such panel — "Manage" hands off to that screen —
// so instead of hiding the number it shows the one that applies to the tier actually in play.
//
// Three cases, in order:
//   1. Measured ceiling over the INSTALLED tiers ⇒ `needs N GB`. Exact: this is what is on disk.
//   2. Not installed, and some declared tier is measured ⇒ `from N GB`. The client cannot know which
//      tier Download will fetch (the catalog omits the manifest `default` flag, and 3 of the 53 matrix
//      models default to a heavier tier than the lightest), so the honest statement is the ENTRY cost —
//      which is also what a "can I use this model at all" decision needs. Deliberately worded as a
//      floor, never as a specific tier's requirement.
//   3. No per-tier measurement ⇒ the blanket floor, exactly as before. Covers every single-variant
//      model and every matrix model epic 15448 has not measured yet.
function needsLabel(model) {
  const installedPeak = installedTierPeakGb(model);
  if (installedPeak !== null) {
    return `needs ${Math.ceil(installedPeak)} GB`;
  }
  if (model?.installState !== "installed") {
    const cheapest = cheapestDeclaredTierPeakGb(model);
    if (cheapest !== null) {
      return `from ${Math.ceil(cheapest)} GB`;
    }
  }
  const floor = model?.mlx?.minMemoryGb;
  return Number.isFinite(floor) ? `needs ${floor} GB` : null;
}
