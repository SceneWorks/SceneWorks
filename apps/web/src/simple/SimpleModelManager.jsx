import React, { useMemo, useState } from "react";
import { Icon } from "../components/Icons.jsx";
import { useAppContext } from "../context/AppContext.js";
import { terminalStatuses } from "../constants.js";
import { useUnifiedMemoryGb } from "../hooks/useUnifiedMemoryGb.js";
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

// The model's declared memory floor, when it has one (mlx.minMemoryGb — the blanket
// floor for its heaviest tier). Absent for models that declare none.
function needsLabel(model) {
  const floor = model?.mlx?.minMemoryGb;
  return Number.isFinite(floor) ? `needs ${floor} GB` : null;
}
