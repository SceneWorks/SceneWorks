import React, { useCallback, useEffect, useMemo, useState } from "react";
import { WorkerProgressCard } from "../components/WorkerProgressCard.jsx";
import { WorkPanel } from "../components/WorkPanel.jsx";
import { WAN_MOE_PAIRED_LORA_MODEL_IDS, terminalStatuses } from "../constants.js";
import { hasPresentCredential, loadCredentials } from "../credentials.js";
import {
  extractFamilies,
  loraHasResolvableFamily,
  modelLoraFamilies,
  normalizeLoraFamily,
  presetLoraId,
  presetLoras,
} from "../presetUtils.js";
import { useAppContext } from "../context/AppContext.js";
import { DEFAULT_MAC_CAPABILITIES, macModelBlock } from "../macGating.js";
import { appConfirm } from "../appConfirm.jsx";
import { formatBytes } from "../formatting.js";
import {
  availabilityBadge,
  canHoldLocalCopy,
  describeRemovalPreview,
  entriesForModel,
  fetchModelCache,
  previewCacheRemoval,
  removalIsAllowed,
  removeCacheEntry,
  setCacheEntryPin,
} from "../modelCache.js";
import { KeywordTagEditor } from "../components/KeywordTagEditor.jsx";
import { useHostMemory } from "../hooks/useHostMemory.js";
import { hostMemoryGbForBackend } from "../hostMemory.js";
import { tierLabel } from "../quantTier.js";
import { blanketFloorGb, suggestTier, tierFits } from "../tierSuggestion.js";
import { safeExternalUrl } from "../urls.js";

function matchesFamily(item, familyFilter) {
  if (familyFilter === "all") {
    return true;
  }
  // Accept either a LoRA catalog entry or a lora_import job snapshot (whose
  // family metadata lives under payload.manifestEntry).
  const families = extractFamilies(item, { includeManifest: true });
  // Import jobs can briefly lack family metadata; completed catalog entries should not.
  return item.type === "lora_import" && families.length === 0 ? true : families.includes(familyFilter);
}

function loraImportKey(job) {
  return job.payload?.loraId ?? job.payload?.sourceUrl ?? job.payload?.sourcePath ?? job.payload?.name ?? null;
}

function completedLoraImportTimes(jobs) {
  const completed = new Map();
  jobs
    .filter((job) => job.type === "lora_import" && job.status === "completed")
    .forEach((job) => {
      const key = loraImportKey(job);
      if (!key || !job.createdAt) {
        return;
      }
      const previous = completed.get(key);
      if (!previous || job.createdAt.localeCompare(previous) > 0) {
        completed.set(key, job.createdAt);
      }
    });
  return completed;
}

function isSupersededLoraImport(job, completedTimes) {
  const key = loraImportKey(job);
  const completedAt = key ? completedTimes.get(key) : null;
  return Boolean(completedAt) && terminalStatuses.has(job.status) && job.status !== "completed" && completedAt.localeCompare(job.createdAt ?? "") > 0;
}

function downloadSizeText(model) {
  if (!model.downloadSizeLabel) {
    return "Unavailable";
  }
  return model.downloadSizeEstimated ? `~${model.downloadSizeLabel}` : model.downloadSizeLabel;
}

// Human-readable size for a per-tier byte count (sc-8509). The catalog gives per-variant sizes as
// raw `downloadSizeBytes` numbers (unlike the model-level `downloadSizeLabel` string), so the tier
// panel formats them here. Binary units (GiB-based) to match the model-level label's convention.
function formatTierSize(bytes) {
  if (typeof bytes !== "number" || !Number.isFinite(bytes) || bytes <= 0) {
    return "Size unavailable";
  }
  const gb = bytes / (1024 * 1024 * 1024);
  if (gb >= 1) {
    return `${gb.toFixed(1)} GB`;
  }
  const mb = bytes / (1024 * 1024);
  return `${mb.toFixed(0)} MB`;
}

// MLX status text, keyed off the macOS catalog's mlxConversionState. Turnkey
// ("ready") models fetch their MLX weights automatically on first generation;
// convert-required models need the native checkpoint downloaded, then converted.
function mlxStatusText(model) {
  switch (model.mlxConversionState) {
    case "ready":
      return model.mlxInstallState === "installed"
        ? "MLX weights installed."
        : "MLX weights download automatically on first generation.";
    case "needs_source":
      return "Download the model first, then convert it to MLX.";
    case "needs_conversion":
      return "Native checkpoint downloaded — ready to convert to MLX.";
    case "converted":
      return "Converted to MLX and ready.";
    default:
      return "";
  }
}

const MODEL_TYPE_OPTIONS = [
  { value: "image", label: "Image" },
  { value: "video", label: "Video" },
  { value: "audio", label: "Audio" },
  { value: "utility", label: "Utility" },
];
const MODEL_TAB_TYPES = [
  ["image", "Image Models"],
  ["video", "Video Models"],
  ["audio", "Audio Models"],
  ["utility", "Utility Models"],
];

// sc-7081 (epic 7080, P0) originally hid this form on every platform because an imported
// checkpoint had no runnable engine and the API refused the request. Epic 14015 (Mac-first
// community-checkpoint import) restores it behind a real compatibility gate: S0d flipped the
// backend `model_import_enabled()` on and gates `POST /api/v1/models/import` on the
// base-weight detector's verdict (supported (family, component, quant) triple → accepted;
// unsupported → 400 with a typed reason). S0e (sc-14020) flips this web mirror on so the
// point-at-file import affordance renders. A rejected checkpoint surfaces the detector's
// typed reason (unsupported family/quant/unrecognized) inline rather than a generic error;
// an accepted one queues a `model_import` job whose completion refetches the catalog and
// surfaces the new `catalogScope:"user"` model (a Krea 2 base checkpoint today).
const MODEL_IMPORT_ENABLED = true;

// Capability descriptors shown as chips on each model card. With models now grouped
// by `type`, the chips are what tell the user what a card actually does (plain
// text-to-image vs editing vs character reference, etc.). Unknown keys fall back to
// a humanized form so a new capability still reads sensibly without a code change.
const CAPABILITY_LABELS = {
  text_to_image: "Text to Image",
  image_to_image: "Image to Image",
  edit_image: "Image Edit",
  character_image: "Character",
  vqa: "Visual Q&A",
  interleave: "Interleaved",
  image_to_video: "Image to Video",
  text_to_video: "Text to Video",
  // sc-8445: Krea Realtime advertises text/image/video-to-video, and without this row its third
  // chip fell to the humanized fallback ("video to video") — visibly out of style beside the two
  // title-cased siblings on the same card.
  video_to_video: "Video to Video",
  first_last_frame: "First / Last Frame",
  extend_clip: "Extend Clip",
  video_bridge: "Video Bridge",
  replace_person: "Replace Person",
};
const RETIRED_MODEL_CAPABILITIES = new Set(["style_variations"]);

function capabilityLabel(capability) {
  return CAPABILITY_LABELS[capability] ?? String(capability).replaceAll("_", " ");
}

// Audio capability chips (epic 13400 / sc-13406): audio-type models describe what they
// do through the manifest `audio` sub-block rather than the generic `capabilities[]`, so
// surface the same at-a-glance chips image/video cards get — derived straight from that
// sub-block. One chip per PRESENT field (voice bank size, languages, edit modes,
// multi-speaker); an absent field emits NO chip, so the chips are capability-driven and
// never per-model. Languages are capped so a many-language model (e.g. ACE-Step's 10)
// stays a single tidy pill.
const AUDIO_LANGUAGE_CHIP_MAX = 4;

function audioCapabilityChips(model) {
  const audio = model?.audio && typeof model.audio === "object" ? model.audio : null;
  if (!audio) {
    return [];
  }
  const chips = [];
  const voiceCount = Array.isArray(audio.voices) ? audio.voices.length : 0;
  if (voiceCount > 0) {
    chips.push(`${voiceCount} ${voiceCount === 1 ? "voice" : "voices"}`);
  }
  const languages = Array.isArray(audio.languages) ? audio.languages.filter(Boolean) : [];
  if (languages.length) {
    const shown = languages.slice(0, AUDIO_LANGUAGE_CHIP_MAX).join(", ");
    chips.push(
      languages.length > AUDIO_LANGUAGE_CHIP_MAX
        ? `${shown} +${languages.length - AUDIO_LANGUAGE_CHIP_MAX}`
        : shown,
    );
  }
  const editModes = Array.isArray(audio.editModes) ? audio.editModes.filter(Boolean) : [];
  if (editModes.length) {
    chips.push(`Edit: ${editModes.join(", ")}`);
  }
  if (audio.supportsMultiSpeaker === true) {
    chips.push("Multi-speaker");
  }
  return chips;
}

// Curated "getting started" models, flagged `recommended: true` in the catalog
// (config/manifests/builtin.models.jsonc). Within each type section these float to a
// "Recommended" subgroup; the rest collapse under "Additional Supported".
function isRecommendedModel(model) {
  return model.recommended === true;
}

// Group key for the family-organized LoRA list. A LoRA can list several compatible
// families; we group under its primary one and bucket family-less entries under a
// trailing "compatible" group.
function loraGroupKey(lora) {
  return lora.family ?? extractFamilies(lora)[0] ?? "";
}

// The Hugging Face page of a gated model's primary download repo — where the user
// clicks "Agree and access" to be granted access with their token (sc-5999). Derived
// from the first HF download repo (or the mlx repo), so it covers every gated model
// without a per-model manifest field. Falls back to `licenseUrl` when no repo is known.
function gatedRepoUrl(model) {
  const host = model.credentialHost || "huggingface.co";
  const repo =
    (model.downloads ?? []).find((entry) => entry.provider === "huggingface" && entry.repo)?.repo ??
    model.mlx?.repo;
  return repo ? `https://${host}/${repo}` : null;
}

// Per-model license acknowledgment (sc-7872). Gated models (FLUX.2 [dev],
// SD3.5 Large/Turbo/Medium, …) carry a non-commercial / community license the
// user must accept on Hugging Face before access is granted; we also require an
// explicit in-app acknowledgment before queuing the download so the license terms
// are surfaced and confirmed at the point of action. The ack is persisted per
// model id in localStorage (origin-scoped, same store as the theme/token caches);
// it's a UX gate only — the server-side download still needs the HF credential +
// granted access. localStorage may be unavailable (private mode, quota), so the
// getters/setters swallow failures and default to "not acknowledged".
const LICENSE_ACK_KEY_PREFIX = "sceneworks-license-ack:";

function readLicenseAck(modelId) {
  if (!modelId) {
    return false;
  }
  try {
    return window.localStorage.getItem(`${LICENSE_ACK_KEY_PREFIX}${modelId}`) === "true";
  } catch {
    return false;
  }
}

function writeLicenseAck(modelId, acknowledged) {
  if (!modelId) {
    return;
  }
  try {
    const key = `${LICENSE_ACK_KEY_PREFIX}${modelId}`;
    if (acknowledged) {
      window.localStorage.setItem(key, "true");
    } else {
      window.localStorage.removeItem(key);
    }
  } catch {
    // localStorage unavailable — the ack just won't persist this session. The
    // in-memory state still gates the download for the current view.
  }
}

// Gated models (e.g. FLUX.1 [dev]) need an accepted license + a saved credential
// before a download can succeed. The catalog flags these with `gated`/
// `credentialHost`/`licenseUrl` (sc-1898). When the matching credential is already
// present we soften the notice to a ready state; otherwise we point the user at the
// Settings credential screen. `present` is undefined while presence is still
// unknown (e.g. the credential list hasn't loaded) — we still show the link then.
// `repoUrl` links the gated repo so the user can request access (sc-5999); shown
// alongside `licenseUrl` only when the license lives on a different page (e.g.
// Ideogram 4, whose terms are on the source repo but access is on the SceneWorks repo).
// `acknowledged`/`onAcknowledgeChange` drive the in-app license-acknowledgment gate
// (sc-7872): the download button stays disabled until the user checks the box.
function GatedModelNotice({
  host,
  repoUrl,
  licenseUrl,
  present,
  acknowledged,
  onAcknowledgeChange,
  onOpenSettings,
}) {
  const hostLabel = host || "the required service";
  const safeRepoUrl = safeExternalUrl(repoUrl);
  const safeLicenseUrl = safeExternalUrl(licenseUrl);
  const showSeparateLicense = safeLicenseUrl && safeLicenseUrl !== safeRepoUrl;
  return (
    <div className={present ? "model-gated-notice ready" : "model-gated-notice"}>
      <p className={present ? "inline-success" : "inline-warning"}>
        {present
          ? `Credential for ${hostLabel} saved — request access on the model page, then download.`
          : `Gated download. Add a ${hostLabel} token, then request access on the model page and accept the license before downloading.`}
      </p>
      <div className="model-gated-actions">
        {present ? null : (
          <button type="button" onClick={onOpenSettings}>
            Add token in Settings
          </button>
        )}
        {safeRepoUrl ? (
          <a href={safeRepoUrl} target="_blank" rel="noreferrer noopener">
            Request access on Hugging Face
          </a>
        ) : null}
        {showSeparateLicense ? (
          <a href={safeLicenseUrl} target="_blank" rel="noreferrer noopener">
            Review license
          </a>
        ) : null}
      </div>
      <label className="model-license-ack">
        <input
          type="checkbox"
          checked={acknowledged}
          onChange={(event) => onAcknowledgeChange(event.target.checked)}
        />
        <span>I have read and accept this model&rsquo;s license.</span>
      </label>
    </div>
  );
}

function referencedPresetNames(recipePresets, kind, id) {
  return recipePresets
    .filter((preset) => {
      if (kind === "model") {
        return preset.model === id;
      }
      return presetLoras(preset).some((lora) => presetLoraId(lora) === id);
    })
    .map((preset) => preset.name ?? preset.id)
    .filter(Boolean);
}

function deleteConfirmation(kind, item, recipePresets) {
  const name = item.name ?? item.id;
  const presetNames = referencedPresetNames(recipePresets, kind, item.id);
  const lines = [
    `Delete ${kind} "${name}"?`,
    "This removes the registry entry and SceneWorks-owned local files when available.",
  ];
  if (presetNames.length) {
    lines.push(`Referenced by presets: ${presetNames.slice(0, 5).join(", ")}.`);
    lines.push("Those presets will keep a broken reference until updated.");
  }
  if (item.scope === "builtin" || item.catalogScope === "builtin") {
    lines.push("Built-in catalog entries stay protected; only local installed files can be removed.");
  }
  return lines.join("\n\n");
}

function deleteResultText(result, name) {
  const removed = result?.removedManifestEntry ? "Removed the registry entry" : "Removed local files";
  const warnings = result?.warnings?.length ? ` ${result.warnings.join(" ")}` : "";
  return `${removed} for ${name}.${warnings}`;
}

// The declared quant tiers of a matrix model, in suggestion-fidelity order (bf16 → q8 → q4, i.e.
// highest first) so the panel lists the biggest/best at the top. `suggestTier` already orders on
// this basis; we surface the same order to the user. Each entry is the raw `variants[]` object.
// "training" (sc-8797) is the extra flat-diffusers LoRA-training-base artifact a tiered model can
// host (lens on macOS); it renders last so the quant tiers stay the visual focus. It is not in
// tierSuggestion's fidelity list, so it is never the suggested/preselected download.
const TIER_DISPLAY_ORDER = ["bf16", "q8", "q4", "training"];

function orderedMatrixVariants(model) {
  if (!model?.hasVariantMatrix || !Array.isArray(model.variants)) {
    return [];
  }
  // Only real quant tiers (a single-variant model's "default" pseudo-tier never renders a matrix).
  const tiers = model.variants.filter((variant) => TIER_DISPLAY_ORDER.includes(variant?.variant));
  return [...tiers].sort(
    (a, b) => TIER_DISPLAY_ORDER.indexOf(a.variant) - TIER_DISPLAY_ORDER.indexOf(b.variant),
  );
}

// Per-tier quant-download panel (sc-8509, epic 8506). Shown instead of the single Download button
// when a model advertises a quant matrix (`hasVariantMatrix`). Lets the user:
//   - see every tier's on-disk size + install state,
//   - see (and start on) a RAM-based SUGGESTED tier (`suggestTier`), highlighted,
//   - multi-select and install MORE THAN ONE tier at once (for A/B), each fetching its own artifact.
// SUGGEST-NEVER-WITHHOLD: every not-installed tier is selectable regardless of RAM; the suggestion
// only preselects/highlights. `onDownloadVariant(model, tier)` wires each selection through the
// existing install path with the tier's `variant`.
function ModelTierDownloadPanel({
  model,
  unifiedMemoryGb,
  backend,
  downloadJobs,
  licenseAckRequired,
  onDownloadVariant,
  onDeleteVariant,
  deletingItem,
}) {
  const variants = orderedMatrixVariants(model);
  const suggested = suggestTier(model, unifiedMemoryGb, { backend });
  // In-flight download job per tier, keyed by the job payload's `variant` (sc-8508 records it).
  const activeJobByTier = new Map();
  for (const job of downloadJobs) {
    const tier = job.payload?.variant;
    if (tier && !terminalStatuses.has(job.status)) {
      activeJobByTier.set(tier, job);
    }
  }
  // Selection defaults to the suggested tier (if it isn't already installed). Recomputed only on
  // mount + when the suggested tier changes (e.g. the memory signal resolves after first paint).
  const [selected, setSelected] = useState(() => new Set());
  const initializedForSuggestion = React.useRef(null);
  useEffect(() => {
    if (suggested && initializedForSuggestion.current !== suggested) {
      initializedForSuggestion.current = suggested;
      const suggestedVariant = variants.find((variant) => variant.variant === suggested);
      // Preselect the suggestion only when it's still installable (not already installed).
      if (suggestedVariant && suggestedVariant.installState !== "installed") {
        setSelected(new Set([suggested]));
      }
    }
  }, [suggested]); // eslint-disable-line react-hooks/exhaustive-deps

  function toggle(tier) {
    setSelected((current) => {
      const next = new Set(current);
      if (next.has(tier)) {
        next.delete(tier);
      } else {
        next.add(tier);
      }
      return next;
    });
  }

  // Tiers we can actually queue now: selected, not already installed, and no in-flight job.
  const downloadable = [...selected].filter((tier) => {
    const variant = variants.find((entry) => entry.variant === tier);
    return variant && variant.installState !== "installed" && !activeJobByTier.has(tier);
  });

  function downloadSelected() {
    for (const tier of downloadable) {
      onDownloadVariant(model, tier);
    }
    // Clear the queued tiers from the selection so the button reflects only still-pending picks.
    setSelected((current) => {
      const next = new Set(current);
      for (const tier of downloadable) {
        next.delete(tier);
      }
      return next;
    });
  }

  return (
    <div className="model-tier-panel">
      <div className="model-tier-panel-heading">
        <span className="eyebrow">Quant tiers</span>
        <span className="helper-copy">Suggested for this machine is highlighted. Pick one or more to A/B.</span>
      </div>
      <ul className="model-tier-list">
        {variants.map((variant) => {
          const tier = variant.variant;
          const installed = variant.installState === "installed";
          const activeJob = activeJobByTier.get(tier);
          const isSuggested = tier === suggested;
          const checked = selected.has(tier);
          // Per-tier RAM guidance (sc-10042): whether THIS tier's peak footprint fits the host with
          // headroom, from the same `tierFits` the suggestion uses (measured `footprint.peakMemoryBytes`
          // when present — e.g. Wan q4 ~24 GiB — else the disk-based estimate). Replaces the blanket
          // model-level warning: q4/q8 that actually fit show nothing; only a genuinely over-budget tier
          // (e.g. bf16 on a small Mac) is flagged. Advisory only — SUGGEST-NEVER-WITHHOLD (epic 8506
          // decision 1) keeps every tier's checkbox enabled regardless.
          const overBudget = !tierFits(variant, unifiedMemoryGb, { backend, model });
          // A torn tier: the cache holds SOME of this tier's declared files but not all. Distinct from
          // both "installed" and "not installed" (sc-12279).
          const incomplete = !installed && variant.cacheState === "incomplete";
          const missingHere = Array.isArray(variant.missingRequiredFiles) ? variant.missingRequiredFiles : [];
          const incompleteHint = missingHere.length
            ? `This tier is partly downloaded and won't load. Missing: ${missingHere.join(", ")}. Select it and download again to repair.`
            : "This tier is partly downloaded and won't load. Select it and download again to repair.";
          const rowClasses = ["model-tier-row"];
          if (isSuggested) {
            rowClasses.push("suggested");
          }
          if (overBudget) {
            rowClasses.push("over-budget");
          }
          if (incomplete) {
            rowClasses.push("incomplete");
          }
          return (
            <li className={rowClasses.join(" ")} key={tier}>
              <label className="model-tier-select">
                <input
                  type="checkbox"
                  checked={checked}
                  disabled={installed || Boolean(activeJob) || licenseAckRequired}
                  onChange={() => toggle(tier)}
                />
                <span className="model-tier-label">
                  {tierLabel(tier)}
                  {isSuggested ? <span className="model-tier-suggested-badge">Suggested</span> : null}
                  {/* Distinct class (NOT `.status-badge`) so it never collides with the per-row
                      install-state status badge query/rendering — this is a separate RAM advisory. */}
                  {overBudget ? (
                    <span
                      className="model-tier-memory-warning"
                      title={`This tier's peak memory is estimated above this machine's ~${Math.round(unifiedMemoryGb)} GB. It can still install, but may run out of memory during generation.`}
                    >
                      may exceed memory
                    </span>
                  ) : null}
                </span>
              </label>
              <span className="model-tier-size">
                {formatTierSize(variant.footprint?.diskSizeBytes ?? variant.downloadSizeBytes)}
              </span>
              {/* sc-12279: a TORN tier (some of its files present, some not) is not the same as an
                  absent one — the API already distinguishes them via `cacheState`, but this row used to
                  render both as "not installed". That reads as "nothing to do here" for a tier that is
                  actually half-downloaded and will fail at load, and the model-level Fix button is
                  suppressed whenever a complete sibling tier exists (sc-9907). Say "incomplete" and
                  point at the repair: the checkbox below is already enabled for it, so re-selecting the
                  tier and downloading repairs it. */}
              <span
                className={
                  installed ? "status-badge installed" : incomplete ? "status-badge incomplete" : "status-badge"
                }
                title={incomplete ? incompleteHint : undefined}
              >
                {activeJob ? activeJob.status : installed ? "installed" : incomplete ? "incomplete" : "not installed"}
              </span>
              {/* Reclaim an installed tier's disk (sc-12024). Only this tier's files/blobs are
                  removed; the model and its other tiers stay installed. Disabled while a download
                  for this tier is in flight or this tier is mid-delete. */}
              {installed && onDeleteVariant ? (
                <button
                  type="button"
                  className="model-tier-delete danger-action"
                  disabled={Boolean(activeJob) || deletingItem === `variant:${model.id}:${tier}`}
                  title={`Delete the ${tierLabel(tier)} tier and reclaim its disk space`}
                  onClick={() =>
                    onDeleteVariant(
                      model,
                      tier,
                      variant.footprint?.diskSizeBytes ?? variant.downloadSizeBytes,
                    )
                  }
                >
                  {deletingItem === `variant:${model.id}:${tier}` ? "Deleting" : "Delete"}
                </button>
              ) : incomplete ? (
                // A torn tier repairs by re-downloading that same tier (sc-13383). One-click Fix wires
                // straight through the tier install path — no separate select-then-download step — and
                // disables while a repair job for this tier is already in flight (its status shows in the
                // badge above and on the button).
                <button
                  type="button"
                  className="model-tier-fix"
                  disabled={Boolean(activeJob) || licenseAckRequired}
                  title={incompleteHint}
                  onClick={() => onDownloadVariant(model, tier)}
                >
                  {activeJob ? activeJob.status : "Fix"}
                </button>
              ) : (
                <span className="model-tier-delete-spacer" aria-hidden="true" />
              )}
            </li>
          );
        })}
      </ul>
      <div className="model-tier-actions">
        <button
          type="button"
          disabled={downloadable.length === 0 || licenseAckRequired}
          title={licenseAckRequired ? "Accept the license above before downloading." : undefined}
          onClick={downloadSelected}
        >
          {downloadable.length > 1
            ? `Download ${downloadable.length} tiers`
            : downloadable.length === 1
              ? `Download ${tierLabel(downloadable[0])}`
              : "Select a tier to download"}
        </button>
      </div>
    </div>
  );
}

// Per-tier delete for convert-at-install models (sc-12025). These models (e.g. Anima) emit every
// tier from ONE convert job and surface them via `mlxTiers` — decoupled from the download matrix, so
// they render no ModelTierDownloadPanel and had NO per-tier control on the Models page (only
// whole-model delete). This compact list lets the user reclaim an unused convert tier's disk. The
// catalog carries no per-tier size for these, so rows show the tier + an installed badge; the backend
// reports the actual bytes reclaimed. Reuses the same `onDeleteVariant(model, tier)` path.
function ConvertedTierList({ model, onDeleteVariant, deletingItem }) {
  const tiers = Array.isArray(model.mlxTiers) ? model.mlxTiers : [];
  const ordered = TIER_DISPLAY_ORDER.filter((tier) => tiers.includes(tier));
  if (ordered.length === 0 || !onDeleteVariant) {
    return null;
  }
  return (
    <div className="model-tier-panel">
      <div className="model-tier-panel-heading">
        <span className="eyebrow">Installed tiers</span>
        <span className="helper-copy">Delete a tier you&apos;re not using to reclaim its disk space.</span>
      </div>
      <ul className="model-tier-list">
        {ordered.map((tier) => {
          const deleting = deletingItem === `variant:${model.id}:${tier}`;
          return (
            <li className="model-tier-row" key={tier}>
              <span className="model-tier-label">{tierLabel(tier)}</span>
              <span className="model-tier-size" />
              <span className="status-badge installed">installed</span>
              <button
                type="button"
                className="model-tier-delete danger-action"
                disabled={deleting}
                title={`Delete the ${tierLabel(tier)} tier and reclaim its disk space`}
                onClick={() => onDeleteVariant(model, tier)}
              >
                {deleting ? "Deleting" : "Delete"}
              </button>
            </li>
          );
        })}
      </ul>
    </div>
  );
}

export function ModelManagerScreen() {
  const {
    activeProject,
    jobs,
    loras,
    models,
    jobAction,
    setActiveView,
    deleteLora: deleteLoraAction,
    deleteModel: deleteModelAction,
    deleteModelVariant: deleteModelVariantAction,
    createModelDownloadJob,
    createLoraDownloadJob,
    createModelConvertJob,
    createLoraImportJob,
    updateLora,
    fetchLoraEmbeddedTags,
    createModelImportJob,
    presets: recipePresets = [],
    macCapabilities = DEFAULT_MAC_CAPABILITIES,
  } = useAppContext();
  // Third-party LyCORIS now applies on every MLX provider (epic 3641), so the LyCORIS upload is no
  // longer Mac-gated.
  const onCancelJob = (job) => jobAction(job, "cancel");
  const onResumeDownloadJob = (job, payload) => jobAction(job, "retry", { body: payload ?? {} });
  const onFreshDownloadJob = (job, payload) => jobAction(job, "retry", { body: payload ?? {} });
  const onConvertModel = createModelConvertJob;
  const onDeleteLora = deleteLoraAction;
  const onDeleteModel = deleteModelAction;
  const onDownloadModel = createModelDownloadJob;
  // Install a specific quant tier of a matrix model (sc-8509). Threads the tier through the existing
  // download path via the `variant` option so each install fetches that tier's own artifact.
  const onDownloadVariant = (model, variant) => createModelDownloadJob(model, { variant });
  const onDownloadLora = createLoraDownloadJob;
  const onImportLora = createLoraImportJob;
  const onUpdateLora = updateLora;
  const onFetchLoraEmbeddedTags = fetchLoraEmbeddedTags;
  const onImportModel = createModelImportJob;
  const onOpenQueue = () => setActiveView("Queue");
  // LoRA families come from each model's LoRA-compatibility set — NOT its model
  // `family` identity. They usually coincide, but distilled variants differ: e.g.
  // FLUX.2 [klein] has family "flux2-klein" yet accepts "flux2" LoRAs. The import
  // validator + generation-time matcher both key off loraCompatibility.families,
  // so the dropdown must too, or the user picks a family the backend rejects.
  const families = useMemo(
    () =>
      Array.from(
        new Set(models.flatMap((model) => modelLoraFamilies(model)).filter(Boolean)),
      ).sort(),
    [models],
  );
  const familiesKey = families.join("|");
  const [familyFilter, setFamilyFilter] = useState("all");
  const [importingLora, setImportingLora] = useState(false);
  const [importMessage, setImportMessage] = useState({ tone: "neutral", text: "" });
  const [importForm, setImportForm] = useState({
    // Default to file upload: most LoRAs are a file the user already downloaded
    // (e.g. from civit.ai); the URL tab remains one click away.
    mode: "file",
    sourceUrl: "",
    file: null,
    secondaryFile: null,
    name: "",
    scope: "global",
    family: "",
    baseModel: "",
    triggerKeywords: [],
    notes: "",
  });
  const [fileInputKey, setFileInputKey] = useState(0);
  // Inline trigger-keyword / notes editor for a LoRA row (epic 10328). editingLora keys
  // the open row by scope:id; the draft holds the working keywords + notes; suggestions
  // come from the LoRA's embedded ss_tag_frequency metadata (fetched on open).
  const [editingLora, setEditingLora] = useState("");
  const [loraEditDraft, setLoraEditDraft] = useState({ triggerWords: [], notes: "" });
  const [loraEditSuggestions, setLoraEditSuggestions] = useState([]);
  const [savingLora, setSavingLora] = useState(false);
  const [loraEditError, setLoraEditError] = useState("");
  const [importingModel, setImportingModel] = useState(false);
  const [modelImportMessage, setModelImportMessage] = useState({ tone: "neutral", text: "" });
  // Point-at-file import (sc-14020) leads with the file picker; URL stays available via the
  // segmented toggle. `type` defaults to image — the only importable base checkpoint today.
  const [modelImportForm, setModelImportForm] = useState({
    mode: "file",
    sourceUrl: "",
    file: null,
    name: "",
    type: "image",
    family: "",
  });
  const [modelFileInputKey, setModelFileInputKey] = useState(0);
  const [deletingItem, setDeletingItem] = useState("");
  const [deleteMessage, setDeleteMessage] = useState({ tone: "neutral", text: "" });
  // Resolved-model hot cache (epic 19703, sc-19711). ONE status read backs every card's local-copy
  // block; it is refreshed on mount and after each keep/remove, never on a timer — a journal
  // listing grows with the number of cached bundles, so polling it would put exactly the per-row
  // read cost back on a screen that sc-19708 took it off.
  const [modelCache, setModelCache] = useState(null);
  // Cache key of the entry whose keep/remove request is in flight, so its buttons disable.
  const [cacheBusyKey, setCacheBusyKey] = useState("");
  // Why the local-copy controls are absent, when they are. Held apart from `deleteMessage` so an
  // action outcome cannot clobber a standing explanation, and vice versa.
  const [cacheError, setCacheError] = useState("");
  // Tabbed interface (epic 10309): the active tab, the tab to restore when a search
  // clears, and the persistent search query. Every model now renders as an always-open
  // card, so the old per-row expand state is gone. Tabs: image | video | utility | lora,
  // plus a transient "search" tab that only exists while `search` is non-empty.
  const [activeTab, setActiveTab] = useState("image");
  const [prevTab, setPrevTab] = useState("image");
  const [search, setSearch] = useState("");
  // Typing into search auto-switches to the transient Search Results tab (remembering the
  // tab we came from); clearing it restores that tab. Clicking a tab just sets it.
  const handleSearchChange = (event) => {
    const value = event.target.value;
    const hasValue = value.trim() !== "";
    setSearch(value);
    if (hasValue && activeTab !== "search") {
      setPrevTab(activeTab);
      setActiveTab("search");
    } else if (!hasValue && activeTab === "search") {
      setActiveTab(prevTab || "image");
    }
  };
  const hostMemory = useHostMemory();
  const memoryBackend = macCapabilities?.macGatingActive ? "mlx" : "candle";
  const unifiedMemoryGb = hostMemoryGbForBackend(hostMemory, memoryBackend);
  // "Update" orchestration for convert-at-install models (epic 10285): re-download the newer
  // source checkpoint, then auto-fire the re-convert once that download completes. Maps a model id
  // to the specific download job id we're waiting on — a specific id (not "any completed download")
  // so a stale prior-version download can't trigger the convert before the new source is in.
  const [pendingUpdate, setPendingUpdate] = useState({});
  // Gated-model credential presence (sc-1898): only fetched when the catalog has a
  // gated model, so non-gated deployments make no extra credential request.
  const [credentials, setCredentials] = useState([]);
  // Per-model license acknowledgments (sc-7872), seeded from localStorage so a
  // returning user keeps prior accepts. Keyed by model id; toggling the gated
  // notice checkbox both updates this map and persists to localStorage.
  const [licenseAcks, setLicenseAcks] = useState(() =>
    models.reduce((acc, model) => {
      if (model.gated && readLicenseAck(model.id)) {
        acc[model.id] = true;
      }
      return acc;
    }, {}),
  );
  const setLicenseAck = (modelId, acknowledged) => {
    writeLicenseAck(modelId, acknowledged);
    setLicenseAcks((current) => ({ ...current, [modelId]: acknowledged }));
  };
  // The `useState` initializer above runs only on first mount, but `models`
  // arrives asynchronously (the catalog fetch resolves after the screen mounts).
  // A returning user's persisted localStorage ack would otherwise never re-seed
  // for gated models that weren't in the initial (often empty) catalog. Re-seed
  // whenever the catalog changes: merge in any persisted ack for a gated model
  // we haven't already recorded, without clobbering an in-session toggle (the
  // user's explicit checkbox state always wins over the persisted value).
  useEffect(() => {
    setLicenseAcks((current) => {
      let next = current;
      for (const model of models) {
        if (model.gated && current[model.id] === undefined && readLicenseAck(model.id)) {
          if (next === current) {
            next = { ...current };
          }
          next[model.id] = true;
        }
      }
      return next;
    });
  }, [models]);
  const hasGatedModel = models.some((model) => model.gated);
  const visibleLoras = useMemo(
    () => loras.filter((lora) => matchesFamily(lora, familyFilter)),
    [familyFilter, loras],
  );
  // Wan A14B MoE paired upload (sc-1991): when the user targets the wan-video
  // family, let them pick the specific base model and (for two-expert A14B models)
  // upload the low-noise expert half alongside the high-noise primary.
  const wanBaseModelOptions = models.filter((model) => model.family === "wan-video");
  const showBaseModelSelect = importForm.family === "wan-video" && wanBaseModelOptions.length > 0;
  const isMoeBaseModel = WAN_MOE_PAIRED_LORA_MODEL_IDS.has(importForm.baseModel);
  const showSecondaryFileSlot = isMoeBaseModel && importForm.mode === "file";
  const moeMissingSecondary = showSecondaryFileSlot && Boolean(importForm.file) && !importForm.secondaryFile;

  useEffect(() => {
    if (familyFilter !== "all" && !families.includes(familyFilter)) {
      setFamilyFilter("all");
    }
  }, [families, familiesKey, familyFilter]);

  useEffect(() => {
    setImportForm((current) => (current.family && !families.includes(current.family) ? { ...current, family: "" } : current));
  }, [families, familiesKey]);

  // The base model + low-noise slot only apply to wan-video imports; clear them
  // when the family changes away so a stale baseModel can't ride along.
  useEffect(() => {
    setImportForm((current) =>
      current.family !== "wan-video" && (current.baseModel || current.secondaryFile)
        ? { ...current, baseModel: "", secondaryFile: null }
        : current,
    );
  }, [importForm.family]);

  useEffect(() => {
    if (!hasGatedModel) {
      return undefined;
    }
    let cancelled = false;
    loadCredentials()
      .then((list) => {
        if (!cancelled) {
          setCredentials(Array.isArray(list) ? list : []);
        }
      })
      // Presence unknown (e.g. not authenticated yet) — the notice still links to Settings.
      .catch(() => {
        if (!cancelled) {
          setCredentials([]);
        }
      });
    return () => {
      cancelled = true;
    };
  }, [hasGatedModel]);

  // Second half of the "Update" flow: when the tracked source re-download completes, kick the
  // re-convert (the convert reads the now-current `convertSourceFile` from the cache). We watch the
  // SPECIFIC download job id, so a previously-completed old-version download never fires this early.
  useEffect(() => {
    const entries = Object.entries(pendingUpdate);
    if (entries.length === 0) {
      return;
    }
    let changed = false;
    const next = { ...pendingUpdate };
    for (const [modelId, downloadJobId] of entries) {
      const download = jobs.find((job) => job.id === downloadJobId);
      if (!download) {
        continue; // job not visible yet — keep waiting
      }
      if (download.status === "completed") {
        const activeConvert = jobs.find(
          (job) =>
            job.type === "model_convert" &&
            job.payload?.modelId === modelId &&
            !terminalStatuses.has(job.status),
        );
        if (!activeConvert) {
          createModelConvertJob({ id: modelId });
        }
        delete next[modelId];
        changed = true;
      } else if (terminalStatuses.has(download.status)) {
        // Download failed/canceled/interrupted — abandon the chain; the download error is surfaced.
        delete next[modelId];
        changed = true;
      }
    }
    if (changed) {
      setPendingUpdate(next);
    }
  }, [jobs, pendingUpdate, createModelConvertJob]);

  // One resolved-cache status read for the whole screen. A failure leaves `modelCache` null, which
  // renders no local-copy blocks at all — never an empty-but-confident "no local copies". The
  // reason is kept separately so the screen can say ONCE why the controls are missing: the failure
  // is global, so repeating it on every model card would be noise, and staying silent would leave
  // the user to guess whether they have no local copies or no answer.
  const refreshModelCache = useCallback(async () => {
    try {
      const snapshot = await fetchModelCache();
      setModelCache(snapshot);
      // A 200 can still carry `error`: the store exists but could not be listed.
      setCacheError(snapshot?.error ? String(snapshot.error) : "");
    } catch (error) {
      setModelCache(null);
      setCacheError(String(error?.message ?? error));
    }
  }, []);

  useEffect(() => {
    refreshModelCache();
  }, [refreshModelCache]);

  // "Keep locally" / "Allow automatic removal" — the artifact pin. Re-reads the authoritative
  // status afterwards rather than optimistically flipping the row: the UI must not claim a state
  // the store hasn't committed.
  async function toggleKeepLocalCopy(entry) {
    setCacheBusyKey(entry.cacheKey);
    try {
      await setCacheEntryPin(entry.cacheKey, !entry.pinned);
      await refreshModelCache();
      setDeleteMessage({
        tone: "neutral",
        text: entry.pinned
          ? "This local copy can now be removed automatically when space is needed."
          : "This local copy will be kept until you allow automatic removal.",
      });
    } catch (error) {
      setDeleteMessage({ tone: "error", text: String(error?.message ?? error) });
    } finally {
      setCacheBusyKey("");
    }
  }

  // "Remove local copy". The preview is fetched FIRST and the confirm is built entirely from it —
  // measured bytes, the typed pin answer (including "can't determine"), and any refusal reason —
  // so the dialog can never promise a removal the store would then refuse. A blocked preview is
  // reported and the destructive path is not offered at all.
  async function removeLocalCopy(model, entry) {
    setCacheBusyKey(entry.cacheKey);
    try {
      const preview = await previewCacheRemoval(entry.cacheKey);
      const lines = describeRemovalPreview(preview);
      if (!removalIsAllowed(preview)) {
        setDeleteMessage({ tone: "error", text: lines.join(" ") });
        return;
      }
      const proceed = await appConfirm({
        title: `Remove the local copy of ${model.name}?`,
        message: lines.join(" "),
        confirmLabel: "Remove local copy",
        tone: "danger",
      });
      if (!proceed) return;
      const outcome = await removeCacheEntry(entry.cacheKey);
      await refreshModelCache();
      setDeleteMessage({
        tone: "neutral",
        text: outcome.sourceUnavailableWarning
          ? `Removed the local copy and freed ${formatBytes(outcome.reclaimedBytes)}. ${model.name} stays unusable until its model library is reconnected.`
          : `Removed the local copy and freed ${formatBytes(outcome.reclaimedBytes)}.`,
      });
    } catch (error) {
      setDeleteMessage({ tone: "error", text: String(error?.message ?? error) });
    } finally {
      setCacheBusyKey("");
    }
  }

  // First half of the "Update" flow: re-download the newer source, then track its job so the effect
  // above can auto-convert once it lands. Reuses the existing download + convert endpoints.
  async function handleUpdateModel(model) {
    const job = await onDownloadModel(model);
    if (job?.id) {
      setPendingUpdate((prev) => ({ ...prev, [model.id]: job.id }));
    }
  }

  function downloadJobsFor(model) {
    return jobs.filter((job) => job.type === "model_download" && job.payload?.modelId === model.id);
  }

  function convertJobsFor(model) {
    return jobs.filter((job) => job.type === "model_convert" && job.payload?.modelId === model.id);
  }

  function loraDownloadJobsFor(lora) {
    return jobs.filter((job) => job.type === "lora_download" && job.payload?.loraId === lora.id);
  }

  async function importLora(event) {
    event.preventDefault();
    const isFileImport = importForm.mode === "file";
    if ((!isFileImport && !importForm.sourceUrl.trim()) || (isFileImport && !importForm.file) || !onImportLora) {
      return;
    }
    setImportingLora(true);
    setImportMessage({
      tone: "neutral",
      text: isFileImport ? "Uploading LoRA file before queueing import." : "",
    });
    try {
      const familyOverride = importForm.family ? { family: importForm.family } : {};
      // Carry the chosen base model (wan-video) and, for an A14B MoE upload, the
      // low-noise expert half so both land in one record (sc-1991).
      const baseModelOverride = showBaseModelSelect && importForm.baseModel ? { baseModel: importForm.baseModel } : {};
      const secondaryOverride =
        isFileImport && showSecondaryFileSlot && importForm.secondaryFile
          ? { secondaryFile: importForm.secondaryFile }
          : {};
      const job = await onImportLora({
        ...(isFileImport ? { file: importForm.file } : { sourceUrl: importForm.sourceUrl.trim() }),
        name: importForm.name.trim() || undefined,
        scope: importForm.scope,
        triggerWords: importForm.triggerKeywords,
        notes: importForm.notes.trim() || undefined,
        ...familyOverride,
        ...baseModelOverride,
        ...secondaryOverride,
      });
      const loraId = job?.payload?.loraId;
      const resolvedFamily = job?.payload?.manifestEntry?.family;
      // Show the detected family in the same normalized vocabulary the dropdown
      // uses (e.g. the backend's canonical `krea_2` displays as `krea-2`), so the
      // note never names a token the manual selector doesn't list.
      const detectionNote =
        !importForm.family && resolvedFamily
          ? ` Detected family: ${normalizeLoraFamily(resolvedFamily)}.`
          : "";
      setImportForm((current) => ({
        ...current,
        sourceUrl: "",
        file: null,
        secondaryFile: null,
        name: "",
        triggerKeywords: [],
        notes: "",
      }));
      // Force a re-mount so choosing the same file again still emits a change event.
      setFileInputKey((current) => current + 1);
      setImportMessage({
        tone: "success",
        text: `${loraId ? `LoRA import queued for ${loraId}.` : "LoRA import queued."}${detectionNote}`,
      });
    } catch (err) {
      setImportMessage({ tone: "error", text: err.message });
    } finally {
      setImportingLora(false);
    }
  }

  async function importModel(event) {
    event.preventDefault();
    const isFileImport = modelImportForm.mode === "file";
    if ((!isFileImport && !modelImportForm.sourceUrl.trim()) || (isFileImport && !modelImportForm.file) || !onImportModel) {
      return;
    }
    setImportingModel(true);
    setModelImportMessage({
      tone: "neutral",
      text: isFileImport ? "Uploading model file before queueing import." : "",
    });
    try {
      const familyOverride = modelImportForm.family ? { family: modelImportForm.family } : {};
      const job = await onImportModel({
        ...(isFileImport ? { file: modelImportForm.file } : { sourceUrl: modelImportForm.sourceUrl.trim() }),
        name: modelImportForm.name.trim() || undefined,
        // Send the model type under `type` — the literal field name the backend's multipart parser
        // reads (models.rs `model_import_request_from_multipart`) and that JSON deserialization
        // accepts via `#[serde(alias = "type")]` on `ModelImportRequest`. Keying this `modelType`
        // was silently dropped on file uploads (multipart has no serde aliasing), defaulting every
        // imported checkpoint to `image` regardless of selection (sc-14020).
        type: modelImportForm.type,
        ...familyOverride,
      });
      const modelId = job?.payload?.modelId;
      const resolvedFamily = job?.payload?.manifestEntry?.family;
      const detectionNote =
        !modelImportForm.family && resolvedFamily
          ? ` Detected family: ${normalizeLoraFamily(resolvedFamily)}.`
          : "";
      setModelImportForm((current) => ({ ...current, sourceUrl: "", file: null, name: "" }));
      setModelFileInputKey((current) => current + 1);
      setModelImportMessage({
        tone: "success",
        text: `${modelId ? `Model import queued for ${modelId}.` : "Model import queued."}${detectionNote}`,
      });
    } catch (err) {
      setModelImportMessage({ tone: "error", text: err.message });
    } finally {
      setImportingModel(false);
    }
  }

  async function deleteModel(model) {
    if (!onDeleteModel || model.removable === false) {
      return;
    }
    // Desktop-safe confirm (sc-12068) — window.confirm no-ops in the Tauri WebView.
    if (
      !(await appConfirm({
        title: "Delete model?",
        message: deleteConfirmation("model", model, recipePresets),
        confirmLabel: "Delete",
        cancelLabel: "Cancel",
        tone: "danger",
      }))
    ) {
      return;
    }
    setDeletingItem(`model:${model.id}`);
    setDeleteMessage({ tone: "neutral", text: "" });
    try {
      const result = await onDeleteModel(model);
      if (result?.cancelled) {
        setDeleteMessage({ tone: "neutral", text: "" });
      } else {
        setDeleteMessage({ tone: "success", text: deleteResultText(result, model.name ?? model.id) });
      }
    } catch (err) {
      setDeleteMessage({ tone: "error", text: err.message });
    } finally {
      setDeletingItem("");
    }
  }

  // Delete ONE installed quant tier of a model to reclaim its disk (sc-12024/sc-12025). Unlike
  // deleteModel this leaves the model (and its other tiers) in place; the catalog refetch flips the
  // tier back to "not installed". `tier` is the tier key; `sizeBytes` (optional) is the tier's
  // declared on-disk size for the confirm — download-matrix rows have it, convert-at-install
  // (mlxTiers) rows don't, so the confirm omits the estimate there and the result reports the real
  // reclaimed bytes from the backend.
  async function deleteModelVariant(model, tier, sizeBytes) {
    if (!deleteModelVariantAction) {
      return;
    }
    const tierName = tierLabel(tier);
    const sizeClause =
      sizeBytes != null
        ? `This permanently frees about ${formatTierSize(sizeBytes)} and removes`
        : "This permanently removes";
    const message = [
      `Delete the ${tierName} tier of "${model.name ?? model.id}"?`,
      `${sizeClause} only this tier — your other tiers stay installed. It skips the Trash and can't be undone.`,
    ].join("\n\n");
    // Desktop-safe confirm (sc-12068) — window.confirm no-ops in the Tauri WebView.
    if (
      !(await appConfirm({
        title: "Delete tier?",
        message,
        confirmLabel: "Delete",
        cancelLabel: "Cancel",
        tone: "danger",
      }))
    ) {
      return;
    }
    setDeletingItem(`variant:${model.id}:${tier}`);
    setDeleteMessage({ tone: "neutral", text: "" });
    try {
      const result = await deleteModelVariantAction(model, tier);
      if (result?.cancelled) {
        setDeleteMessage({ tone: "neutral", text: "" });
      } else {
        const freed = result?.reclaimedLabel ?? "";
        setDeleteMessage({
          tone: "success",
          text: `Deleted the ${tierName} tier of ${model.name ?? model.id}${freed ? ` and reclaimed ${freed}` : ""}.`,
        });
      }
    } catch (err) {
      setDeleteMessage({ tone: "error", text: err.message });
    } finally {
      setDeletingItem("");
    }
  }

  async function deleteLora(lora) {
    if (!onDeleteLora || lora.removable === false) {
      return;
    }
    // Desktop-safe confirm (sc-12068) — window.confirm no-ops in the Tauri WebView.
    if (
      !(await appConfirm({
        title: "Delete LoRA?",
        message: deleteConfirmation("lora", lora, recipePresets),
        confirmLabel: "Delete",
        cancelLabel: "Cancel",
        tone: "danger",
      }))
    ) {
      return;
    }
    setDeletingItem(`lora:${lora.scope ?? "global"}:${lora.id}`);
    setDeleteMessage({ tone: "neutral", text: "" });
    try {
      const result = await onDeleteLora(lora);
      if (result?.cancelled) {
        setDeleteMessage({ tone: "neutral", text: "" });
      } else {
        setDeleteMessage({ tone: "success", text: deleteResultText(result, lora.name ?? lora.id) });
      }
    } catch (err) {
      setDeleteMessage({ tone: "error", text: err.message });
    } finally {
      setDeletingItem("");
    }
  }

  const completedImportTimes = completedLoraImportTimes(jobs);
  const pendingLoraImportJobs = jobs.filter((job) => job.type === "lora_import" && !isSupersededLoraImport(job, completedImportTimes));
  const localLoraImportJobs = pendingLoraImportJobs.filter((job) => job.status !== "completed" && matchesFamily(job, familyFilter));
  const pendingModelImportJobs = jobs.filter((job) => job.type === "model_import" && job.status !== "completed");
  const isModelFileImport = modelImportForm.mode === "file";
  const modelImportDisabled =
    importingModel ||
    !onImportModel ||
    (isModelFileImport ? !modelImportForm.file : !modelImportForm.sourceUrl.trim());
  const hiddenImportCount =
    familyFilter === "all" ? 0 : pendingLoraImportJobs.filter((job) => job.status !== "completed" && !matchesFamily(job, familyFilter)).length;
  const isFileImport = importForm.mode === "file";
  const importDisabled =
    importingLora ||
    !onImportLora ||
    (importForm.scope === "project" && !activeProject) ||
    (isFileImport ? !importForm.file : !importForm.sourceUrl.trim());

  // LoRAs split into Built-In (catalog `scope: "builtin"`) and User (global/project)
  // containers. Built-in entries are a flat list with a Download affordance; user
  // entries keep the family-organized grouping below.
  const builtinLoras = visibleLoras.filter((lora) => lora.scope === "builtin");
  const userLoras = visibleLoras.filter((lora) => lora.scope !== "builtin");

  // Visible user LoRAs grouped by family for the family-organized list. The family
  // dropdown still narrows `visibleLoras` upstream; when a specific family is
  // selected this collapses to a single group.
  const loraGroupMap = new Map();
  userLoras.forEach((lora) => {
    const key = loraGroupKey(lora) || "compatible";
    if (!loraGroupMap.has(key)) {
      loraGroupMap.set(key, []);
    }
    loraGroupMap.get(key).push(lora);
  });
  const loraGroups = [...loraGroupMap.entries()]
    .sort(([a], [b]) => (a === "compatible" ? 1 : b === "compatible" ? -1 : a.localeCompare(b)))
    .map(([family, items]) => ({ family, items }));

  function renderModelCard(model) {
    const downloadJobs = downloadJobsFor(model);
    const downloadJob = downloadJobs.find((job) => !terminalStatuses.has(job.status));
    const installed = model.installState === "installed";
    const incomplete = model.cacheState === "incomplete" || model.repairAvailable;
    const missingRequiredFiles = Array.isArray(model.missingRequiredFiles) ? model.missingRequiredFiles : [];
    const localDownloadJob = installed ? null : downloadJobs.find((job) => job.status !== "completed");
    const failedDownload = localDownloadJob && terminalStatuses.has(localDownloadJob.status);
    const downloadSize = downloadSizeText(model);
    const unassociated = !model.family;
    const capabilities = Array.isArray(model.capabilities)
      ? model.capabilities.filter((capability) => !RETIRED_MODEL_CAPABILITIES.has(capability))
      : [];
    // Audio-type cards describe themselves via the `audio` sub-block (voice count,
    // languages, edit modes, multi-speaker) instead of the generic capabilities[].
    const audioChips = audioCapabilityChips(model);
    const deleteKey = `model:${model.id}`;
    const canDelete = Boolean(onDeleteModel) && model.removable !== false;
    // MLX (macOS) variant: only present when the catalog computed mlxConversionState.
    // Conversion state is a platform capability supplied by the API, not a memory measurement.
    // Keep that control surface intact while guarding the MLX memory block by the active lane.
    const mlxState = model.mlxConversionState;
    const mlxMinGb = memoryBackend === "mlx" ? blanketFloorGb(model, "mlx") : null;
    const mlxEnoughMemory = unifiedMemoryGb == null || mlxMinGb == null || unifiedMemoryGb >= mlxMinGb;
    const convertJobs = convertJobsFor(model);
    const convertJob = convertJobs.find((job) => !terminalStatuses.has(job.status));
    const failedConvert = convertJobs.find((job) => terminalStatuses.has(job.status) && job.status !== "completed");
    const showConvertButton = mlxState === "needs_conversion" || mlxState === "converted";
    const gated = Boolean(model.gated);
    const credentialPresent = gated && hasPresentCredential(credentials, model.credentialHost);
    // License-acknowledgment gate (sc-7872): an uninstalled gated model can't be
    // downloaded until the user accepts its license in-app. Already-installed
    // gated models (no notice shown) are never blocked.
    const licenseAcknowledged = licenseAcks[model.id] === true;
    const licenseAckRequired = gated && !installed && !licenseAcknowledged;
    // Quant-matrix models (sc-8509): render the per-tier download panel with a RAM-based suggestion
    // + multi-select instead of the single Download button. Single-variant models are unchanged.
    const hasTierMatrix = model.hasVariantMatrix === true && orderedMatrixVariants(model).length > 0;
    const firstCapability = capabilities.length ? capabilityLabel(capabilities[0]) : null;
    const familyMeta = [model.family ?? "unassociated", firstCapability].filter(Boolean).join(" · ");
    const macBlock = macModelBlock(model, macCapabilities);
    // Header install-status badge: warn tones for an incomplete cache, accent for installed,
    // neutral for missing.
    const statusClass = incomplete ? "status-badge warning" : installed ? "status-badge installed" : "status-badge";
    const statusText = incomplete ? "incomplete" : installed ? "installed" : "missing";
    // Where this model's files actually resolve from, straight off the typed judgement the one
    // shared resolver produced (sc-19708). NEVER re-derived here from paths or error text — that
    // discipline is the whole reason a second, drifting availability opinion can't exist.
    const availability = availabilityBadge(model);
    // Local copies of THIS model, joined by the backend.
    const localCopies = entriesForModel(modelCache, model.id);
    // The block renders ONLY when the cache state is actually known. `modelCache` is null when the
    // status read failed outright, and a returned snapshot carries `error` when the store exists
    // but could not be listed — in both cases the entry list is empty for want of an answer, not
    // because there are no copies. Gating on `cacheKnown` is what stops "No local copy yet." from
    // being rendered as a confident claim over a read that never succeeded.
    const cacheKnown = Boolean(modelCache) && !modelCache.error;
    const showLocalCopySection = cacheKnown && (localCopies.length > 0 || canHoldLocalCopy(model));
    return (
      <article className={model.recommended ? "model-card recommended" : "model-card"} key={model.id}>
        <div className="model-card-head">
          <span className="model-card-title">
            <strong>{model.name}</strong>
            <small>{familyMeta}</small>
          </span>
          <span className="model-card-status">
            <span className={statusClass}>{statusText}</span>
            {availability ? (
              <span
                className={availability.tone ? `status-badge ${availability.tone}` : "status-badge"}
                title={availability.title}
              >
                {availability.text}
              </span>
            ) : null}
            {model.updateAvailable ? <span className="status-badge warning">update available</span> : null}
            {unassociated ? (
              <span className="status-badge warning" title="Set this model's family in user.models.jsonc before using it for generation.">
                needs family
              </span>
            ) : null}
            {macBlock ? (
              <span className="status-badge warning" title={macBlock.text}>
                not on Mac
              </span>
            ) : null}
          </span>
        </div>
        {isRecommendedModel(model) ? <span className="model-card-rec-chip">★ Recommended</span> : null}
        <p className="model-card-description">{model.ui?.description ?? model.family ?? model.id}</p>
        {capabilities.length ? (
          <ul className="model-capabilities">
            {capabilities.map((capability) => (
              <li className="chip" key={capability}>
                {capabilityLabel(capability)}
              </li>
            ))}
          </ul>
        ) : null}
        {audioChips.length ? (
          <ul className="model-capabilities model-audio-capabilities">
            {audioChips.map((chip) => (
              <li className="chip" key={chip}>
                {chip}
              </li>
            ))}
          </ul>
        ) : null}
        {gated && !installed ? (
          <GatedModelNotice
            host={model.credentialHost}
            repoUrl={gatedRepoUrl(model) ?? model.licenseUrl ?? null}
            licenseUrl={model.licenseUrl}
            present={credentialPresent}
            acknowledged={licenseAcknowledged}
            onAcknowledgeChange={(checked) => setLicenseAck(model.id, checked)}
            onOpenSettings={() => setActiveView("Settings")}
          />
        ) : null}
        {incomplete ? (
          <p className="inline-warning">
            Cached files are incomplete
            {missingRequiredFiles.length ? `: ${missingRequiredFiles.slice(0, 3).join(", ")}${missingRequiredFiles.length > 3 ? "..." : ""}` : ""}.
          </p>
        ) : null}
        {localDownloadJob ? (
          <WorkerProgressCard
            job={localDownloadJob}
            onCancel={onCancelJob}
            onRetry={onResumeDownloadJob}
            onFreshRetry={onFreshDownloadJob}
            onOpenQueue={onOpenQueue}
          />
        ) : null}
        {mlxState ? (
          <div className="mlx-status">
            <div className="mlx-status-badges">
              <span className="status-badge">MLX</span>
              {/* The model-level `mlx.minMemoryGb` is a single blanket floor = the HEAVIEST tier's
                  worst case (e.g. Wan A14B bf16, both MoE experts dense = 133 GB). Showing it
                  tier-agnostically over-warns quant-matrix models whose default/installed tier is q4
                  (measured ~24 GiB, one MoE expert resident at a time) — it reads "won't run" on Macs
                  that run q4/q8 fine. For a matrix model the per-tier download panel below drives the
                  per-tier RAM guidance instead (sc-10042); keep the blanket badge/warning only for
                  single-variant MLX models, which have one legitimate footprint. */}
              {mlxMinGb != null && !hasTierMatrix ? (
                <span className={mlxEnoughMemory ? "status-badge" : "status-badge warning"}>needs ≥ {mlxMinGb} GB</span>
              ) : null}
            </div>
            <p>{mlxStatusText(model)}</p>
            {!mlxEnoughMemory && !hasTierMatrix ? (
              <p className="inline-warning">
                Needs ≥ {mlxMinGb} GB unified memory; this Mac has ~{Math.round(unifiedMemoryGb)} GB. It may run out of memory.
              </p>
            ) : null}
            {convertJob ? <WorkerProgressCard job={convertJob} onCancel={onCancelJob} onOpenQueue={onOpenQueue} /> : null}
            {model.updateAvailable ? (
              <>
                <p className="inline-warning">
                  A newer checkpoint is available. Update re-downloads it and re-converts to MLX.
                </p>
                <button
                  disabled={
                    Boolean(downloadJob) ||
                    Boolean(convertJob) ||
                    Boolean(pendingUpdate[model.id]) ||
                    !mlxEnoughMemory
                  }
                  onClick={() => handleUpdateModel(model)}
                  type="button"
                >
                  {convertJob
                    ? convertJob.status
                    : downloadJob || pendingUpdate[model.id]
                      ? "Updating…"
                      : "Update"}
                </button>
              </>
            ) : showConvertButton ? (
              <button
                disabled={mlxState === "converted" || Boolean(convertJob) || !mlxEnoughMemory}
                onClick={() => onConvertModel?.(model)}
                type="button"
              >
                {convertJob
                  ? convertJob.status
                  : mlxState === "converted"
                    ? "MLX ready"
                    : failedConvert
                      ? "Retry MLX Conversion"
                      : "Convert to MLX"}
              </button>
            ) : null}
          </div>
        ) : null}
        {hasTierMatrix ? (
          <ModelTierDownloadPanel
            model={model}
            unifiedMemoryGb={unifiedMemoryGb}
            backend={memoryBackend}
            downloadJobs={downloadJobs}
            licenseAckRequired={licenseAckRequired}
            onDownloadVariant={onDownloadVariant}
            onDeleteVariant={deleteModelVariant}
            deletingItem={deletingItem}
          />
        ) : null}
        {/* Convert-at-install models (mlxTiers) have no download panel; surface their installed
            tiers with a per-tier delete so unused convert outputs can be reclaimed (sc-12025). */}
        {!hasTierMatrix ? (
          <ConvertedTierList
            model={model}
            onDeleteVariant={deleteModelVariant}
            deletingItem={deletingItem}
          />
        ) : null}
        {/* Local copies of this model in the resolved-model cache (sc-19711). Separate from the
            download/install job surface above on purpose: promoting or reclaiming a local copy is
            not an install, and conflating them would make "removed" read as "uninstalled". */}
        {showLocalCopySection ? (
          <div className="model-local-copy">
            <div className="model-local-copy-head">
              <span className="status-badge">Local copy</span>
              {model.modelAvailability === "installed_external_unavailable" && !localCopies.length ? (
                <span className="status-badge warning">library disconnected</span>
              ) : null}
            </div>
            {localCopies.length === 0 ? (
              <p>
                No local copy yet. SceneWorks makes one the next time it loads this model, if local
                copies are turned on in Settings.
              </p>
            ) : (
              <ul className="model-local-copy-list">
                {localCopies.map((entry) => {
                  const busy = cacheBusyKey === entry.cacheKey;
                  const ready = entry.state === "complete";
                  return (
                    <li className="model-local-copy-entry" key={entry.cacheKey}>
                      <span className="model-local-copy-text">
                        <span>
                          {entry.tier ? `${tierLabel(entry.tier)} · ` : ""}
                          {formatBytes(entry.bytes)}
                        </span>
                        <small>
                          {ready
                            ? entry.pinned
                              ? "Kept — never removed automatically"
                              : "Can be removed automatically when space is needed"
                            : `Not ready to use (${entry.state}) — SceneWorks finishes or clears it on the next check`}
                        </small>
                      </span>
                      <span className="model-local-copy-actions">
                        <button
                          disabled={busy || !ready}
                          onClick={() => toggleKeepLocalCopy(entry)}
                          type="button"
                        >
                          {entry.pinned ? "Allow automatic removal" : "Keep locally"}
                        </button>
                        <button
                          className="danger-action"
                          disabled={busy}
                          onClick={() => removeLocalCopy(model, entry)}
                          type="button"
                        >
                          {busy ? "Working…" : "Remove local copy"}
                        </button>
                      </span>
                    </li>
                  );
                })}
              </ul>
            )}
            <p>
              Removing a local copy never uninstalls the model — the original files in the model
              library are untouched.
            </p>
          </div>
        ) : null}
        {model.updateAvailable && !mlxState ? (
          <p className="inline-warning">A newer model download is available; the installed version remains usable.</p>
        ) : null}
        <div className="model-card-footer">
          <span className="model-card-size">{downloadSize}</span>
          <div className="model-card-footer-actions">
            {hasTierMatrix ? (
              // A quant-matrix model installs its tiers from the panel above. Keep only a Fix
              // affordance here for an incomplete cache or soft co-requisite update; otherwise
              // there's no single-tier button. The default-tier job fetches every co-requisite.
              incomplete || model.updateAvailable ? (
                <button
                  className="model-card-primary"
                  disabled={!model.downloadable || Boolean(downloadJob) || licenseAckRequired}
                  title={licenseAckRequired ? "Accept the license above before downloading." : undefined}
                  onClick={() =>
                    failedDownload
                      ? onResumeDownloadJob(localDownloadJob, { payloadChanges: { downloadAction: "resume" } })
                      : onDownloadModel(model)
                  }
                  type="button"
                >
                  {downloadJob ? downloadJob.status : failedDownload ? "Resume Download" : model.updateAvailable ? "Update" : "Fix"}
                </button>
              ) : null
            ) : (
              <button
                className="model-card-primary"
                disabled={(installed && !incomplete && !model.updateAvailable) || !model.downloadable || Boolean(downloadJob) || licenseAckRequired}
                title={licenseAckRequired ? "Accept the license above before downloading." : undefined}
                onClick={() =>
                  failedDownload
                    ? onResumeDownloadJob(localDownloadJob, { payloadChanges: { downloadAction: "resume" } })
                    : onDownloadModel(model)
                }
                type="button"
              >
                {downloadJob
                  ? downloadJob.status
                  : failedDownload
                      ? "Resume Download"
                      : incomplete
                        ? "Fix"
                        : model.updateAvailable
                          ? "Update"
                        : installed
                          ? "Ready"
                          : model.downloadSizeLabel
                            ? `Download ${downloadSize}`
                            : "Download"}
              </button>
            )}
            <button className="danger-action" disabled={!canDelete || deletingItem === deleteKey} onClick={() => deleteModel(model)} type="button">
              {model.removable === false ? "Protected" : deletingItem === deleteKey ? "Deleting" : "Delete"}
            </button>
          </div>
        </div>
      </article>
    );
  }

  const loraEditKey = (lora) => `${lora.scope ?? "global"}:${lora.id}`;

  function startEditLora(lora) {
    setEditingLora(loraEditKey(lora));
    setLoraEditDraft({
      triggerWords: Array.isArray(lora.triggerWords) ? lora.triggerWords : [],
      notes: typeof lora.notes === "string" ? lora.notes : "",
    });
    setLoraEditSuggestions([]);
    setLoraEditError("");
    if (onFetchLoraEmbeddedTags) {
      onFetchLoraEmbeddedTags(lora)
        .then((tags) => setLoraEditSuggestions(Array.isArray(tags) ? tags : []))
        .catch(() => setLoraEditSuggestions([]));
    }
  }

  function cancelEditLora() {
    setEditingLora("");
    setLoraEditError("");
  }

  async function saveEditLora(lora) {
    if (!onUpdateLora) {
      return;
    }
    setSavingLora(true);
    setLoraEditError("");
    try {
      await onUpdateLora(lora, {
        triggerWords: loraEditDraft.triggerWords,
        notes: loraEditDraft.notes.trim(),
      });
      setEditingLora("");
    } catch (error) {
      setLoraEditError(error?.message ?? "Failed to save LoRA metadata.");
    } finally {
      setSavingLora(false);
    }
  }

  function renderLoraRow(lora) {
    const installed = lora.installState === "installed";
    const missing = lora.installState === "missing";
    const isBuiltin = lora.scope === "builtin";
    // A built-in that isn't downloaded reads "not installed" (neutral) — it can be fetched.
    // A user LoRA with no local files reads "unavailable" (danger) — it's registered but broken.
    const userUnavailable = missing && !isBuiltin;
    const statusText = installed ? "installed" : missing ? (isBuiltin ? "not installed" : "unavailable") : "pending";
    const statusClass = installed ? "status-badge installed" : userUnavailable ? "status-badge danger" : "status-badge";
    const rowClass = userUnavailable ? "lora-row unavailable" : "lora-row";
    const deleteKey = `lora:${lora.scope ?? "global"}:${lora.id}`;
    // Built-in LoRAs with a Hugging Face source can be fetched on demand (sc-5944) —
    // user LoRAs are installed via the import form, so they get no Download affordance.
    const hfSource = (lora.source?.provider ?? lora.provider) === "huggingface";
    const canDownload = Boolean(onDownloadLora) && lora.scope === "builtin" && (!installed || lora.updateAvailable) && hfSource;
    const downloadJob = loraDownloadJobsFor(lora).find((job) => !terminalStatuses.has(job.status));
    const keywords = Array.isArray(lora.triggerWords) ? lora.triggerWords : [];
    const notes = typeof lora.notes === "string" ? lora.notes : "";
    const isEditing = editingLora === loraEditKey(lora);
    // A LoRA whose architecture family couldn't be resolved (no manifest family, no
    // detected signature — e.g. an external ComfyUI adapter in a format we don't yet
    // recognize) is listed so the user sees the file was found, but flagged unusable:
    // the generate-time gate refuses it for every model, so no picker offers it
    // (sc-10509). Built-in catalog LoRAs always declare a family, so this only marks
    // user/external rows.
    const unusable = !isBuiltin && !loraHasResolvableFamily(lora);
    // Built-in entries are read-only (their manifest is compiled in); the backend
    // rejects PATCH on them, so no Edit affordance is offered.
    const canEdit = Boolean(onUpdateLora) && lora.scope !== "builtin";
    // What the adapter file itself declared in its safetensors `__metadata__` at import
    // (sc-14057). Each part is shown only when the file stated it — a great many third-party
    // adapters declare no rank/alpha, and showing an inferred value would be a lie about the
    // file. `rank`/`alpha` are numbers; `== null` keeps a legitimate alpha of 0.
    const declared = [
      lora.networkType,
      lora.rank == null ? null : `rank ${lora.rank}`,
      lora.alpha == null ? null : `alpha ${lora.alpha}`,
    ].filter(Boolean);
    return (
      <article className={rowClass} key={lora.id ?? lora.name}>
        <span>
          <strong>{lora.name ?? lora.id}</strong>
          <small>{[lora.scope, unusable ? "unrecognized format" : lora.family ?? "compatible", ...declared].filter(Boolean).join(" | ")}</small>
        </span>
        {unusable ? (
          <span className="status-badge warning" title="This file's architecture family couldn't be identified, so it can't be applied to any model.">
            unusable
          </span>
        ) : (
          <>
            <span className={statusClass}>{statusText}</span>
            {lora.updateAvailable ? <span className="status-badge warning">update available</span> : null}
          </>
        )}
        <span className="lora-row-actions">
          {canDownload ? (
            <button disabled={Boolean(downloadJob)} onClick={() => onDownloadLora(lora)} type="button">
              {downloadJob ? downloadJob.status : lora.updateAvailable ? "Update" : "Download"}
            </button>
          ) : null}
          {canEdit && !isEditing ? (
            <button onClick={() => startEditLora(lora)} type="button">
              Edit
            </button>
          ) : null}
          <button
            className="danger-action"
            disabled={!onDeleteLora || lora.removable === false || deletingItem === deleteKey}
            onClick={() => deleteLora(lora)}
            type="button"
          >
            {lora.removable === false ? "Protected" : deletingItem === deleteKey ? "Deleting" : "Delete"}
          </button>
        </span>
        {!isEditing && (keywords.length || notes) ? (
          <div className="lora-row-meta">
            {keywords.length ? (
              <div className="lora-keywords">
                {keywords.map((keyword) => (
                  <span className="kw-chip" key={keyword}>
                    {keyword}
                  </span>
                ))}
              </div>
            ) : null}
            {notes ? <p className="lora-notes">{notes}</p> : null}
          </div>
        ) : null}
        {isEditing ? (
          <div className="lora-row-editor">
            <label>
              Trigger keywords
              <KeywordTagEditor
                disabled={savingLora}
                onChange={(triggerWords) => setLoraEditDraft((current) => ({ ...current, triggerWords }))}
                suggestions={loraEditSuggestions}
                value={loraEditDraft.triggerWords}
              />
            </label>
            <label>
              Notes
              <textarea
                disabled={savingLora}
                onChange={(event) => setLoraEditDraft((current) => ({ ...current, notes: event.target.value }))}
                placeholder="How to use this LoRA — keyword combinations, recommended weight, and other tips."
                rows={2}
                value={loraEditDraft.notes}
              />
            </label>
            <div className="lora-row-editor-actions">
              <button disabled={savingLora} onClick={() => saveEditLora(lora)} type="button">
                {savingLora ? "Saving…" : "Save"}
              </button>
              <button disabled={savingLora} onClick={cancelEditLora} type="button">
                Cancel
              </button>
            </div>
            {loraEditError ? <p className="inline-warning">{loraEditError}</p> : null}
          </div>
        ) : null}
        {downloadJob ? (
          <div className="lora-row-progress">
            <WorkerProgressCard job={downloadJob} onCancel={onCancelJob} onOpenQueue={onOpenQueue} />
          </div>
        ) : null}
      </article>
    );
  }

  // --- Tabbed interface derivation (epic 10309) ---
  const query = search.trim().toLowerCase();
  const hasSearch = query !== "";
  // When search is cleared but the transient Search tab is still nominally active, fall back
  // to the tab we came from (handleSearchChange normally does this; this guards other paths).
  const effectiveTab = activeTab === "search" && !hasSearch ? prevTab || "image" : activeTab;

  // Family filter applies to models via their LoRA-compatibility families (the same union the
  // dropdown is built from) and to LoRAs via matchesFamily — so a selected token never hides
  // everything. Search is case-insensitive over name/family/capability labels (models) and
  // name/family (LoRAs).
  // Derive every model tab, count, and search group in one memoized catalog pass.
  // Rendering the Recommended band and the full tab panel then shares the same
  // arrays instead of repeating type/search/family filters for each consumer.
  const { modelGroupsByType, modelTypeCounts, searchModelMatches } = useMemo(() => {
    const activeByType = Object.fromEntries(MODEL_TAB_TYPES.map(([type]) => [type, []]));
    const counts = Object.fromEntries(MODEL_TAB_TYPES.map(([type]) => [type, 0]));
    const searchMatches = [];
    for (const model of models) {
      if (counts[model.type] !== undefined) {
        counts[model.type] += 1;
      }
      const familyMatches =
        familyFilter === "all" || modelLoraFamilies(model).includes(familyFilter);
      const queryMatches =
        !hasSearch ||
        (model.name ?? model.id ?? "").toLowerCase().includes(query) ||
        (model.family ?? "").toLowerCase().includes(query) ||
        (Array.isArray(model.capabilities) ? model.capabilities : [])
          .map(capabilityLabel)
          .some((label) => label.toLowerCase().includes(query));
      if (!familyMatches || !queryMatches) {
        continue;
      }
      searchMatches.push(model);
      activeByType[model.type]?.push(model);
    }

    const groups = {};
    for (const [type] of MODEL_TAB_TYPES) {
      const recommended = [];
      const others = [];
      for (const model of activeByType[type]) {
        (isRecommendedModel(model) ? recommended : others).push(model);
      }
      const typeLabel = type;
      groups[type] = {
        activeModels: activeByType[type],
        recommended,
        others,
        othersHeading:
          recommended.length > 0
            ? `All ${typeLabel} models`
            : `${typeLabel.charAt(0).toUpperCase()}${typeLabel.slice(1)} models`,
      };
    }
    return {
      modelGroupsByType: groups,
      modelTypeCounts: counts,
      searchModelMatches: searchMatches,
    };
  }, [familyFilter, hasSearch, models, query]);
  const searchLoraMatches = useMemo(
    () =>
      visibleLoras.filter(
        (lora) =>
          !hasSearch ||
          (lora.name ?? lora.id ?? "").toLowerCase().includes(query) ||
          (lora.family ?? "").toLowerCase().includes(query),
      ),
    [hasSearch, query, visibleLoras],
  );
  const searchTotalCount = searchModelMatches.length + searchLoraMatches.length;

  // Tab definitions — counts are TOTALS per type (not the filtered view). The transient Search
  // Results tab is appended only while a search is active; its badge is the live match count.
  const tabDefs = useMemo(() => {
    const definitions = [
      ...MODEL_TAB_TYPES.map(([type, label]) => [type, label, modelTypeCounts[type]]),
      ["lora", "LoRAs", loras.length],
    ];
    if (hasSearch) {
      definitions.push(["search", "⌕ Search Results", searchTotalCount]);
    }
    return definitions;
  }, [hasSearch, loras.length, modelTypeCounts, searchTotalCount]);

  const isSearchTab = effectiveTab === "search";
  const isLoraTab = effectiveTab === "lora";
  const isModelTab = !isSearchTab && !isLoraTab;

  // A model type panel: an accent Recommended band (when any recommended match) over an
  // "All {type} models" grid of the rest. Both filtered by the active search + family.
  // Recommended picks, rendered flush inside the work-panel (Page-Frame standard:
  // curated getting-started content belongs to the action, not floating on the
  // canvas). No accent-band card here — the work-panel is already the one card.
  function renderRecommendedPicks(type) {
    const { recommended } = modelGroupsByType[type];
    if (!recommended.length) {
      return null;
    }
    return (
      <div className="models-recommended">
        <div className="models-accent-band-head">
          <span className="models-accent-dot" aria-hidden="true" />
          <p className="eyebrow">Recommended</p>
          <span className="models-accent-band-count">{recommended.length}</span>
          <span className="models-accent-band-caption">Curated getting-started picks</span>
        </div>
        <div className="models-card-grid">{recommended.map((model) => renderModelCard(model))}</div>
      </div>
    );
  }

  // Base-checkpoint import affordance (epic 14015, sc-14020). Surfaced on the Image Models
  // tab (sc-14069) rather than the LoRAs tab where the sc-7080 scaffold first placed it — a
  // base checkpoint IS an image model (a Krea 2 DiT today), so this is where a user looks for
  // it. Gated on MODEL_IMPORT_ENABLED (mirrors the backend kill-switch S0d opened); the
  // section also surfaces any in-flight model-import progress. Factored out of the tab body so
  // it can mount under any relevant model-type tab once more base-checkpoint types become
  // importable — image is the only relevant tab today (the Type selector is image-locked).
  function renderModelImportPanel() {
    if (!MODEL_IMPORT_ENABLED && pendingModelImportJobs.length === 0) {
      return null;
    }
    return (
      <section className="model-import-panel-section">
        {MODEL_IMPORT_ENABLED && (
          <form className="models-accent-band models-import-panel" aria-label="Import model" onSubmit={importModel}>
            <div className="models-accent-band-head">
              <span className="models-accent-dot" aria-hidden="true" />
              <p className="eyebrow">Import model</p>
              <span className="models-accent-band-caption">
                Point at a base checkpoint file — auto-detects family (Krea 2 today)
              </span>
            </div>
            <div className="segmented-control compact-segment" aria-label="Model import source">
              <button
                className={modelImportForm.mode === "url" ? "active" : ""}
                disabled={importingModel}
                onClick={() => setModelImportForm((current) => ({ ...current, mode: "url" }))}
                type="button"
              >
                URL
              </button>
              <button
                className={modelImportForm.mode === "file" ? "active" : ""}
                disabled={importingModel}
                onClick={() => setModelImportForm((current) => ({ ...current, mode: "file" }))}
                type="button"
              >
                Upload
              </button>
            </div>
            <div className="models-import-grid">
              <label>
                Type
                {/* Base-checkpoint import only produces image models today (a Krea 2 DiT).
                    `queue_model_import_job` (models.rs) writes this type verbatim into the
                    user manifest and never reconciles it against the detected family, so
                    offering video/audio/utility here would let an image checkpoint be
                    mis-typed. Constrain to Image (disabled) until more base-checkpoint types
                    are importable, then drop the image-only filter to restore the full
                    MODEL_TYPE_OPTIONS selector (sc-14020). */}
                <select disabled value={modelImportForm.type} aria-readonly="true">
                  {MODEL_TYPE_OPTIONS.filter((option) => option.value === "image").map((option) => (
                    <option key={option.value} value={option.value}>
                      {option.label}
                    </option>
                  ))}
                </select>
              </label>
              <label>
                Family
                <select
                  disabled={importingModel || !families.length}
                  onChange={(event) => setModelImportForm((current) => ({ ...current, family: event.target.value }))}
                  value={modelImportForm.family}
                >
                  {families.length ? (
                    <>
                      <option value="">Auto-detect</option>
                      {families.map((family) => (
                        <option key={family} value={family}>
                          {family}
                        </option>
                      ))}
                    </>
                  ) : (
                    <option value="">No known families</option>
                  )}
                </select>
              </label>
              {isModelFileImport ? (
                <label>
                  Model File
                  <span className="file-picker-row">
                    <span className="file-upload-button">
                      Choose
                      <input
                        accept=".safetensors,.ckpt,.pt,.bin"
                        disabled={importingModel}
                        key={modelFileInputKey}
                        onChange={(event) => setModelImportForm((current) => ({ ...current, file: event.target.files?.[0] ?? null }))}
                        type="file"
                      />
                    </span>
                    <span className="selected-file-name">{modelImportForm.file?.name ?? "No file selected"}</span>
                  </span>
                </label>
              ) : (
                <label>
                  Source URL
                  <input
                    disabled={importingModel}
                    onChange={(event) => setModelImportForm((current) => ({ ...current, sourceUrl: event.target.value }))}
                    placeholder="https://..."
                    value={modelImportForm.sourceUrl}
                  />
                </label>
              )}
              <label>
                Name
                <input
                  disabled={importingModel}
                  onChange={(event) => setModelImportForm((current) => ({ ...current, name: event.target.value }))}
                  placeholder="Optional"
                  value={modelImportForm.name}
                />
              </label>
              <button disabled={modelImportDisabled} type="submit">
                {importingModel ? (isModelFileImport ? "Uploading" : "Queueing...") : "Queue Import"}
              </button>
            </div>
            {modelImportMessage.text ? <p className={modelImportMessage.tone === "success" ? "inline-success" : "inline-warning"}>{modelImportMessage.text}</p> : null}
          </form>
        )}
        {pendingModelImportJobs.length ? (
          <div className="lora-import-progress">
            <strong>Model imports in progress</strong>
            <div className="local-job-stack">
              {pendingModelImportJobs.map((job) => (
                <WorkerProgressCard job={job} key={job.id} onCancel={onCancelJob} onOpenQueue={onOpenQueue} />
              ))}
            </div>
          </div>
        ) : null}
      </section>
    );
  }

  // The catalog grid on the canvas: "All {type} models" (the non-recommended
  // rest). Recommended picks render separately in the work-panel above.
  function renderModelTabPanel(type) {
    const { recommended, others, othersHeading } = modelGroupsByType[type];
    return (
      <div className="models-tab-panel">
        {type === "image" ? renderModelImportPanel() : null}
        {others.length ? (
          <div className="models-section">
            <div className="models-section-heading">
              <h3>{othersHeading}</h3>
              <span>{others.length}</span>
            </div>
            <div className="models-card-grid">{others.map((model) => renderModelCard(model))}</div>
          </div>
        ) : null}
        {recommended.length === 0 && others.length === 0 ? (
          <div className="models-empty">No models match your search.</div>
        ) : null}
      </div>
    );
  }

  // The transient Search Results tab: cross-type model matches grouped by type, then a LoRA
  // section using the shared row. Only non-empty groups render.
  function renderSearchTabPanel() {
    const groups = MODEL_TAB_TYPES
      .map(([type, label]) => ({ label, items: modelGroupsByType[type].activeModels }))
      .filter((group) => group.items.length > 0);
    const isEmpty = groups.length === 0 && searchLoraMatches.length === 0;
    return (
      <div className="models-tab-panel">
        {groups.map((group) => (
          <div className="models-section" key={group.label}>
            <div className="models-section-heading">
              <h3>{group.label}</h3>
              <span>{group.items.length}</span>
            </div>
            <div className="models-card-grid">{group.items.map((model) => renderModelCard(model))}</div>
          </div>
        ))}
        {searchLoraMatches.length ? (
          <div className="models-section">
            <div className="models-section-heading">
              <h3>LoRAs</h3>
              <span>{searchLoraMatches.length}</span>
            </div>
            <div className="lora-list">{searchLoraMatches.map((lora) => renderLoraRow(lora))}</div>
          </div>
        ) : null}
        {isEmpty ? <div className="models-empty">No models or LoRAs match &ldquo;{search}&rdquo;.</div> : null}
      </div>
    );
  }

  return (
    <section className="page-frame models-surface">
      <WorkPanel>
        <div className="models-tabbar">
        <div className="mode-tabs" role="tablist" aria-label="Model type">
          {tabDefs.map(([key, label, count]) => {
            const active = effectiveTab === key;
            return (
              <button
                key={key}
                type="button"
                role="tab"
                aria-selected={active}
                className={active ? "mode-tab active" : "mode-tab"}
                onClick={() => setActiveTab(key)}
              >
                {label}
                <span className="models-tab-count">{count}</span>
              </button>
            );
          })}
        </div>
        <div className="models-tabbar-controls">
          <div className="models-search">
            <span className="models-search-glyph" aria-hidden="true">
              ⌕
            </span>
            <input
              type="search"
              value={search}
              onChange={handleSearchChange}
              placeholder="Search models"
              aria-label="Search models"
            />
          </div>
          <select
            className="models-family-select"
            value={familyFilter}
            onChange={(event) => setFamilyFilter(event.target.value)}
            aria-label="Filter by family"
          >
            <option value="all">All families</option>
            {families.map((family) => (
              <option key={family} value={family}>
                {family}
              </option>
            ))}
          </select>
        </div>
        </div>
        {isModelTab ? renderRecommendedPicks(effectiveTab) : null}
      </WorkPanel>

      {/* Live region so a delete / local-copy outcome is announced, not only painted — the
          keep/remove buttons sit far down a long card and their result lands up here. */}
      {deleteMessage.text ? (
        <p
          aria-live="polite"
          className={deleteMessage.tone === "success" ? "inline-success" : "inline-warning"}
          role="status"
        >
          {deleteMessage.text}
        </p>
      ) : null}

      {/* Stated once for the whole screen, because the read that failed was one read for the whole
          screen. Says what is unavailable and why, without implying anything about what is or is
          not cached — that is precisely what could not be determined. */}
      {cacheError ? (
        <p className="inline-warning">
          Local model copies can’t be shown right now, so their controls are hidden: {cacheError}
        </p>
      ) : null}

      {isModelTab ? renderModelTabPanel(effectiveTab) : null}
      {isSearchTab ? renderSearchTabPanel() : null}

      {isLoraTab ? (
        <div className="models-tab-panel">
          {/* Import LoRA — same accent-band treatment as the Recommended band. */}
          <form className="models-accent-band models-import-panel" aria-label="Import LoRA" onSubmit={importLora}>
            <div className="models-accent-band-head">
              <span className="models-accent-dot" aria-hidden="true" />
              <p className="eyebrow">Import LoRA</p>
              <span className="models-accent-band-caption">Add a LoRA from a URL or file — auto-detects family</span>
            </div>
            <div className="segmented-control compact-segment" aria-label="LoRA import source">
              <button
                className={importForm.mode === "url" ? "active" : ""}
                disabled={importingLora}
                onClick={() => setImportForm((current) => ({ ...current, mode: "url" }))}
                type="button"
              >
                URL
              </button>
              <button
                className={importForm.mode === "file" ? "active" : ""}
                disabled={importingLora}
                onClick={() => setImportForm((current) => ({ ...current, mode: "file" }))}
                type="button"
              >
                Upload
              </button>
            </div>
            <div className="models-import-grid">
              <label>
                Scope
                <select
                  disabled={importingLora}
                  onChange={(event) => setImportForm((current) => ({ ...current, scope: event.target.value }))}
                  value={importForm.scope}
                >
                  <option value="global">Global</option>
                  <option disabled={!activeProject} value="project">
                    Project
                  </option>
                </select>
              </label>
              <label>
                Family
                <select
                  disabled={importingLora || !families.length}
                  onChange={(event) => setImportForm((current) => ({ ...current, family: event.target.value }))}
                  value={importForm.family}
                >
                  {families.length ? (
                    <>
                      <option value="">Auto-detect</option>
                      {families.map((family) => (
                        <option key={family} value={family}>
                          {family}
                        </option>
                      ))}
                    </>
                  ) : (
                    <option value="">No model families</option>
                  )}
                </select>
              </label>
              {showBaseModelSelect ? (
                <label>
                  Base model
                  <select
                    disabled={importingLora}
                    onChange={(event) => setImportForm((current) => ({ ...current, baseModel: event.target.value }))}
                    value={importForm.baseModel}
                  >
                    <option value="">Auto / unspecified</option>
                    {wanBaseModelOptions.map((model) => (
                      <option key={model.id} value={model.id}>
                        {model.name ?? model.id}
                      </option>
                    ))}
                  </select>
                </label>
              ) : null}
              {isFileImport ? (
                <>
                  <label>
                    LoRA File
                    <span className="file-picker-row">
                      <span className="file-upload-button">
                        Choose
                        <input
                          accept=".safetensors,.ckpt,.pt,.bin"
                          disabled={importingLora}
                          key={fileInputKey}
                          onChange={(event) => setImportForm((current) => ({ ...current, file: event.target.files?.[0] ?? null }))}
                          type="file"
                        />
                      </span>
                      <span className="selected-file-name">{importForm.file?.name ?? "No file selected"}</span>
                    </span>
                  </label>
                  {showSecondaryFileSlot ? (
                    <label>
                      Low-noise expert (Wan A14B MoE)
                      <span className="file-picker-row">
                        <span className="file-upload-button">
                          Choose
                          <input
                            accept=".safetensors,.ckpt,.pt,.bin"
                            disabled={importingLora}
                            key={`secondary-${fileInputKey}`}
                            onChange={(event) => setImportForm((current) => ({ ...current, secondaryFile: event.target.files?.[0] ?? null }))}
                            type="file"
                          />
                        </span>
                        <span className="selected-file-name">{importForm.secondaryFile?.name ?? "No file selected"}</span>
                      </span>
                    </label>
                  ) : null}
                </>
              ) : (
                <label>
                  Source URL
                  <input
                    disabled={importingLora}
                    onChange={(event) => setImportForm((current) => ({ ...current, sourceUrl: event.target.value }))}
                    placeholder="https://..."
                    value={importForm.sourceUrl}
                  />
                </label>
              )}
              <label>
                Name
                <input
                  disabled={importingLora}
                  onChange={(event) => setImportForm((current) => ({ ...current, name: event.target.value }))}
                  placeholder="Optional"
                  value={importForm.name}
                />
              </label>
              <button disabled={importDisabled} type="submit">
                {importingLora ? (isFileImport ? "Uploading" : "Queueing...") : "Queue Import"}
              </button>
            </div>
            <div className="lora-import-metadata">
              <label>
                Trigger keywords
                <KeywordTagEditor
                  disabled={importingLora}
                  onChange={(triggerKeywords) =>
                    setImportForm((current) => ({ ...current, triggerKeywords }))
                  }
                  placeholder="e.g. sksStyle — press Enter or comma to add"
                  value={importForm.triggerKeywords}
                />
              </label>
              <label>
                Notes
                <textarea
                  disabled={importingLora}
                  onChange={(event) => setImportForm((current) => ({ ...current, notes: event.target.value }))}
                  placeholder="How to use this LoRA — keyword combinations, recommended weight, and other tips."
                  rows={2}
                  value={importForm.notes}
                />
              </label>
            </div>
            {showSecondaryFileSlot ? (
              <p className="helper-copy">
                Wan A14B is a two-expert model. Upload both the high-noise file and the low-noise expert so each expert
                gets its own weights.
              </p>
            ) : null}
            {moeMissingSecondary ? (
              <p className="inline-warning">
                No low-noise expert selected — this LoRA will load into the high-noise expert only, leaving the
                low-noise expert un-adapted.
              </p>
            ) : null}
            {importForm.scope === "project" && !activeProject ? <p className="helper-copy">Open a project before importing a project LoRA.</p> : null}
            {importMessage.text ? <p className={importMessage.tone === "success" ? "inline-success" : "inline-warning"}>{importMessage.text}</p> : null}
          </form>
          {localLoraImportJobs.length ? (
            <div className="lora-import-progress">
              <strong>LoRA imports in progress</strong>
              <div className="local-job-stack">
                {localLoraImportJobs.map((job) => (
                  <WorkerProgressCard job={job} key={job.id} onCancel={onCancelJob} onOpenQueue={onOpenQueue} />
                ))}
              </div>
            </div>
          ) : null}
          {hiddenImportCount ? <p className="helper-copy">{hiddenImportCount} LoRA import{hiddenImportCount === 1 ? " is" : "s are"} hidden by this family filter.</p> : null}

          {builtinLoras.length ? (
            <div className="models-section">
              <div className="models-section-heading">
                <h3>Built-In LoRAs</h3>
                <span>{builtinLoras.length}</span>
              </div>
              <div className="lora-list">{builtinLoras.map((lora) => renderLoraRow(lora))}</div>
            </div>
          ) : null}

          <div className="models-section">
            <div className="models-section-heading">
              <h3>User LoRAs</h3>
              <span>{userLoras.length}</span>
            </div>
            {userLoras.length ? (
              <div className="models-subgroups">
                {loraGroups.map((group) => (
                  <div className="models-subsection" key={group.family}>
                    <div className="models-section-heading">
                      <h4>{group.family === "compatible" ? "Other / compatible" : group.family}</h4>
                      <span>{group.items.length}</span>
                    </div>
                    <div className="lora-list">{group.items.map((lora) => renderLoraRow(lora))}</div>
                  </div>
                ))}
              </div>
            ) : localLoraImportJobs.length ? null : loras.length && familyFilter !== "all" ? (
              <div className="models-empty">No user LoRAs match {familyFilter}.</div>
            ) : (
              <div className="models-empty">No user LoRAs yet — import one above.</div>
            )}
          </div>
        </div>
      ) : null}
    </section>
  );
}
