import React, { useCallback, useRef, useState } from "react";
import { Icon } from "../components/Icons.jsx";
import { AssetThumbnail, assetUrl } from "../components/assetMedia.jsx";
import { resolveJobResultAssets } from "../jobResultAssets.js";
import { terminalStatuses } from "../constants.js";
import { STYLE_GROUPS } from "../data/styleCatalog.js";
import { useAppStatic } from "../context/AppContext.js";
import { SimpleJobCard } from "./SimpleJobCard.jsx";
import { useSimpleUi } from "./SimpleUiContext.js";

// Shared building blocks for the three Simple studios (design handoff). Each one is
// the design's control re-expressed over the app's real data — no mock state.

const STYLE_THUMB_BASE = "/style-thumbs";

// Style strip: "None" plus the eight authored Style Catalog groups (the README's
// "back this with the real styleCatalog groups"). Selecting a group stores the GROUP
// id, which styleTextForId resolves to that group's overall description — the same
// selection contract the advanced StylePicker uses, so a Simple run replays into the
// advanced picker unchanged.
export function StyleStrip({ value, onChange }) {
  const options = [{ id: null, name: "None" }, ...STYLE_GROUPS.map((g) => ({ id: g.id, name: g.name }))];
  return (
    <div>
      <span className="su-section-label">Style &amp; Presets</span>
      <div className="su-style-strip su-scroll" role="radiogroup" aria-label="Style">
        {options.map((option) => (
          <StyleTile
            active={(value ?? null) === option.id}
            key={option.id ?? "none"}
            name={option.name}
            onSelect={() => onChange(option.id)}
            styleId={option.id}
          />
        ))}
      </div>
    </div>
  );
}

function StyleTile({ active, name, onSelect, styleId }) {
  // Thumbnails are generated locally (scripts/generate-style-thumbnails.mjs) and are
  // git-ignored, so they may be absent at runtime — a 404 collapses to the neutral slot
  // exactly as the advanced picker does, never a broken image.
  const [failed, setFailed] = useState(false);
  return (
    <button
      aria-checked={active}
      className={active ? "su-style active" : "su-style"}
      onClick={onSelect}
      role="radio"
      type="button"
    >
      <span className="su-style-thumb">
        {styleId && !failed ? (
          <img alt="" loading="lazy" onError={() => setFailed(true)} src={`${STYLE_THUMB_BASE}/${styleId}.png`} />
        ) : null}
      </span>
      <span className="su-style-name">{name}</span>
    </button>
  );
}

// Reference / source-image tile. Armed state shows the design's "SET" badge plus a
// thumbnail and the asset's display name; tapping an armed tile clears it, tapping an
// empty one opens the picker sheet over the project's recent images.
export function ReferenceTile({ label, hint, asset, assets, onChange, required = false, disabled = false }) {
  const { openSheet, toast } = useSimpleUi();
  const pick = useCallback(() => {
    if (asset) {
      onChange(null);
      toast("Reference removed");
      return;
    }
    if (!assets.length) {
      toast("No images in this project yet");
      return;
    }
    openSheet({
      title: "Choose a reference",
      kind: "list",
      options: assets.map((item) => ({
        value: item.id,
        label: item.displayName ?? item.id,
        thumbnail: assetUrl(item),
      })),
      onSelect: (id) => {
        onChange(id);
        toast("Reference attached");
      },
    });
  }, [asset, assets, onChange, openSheet, toast]);

  return (
    <button
      className={asset && !disabled ? "su-tile active" : "su-tile"}
      disabled={disabled}
      onClick={pick}
      type="button"
    >
      <span className="su-tile-head">
        <Icon.Image size={15} />
        {label}
        {asset && !disabled ? <span className="su-set-badge">SET</span> : null}
      </span>
      {/* A disabled tile shows its REASON, never the armed thumbnail — an armed reference the
          model can't consume must not read as active. */}
      {asset && !disabled ? (
        <span className="su-tile-ref">
          <AssetThumbnail asset={asset} />
          <span>{asset.displayName ?? asset.id}</span>
        </span>
      ) : (
        <span className="su-tile-hint">{hint ?? (required ? "Required — tap to attach" : "Tap to attach an image")}</span>
      )}
    </button>
  );
}

// Inline refine panel (design): one "Rewrite prompt" button that replaces the prompt with
// the Anubis-8B rewrite and closes. Unlike the advanced RefinePromptControl there is no
// review step — that is the simplification the design asks for — but the failure paths it
// owns are preserved: a missing refiner model offers its download instead of a raw error.
export function RefinePanel({ blurb, busy, error, onRun, onDownloadModel, modelMissing }) {
  return (
    <div className="su-inline-panel">
      <p>{blurb}</p>
      {error ? (
        <p className="su-error">{error}</p>
      ) : null}
      {modelMissing && onDownloadModel ? (
        <button className="su-accent-btn" onClick={onDownloadModel} type="button">
          Download refinement model
        </button>
      ) : (
        <button
          className={busy ? "su-accent-btn busy" : "su-accent-btn"}
          disabled={busy}
          onClick={onRun}
          type="button"
        >
          {busy ? <span aria-hidden="true" className="su-spinner su-spinner--sm" /> : null}
          {busy ? "Refining…" : "Rewrite prompt"}
        </button>
      )}
    </div>
  );
}

// A settings-bar select that opens the shared picker sheet.
export function SheetSelect({ label, value, options, onSelect, title, kind = "list", disabled }) {
  const { openSheet } = useSimpleUi();
  return (
    <div className="su-field">
      <label htmlFor={undefined}>{label}</label>
      <button
        className="su-select"
        disabled={disabled || options.length === 0}
        onClick={() =>
          openSheet({
            title: title ?? label,
            kind,
            options,
            onSelect,
          })
        }
        type="button"
      >
        <span>{value}</span>
        <Icon.ChevDown size={12} />
      </button>
    </div>
  );
}

// `label` doubles as the visible field label and the group's accessible name. A caller
// that already has a heading above the row (Settings' "Default quality" card) passes only
// `ariaLabel`, so no empty <label> is emitted.
export function Chips({ label, ariaLabel, options, value, onChange, capped = false }) {
  return (
    <div className="su-field">
      {label ? <label>{label}</label> : null}
      <div
        aria-label={label || ariaLabel}
        className={capped ? "su-chips su-chips--capped" : "su-chips"}
        role="radiogroup"
      >
        {options.map((option) => (
          <button
            aria-checked={option.value === value}
            className={option.value === value ? "su-chip active" : "su-chip"}
            key={String(option.value)}
            onClick={() => onChange(option.value)}
            role="radio"
            title={option.title}
            type="button"
          >
            {option.label}
          </button>
        ))}
      </div>
    </div>
  );
}

// The newest run out of a studio's local job stack. The stack is ordered oldest→newest
// (buildLocalJobStack), so the last entry is the run the Results block reflects.
export function newestLocalJob(localJobs) {
  return localJobs.length ? localJobs[localJobs.length - 1] : null;
}

export function jobIsRunning(job) {
  return Boolean(job) && !terminalStatuses.has(job.status);
}


// The studio's live run strip: the SAME job card the Queue screen lists, rendered directly
// under Generate. That is where the user's attention already is, so progress and Cancel are
// in front of them instead of one navigation away — and because it is one component, the two
// surfaces cannot drift.
//
// It stays mounted after the run ends so the outcome (Done / the failure reason) is readable
// where the run was started; the results grid renders below it.
export function StudioRunStatus({ job }) {
  const { toast } = useSimpleUi();
  const { jobAction } = useAppStatic();
  if (!job) {
    return null;
  }
  return (
    <SimpleJobCard
      job={job}
      onCancel={async (target) => {
        await jobAction?.(target, "cancel");
        toast("Run canceled");
      }}
    />
  );
}

// Results block. Renders the newest run's outputs; while that run is still in flight it
// renders one placeholder tile per expected output so the grid doesn't pop in.
export function StudioResults({ job, assets, type, columns, meta, pendingCount = 1, wide = false }) {
  const { toast, openPreview } = useSimpleUi();
  const { updateAssetStatus } = useAppStatic();
  const results = job ? resolveJobResultAssets(job, assets, { type }) : [];
  const running = jobIsRunning(job);
  // Failure/outcome reporting lives in StudioRunStatus (the run strip above this), so this
  // block is purely the output grid — a failed run simply has none.
  if (!job || (!running && results.length === 0)) {
    return null;
  }
  const tiles = results.length ? results : Array.from({ length: pendingCount }, () => null);
  return (
    <div>
      <div className="su-results-head">
        <strong>{results.length === 1 ? "Result" : "Results"}</strong>
        {meta ? <span>{meta}</span> : null}
      </div>
      <div className="su-results-grid" style={{ "--su-result-cols": columns }}>
        {tiles.map((asset, index) => (
          <div
            className={wide ? "su-result su-result--wide" : "su-result"}
            key={asset?.id ?? `pending-${index}`}
          >
            {asset ? (
              <>
                <button
                  aria-label="Open preview"
                  className="su-result-open"
                  onClick={() => openPreview(asset)}
                  type="button"
                >
                  <AssetThumbnail asset={asset} />
                </button>
                <button
                  aria-label={asset.status?.favorite ? "Remove from favorites" : "Add to favorites"}
                  className={
                    asset.status?.favorite ? "su-result-action su-result-action--fav on" : "su-result-action su-result-action--fav"
                  }
                  onClick={() => {
                    updateAssetStatus(asset, { favorite: !asset.status?.favorite });
                    toast(asset.status?.favorite ? "Removed from favorites" : "Added to favorites");
                  }}
                  type="button"
                >
                  <Icon.Star filled={Boolean(asset.status?.favorite)} size={14} />
                </button>
                <DownloadButton asset={asset} />
              </>
            ) : (
              <span className="su-result-pending">
                {running ? <span aria-hidden="true" className="su-spinner" /> : null}
              </span>
            )}
          </div>
        ))}
      </div>
    </div>
  );
}

// Download uses a real anchor (the media URL already carries the auth ticket in
// remote-auth mode, sc-8810) rather than a fetch — the browser owns the save dialog.
export function DownloadButton({ asset, className = "su-result-action su-result-action--dl", label = null }) {
  const anchorRef = useRef(null);
  const { toast } = useSimpleUi();
  return (
    <>
      <a
        aria-hidden="true"
        download={asset.displayName ?? ""}
        href={assetUrl(asset)}
        ref={anchorRef}
        style={{ display: "none" }}
        tabIndex={-1}
      >
        download
      </a>
      <button
        aria-label="Download"
        className={className}
        onClick={() => {
          anchorRef.current?.click();
          toast("Downloading…");
        }}
        type="button"
      >
        <Icon.Download size={label ? 15 : 14} />
        {label}
      </button>
    </>
  );
}
