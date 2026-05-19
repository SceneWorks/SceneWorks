import React, { useEffect, useMemo, useRef, useState } from "react";
import { AssetThumbnail, assetCanRenderAsImage, assetCanRenderAsVideo } from "./assetMedia.jsx";

const categoryOptions = [
  ["all", "All"],
  ["image", "Images"],
  ["video", "Video"],
  ["upload", "Uploads"],
  ["render", "Renders"],
];

function compactDate(value) {
  if (!value) {
    return "No date";
  }
  const date = new Date(value);
  if (Number.isNaN(date.getTime())) {
    return "No date";
  }
  return new Intl.DateTimeFormat(undefined, {
    month: "short",
    day: "numeric",
    hour: "numeric",
    minute: "2-digit",
  }).format(date);
}

function titleFor(asset) {
  return asset?.displayName ?? asset?.title ?? asset?.name ?? asset?.id ?? "Untitled asset";
}

function typeLabel(asset) {
  if (assetCanRenderAsImage(asset)) {
    return asset.type === "frame" ? "Frame" : "Image";
  }
  if (assetCanRenderAsVideo(asset)) {
    return "Video";
  }
  return asset?.type ? asset.type[0].toUpperCase() + asset.type.slice(1) : "Asset";
}

function sourceLabel(asset) {
  if (asset?.type === "upload") {
    return "Upload";
  }
  if (asset?.recipe?.mode) {
    return asset.recipe.mode.replaceAll("_", " ");
  }
  if (asset?.generationSetId) {
    return "Render";
  }
  return asset?.recipe?.model ?? asset?.file?.mimeType ?? "Library";
}

function statusLabel(asset) {
  if (asset?.status?.trashed) {
    return "Trashed";
  }
  if (asset?.status?.rejected) {
    return "Rejected";
  }
  if (asset?.status?.favorite) {
    return "Favorite";
  }
  return asset?.status?.state ?? asset?.status?.label ?? "";
}

function categoryMatches(asset, category) {
  if (category === "all") {
    return true;
  }
  if (category === "image") {
    return assetCanRenderAsImage(asset);
  }
  if (category === "video") {
    return assetCanRenderAsVideo(asset);
  }
  if (category === "upload") {
    return asset?.type === "upload" || asset?.source === "upload" || asset?.file?.source === "upload";
  }
  if (category === "render") {
    return Boolean(asset?.generationSetId || asset?.recipe || asset?.type === "render");
  }
  return true;
}

function searchableText(asset) {
  return [
    titleFor(asset),
    asset?.id,
    asset?.type,
    asset?.file?.mimeType,
    asset?.recipe?.mode,
    asset?.recipe?.model,
    asset?.recipe?.prompt,
    statusLabel(asset),
  ]
    .filter(Boolean)
    .join(" ")
    .toLowerCase();
}

function normalizeSelection(ids, assets, multiple) {
  const available = new Set(assets.map((asset) => asset.id));
  const kept = ids.filter((id) => available.has(id));
  return multiple ? kept : kept.slice(0, 1);
}

export function AssetPreviewChips({ assets, emptyLabel = "No asset selected" }) {
  if (!assets.length) {
    return <div className="asset-picker-empty">{emptyLabel}</div>;
  }

  return (
    <div className="asset-preview-chips">
      {assets.map((asset) => (
        <div className="asset-preview-chip" key={asset.id}>
          <AssetThumbnail asset={asset} />
          <span>
            <strong>{titleFor(asset)}</strong>
            <small>
              {typeLabel(asset)} | {compactDate(asset.createdAt ?? asset.updatedAt)} | {String(asset.id).slice(-6)}
            </small>
          </span>
        </div>
      ))}
    </div>
  );
}

export function AssetPickerField({
  assets,
  buttonLabel = "Select",
  emptyLabel,
  label,
  multiple = false,
  onChange,
  value = "",
  values = [],
}) {
  const [open, setOpen] = useState(false);
  const selectedIds = multiple ? values : value ? [value] : [];
  const selectedAssets = selectedIds.map((id) => assets.find((asset) => asset.id === id)).filter(Boolean);

  function confirm(ids) {
    onChange(multiple ? ids : ids[0] ?? "");
    setOpen(false);
  }

  return (
    <div className="asset-picker-field">
      <div className="asset-picker-head">
        <span className="asset-picker-label">{label}</span>
        <button aria-haspopup="dialog" onClick={() => setOpen(true)} type="button">
          {selectedAssets.length ? "Change" : buttonLabel}
        </button>
      </div>
      <AssetPreviewChips assets={selectedAssets} emptyLabel={emptyLabel ?? (multiple ? "No assets selected" : "No asset selected")} />
      {open ? (
        <AssetPickerModal
          assets={assets}
          initialSelectedIds={selectedIds}
          multiple={multiple}
          onCancel={() => setOpen(false)}
          onConfirm={confirm}
          title={label}
        />
      ) : null}
    </div>
  );
}

export function AssetPickerModal({ assets, initialSelectedIds, multiple = false, onCancel, onConfirm, title = "Select assets" }) {
  const [category, setCategory] = useState("all");
  const [query, setQuery] = useState("");
  const [selectedIds, setSelectedIds] = useState(() => normalizeSelection(initialSelectedIds, assets, multiple));
  const dialogRef = useRef(null);

  useEffect(() => {
    setSelectedIds(normalizeSelection(initialSelectedIds, assets, multiple));
  }, [assets, initialSelectedIds, multiple]);

  useEffect(() => {
    dialogRef.current?.focus();
  }, []);

  useEffect(() => {
    function onKeyDown(event) {
      if (event.key === "Escape") {
        event.preventDefault();
        onCancel();
      }
    }
    window.addEventListener("keydown", onKeyDown);
    return () => window.removeEventListener("keydown", onKeyDown);
  }, [onCancel]);

  const categoryCounts = useMemo(() => {
    return Object.fromEntries(categoryOptions.map(([key]) => [key, assets.filter((asset) => categoryMatches(asset, key)).length]));
  }, [assets]);

  const visibleAssets = useMemo(() => {
    const needle = query.trim().toLowerCase();
    return assets.filter((asset) => categoryMatches(asset, category) && (!needle || searchableText(asset).includes(needle)));
  }, [assets, category, query]);

  function toggleAsset(asset) {
    setSelectedIds((ids) => {
      if (multiple) {
        return ids.includes(asset.id) ? ids.filter((id) => id !== asset.id) : [...ids, asset.id];
      }
      return [asset.id];
    });
  }

  return (
    <div className="modal-backdrop" onMouseDown={(event) => event.target === event.currentTarget && onCancel()}>
      <section
        aria-label={title}
        aria-modal="true"
        className="asset-picker-modal"
        onMouseDown={(event) => event.stopPropagation()}
        ref={dialogRef}
        role="dialog"
        tabIndex={-1}
      >
        <header className="asset-picker-modal-head">
          <div>
            <p className="eyebrow">{multiple ? "Choose assets" : "Choose asset"}</p>
            <h2>{title}</h2>
          </div>
          <button className="modal-close" onClick={onCancel} type="button">
            Close
          </button>
        </header>

        <div className="asset-picker-toolbar">
          <div className="segmented-control compact-segment" role="tablist" aria-label="Asset category">
            {categoryOptions.map(([key, label]) => (
              <button className={category === key ? "active" : ""} key={key} onClick={() => setCategory(key)} type="button">
                {label} <span>{categoryCounts[key]}</span>
              </button>
            ))}
          </div>
          <input
            aria-label="Search assets"
            onChange={(event) => setQuery(event.target.value)}
            placeholder="Search title, type, prompt"
            value={query}
          />
        </div>

        <div aria-multiselectable={multiple || undefined} className="asset-picker-grid" role="listbox">
          {visibleAssets.map((asset) => {
            const selected = selectedIds.includes(asset.id);
            const status = statusLabel(asset);
            return (
              <button
                aria-selected={selected}
                className={selected ? "asset-picker-card selected" : "asset-picker-card"}
                key={asset.id}
                onClick={() => toggleAsset(asset)}
                onDoubleClick={() => !multiple && onConfirm([asset.id])}
                role="option"
                type="button"
              >
                <AssetThumbnail asset={asset} />
                <span className="asset-picker-card-copy">
                  <strong>{titleFor(asset)}</strong>
                  <small>
                    {typeLabel(asset)} | {sourceLabel(asset)}
                  </small>
                  <small>
                    {compactDate(asset.createdAt ?? asset.updatedAt)} | ID {String(asset.id).slice(-6)}
                    {status ? ` | ${status}` : ""}
                  </small>
                </span>
              </button>
            );
          })}
          {visibleAssets.length ? null : <div className="empty-panel compact-panel">No assets match this view</div>}
        </div>

        <footer className="asset-picker-footer">
          <span>{selectedIds.length ? `${selectedIds.length} selected` : "No selection"}</span>
          <div className="detail-actions">
            <button onClick={onCancel} type="button">
              Cancel
            </button>
            <button disabled={!selectedIds.length} onClick={() => onConfirm(selectedIds)} type="button">
              Use Selection
            </button>
          </div>
        </footer>
      </section>
    </div>
  );
}
