import React, { useEffect, useMemo, useState } from "react";
import { AssetThumbnail, assetCanRenderAsImage, assetCanRenderAsVideo } from "./assetMedia.jsx";
import { assetMatchesCharacter } from "../characterMembership.js";
import { Modal } from "./Modal.jsx";

const categoryOptions = [
  ["all", "All"],
  ["image", "Images"],
  ["video", "Video"],
  ["upload", "Uploads"],
  ["render", "Renders"],
];

const sourceImageTabs = [
  ["assets", "Assets"],
  ["upload", "File Upload"],
  ["character", "Character"],
];

// sc-6042: Character Assets "Import" dialog tabs. Images/Videos pick from the
// Project library; Upload brings files in from the local computer. All three add
// the chosen media to the character's asset library.
const characterImportTabs = [
  ["images", "Images"],
  ["videos", "Videos"],
  ["upload", "Upload"],
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

function assetIdentity(asset) {
  const id = String(asset?.id ?? "");
  return id.length <= 8 ? id : `...${id.slice(-6)}`;
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

function assetSearchIndex(assets) {
  return new Map(assets.map((asset) => [asset.id, searchableText(asset)]));
}

function normalizeSelection(ids, assets, multiple) {
  const available = new Set(assets.map((asset) => asset.id));
  const kept = ids.filter((id) => available.has(id));
  return multiple ? kept : kept.slice(0, 1);
}

function projectMatches(asset, projectId) {
  return Boolean(projectId) && asset?.projectId === projectId;
}

function activeProjectImageAsset(asset, projectId) {
  return (
    projectMatches(asset, projectId) &&
    (assetCanRenderAsImage(asset) || asset?.type === "frame") &&
    !asset?.status?.trashed &&
    !asset?.status?.rejected
  );
}

function activeProjectVideoAsset(asset, projectId) {
  return (
    projectMatches(asset, projectId) &&
    assetCanRenderAsVideo(asset) &&
    !asset?.status?.trashed &&
    !asset?.status?.rejected
  );
}

function filterPickerAssets(assets, query, searchIndex) {
  const needle = query.trim().toLowerCase();
  return assets.filter((asset) => !needle || searchIndex.get(asset.id)?.includes(needle));
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
            <small title={asset.id}>
              {typeLabel(asset)} | {compactDate(asset.createdAt ?? asset.updatedAt)} | ID {assetIdentity(asset)}
            </small>
          </span>
        </div>
      ))}
    </div>
  );
}

export function ImageEditSourcePickerField({
  assets,
  buttonLabel = "Select image",
  characters = [],
  emptyLabel = "No source image selected",
  importAsset,
  label = "Source image",
  onChange,
  projectId,
  value = "",
}) {
  const [open, setOpen] = useState(false);
  const selectableAssets = useMemo(
    () => assets.filter((asset) => activeProjectImageAsset(asset, projectId)),
    [assets, projectId],
  );
  const selectedAsset = selectableAssets.find((asset) => asset.id === value) ?? assets.find((asset) => asset.id === value);

  function confirm(id) {
    onChange(id ?? "");
    setOpen(false);
  }

  return (
    <div className="asset-picker-field">
      <div className="asset-picker-head">
        <span className="asset-picker-label">{label}</span>
        <button aria-haspopup="dialog" onClick={() => setOpen(true)} type="button">
          {selectedAsset ? "Change" : buttonLabel}
        </button>
      </div>
      <AssetPreviewChips assets={selectedAsset ? [selectedAsset] : []} emptyLabel={emptyLabel} />
      {open ? (
        <ImageEditSourcePickerModal
          assets={selectableAssets}
          characters={characters}
          importAsset={importAsset}
          initialSelectedId={value}
          onCancel={() => setOpen(false)}
          onConfirm={confirm}
        />
      ) : null}
    </div>
  );
}

function ImageEditSourcePickerModal({ assets, characters, importAsset, initialSelectedId, onCancel, onConfirm }) {
  const [tab, setTab] = useState("assets");
  const [query, setQuery] = useState("");
  const [selectedId, setSelectedId] = useState(() => (assets.some((asset) => asset.id === initialSelectedId) ? initialSelectedId : ""));
  const [characterId, setCharacterId] = useState(characters[0]?.id ?? "");
  const [dragActive, setDragActive] = useState(false);
  const [uploading, setUploading] = useState(false);
  const [uploadError, setUploadError] = useState("");

  useEffect(() => {
    setSelectedId((id) => (id && assets.some((asset) => asset.id === id) ? id : ""));
  }, [assets]);

  useEffect(() => {
    if (characterId && characters.some((character) => character.id === characterId)) {
      return;
    }
    setCharacterId(characters[0]?.id ?? "");
  }, [characters, characterId]);

  const selectedCharacter = characters.find((character) => character.id === characterId) ?? null;
  const characterAssets = useMemo(
    () => assets.filter((asset) => assetMatchesCharacter(asset, characterId, selectedCharacter)),
    [assets, characterId, selectedCharacter],
  );
  const allCharacterAssetCount = useMemo(
    () =>
      assets.filter((asset) =>
        characters.some((character) => assetMatchesCharacter(asset, character.id, character)),
      ).length,
    [assets, characters],
  );
  const tabAssets = tab === "character" ? characterAssets : assets;
  const searchIndex = useMemo(() => assetSearchIndex(tabAssets), [tabAssets]);
  const visibleAssets = useMemo(() => filterPickerAssets(tabAssets, query, searchIndex), [tabAssets, query, searchIndex]);

  async function handleUpload(file) {
    if (!file || uploading) {
      return;
    }
    if (!importAsset) {
      setUploadError("File upload is unavailable in this context.");
      return;
    }
    setUploading(true);
    setUploadError("");
    try {
      const imported = await importAsset(file, { throwOnError: true });
      if (imported?.id) {
        onConfirm(imported.id);
      }
    } catch (err) {
      setUploadError(err.message);
    } finally {
      setUploading(false);
    }
  }

  function handleDrop(event) {
    event.preventDefault();
    setDragActive(false);
    handleUpload(event.dataTransfer?.files?.[0] ?? null);
  }

  function switchTab(nextTab) {
    setTab(nextTab);
    setQuery("");
    setUploadError("");
  }

  function renderAssetGrid(emptyLabel) {
    return (
      <div aria-multiselectable={undefined} className="asset-picker-grid" role="listbox">
        {visibleAssets.map((asset) => {
          const selected = selectedId === asset.id;
          const status = statusLabel(asset);
          return (
            <button
              aria-selected={selected}
              className={selected ? "asset-picker-card selected" : "asset-picker-card"}
              key={asset.id}
              onClick={() => setSelectedId(asset.id)}
              onDoubleClick={() => onConfirm(asset.id)}
              role="option"
              type="button"
            >
              <AssetThumbnail asset={asset} />
              <span className="asset-picker-card-copy">
                <strong>{titleFor(asset)}</strong>
                <small>
                  {typeLabel(asset)} | {sourceLabel(asset)}
                </small>
                <small title={asset.id}>
                  {compactDate(asset.createdAt ?? asset.updatedAt)} | ID {assetIdentity(asset)}
                  {status ? ` | ${status}` : ""}
                </small>
              </span>
            </button>
          );
        })}
        {visibleAssets.length ? null : <div className="empty-panel compact-panel">{emptyLabel}</div>}
      </div>
    );
  }

  return (
    <Modal className="asset-picker-modal image-edit-source-modal" labelledBy="asset-picker-title" onClose={onCancel}>
      <header className="asset-picker-modal-head">
        <div>
          <p className="eyebrow">Image Edit</p>
          <h2 id="asset-picker-title">Choose source image</h2>
        </div>
        <button className="modal-close" onClick={onCancel} type="button">
          Close
        </button>
      </header>

      <div className="asset-picker-toolbar">
        <div className="segmented-control compact-segment" role="tablist" aria-label="Source image source">
          {sourceImageTabs.map(([key, label]) => (
            <button
              aria-selected={tab === key}
              className={tab === key ? "active" : ""}
              key={key}
              onClick={() => switchTab(key)}
              role="tab"
              type="button"
            >
              {label}
              {key === "assets" ? <span>{assets.length}</span> : null}
              {key === "character" ? <span>{allCharacterAssetCount}</span> : null}
            </button>
          ))}
        </div>
        {tab === "upload" ? null : (
          <input
            aria-label="Search source images"
            onChange={(event) => setQuery(event.target.value)}
            placeholder={tab === "character" ? "Search character assets" : "Search project assets"}
            value={query}
          />
        )}
      </div>

      {tab === "assets" ? renderAssetGrid("No active project images match this view") : null}

      {tab === "upload" ? (
        <div
          className={dragActive ? "dataset-add-dropzone active" : "dataset-add-dropzone"}
          onDragLeave={() => setDragActive(false)}
          onDragOver={(event) => {
            event.preventDefault();
            setDragActive(true);
          }}
          onDrop={handleDrop}
        >
          <p>{uploading ? "Importing image..." : "Drop an image here, or"}</p>
          <label className="file-upload-button">
            <input
              accept="image/*"
              disabled={uploading}
              onChange={(event) => {
                handleUpload(event.target.files?.[0] ?? null);
                event.target.value = "";
              }}
              type="file"
            />
            {uploading ? "Importing" : "Browse files"}
          </label>
          {uploadError ? <p className="inline-warning">{uploadError}</p> : null}
        </div>
      ) : null}

      {tab === "character" ? (
        <div className="dataset-add-character">
          <label>
            Character
            <select aria-label="Character" onChange={(event) => setCharacterId(event.target.value)} value={characterId}>
              {characters.length ? null : <option value="">No characters yet</option>}
              {characters.map((character) => (
                <option key={character.id} value={character.id}>
                  {character.name ?? character.id}
                </option>
              ))}
            </select>
          </label>
          {renderAssetGrid(characterId ? "No active images for this character" : "Select a character")}
        </div>
      ) : null}

      <footer className="asset-picker-footer">
        <span>{selectedId ? "1 selected" : tab === "upload" ? "Upload an image to use it" : "No selection"}</span>
        <div className="detail-actions">
          <button onClick={onCancel} type="button">
            Cancel
          </button>
          <button disabled={!selectedId} onClick={() => onConfirm(selectedId)} type="button">
            Use Selection
          </button>
        </div>
      </footer>
    </Modal>
  );
}

export function AssetPickerField({
  assets,
  buttonLabel = "Select",
  emptyLabel,
  label,
  changeLabel = "Change",
  multiple = false,
  onChange,
  // showCategories=false drops the All/Images/Video/Uploads/Renders segmented
  // control and renders just a scoped multi-select grid (sc-6042). Used by the
  // character reference picker, which is already scoped to one character's
  // assets, so the category tabs were redundant/empty noise there.
  showCategories = true,
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
          {selectedAssets.length ? changeLabel : buttonLabel}
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
          showCategories={showCategories}
          title={label}
        />
      ) : null}
    </div>
  );
}

export function AssetPickerModal({
  assets,
  initialSelectedIds,
  multiple = false,
  onCancel,
  onConfirm,
  showCategories = true,
  title = "Select assets",
}) {
  const [category, setCategory] = useState("all");
  const [query, setQuery] = useState("");
  const [selectedIds, setSelectedIds] = useState(() => normalizeSelection(initialSelectedIds, assets, multiple));

  useEffect(() => {
    setSelectedIds((ids) => normalizeSelection(ids, assets, multiple));
  }, [assets, multiple]);

  const categoryCounts = useMemo(() => {
    return Object.fromEntries(categoryOptions.map(([key]) => [key, assets.filter((asset) => categoryMatches(asset, key)).length]));
  }, [assets]);

  const searchIndex = useMemo(() => assetSearchIndex(assets), [assets]);

  // With the category control hidden the grid is unfiltered (the caller already
  // scoped `assets`), so pin the active category to "all".
  const activeCategory = showCategories ? category : "all";
  const visibleAssets = useMemo(() => {
    const needle = query.trim().toLowerCase();
    return assets.filter((asset) => categoryMatches(asset, activeCategory) && (!needle || searchIndex.get(asset.id)?.includes(needle)));
  }, [assets, activeCategory, query, searchIndex]);

  function toggleAsset(asset) {
    setSelectedIds((ids) => {
      if (multiple) {
        return ids.includes(asset.id) ? ids.filter((id) => id !== asset.id) : [...ids, asset.id];
      }
      return [asset.id];
    });
  }

  return (
    <Modal className="asset-picker-modal" labelledBy="asset-picker-title" onClose={onCancel}>
        <header className="asset-picker-modal-head">
          <div>
            <p className="eyebrow">Library</p>
            <h2 id="asset-picker-title">{title}</h2>
          </div>
          <button className="modal-close" onClick={onCancel} type="button">
            Close
          </button>
        </header>

        <div className="asset-picker-toolbar">
          {showCategories ? (
            <div className="segmented-control compact-segment" role="tablist" aria-label="Asset category">
              {categoryOptions.map(([key, label]) => (
                <button className={category === key ? "active" : ""} key={key} onClick={() => setCategory(key)} type="button">
                  {label} <span>{categoryCounts[key]}</span>
                </button>
              ))}
            </div>
          ) : null}
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
                  <small title={asset.id}>
                    {compactDate(asset.createdAt ?? asset.updatedAt)} | ID {assetIdentity(asset)}
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
    </Modal>
  );
}

// sc-6042: Import media into a character's asset library. Three tabs, all
// multi-select: Images and Videos pull from the Project asset library (of the
// matching type, excluding media already in this character's library); Upload
// brings file(s) in from the local computer. Project selections are attached via
// onImport(assetIds); uploads are imported into the project first (importAsset)
// then handed to the same onImport so both paths converge on one "attach" step.
export function CharacterImportDialog({
  assets = [],
  projectId,
  characterId,
  character,
  characterName,
  importAsset,
  onImport,
  onClose,
}) {
  const [tab, setTab] = useState("images");
  const [query, setQuery] = useState("");
  const [selectedIds, setSelectedIds] = useState([]);
  const [dragActive, setDragActive] = useState(false);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState("");

  const imageCandidates = useMemo(
    () =>
      assets.filter(
        (asset) => activeProjectImageAsset(asset, projectId) && !assetMatchesCharacter(asset, characterId, character),
      ),
    [assets, projectId, characterId, character],
  );
  const videoCandidates = useMemo(
    () =>
      assets.filter(
        (asset) => activeProjectVideoAsset(asset, projectId) && !assetMatchesCharacter(asset, characterId, character),
      ),
    [assets, projectId, characterId, character],
  );

  const tabAssets = tab === "videos" ? videoCandidates : imageCandidates;
  const searchIndex = useMemo(() => assetSearchIndex(tabAssets), [tabAssets]);
  const visibleAssets = useMemo(() => filterPickerAssets(tabAssets, query, searchIndex), [tabAssets, query, searchIndex]);

  function toggleAsset(id) {
    setSelectedIds((ids) => (ids.includes(id) ? ids.filter((value) => value !== id) : [...ids, id]));
  }

  function switchTab(next) {
    setTab(next);
    setQuery("");
    setSelectedIds([]);
    setError("");
  }

  async function attach(assetIds) {
    await onImport(assetIds);
    onClose();
  }

  async function commitSelection() {
    if (!selectedIds.length || busy) {
      return;
    }
    setBusy(true);
    setError("");
    try {
      await attach(selectedIds);
    } catch (err) {
      setError(err?.message ?? "Could not import the selection.");
      setBusy(false);
    }
  }

  async function handleUploadFiles(fileList) {
    const files = Array.from(fileList ?? []);
    if (!files.length || busy) {
      return;
    }
    if (!importAsset) {
      setError("File upload is unavailable in this context.");
      return;
    }
    setBusy(true);
    setError("");
    try {
      const importedIds = [];
      for (const file of files) {
        const imported = await importAsset(file, { throwOnError: true });
        if (imported?.id) {
          importedIds.push(imported.id);
        }
      }
      if (importedIds.length) {
        await attach(importedIds);
      } else {
        setError("Upload failed — try another file.");
        setBusy(false);
      }
    } catch (err) {
      setError(err?.message ?? "Upload failed — try another file.");
      setBusy(false);
    }
  }

  function handleDrop(event) {
    event.preventDefault();
    setDragActive(false);
    handleUploadFiles(event.dataTransfer?.files);
  }

  function renderGrid(emptyLabel) {
    return (
      <div aria-multiselectable className="asset-picker-grid" role="listbox">
        {visibleAssets.map((asset) => {
          const selected = selectedIds.includes(asset.id);
          const status = statusLabel(asset);
          return (
            <button
              aria-selected={selected}
              className={selected ? "asset-picker-card selected" : "asset-picker-card"}
              key={asset.id}
              onClick={() => toggleAsset(asset.id)}
              role="option"
              type="button"
            >
              <AssetThumbnail asset={asset} />
              <span className="asset-picker-card-copy">
                <strong>{titleFor(asset)}</strong>
                <small>
                  {typeLabel(asset)} | {sourceLabel(asset)}
                </small>
                <small title={asset.id}>
                  {compactDate(asset.createdAt ?? asset.updatedAt)} | ID {assetIdentity(asset)}
                  {status ? ` | ${status}` : ""}
                </small>
              </span>
            </button>
          );
        })}
        {visibleAssets.length ? null : <div className="empty-panel compact-panel">{emptyLabel}</div>}
      </div>
    );
  }

  return (
    <Modal className="asset-picker-modal character-import-modal" labelledBy="character-import-title" onClose={onClose}>
      <header className="asset-picker-modal-head">
        <div>
          <p className="eyebrow">Character assets</p>
          <h2 id="character-import-title">Import to {characterName || "character"}</h2>
        </div>
        <button className="modal-close" onClick={onClose} type="button">
          Close
        </button>
      </header>

      <div className="asset-picker-toolbar">
        <div className="segmented-control compact-segment" role="tablist" aria-label="Import source">
          {characterImportTabs.map(([key, label]) => (
            <button
              aria-selected={tab === key}
              className={tab === key ? "active" : ""}
              key={key}
              onClick={() => switchTab(key)}
              role="tab"
              type="button"
            >
              {label}
              {key === "images" ? <span>{imageCandidates.length}</span> : null}
              {key === "videos" ? <span>{videoCandidates.length}</span> : null}
            </button>
          ))}
        </div>
        {tab === "upload" ? null : (
          <input
            aria-label="Search project assets"
            onChange={(event) => setQuery(event.target.value)}
            placeholder="Search project assets"
            value={query}
          />
        )}
      </div>

      {tab === "images" ? renderGrid("No project images to import") : null}
      {tab === "videos" ? renderGrid("No project videos to import") : null}

      {tab === "upload" ? (
        <div
          className={dragActive ? "dataset-add-dropzone active" : "dataset-add-dropzone"}
          onDragLeave={() => setDragActive(false)}
          onDragOver={(event) => {
            event.preventDefault();
            setDragActive(true);
          }}
          onDrop={handleDrop}
        >
          <p>{busy ? "Importing…" : "Drop images or videos here, or"}</p>
          <label className="file-upload-button">
            <input
              accept="image/*,video/*"
              disabled={busy}
              multiple
              onChange={(event) => {
                handleUploadFiles(event.target.files);
                event.target.value = "";
              }}
              type="file"
            />
            {busy ? "Importing" : "Browse files"}
          </label>
        </div>
      ) : null}

      {error ? <p className="inline-warning">{error}</p> : null}

      {tab === "upload" ? null : (
        <footer className="asset-picker-footer">
          <span>{selectedIds.length ? `${selectedIds.length} selected` : "No selection"}</span>
          <div className="detail-actions">
            <button onClick={onClose} type="button">
              Cancel
            </button>
            <button className="primary-action" disabled={!selectedIds.length || busy} onClick={commitSelection} type="button">
              {busy ? "Importing…" : `Import ${selectedIds.length || ""}`.trim()}
            </button>
          </div>
        </footer>
      )}
    </Modal>
  );
}
