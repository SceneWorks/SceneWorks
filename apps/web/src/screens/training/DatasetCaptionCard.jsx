import React from "react";
import { AssetThumbnail } from "../../components/assetMedia.jsx";
import { imageAssetName } from "../../training/datasetHelpers.js";
import { SUBJECT_MASK_ACCEPT } from "../../training/subjectMasks.js";
import { EditReferenceRail } from "./EditReferenceRail.jsx";
import { ReadinessBadge, ReadinessFlagDetails } from "./DatasetDoctor.jsx";

const EMPTY_OBJECT = Object.freeze({});
const EMPTY_ARRAY = Object.freeze([]);
// Human label for the detected caption source (sc-2025) — read-only on the card. An auto caption
// made in a non-default mode (sc-24829) names that mode.
const captionModeLabels = { subjectOnly: "subject only", triggerOnly: "trigger words" };

function captionSourceLabel(source, mode) {
  if (source === "imported") {
    return "Imported";
  }
  if (source === "auto") {
    return captionModeLabels[mode] ? `Auto · ${captionModeLabels[mode]}` : "Auto";
  }
  return "Manual";
}

// Every changing value is item-scoped. Shared handlers retain identity and read
// the current screen/panel closures; React's default shallow comparison includes them.
export const DatasetCaptionCard = React.memo(function DatasetCaptionCard({
  asset, captionDraft, namedCaptionDraft, savedItem, readinessEntry, readinessPending,
  maskStatus, maskSrc, maskError, showEditPairs, assetsById, referenceCap, referenceIds,
  holdout, canToggleHoldout, canToggleItemAck, canUploadMask, preparedUploading,
  maskUploading, captioning, handlers,
}) {
  const disabled = asset.status?.rejected || asset.status?.trashed;
  const draft = captionDraft ?? EMPTY_OBJECT;
  const source = draft.source ?? "manual";
  const name = asset.displayName ?? imageAssetName(asset);

  const preparedPath = savedItem?.ltxPreparedBundlePath;
  // Named caption fields (sc-25685): a fresh `.json` import's, else the saved item's.
  const namedCaptions = namedCaptionDraft ?? savedItem?.captions;
  const namedCaptionFields =
    namedCaptions && typeof namedCaptions === "object" && !Array.isArray(namedCaptions)
      ? Object.keys(namedCaptions)
      : [];

  return (
    <article
      className={["training-caption-card", disabled ? "disabled" : ""].filter(Boolean).join(" ")}
      data-caption-id={asset.id}
    >
      <button className="training-caption-card-thumb" onClick={() => handlers.preview(asset)} type="button">
        <AssetThumbnail asset={asset} />
        {maskSrc ? (
          <img
            alt={`Subject mask for ${name}`}
            className="training-subject-mask-overlay"
            decoding="async"
            src={maskSrc}
          />
        ) : null}
        {/* Only flash pending on the first load — during an ack-triggered refetch the
            prior report's badges hold steady rather than blinking to "·". */}
        <ReadinessBadge entry={readinessEntry} loading={readinessPending} />
      </button>
      <div className="training-caption-card-body">
        <div className="training-caption-card-meta">
          <strong title={name}>{name}</strong>
          <span className={`training-caption-source source-${source}`}>{captionSourceLabel(source, draft.mode)}</span>
          {disabled ? (
            <span className="training-asset-badge">{asset.status?.trashed ? "Trashed" : "Rejected"}</span>
          ) : null}
          {maskStatus?.hasMask && maskStatus.empty ? (
            <span
              className="training-asset-badge"
              title="No person was detected — the mask is empty. Upload a replacement mask."
            >
              No subject in mask
            </span>
          ) : null}
          {maskStatus?.hasMask && maskStatus.source === "upload" ? (
            <span className="training-asset-badge">Mask uploaded</span>
          ) : null}
        </div>
        <ReadinessFlagDetails
          entry={readinessEntry}
          onToggle={
            readinessEntry && canToggleItemAck
              ? (check, dismissed) => handlers.toggleItemAck(readinessEntry, check, dismissed)
              : undefined
          }
        />
        {showEditPairs ? (
          <EditReferenceRail
            assetsById={assetsById}
            cap={referenceCap}
            itemName={name}
            onAdd={() => handlers.pickReferences({ selectionId: asset.id, name })}
            onChange={(ids) => handlers.setItemReferences(asset.id, ids)}
            referenceIds={referenceIds ?? EMPTY_ARRAY}
          />
        ) : null}
        <textarea
          aria-label={`Caption for ${name}`}
          className={[
            "training-caption-card-text",
            !String(draft.text ?? "").trim() ? "is-empty" : "",
          ]
            .filter(Boolean)
            .join(" ")}
          onChange={(event) => handlers.updateCaption(asset.id, event.target.value)}
          placeholder={
            showEditPairs
              ? "Edit instruction"
              : "Describe this image…"
          }
          rows={3}
          value={draft.text ?? ""}
        />
        {canToggleHoldout ? (
          <label className="training-checkbox-field" title="Iris training: a holdout item is never trained on; it is the holdout set the run validates against.">
            <input
              checked={holdout}
              onChange={(event) => handlers.toggleHoldout(asset.id, event.target.checked)}
              type="checkbox"
            />
            Holdout (validation only)
          </label>
        ) : null}
        {namedCaptionFields.length ? (
          <span
            className="training-field-hint training-caption-named-fields"
            title={namedCaptionFields.map((field) => `${field}: ${namedCaptions[field]}`).join("\n")}
          >
            +{namedCaptionFields.length} caption field{namedCaptionFields.length === 1 ? "" : "s"}:{" "}
            {namedCaptionFields.join(", ")}
          </span>
        ) : null}
        <div className="training-caption-card-actions">
          <label className="secondary-action strong">
            {preparedUploading
              ? "Uploading…"
              : preparedPath
                ? "Replace LTX bundle"
                : "Upload LTX bundle"}
            <input
              accept=".safetensors,application/octet-stream"
              aria-label={`Upload LTX prepared bundle for ${name}`}
              disabled={!savedItem?.id || preparedUploading}
              hidden
              onChange={(event) => handlers.uploadPrepared(asset, event.target.files?.[0])}
              type="file"
            />
          </label>
          {preparedPath ? <span className="training-asset-badge">Prepared</span> : null}
          {canUploadMask ? (
            <label className="secondary-action">
              {maskUploading
                ? "Uploading…"
                : maskStatus?.hasMask
                  ? "Replace mask"
                  : "Upload mask"}
              <input
                accept={SUBJECT_MASK_ACCEPT}
                aria-label={`Upload subject mask for ${name}`}
                disabled={!savedItem?.id || maskUploading}
                hidden
                onChange={(event) => {
                  handlers.uploadMask(asset, event.target.files?.[0]);
                  event.target.value = "";
                }}
                type="file"
              />
            </label>
          ) : null}
          <button
            aria-label={`Remove ${name}`}
            className="secondary-action"
            onClick={() => handlers.removeAsset(asset.id)}
            type="button"
          >
            Remove
          </button>
          <button
            aria-label={`Re-caption ${name}`}
            className="secondary-action strong"
            disabled={captioning}
            onClick={() => handlers.recaption(asset)}
            type="button"
          >
            Re-Caption
          </button>
        </div>
        {maskError ? (
          <p className="inline-warning" role="alert">
            {maskError}
          </p>
        ) : null}
      </div>
    </article>
  );
});
