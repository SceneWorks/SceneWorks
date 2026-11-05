import { MAX_SCAIL2_REFERENCE_CHARACTERS } from "./videoTierHelpers.js";
import React from "react";
import { AssetPickerField, ImageEditSourcePickerField, VideoSourcePickerField } from "../../components/AssetPicker.jsx";
import { FitModeControl, effectiveFitMode } from "../../components/FitModeControl.jsx";
import { ReplacePersonPanel } from "../ReplacePersonPanel.jsx";

export function VideoSourcePanel({
  mode,
  imageAssets,
  characters,
  importAsset,
  setSourceAssetOrigin,
  setSourceAssetId,
  activeProject,
  sourceAssetId,
  acceptsLastOnly,
  setLastFrameAssetId,
  importedLastOnly,
  sourceAssetOrigin,
  lastFrameAssetId,
  fitMode,
  setFitMode,
  videoAssets,
  setSourceClipAssetId,
  sourceClipAssetId,
  setBridgeRightClipAssetId,
  bridgeRightClipAssetId,
  setSourceClipAssetIds,
  sourceClipAssetIds,
  setReferenceClipAssetId,
  referenceClipAssetId,
  setReferenceAssetIds,
  referenceAssetIds,
  showReferenceClips,
  refCaps,
  showAudioReferences,
  audioAssets,
  setReferenceAudioAssetIds,
  referenceAudioAssetIds,
  scail2MultiReferenceEnabled,
  scail2ReferenceOverflow,
  createPersonDetectionJob,
  createPersonTrackJob,
  createModelDownloadJob,
  personReadiness,
  detectionResult,
  matchingTracks,
  personTrackId,
  replacementMode,
  representativeFrame,
  saveTrackCorrections,
  selectedDetection,
  selectedTrack,
  setPersonTrackId,
  setReplacementMode,
  setSelectedDetectionId,
  setTrackName,
  trackName,
  videoModels,
  model,
  setUserModel
}) {
  return (
<>
{mode !== "text_to_video" ? (
          <div className="studio-source-band">
            {mode === "image_to_video" || mode === "first_last_frame" ? (
              <ImageEditSourcePickerField
                assets={imageAssets}
                buttonLabel="Select image"
                characters={characters}
                emptyLabel="No first frame selected"
                eyebrow="Video Studio"
                importAsset={importAsset}
                label="First frame"
                onChange={(value) => {
                  // The user picked the first frame here: it is theirs, and it stays theirs.
                  setSourceAssetOrigin(value ? "explicit" : "none");
                  setSourceAssetId(value);
                }}
                projectId={activeProject?.id}
                value={sourceAssetId}
              />
            ) : null}

            {mode === "first_last_frame" || (acceptsLastOnly && mode === "image_to_video") ? (
              <ImageEditSourcePickerField
                assets={imageAssets}
                buttonLabel="Select image"
                characters={characters}
                emptyLabel="No last frame selected"
                eyebrow="Video Studio"
                importAsset={importAsset}
                label="Last frame"
                onChange={(value) => {
                  setLastFrameAssetId(value);
                  // Choosing the LAST frame frames the request around it. In Image-to-Video on an
                  // IMPORTED H3 partition that accepts a last-only request, the studio's ambient
                  // first frame (the app's current asset, never chosen here) is dropped rather
                  // than sent as an invented first frame. A first frame the user DID choose stays,
                  // and so does the ambient one in First -> Last, where the first frame is REQUIRED.
                  // The request itself is guarded at submit time as well, because this picker is
                  // not the only way the ambient selection can arrive.
                  if (value && mode === "image_to_video" && importedLastOnly && sourceAssetOrigin === "ambient") {
                    setSourceAssetOrigin("none");
                    setSourceAssetId("");
                  }
                }}
                projectId={activeProject?.id}
                value={lastFrameAssetId}
              />
            ) : null}

            {mode === "image_to_video" || mode === "first_last_frame" ? (
              <FitModeControl
                value={effectiveFitMode(fitMode, false)}
                onChange={setFitMode}
                inpaintCapable={false}
              />
            ) : null}

            {mode === "extend_clip" ? (
              <VideoSourcePickerField
                assets={videoAssets}
                buttonLabel="Select clip"
                characters={characters}
                emptyLabel="No source clip selected"
                importAsset={importAsset}
                label="Source clip"
                onChange={setSourceClipAssetId}
                projectId={activeProject?.id}
                value={sourceClipAssetId}
              />
            ) : null}

            {mode === "video_bridge" ? (
              <>
                <VideoSourcePickerField
                  assets={videoAssets}
                  buttonLabel="Select clip"
                  characters={characters}
                  emptyLabel="No left clip selected"
                  importAsset={importAsset}
                  label="Left clip"
                  onChange={setSourceClipAssetId}
                  projectId={activeProject?.id}
                  value={sourceClipAssetId}
                />
                <VideoSourcePickerField
                  assets={videoAssets}
                  buttonLabel="Select clip"
                  characters={characters}
                  emptyLabel="No right clip selected"
                  importAsset={importAsset}
                  label="Right clip"
                  onChange={setBridgeRightClipAssetId}
                  projectId={activeProject?.id}
                  value={bridgeRightClipAssetId}
                />
              </>
            ) : null}

            {["video_to_video", "reference_video_to_video", "ads2v"].includes(mode) ? (
              <VideoSourcePickerField
                assets={videoAssets}
                buttonLabel="Select clip"
                characters={characters}
                emptyLabel="No source clip selected"
                importAsset={importAsset}
                label="Source clip"
                onChange={setSourceClipAssetId}
                projectId={activeProject?.id}
                value={sourceClipAssetId}
              />
            ) : null}

            {mode === "multi_video_to_video" ? (
              <VideoSourcePickerField
                assets={videoAssets}
                buttonLabel="Select clips"
                characters={characters}
                changeLabel="Edit clips"
                emptyLabel="No source clips selected"
                importAsset={importAsset}
                label="Source clips"
                multiple
                onChange={setSourceClipAssetIds}
                projectId={activeProject?.id}
                values={sourceClipAssetIds}
              />
            ) : null}

            {mode === "ads2v" ? (
              <VideoSourcePickerField
                assets={videoAssets}
                buttonLabel="Select clip"
                characters={characters}
                emptyLabel="No reference video selected"
                importAsset={importAsset}
                label="Reference video"
                onChange={setReferenceClipAssetId}
                projectId={activeProject?.id}
                value={referenceClipAssetId}
              />
            ) : null}

            {["reference_to_video", "reference_video_to_video", "ads2v"].includes(mode) ? (
              <ImageEditSourcePickerField
                assets={imageAssets}
                buttonLabel="Select images"
                characters={characters}
                changeLabel="Edit references"
                emptyLabel="No reference images selected"
                eyebrow="Video Studio"
                importAsset={importAsset}
                label="Reference images"
                multiple
                onChange={setReferenceAssetIds}
                projectId={activeProject?.id}
                values={referenceAssetIds}
              />
            ) : null}

            {/* Reference VIDEO clips (sc-17160 / sc-17161). Only for a model that declares a
                reference-clip cap: MiniMax-H3 Ref2VA conditions on motion and pacing from up to 3
                clips, where Bernini's r2v encodes reference images alone and would silently ignore
                them — the invisible-in-the-output failure the per-model caps exist to prevent. */}
            {showReferenceClips ? (
              <VideoSourcePickerField
                assets={videoAssets}
                buttonLabel="Select clips"
                characters={characters}
                changeLabel="Edit clips"
                emptyLabel={`No reference clips selected (up to ${refCaps.clips})`}
                importAsset={importAsset}
                label="Reference clips"
                multiple
                onChange={setSourceClipAssetIds}
                projectId={activeProject?.id}
                values={sourceClipAssetIds}
              />
            ) : null}

            {/* Reference AUDIO clips (sc-17160 landed the payload field; this is the control that
                makes it reachable). Gated on the declared cap alone — it defaults to 0, so the
                picker appears only for a model that says it conditions on audio. Uses the plain
                AssetPickerField with categories hidden, the same shape Audio Studio's reference
                voice uses: the media pickers' All/Images/Video tabs carry no audio bucket.
                `importAsset` + `mediaKind` give it the same local-file import the image/clip
                pickers above carry (sc-17137 review B3): a project with no audio assets would
                otherwise offer an empty grid with no way in. */}
            {showAudioReferences ? (
              <AssetPickerField
                assets={audioAssets}
                buttonLabel="Select audio"
                changeLabel="Edit audio"
                emptyLabel={`No reference audio selected (up to ${refCaps.audio})`}
                importAsset={importAsset}
                label="Reference audio"
                mediaKind="audio"
                multiple
                onChange={setReferenceAudioAssetIds}
                showCategories={false}
                values={referenceAudioAssetIds}
              />
            ) : null}

            {mode === "animate_character" ? (
              <>
                <VideoSourcePickerField
                  assets={videoAssets}
                  buttonLabel="Select clip"
                  characters={characters}
                  emptyLabel="No driving video selected"
                  importAsset={importAsset}
                  label="Driving video"
                  onChange={setSourceClipAssetId}
                  projectId={activeProject?.id}
                  value={sourceClipAssetId}
                />
                {/* The paired-reference UI stays descriptor-gated until the matching inference pin is
                    live. Its ordered array maps to strict Reference,Mask pairs in both workers. */}
                <ImageEditSourcePickerField
                  assets={imageAssets}
                  buttonLabel={scail2MultiReferenceEnabled ? "Select images" : "Select image"}
                  characters={characters}
                  changeLabel={scail2MultiReferenceEnabled ? "Edit characters" : "Change character"}
                  emptyLabel={scail2MultiReferenceEnabled ? "No reference characters selected" : "No reference character selected"}
                  eyebrow="Video Studio"
                  importAsset={importAsset}
                  label={scail2MultiReferenceEnabled ? "Reference characters (ordered, up to 6)" : "Reference character"}
                  multiple={scail2MultiReferenceEnabled}
                  onChange={(ids) =>
                    setReferenceAssetIds(
                      scail2MultiReferenceEnabled ? ids : ids ? [ids] : [],
                    )
                  }
                  projectId={activeProject?.id}
                  value={referenceAssetIds[0] ?? ""}
                  values={referenceAssetIds}
                />
                {scail2ReferenceOverflow ? (
                  <p className="inline-warning" role="alert">
                    SCAIL-2 supports at most {MAX_SCAIL2_REFERENCE_CHARACTERS} reference characters. Remove one before rendering.
                  </p>
                ) : null}
              </>
            ) : null}

            {mode === "replace_person" ? (
              <ReplacePersonPanel
                characters={characters}
                createPersonDetectionJob={createPersonDetectionJob}
                createPersonTrackJob={createPersonTrackJob}
                createModelDownloadJob={createModelDownloadJob}
                importAsset={importAsset}
                personReadiness={personReadiness}
                projectId={activeProject?.id}
                detectionResult={detectionResult}
                matchingTracks={matchingTracks}
                personTrackId={personTrackId}
                replacementMode={replacementMode}
                representativeFrame={representativeFrame}
                saveTrackCorrections={saveTrackCorrections}
                selectedDetection={selectedDetection}
                selectedTrack={selectedTrack}
                setPersonTrackId={setPersonTrackId}
                setReplacementMode={setReplacementMode}
                setSelectedDetectionId={setSelectedDetectionId}
                setSourceClipAssetId={setSourceClipAssetId}
                setTrackName={setTrackName}
                sourceClipAssetId={sourceClipAssetId}
                trackName={trackName}
                videoAssets={videoAssets}
                videoModels={videoModels}
                model={model}
                setModel={setUserModel}
              />
            ) : null}
          </div>
        ) : null}
</>
  );
}
