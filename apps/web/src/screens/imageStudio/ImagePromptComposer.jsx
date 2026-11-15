import { StudioGuideButton } from "../../components/StudioFields.jsx";
import React from "react";
import { ImageEditSourcePickerField } from "../../components/AssetPicker.jsx";
import { Icon } from "../../components/Icons.jsx";
import { RefinePromptControl } from "../../components/RefinePromptControl.jsx";
import { EditPromptTemplates } from "../../components/EditPromptTemplates.jsx";
import StructuredPromptBuilder from "../../components/StructuredPromptBuilder.jsx";
import ReferenceCaptionPicker from "../../components/ReferenceCaptionPicker.jsx";
import BatchPromptPanel from "../../components/BatchPromptPanel.jsx";
import { QwenRewritePromptControl } from "../../components/QwenRewritePromptControl.jsx";
import { Flux3LayoutEditor } from "../../components/Flux3LayoutEditor.jsx";
import { assetUrl } from "../../components/assetMedia.jsx";
import { onPromptKeyDown, ModeTabs } from "../../components/generationStudio.jsx";
import { tierLabel } from "../../quantTier.js";

export function ImagePromptComposer({
  mode,
  handleModeChange,
  macModeTabBlock,
  batchMode,
  setBatchMode,
  openGuide,
  onOpenPresets,
  batchPromptsText,
  resolutionOptions,
  setBatchPromptsText,
  batchVariableValues,
  setBatchVariableValues,
  batchImagesPerPrompt,
  promptBatches,
  activeProject,
  batchName,
  setBatchName,
  batchScope,
  setBatchScope,
  loadedBatchId,
  handleSaveBatch,
  handleNewBatch,
  handleLoadBatch,
  handleDeleteBatch,
  handleImportBatch,
  batchBusy,
  batchError,
  batchValidity,
  batchTotal,
  batchRun,
  batchRunProgress,
  cancelBatchRun,
  replaceBatchRun,
  batchConfirmPending,
  runBatch,
  setBatchConfirmPending,
  batchRunDisabled,
  structuredPromptModel,
  caption,
  setCaption,
  captionValidation,
  promptMode,
  setPromptMode,
  prompt,
  setPromptFromUser,
  magicPrompt,
  onMagicExpand,
  magicModelMissing,
  refineModel,
  createModelDownloadJob,
  onReferenceImageLoaded,
  resolutionSnapKey,
  imageCaption,
  onImageCaption,
  editImageAssets,
  characters,
  importAsset,
  visionCaptionReady,
  visionCaptionOffers,
  modelDownloadJobs,
  setActiveView,
  onOpenQueue,
  onCancelJob,
  generateDisabled,
  submitting,
  expanding,
  replayTierUnavailable,
  quantTier,
  setRecipeTierRequest,
  submitError,
  promptTokenNotice,
  promptHint,
  suggestions,
  supportsImg2img,
  promptTool,
  togglePromptTool,
  img2imgReferenceAssetId,
  imageReference = { active: false },
  changeImg2imgReferenceAssetId,
  img2imgStrengthConfig,
  setImg2imgStrength,
  img2imgStrength,
  imageDescribe,
  qwenRewriteAvailable,
  onImageDescribe,
  promptGuide,
  model,
  refinePrompt,
  handleResolutionChange,
  qwenRewriteReferenceIds,
  qwenRewriteModel,
  qwenRewritePrompt,
  flux3LayoutAvailable,
  flux3Layout,
  setFlux3Layout,
  width,
  height,
  layoutReferences,
  assets
}) {
  return (
<>
<div className="prompt-hero-top">
            <ModeTabs
              label="Image mode"
              options={[
                ["text_to_image", "Text"],
                ["edit_image", "Edit"],
                ["character_image", "With character"],
              ]}
              mode={mode}
              onChange={handleModeChange}
              blockFor={(value) => macModeTabBlock(value)}
              leadingIcon={(value) => value === "text_to_image" ? <Icon.Sparkle size={13} /> : null}
            />
            <button
              aria-pressed={batchMode}
              className={batchMode ? "batch-toggle active" : "batch-toggle"}
              onClick={() => setBatchMode((on) => !on)}
              title="Run a list of prompts as one batch with the current settings"
              type="button"
            >
              <Icon.Stars size={13} /> Batch
            </button>
            <div className="prompt-hero-links">
              <StudioGuideButton onOpen={openGuide} />
              {onOpenPresets ? (
                <button className="hero-link" onClick={onOpenPresets} type="button">
                  <Icon.Folder size={14} /> Saved presets
                </button>
              ) : null}
            </div>
          </div>

          {imageReference.active ? (
            <div className="prompt-tool-panel image-reference-summary" aria-label="Active image reference">
              <div className="asset-picker-head">
                <span><strong>Image reference: On</strong> · {imageReference.label} · strength {Number(img2imgStrength).toFixed(2)}</span>
                <div className="asset-picker-actions">
                  <button type="button" onClick={() => changeImg2imgReferenceAssetId("")}>Remove image reference</button>
                  <button type="button" onClick={() => { if (promptTool !== "imageReference") togglePromptTool("imageReference"); }}>Reference settings</button>
                </div>
              </div>
              {imageReference.error ? <p className="structured-error" role="alert">{imageReference.error}</p> : null}
              {imageReference.pending ? <p role="status">Checking the reference image…</p> : null}
              {imageReference.error ? <button type="button" onClick={imageReference.retry}>Retry reference check</button> : null}
            </div>
          ) : null}

          {batchMode ? (
            <div className="prompt-input-row batch">
              <BatchPromptPanel
                promptsText={batchPromptsText}
                resolutionOptions={resolutionOptions}
                onPromptsTextChange={setBatchPromptsText}
                variableValues={batchVariableValues}
                onVariableValuesChange={setBatchVariableValues}
                count={batchImagesPerPrompt}
                batches={promptBatches}
                projectId={activeProject?.id ?? null}
                name={batchName}
                onNameChange={setBatchName}
                scope={batchScope}
                onScopeChange={setBatchScope}
                loadedBatchId={loadedBatchId}
                onSave={handleSaveBatch}
                onNew={handleNewBatch}
                onLoad={handleLoadBatch}
                onDelete={handleDeleteBatch}
                onImport={handleImportBatch}
                busy={batchBusy}
                error={batchError}
              />
              <div className="batch-run">
                {/* The batch's blocking problems, still one at a time in priority order —
                    but now the winning message and the disabled button read from the same
                    summary, so they cannot disagree (sc-10649). The empty-batch hint is a
                    silent requirement, rendered here as its own empty-state affordance. */}
                {batchValidity.surfaced.length ? (
                  <p className="batch-warning">{batchValidity.surfaced[0].message}</p>
                ) : batchTotal === 0 ? (
                  <p className="batch-hint">Add at least one prompt to run a batch.</p>
                ) : null}
                {batchRun ? (
                  <div className="batch-run-progress" aria-live="polite">
                    <span>
                      {batchRun.submitting
                        ? `Processed ${batchRunProgress.total - batchRunProgress.pending}/${batchRunProgress.total}`
                        : `${batchRunProgress.done}/${batchRunProgress.total} done`}
                      {batchRunProgress.failed ? ` · ${batchRunProgress.failed} failed` : ""}
                      {!batchRun.submitting && batchRunProgress.pending ? ` · ${batchRunProgress.pending} not queued` : ""}
                    </span>
                    {batchRun.firstError ? <span role="alert">First enqueue error: {batchRun.firstError}{batchRun.lastError !== batchRun.firstError ? ` · Last enqueue error: ${batchRun.lastError}` : ""}</span> : null}
                    {batchRun.submitting ? (
                      <button className="batch-btn ghost" onClick={cancelBatchRun} type="button">
                        Stop
                      </button>
                    ) : batchRunProgress.active > 0 ? (
                      <button className="batch-btn ghost" onClick={cancelBatchRun} type="button">
                        Cancel remaining
                      </button>
                    ) : (
                      <button className="batch-btn ghost" onClick={() => replaceBatchRun(null)} type="button">
                        Clear
                      </button>
                    )}
                  </div>
                ) : null}
                {batchConfirmPending ? (
                  <div className="batch-confirm" role="alertdialog">
                    <span>Queue {batchTotal} images? That’s a large batch.</span>
                    <button className="prompt-cta" onClick={() => runBatch(true)} type="button">
                      <Icon.Play size={14} /> Queue {batchTotal}
                    </button>
                    <button className="batch-btn ghost" onClick={() => setBatchConfirmPending(false)} type="button">
                      Cancel
                    </button>
                  </div>
                ) : (
                  <button
                    className="prompt-cta"
                    disabled={batchRunDisabled}
                    onClick={() => runBatch(false)}
                    type="button"
                  >
                    <Icon.Play size={14} />
                    {batchRun?.submitting ? "Queueing…" : `Run batch · ${batchTotal}`}
                  </button>
                )}
              </div>
            </div>
          ) : (
          <div className={`prompt-input-row${structuredPromptModel ? " structured" : ""}`}>
            {structuredPromptModel ? (
              <StructuredPromptBuilder
                caption={caption}
                onCaptionChange={setCaption}
                validation={captionValidation}
                mode={promptMode}
                onModeChange={setPromptMode}
                plainText={prompt}
                onPlainTextChange={setPromptFromUser}
                onMagicExpand={magicPrompt ? onMagicExpand : undefined}
                magicModelMissing={magicModelMissing}
                onDownloadMagicModel={refineModel ? () => createModelDownloadJob(refineModel) : undefined}
                // sc-8109 seam: the reference-image picker calls this with the uploaded image's
                // natural dimensions to auto-preset the resolution to the nearest aspect.
                onReferenceImageLoaded={onReferenceImageLoaded}
                referenceSnapKey={resolutionSnapKey}
                // Reference-image → JSON caption (epic 8102, sc-8108). Gated to text-to-image ONLY:
                // edit/character modes condition on their own source/identity image, so a fresh
                // scene caption written from a different reference would conflict. The image is
                // captioning-only (C1) — never sent to generation.
                onImageCaption={mode === "text_to_image" && imageCaption ? onImageCaption : undefined}
                referenceAssets={editImageAssets}
                referenceCharacters={characters}
                importAsset={importAsset}
                projectId={activeProject?.id ?? ""}
                // Reference-image caption gate (sc-8110): the section's availability is now driven by the
                // catalog (visionCaptionReady) through the shared ModelAvailabilityGate, not an inline
                // error-after-click affordance. When the captioner is missing, the gate offers a download.
                visionCaptionReady={visionCaptionReady}
                visionCaptionOffers={visionCaptionOffers}
                visionCaptionDownloadJobs={modelDownloadJobs}
                onDownloadModel={createModelDownloadJob}
                onOpenModels={() => setActiveView("Models")}
                onOpenQueue={onOpenQueue}
                onCancelJob={onCancelJob}
              />
            ) : (
              <textarea
                aria-label="Prompt"
                className="prompt-input"
                onChange={(event) => setPromptFromUser(event.target.value)}
                onKeyDown={onPromptKeyDown}
                placeholder="Describe your shot — subject, lighting, mood, lens…"
                value={prompt}
              />
            )}
            <button className="prompt-cta" disabled={generateDisabled} type="submit">
              <Icon.Sparkle size={14} />
              {submitting ? (expanding ? "Expanding…" : "Queueing…") : "Generate"}
            </button>
          </div>
          )}
          {/* Auto-expand failure (sc-6501): a structured model couldn't turn the plain-text idea
              into a caption (e.g. the prompt-refiner model isn't installed). We never fall back to
              sending raw plain text, so surface the reason and the path forward. */}
          {replayTierUnavailable ? (
            <p role="alert">Recipe {tierLabel(quantTier)} unavailable. Install it on a supported worker, or <button type="button" onClick={() => setRecipeTierRequest(null)}>use current settings</button>.</p>
          ) : null}
          {submitError ? (
            <p className="structured-error" role="alert">
              {submitError}
            </p>
          ) : null}

          {/* Booru-convention prompt hint (sc-10760): danbooru-tag models (Anima, Illustrious) declare
              `ui.promptHint`, so the studio nudges toward the quality prefix + tag-style prompting the model
              was trained on (a bare sentence renders low-effort art). Opens the existing prompt-guide modal.
              Free-text models only. */}
          {!structuredPromptModel && promptTokenNotice ? (
            <p className="prompt-hint" role="note">
              {promptTokenNotice}
            </p>
          ) : null}
          {!structuredPromptModel && promptHint ? (
            <p className="prompt-hint">
              {promptHint}{" "}
              <StudioGuideButton onOpen={openGuide} hint />
            </p>
          ) : null}

          {/* Scene suggestions sit directly under the prompt (UI-refinement 4a). Free-text
              prompts only; structured models get the builder + (later) magic-prompt.
              Edit mode swaps them for the built-in edit instructions: a scene description
              ("Fox watching from the edge of a snowy forest") is not an instruction, so the
              generation pills are actively misleading on the Edit tab. */}
          {structuredPromptModel ? null : mode === "edit_image" ? (
            <EditPromptTemplates onApply={setPromptFromUser} variant="studio" />
          ) : (
            <div className="suggestion-row">
              <span className="suggestion-row-label">Try:</span>
              {suggestions.map((suggestion) => (
                <button
                  className="suggestion"
                  key={suggestion}
                  onClick={() => setPromptFromUser(suggestion)}
                  type="button"
                >
                  <Icon.Sparkle size={11} />
                  {suggestion}
                </button>
              ))}
            </div>
          )}

          {/* Style Catalog picker + its composed-prompt preview moved into the Style axis row of the
              settings bar (sc-13135) — see the .settings-bar-style-axis block below. */}

          {/* Prompt tools (UI-refinement 1b; restructured sc-10195): a framed strip of up to THREE
              distinct tiles, one panel open at a time (all free-text only — structured models excluded).
              1) "Image reference" — img2img reference-guided generation (a picked image + strength slider
                 VISUALLY guides the render). Shown only for img2img-capable models (`supportsImg2img`).
              2) "Prompt from image" — the reference→text describe flow + mood board (epic 8203/8595): the
                 vision captioner writes prompt TEXT (captioning-only, never sent to generation). Gated on
                 the macOS-first captioner being platform-eligible.
              3) "Refine my prompt" — RefinePromptControl (sc-2041).
              img2img and describe used to share ONE overloaded tile (sc-8593); splitting them makes the
              "guide the render vs. write a prompt" choice explicit (Michael, on-device). */}
          {(() => {
            // img2img "Image reference" tile — a picked image + strength that SEED the render
            // (latent-init). Available for every `ui.img2img` model in text-to-image mode, INCLUDING
            // structured-prompt models (Ideogram, epic 8588 A4.4 sc-10192): the caption builder replaces
            // the free-text prompt tools, but reference-guided generation is orthogonal to how the prompt
            // is authored, so the tile coexists with the JSON-caption builder. Needs no vision captioner.
            const img2imgAvailable = supportsImg2img && mode === "text_to_image";
            const imageRefActive = img2imgAvailable && promptTool === "imageReference";
            const img2imgTile = img2imgAvailable ? (
              <button
                type="button"
                className={imageRefActive ? "prompt-tool active" : "prompt-tool"}
                aria-pressed={imageRefActive}
                onClick={() => togglePromptTool("imageReference")}
              >
                <span className="prompt-tool-title">
                  <Icon.Image size={15} /> Image reference
                  {/* Armed indicator: the panel is a collapsible accordion, so once a reference is
                      picked and the tile is collapsed there's otherwise no sign it's still driving
                      every generation. Show "On" whenever a reference is set, regardless of expansion. */}
                  {img2imgReferenceAssetId ? <span className="prompt-tool-flag">On</span> : null}
                </span>
                <span className="prompt-tool-desc">Guide the render with an image (image-to-image)</span>
              </button>
            ) : null;
            const img2imgPanel = imageRefActive ? (
              <div className="prompt-tool-panel">
                <div className="structured-reference">
                  <p className="structured-hint">
                    Pick an image to guide the render (image-to-image). A higher reference strength
                    stays closer to it; lower lets the prompt take over. Any strength above 0 starts
                    the denoise partway through the schedule, so progress counts only the remaining
                    steps — a stronger reference runs fewer than the Steps you set.
                  </p>
                  <ImageEditSourcePickerField
                    assets={editImageAssets}
                    buttonLabel="Select reference image"
                    changeLabel="Change reference"
                    characters={characters}
                    clearable
                    emptyLabel="No reference image selected"
                    importAsset={importAsset}
                    label="Reference image"
                    onChange={changeImg2imgReferenceAssetId}
                    projectId={activeProject?.id}
                    value={img2imgReferenceAssetId}
                  />
                  {img2imgReferenceAssetId ? (
                    <label className="reference-strength img2img-strength">
                      {img2imgStrengthConfig?.label ?? "Reference strength"}
                      <input
                        max={img2imgStrengthConfig?.max ?? 1}
                        min={img2imgStrengthConfig?.min ?? 0}
                        onChange={(event) => setImg2imgStrength(Number(event.target.value))}
                        step={img2imgStrengthConfig?.step ?? 0.05}
                        type="range"
                        value={img2imgStrength}
                      />
                      <span>{Number(img2imgStrength).toFixed(2)}</span>
                    </label>
                  ) : null}
                </div>
              </div>
            ) : null;

            // Structured-prompt models (Ideogram) get ONLY the img2img tile in this strip: "Prompt from
            // image" is served by the caption builder's own image→caption picker (epic 8102) and "Refine
            // my prompt" by its magic-expand, so rendering them here would duplicate those. Nothing to
            // show when the model doesn't advertise img2img.
            if (structuredPromptModel) {
              return img2imgAvailable ? (
                <div className="prompt-tools">
                  <div className="prompt-tools-head">
                    <span className="prompt-tools-title">Prompt tools</span>
                    <span className="hairline" />
                  </div>
                  <div className="prompt-tools-tiles">{img2imgTile}</div>
                  {img2imgPanel}
                </div>
              ) : null;
            }

            const describeAvailable =
              mode === "text_to_image" &&
              typeof imageDescribe === "function" &&
              (visionCaptionReady || visionCaptionOffers.length > 0);
            const describeActive = describeAvailable && promptTool === "describe";
            const refineActive = promptTool === "refine";
            // sc-24113 — Qwen-Image 2.1's OFFICIAL rewriter, offered as its own prompt tool beside
            // the generic refiner rather than replacing it. Three gates, and each one is an
            // acceptance criterion rather than defensive coding:
            //
            //  * `qwenRewriteModel` is the catalog entry for whichever rewriter THIS request
            //    selects — the editing one when references are attached, the text-to-image one
            //    otherwise — resolved from the request, never from a picker.
            //  * The tile only renders when that entry is INSTALLED. "Direct prompting works with
            //    neither rewriter installed — no download, no prompt to install, no degraded path"
            //    means the affordance is absent, not disabled-with-a-nag.
            //  * Nothing auto-starts (unlike the generic refiner's tile): rewriting is a user
            //    action all the way down, and even once it runs the result is a draft the user
            //    edits and applies.
            const qwenRewriteActive = qwenRewriteAvailable && promptTool === "qwenRewrite";
            return (
              <div className="prompt-tools">
                <div className="prompt-tools-head">
                  <span className="prompt-tools-title">Prompt tools</span>
                  <span className="hairline" />
                </div>
                <div className="prompt-tools-tiles">
                  {img2imgTile}
                  {describeAvailable ? (
                    <button
                      type="button"
                      className={describeActive ? "prompt-tool active" : "prompt-tool"}
                      aria-pressed={describeActive}
                      onClick={() => togglePromptTool("describe")}
                    >
                      <span className="prompt-tool-title">
                        <Icon.Image size={15} /> Prompt from image
                      </span>
                      <span className="prompt-tool-desc">Caption a reference (or mood board) into an editable prompt</span>
                    </button>
                  ) : null}
                  <button
                    type="button"
                    className={refineActive ? "prompt-tool active" : "prompt-tool"}
                    aria-pressed={refineActive}
                    onClick={() => togglePromptTool("refine")}
                  >
                    <span className="prompt-tool-title">
                      <Icon.Wand size={15} /> Refine my prompt
                    </span>
                    <span className="prompt-tool-desc">Rewrite what you typed for clarity &amp; detail</span>
                  </button>
                  {qwenRewriteAvailable ? (
                    <button
                      type="button"
                      className={qwenRewriteActive ? "prompt-tool active" : "prompt-tool"}
                      aria-pressed={qwenRewriteActive}
                      onClick={() => togglePromptTool("qwenRewrite")}
                    >
                      <span className="prompt-tool-title">
                        <Icon.Wand size={15} /> Qwen rewriter
                      </span>
                      <span className="prompt-tool-desc">
                        Qwen’s own rewriter — a long descriptive prompt plus an aspect-ratio
                        suggestion you can accept or ignore
                      </span>
                    </button>
                  ) : null}
                </div>
                {img2imgPanel}
                {describeActive ? (
                  <div className="prompt-tool-panel">
                    <ReferenceCaptionPicker
                      onCaption={onImageDescribe}
                      onApply={(text) => setPromptFromUser(text)}
                      onReferenceImageLoaded={onReferenceImageLoaded}
                      referenceSnapKey={resolutionSnapKey}
                      referenceAssets={editImageAssets}
                      referenceCharacters={characters}
                      importAsset={importAsset}
                      projectId={activeProject?.id ?? ""}
                      hint="The image is only used to write the prompt — it isn’t sent to generation."
                      buttonLabel="✨ Describe image"
                      busyLabel="Describing…"
                      showMoodBoard={visionCaptionReady}
                      emptyMessage="The image did not produce a usable description. Try another reference."
                      errorFallback="Could not describe the image."
                      gateDescription="Download the vision captioner to turn a reference image into a prompt. It runs locally on the native worker; the image is only used to write the prompt."
                      visionCaptionReady={visionCaptionReady}
                      visionCaptionOffers={visionCaptionOffers}
                      visionCaptionDownloadJobs={modelDownloadJobs}
                      onDownloadModel={createModelDownloadJob}
                      onOpenModels={() => setActiveView("Models")}
                      onOpenQueue={onOpenQueue}
                      onCancelJob={onCancelJob}
                    />
                  </div>
                ) : null}
                {refineActive ? (
                  <div className="prompt-tool-panel">
                    <RefinePromptControl
                      autoStart
                      guidePath={promptGuide.path}
                      modelId={model}
                      onApply={setPromptFromUser}
                      prompt={prompt}
                      refinePrompt={refinePrompt}
                      refineModel={refineModel}
                      onDownloadRefineModel={refineModel ? () => createModelDownloadJob(refineModel) : undefined}
                      workflow="image"
                    />
                  </div>
                ) : null}
                {qwenRewriteActive ? (
                  <div className="prompt-tool-panel">
                    <QwenRewritePromptControl
                      modelId={model}
                      onApply={setPromptFromUser}
                      // The aspect suggestion lands on the SAME resolution control the Aspect menu
                      // drives, and only ever with a value that menu already offers — accepting it
                      // is indistinguishable from the user picking it by hand.
                      onApplyResolution={handleResolutionChange}
                      projectId={activeProject?.id ?? ""}
                      prompt={prompt}
                      // The ORDERED list the render will condition on. Same array, same order —
                      // this is what makes the rewrite's `<imageN>` numbering refer to the pictures
                      // the user actually attached.
                      referenceAssetIds={qwenRewriteReferenceIds}
                      rewriteModel={qwenRewriteModel}
                      rewritePrompt={qwenRewritePrompt}
                    />
                  </div>
                ) : null}
              </div>
            );
          })()}

        {flux3LayoutAvailable ? <Flux3LayoutEditor
          value={flux3Layout}
          onChange={setFlux3Layout}
          width={width || 1024}
          height={height || 1024}
          references={layoutReferences.map((slot) => {
            const asset = assets.find(({ id }) => id === slot.assetId);
            return { ...slot, label: asset?.displayName ?? asset?.name ?? slot.assetId,
              url: asset ? assetUrl(asset) : null, width: asset?.width, height: asset?.height };
          })}
        /> : null}
</>
  );
}
