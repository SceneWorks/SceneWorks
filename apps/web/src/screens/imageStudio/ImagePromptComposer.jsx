import { StudioGuideButton } from "../../components/StudioFields.jsx";
import React from "react";
import { ImageEditSourcePickerField } from "../../components/AssetPicker.jsx";
import { Icon } from "../../components/Icons.jsx";
import { RefinePromptControl } from "../../components/RefinePromptControl.jsx";
import StructuredPromptBuilder from "../../components/StructuredPromptBuilder.jsx";
import ReferenceCaptionPicker from "../../components/ReferenceCaptionPicker.jsx";
import BatchPromptPanel from "../../components/BatchPromptPanel.jsx";
import { QwenRewritePromptControl } from "../../components/QwenRewritePromptControl.jsx";
import { onPromptKeyDown, ModeTabs } from "../../components/generationStudio.jsx";
import { tierLabel } from "../../quantTier.js";
import { H3_IMAGE_EXPAND_COPY, h3ImageProposedResolution } from "../../minimaxH3ImageExpand.js";
import { H3ExpanderModelPicker } from "../../components/H3ExpanderModelPicker.jsx";

import { SUBMIT_SHORTCUT_ARIA, SUBMIT_SHORTCUT_LABEL } from "../../components/submitShortcut.js";

// Image Studio compose panel (studio layout simplification): a top row (mode switch left; Batch,
// Prompt guide, Saved presets right), then ONE prompt box — the textarea (or the structured
// builder / batch editor) with a footer bar holding the prompt tools, the Try: suggestions, and
// Generate — then the status lines, then the one inline assist panel for the active prompt tool.
// Inputs and the settings tab strip are rendered by ImageStudio below this.
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
  sizeEnvelope,
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
  portableBatchRefusal,
  portableBatchRefusalIndex,
  portableBatchCheckedCount,
  batchClipNotice,
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
  // SW-7101: ref + onSelect from usePromptTokenInsertion, so reference tokens insert at the caret.
  promptTextareaProps,
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
  loopModel,
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
  handleApplyProposedSize,
  qwenRewriteReferenceIds,
  qwenRewriteModel,
  qwenRewritePrompt,
  h3ExpandAvailable,
  h3ExpandModel,
  h3ExpandPrompt,
  h3ExpandPicker,
  h3BatchExpandOn = false,
  batchJobCount = 0,
  h3BatchExpandTooMany = "",
  h3BatchUseSizes = false,
  setH3BatchUseSizes,
  resolution,
  selectedModel,
  // The strict-control "Structure" prompt tool (epic 8236): ImageStudio owns the ControlPanel
  // and its wiring; this only decides when its button shows and hosts it in the assist panel.
  structureAvailable = false,
  structurePanel = null,
  // Screen-level status lines (Mac gating, decoder recovery, Generate errors, portable status)
  // that sit with the composer's own, directly under the prompt box.
  statusLines = null,
}) {
  // ── Prompt tools (UI-refinement 1b; restructured sc-10195; studio layout simplification) ──
  // Compact buttons in the prompt box footer, one inline assist panel open at a time.
  // 1) "Image reference" — img2img reference-guided generation (a picked image + strength slider
  //    VISUALLY guides the render). Available for every `ui.img2img` model in text-to-image mode,
  //    INCLUDING structured-prompt models (Ideogram, epic 8588 A4.4 sc-10192): the caption builder
  //    replaces the free-text prompt tools, but reference-guided generation is orthogonal to how the
  //    prompt is authored. Needs no vision captioner.
  // 2) "Structure" — the strict-control panel (pose / canny / depth), text-to-image backbones whose
  //    `ui.controlModes` advertises it. Also orthogonal to prompt authoring, so structured models get it.
  // 3) "Prompt from image" — the reference→text describe flow + mood board (epic 8203/8595): the
  //    vision captioner writes prompt TEXT (captioning-only, never sent to generation).
  // 4) "Refine" — RefinePromptControl (sc-2041).
  // 5) "Qwen rewrite" — sc-24113, Qwen-Image 2.1's OFFICIAL rewriter, offered beside the generic
  //    refiner rather than replacing it. `qwenRewriteModel` is the catalog entry for whichever
  //    rewriter THIS request selects (editing one with references, text-to-image otherwise); the
  //    button only renders when that entry is INSTALLED, and nothing auto-starts.
  // 6) "Expand" — SW-7102, MiniMax-H3-Image's own prompt expander, under the same three gates.
  // Structured-prompt models (Ideogram) get only 1) and 2): "Prompt from image" is served by the
  // caption builder's own image→caption picker (epic 8102) and refining by its magic-expand.
  const img2imgAvailable = supportsImg2img && mode === "text_to_image";
  const describeAvailable =
    !structuredPromptModel &&
    mode === "text_to_image" &&
    typeof imageDescribe === "function" &&
    (loopModel || visionCaptionReady || visionCaptionOffers.length > 0);
  const tools = [
    img2imgAvailable && {
      id: "imageReference",
      label: "Image reference",
      title: "Image reference",
      hint: "Guide the render with an image (image-to-image)",
      icon: <Icon.Image size={15} />,
      // Armed indicator: once a reference is picked and the panel is closed there is otherwise no
      // sign it is still driving every generation, so show "On" whenever a reference is set.
      flag: img2imgReferenceAssetId ? "On" : null,
    },
    structureAvailable && {
      id: "structure",
      label: "Structure",
      title: "Structure control",
      hint: "Lock pose, edges, or depth to a reference",
      icon: <Icon.Maximize size={15} />,
    },
    describeAvailable && {
      id: "describe",
      label: "Prompt from image",
      title: "Prompt from image",
      hint: "Caption a reference (or mood board) into an editable prompt",
      icon: <Icon.Image size={15} />,
    },
    !structuredPromptModel && {
      id: "refine",
      label: "Refine",
      title: "Refine my prompt",
      hint: "Rewrite what you typed for clarity & detail",
      icon: <Icon.Wand size={15} />,
    },
    !structuredPromptModel && qwenRewriteAvailable && {
      id: "qwenRewrite",
      label: "Qwen rewrite",
      title: "Qwen rewriter",
      hint: "Qwen’s own rewriter — a long descriptive prompt plus an aspect-ratio suggestion you can accept or ignore",
      icon: <Icon.Wand size={15} />,
    },
    !structuredPromptModel && h3ExpandAvailable && {
      id: "h3Expand",
      label: "Expand",
      title: "Expand prompt",
      hint: batchMode
        ? H3_IMAGE_EXPAND_COPY.batch.tileDescription
        : "MiniMax-H3-Image’s own expander — a long caption plus a proposed size you can accept or ignore",
      icon: <Icon.Wand size={15} />,
    },
  ].filter(Boolean);
  const activeTool = tools.find((tool) => tool.id === promptTool) ?? null;

  // Scene suggestions sit in the prompt box footer (UI-refinement 4a). Free-text prompts only;
  // structured models get the builder + magic-prompt. Edit mode swaps them for the built-in edit
  // instructions, which live with the Edit inputs (a scene description is not an instruction).
  const showSuggestions = !structuredPromptModel && mode !== "edit_image" && suggestions.length > 0;

  const generateButton = batchMode ? (
    batchConfirmPending ? (
      <div className="batch-confirm" role="alertdialog">
        <span>
          Queue {batchTotal} images? That’s a large batch.
          {h3BatchExpandOn ? H3_IMAGE_EXPAND_COPY.batch.confirm(batchJobCount) : ""}
        </span>
        <button className="prompt-cta studio-generate" onClick={() => runBatch(true)} type="button">
          <Icon.Play size={14} /> Queue {batchTotal}
        </button>
        <button className="batch-btn ghost" onClick={() => setBatchConfirmPending(false)} type="button">
          Cancel
        </button>
      </div>
    ) : (
      <button
        className="studio-generate"
        disabled={batchRunDisabled}
        onClick={() => runBatch(false)}
        type="button"
      >
        <Icon.Play size={14} />
        {batchRun?.expanding
          ? H3_IMAGE_EXPAND_COPY.batch.running
          : batchRun?.submitting
            ? "Queueing…"
            : h3BatchExpandOn
              ? H3_IMAGE_EXPAND_COPY.batch.run(batchTotal)
              : `Run batch · ${batchTotal}`}
      </button>
    )
  ) : (
    <button
      aria-keyshortcuts={SUBMIT_SHORTCUT_ARIA}
      className="studio-generate"
      disabled={generateDisabled}
      type="submit"
    >
      <Icon.Sparkle size={14} />
      {submitting ? (expanding ? "Expanding…" : "Queueing…") : "Generate"}
      <span className="kbd" aria-hidden="true">{SUBMIT_SHORTCUT_LABEL}</span>
    </button>
  );

  return (
<>
<div className="studio-compose-top">
            <ModeTabs
              label="Image mode"
              className="mode-tabs studio-mode-segment"
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
            <span className="studio-compose-top-spacer" aria-hidden="true" />
            <button
              aria-pressed={batchMode}
              className={batchMode ? "batch-toggle studio-ghost-button active" : "batch-toggle studio-ghost-button"}
              onClick={() => setBatchMode((on) => !on)}
              title="Run a list of prompts as one batch with the current settings"
              type="button"
            >
              <Icon.Stars size={13} /> Batch
            </button>
            <StudioGuideButton onOpen={openGuide} />
            {onOpenPresets ? (
              <button className="hero-link" onClick={onOpenPresets} type="button">
                <Icon.Folder size={14} /> Saved presets
              </button>
            ) : null}
          </div>

          <div className={`studio-prompt-box${batchMode ? " batch" : structuredPromptModel ? " structured" : ""}`}>
            {batchMode ? (
              <BatchPromptPanel
                promptsText={batchPromptsText}
                resolutionOptions={resolutionOptions}
                sizeEnvelope={sizeEnvelope}
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
            ) : structuredPromptModel ? (
              <div className="studio-prompt-box-body">
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
                  // Reference-image caption gate (sc-8110): the section's availability is driven by the
                  // catalog (visionCaptionReady) through the shared ModelAvailabilityGate, not an inline
                  // error-after-click affordance. When the captioner is missing, the gate offers a download.
                  visionCaptionReady={visionCaptionReady}
                  loopModel={loopModel}
                  visionCaptionOffers={visionCaptionOffers}
                  visionCaptionDownloadJobs={modelDownloadJobs}
                  onDownloadModel={createModelDownloadJob}
                  onOpenModels={() => setActiveView("Models")}
                  onOpenQueue={onOpenQueue}
                  onCancelJob={onCancelJob}
                />
              </div>
            ) : (
              <textarea
                {...promptTextareaProps}
                aria-label="Prompt"
                className="prompt-input"
                onChange={(event) => setPromptFromUser(event.target.value)}
                onKeyDown={onPromptKeyDown}
                placeholder="Describe your shot — subject, lighting, mood, lens…"
                value={prompt}
              />
            )}
            <div className="studio-prompt-box-foot">
              {tools.map((tool) => (
                <button
                  aria-pressed={activeTool?.id === tool.id}
                  className="studio-prompt-tool"
                  data-tool={tool.id}
                  key={tool.id}
                  onClick={() => togglePromptTool(tool.id)}
                  title={tool.hint}
                  type="button"
                >
                  {tool.icon}
                  {tool.label}
                  {tool.flag ? <span className="studio-prompt-tool-flag">{tool.flag}</span> : null}
                </button>
              ))}
              {tools.length && showSuggestions ? <span className="studio-prompt-box-divider" aria-hidden="true" /> : null}
              {showSuggestions ? (
                <div className="suggestion-row studio-prompt-suggestions">
                  <span className="suggestion-row-label">Try:</span>
                  {suggestions.map((suggestion) => (
                    <button
                      className="suggestion"
                      key={suggestion}
                      onClick={() => setPromptFromUser(suggestion)}
                      title={suggestion}
                      type="button"
                    >
                      <Icon.Sparkle size={11} />
                      {suggestion}
                    </button>
                  ))}
                </div>
              ) : null}
              <span className="studio-prompt-box-spacer" aria-hidden="true" />
              {generateButton}
            </div>
          </div>

          <div className="studio-prompt-status">
            {imageReference.active ? (
              <div className="image-reference-summary" aria-label="Active image reference">
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
              <div className="batch-run">
                {/* The batch's blocking problems, still one at a time in priority order —
                    but now the winning message and the disabled button read from the same
                    summary, so they cannot disagree (sc-10649). The empty-batch hint is a
                    silent requirement, rendered here as its own empty-state affordance. */}
                {batchValidity.surfaced.length ? (
                  <p className="batch-warning">{batchValidity.surfaced[0].message}</p>
                ) : portableBatchRefusal ? (
                  <p className="batch-warning" role="alert">Batch item {portableBatchRefusalIndex}: {portableBatchRefusal.detail}</p>
                ) : batchTotal === 0 ? (
                  <p className="batch-hint">Add at least one prompt to run a batch.</p>
                ) : portableBatchCheckedCount ? (
                  <p className="batch-hint" role="status">Options checked for {portableBatchCheckedCount} requests. Live availability is checked when you run.</p>
                ) : null}
                {/* Items past the model's prompt ceiling run shortened rather than blocking the
                    batch; this says which ones before the run. */}
                {batchClipNotice ? <p className="batch-hint" role="note">{batchClipNotice}</p> : null}
                {batchRun ? (
                  <div className="batch-run-progress" aria-live="polite">
                    <span>
                      {batchRun.expanding
                        ? `${H3_IMAGE_EXPAND_COPY.batch.progress(Math.min(batchRun.expanding.done + 1, batchRun.expanding.total), batchRun.expanding.total)}${batchRunProgress.failed ? ` · ${batchRunProgress.failed} not expanded` : ""}`
                        : batchRun.submitting
                        ? `Processed ${batchRunProgress.total - batchRunProgress.pending}/${batchRunProgress.total}`
                        : `${batchRunProgress.done}/${batchRunProgress.total} done`}
                      {!batchRun.expanding && batchRunProgress.failed ? ` · ${batchRunProgress.failed} failed` : ""}
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
              </div>
            ) : null}

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
            {statusLines}
          </div>

          {activeTool ? (
            <section className="studio-tool-card" aria-label={activeTool.title} data-tool={activeTool.id}>
              <div className="studio-tool-card-head">
                <span className="studio-tool-card-title">{activeTool.title}</span>
                <span className="studio-tool-card-hint">{activeTool.hint}</span>
                <span className="studio-prompt-box-spacer" aria-hidden="true" />
                <button
                  aria-label={`Close ${activeTool.title}`}
                  className="studio-tool-card-close"
                  onClick={() => togglePromptTool(activeTool.id)}
                  type="button"
                >
                  <Icon.Close size={15} />
                </button>
              </div>
              {activeTool.id === "imageReference" ? (
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
              ) : null}
              {activeTool.id === "structure" ? structurePanel : null}
              {activeTool.id === "describe" ? (
                <ReferenceCaptionPicker
                  onCaption={onImageDescribe}
                  showPromptOptions
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
                  loopModel={loopModel}
                  visionCaptionOffers={visionCaptionOffers}
                  visionCaptionDownloadJobs={modelDownloadJobs}
                  onDownloadModel={createModelDownloadJob}
                  onOpenModels={() => setActiveView("Models")}
                  onOpenQueue={onOpenQueue}
                  onCancelJob={onCancelJob}
                />
              ) : null}
              {activeTool.id === "refine" ? (
                <RefinePromptControl
                  guidePath={promptGuide.path}
                  modelId={model}
                  onApply={setPromptFromUser}
                  prompt={prompt}
                  refinePrompt={refinePrompt}
                  refineModel={refineModel}
                  loopModel={loopModel}
                  onDownloadLoopModel={createModelDownloadJob}
                  onDownloadRefineModel={refineModel ? () => createModelDownloadJob(refineModel) : undefined}
                  workflow="image"
                />
              ) : null}
              {activeTool.id === "qwenRewrite" ? (
                <QwenRewritePromptControl
                  modelId={model}
                  onApply={setPromptFromUser}
                  // The aspect suggestion lands on the SAME resolution control the Aspect menu
                  // drives, and only ever with a value that menu already offers — accepting it
                  // is indistinguishable from the user picking it by hand.
                  onApplyResolution={handleApplyProposedSize}
                  projectId={activeProject?.id ?? ""}
                  prompt={prompt}
                  // The ORDERED list the render will condition on. Same array, same order —
                  // this is what makes the rewrite's `<imageN>` numbering refer to the pictures
                  // the user actually attached.
                  referenceAssetIds={qwenRewriteReferenceIds}
                  rewriteModel={qwenRewriteModel}
                  rewritePrompt={qwenRewritePrompt}
                />
              ) : null}
              {activeTool.id === "h3Expand" && batchMode ? (
                <>
                  {/* Batch mode: expansion is part of Run batch (useImageBatchRun), so there
                      is nothing to start or apply here — only what it will do, and the size
                      opt-in the single-prompt control offers per prompt. */}
                  {h3ExpandPicker ? <H3ExpanderModelPicker {...h3ExpandPicker} /> : null}
                  <div className="refine-control batch-expand-control">
                    <p className="refine-hint">{H3_IMAGE_EXPAND_COPY.batch.hint}</p>
                    <label className="batch-expand-sizes">
                      <input
                        checked={h3BatchUseSizes}
                        disabled={Boolean(batchRun?.submitting)}
                        onChange={(event) => setH3BatchUseSizes?.(event.target.checked)}
                        type="checkbox"
                      />{" "}
                      {H3_IMAGE_EXPAND_COPY.batch.useSizes}
                    </label>
                    {h3BatchExpandTooMany ? (
                      <p className="refine-error" role="alert">
                        {h3BatchExpandTooMany}
                      </p>
                    ) : null}
                  </div>
                </>
              ) : activeTool.id === "h3Expand" ? (
                <>
                  {h3ExpandPicker ? <H3ExpanderModelPicker {...h3ExpandPicker} /> : null}
                  <QwenRewritePromptControl
                    copy={H3_IMAGE_EXPAND_COPY}
                    modelId={model}
                    onApply={setPromptFromUser}
                    onApplyResolution={handleApplyProposedSize}
                    projectId={activeProject?.id ?? ""}
                    prompt={prompt}
                    referenceAssetIds={qwenRewriteReferenceIds}
                    rewriteModel={h3ExpandModel}
                    rewritePrompt={h3ExpandPrompt}
                    suggestedResolutionFor={(suggestion) =>
                      h3ImageProposedResolution(suggestion?.whRatio, selectedModel, resolution)
                    }
                  />
                </>
              ) : null}
            </section>
          ) : null}
</>
  );
}
