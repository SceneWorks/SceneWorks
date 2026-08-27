import React from "react";
import { EditPromptTemplates } from "../../components/EditPromptTemplates.jsx";
import { assetDisplayUrl } from "../../components/assetMedia.jsx";

let renderObserverForTests = null;

export function setImageEditorToolPanelRenderObserverForTests(observer) {
  renderObserverForTests = observer;
}

export function useStableImageEditorToolPanelScope(scope) {
  const latestScope = React.useRef(scope);
  latestScope.current = scope;
  const stableFunctions = React.useRef(null);
  if (stableFunctions.current === null) {
    stableFunctions.current = Object.fromEntries(
      Object.entries(scope)
        .filter(([, value]) => typeof value === "function")
        .map(([key, value]) => [
          key,
          /^[A-Z]/.test(key)
            ? value
            : (...args) => latestScope.current[key](...args),
        ]),
    );
  }
  return Object.fromEntries(
    Object.entries(scope).map(([key, value]) => [
      key,
      typeof value === "function" ? stableFunctions.current[key] : value,
    ]),
  );
}

function samePanelProps(previous, next) {
  if (previous.panelKey !== next.panelKey) return false;
  const previousScope = previous.scope;
  const nextScope = next.scope;
  const keys = Object.keys(previousScope);
  if (keys.length !== Object.keys(nextScope).length) return false;
  return keys.every((key) => Object.is(previousScope[key], nextScope[key]));
}

export const ImageEditorEditPanel = React.memo(function ImageEditorEditPanel({ scope }) {
  const { EDIT_OUTPUT_ASPECTS, EditorLoraPanel, FitModeControl, MAX_EDIT_REFERENCES, StudioUpdateBadge, StudioUpdateNotice, aiOp, canMask, clearMask, createLoraDownloadJob, createModelDownloadJob, editAspect, editFitMode, editGuidance, editLora, editLoraDownloadRequested, editLoraInstalled, editLoraRequiredMissing, editLoraSelection, editModel, editModels, editPrompt, editSeed, editorPickerLoras, effectiveFitMode, guidanceDefaultFromModel, imageAssets, maskActive, maskBaseImage, maskBrush, maskErase, maskHasContent, maskLines, maskMode, maskRefineRadius, maskSubTool, multiRefCapable, refAssetIds, refineMask, requestEditLoraDownload, runEdit, selectedEditLoras, selectedEditModel, setEditAspect, setEditFitMode, setEditGuidance, setEditModel, setEditPrompt, setEditSeed, setMaskBrush, setMaskErase, setMaskMode, setMaskRefineRadius, setMaskSubTool, setRefAssetIds, setRefPickerOpen, setShowIncompatibleEditLoras, showIncompatibleEditLoras, smartSelectSupported, updateOptionLabel } = scope;
  const renderPanel = () => {
    if (editModels.length === 0) {
      return (
        <div className="ie-section">
          <p className="ie-note">No edit-capable models installed.</p>
        </div>
      );
    }
    return (
      <>
        <div className="ie-section">
          <div className="ie-sec-title">Model</div>
          <select className="ie-select" onChange={(event) => setEditModel(event.target.value)} value={editModel}>
            {editModels.map((model) => (
              <option key={model.id} value={model.id}>
                {updateOptionLabel(model)}
              </option>
            ))}
          </select>
          <StudioUpdateBadge item={selectedEditModel} />
          <StudioUpdateNotice item={selectedEditModel} onUpdate={createModelDownloadJob} />
          <div className="ie-field">
            <div className="ie-field-top">
              <span className="ie-field-label">Instruction</span>
            </div>
            <textarea
              className="ie-textarea"
              onChange={(event) => setEditPrompt(event.target.value)}
              placeholder="Describe the edit — e.g. “replace the background with a foggy pine forest at dawn”"
              value={editPrompt}
            />
          </div>
          {/* Built-in edit recipes — the same five the studios offer, so the two Edit
              surfaces stay in step. Each REPLACES the instruction (see the module note). */}
          <EditPromptTemplates label="" onApply={setEditPrompt} variant="editor" />
        </div>

        {/* Managed image-edit LoRA (epic 10871, sc-11069): auto-applied for the user — a status note
            when installed, a one-click download that gates the run when not. Inert for edit models
            that need none. Placed right under the model so it reads as part of the edit surface. */}
        {editLora ? (
          editLoraInstalled ? (
            <div className="ie-section">
              <p className="ie-note">✨ {editLora.name} is applied automatically for editing.</p>
              <StudioUpdateBadge item={editLora} />
              <StudioUpdateNotice item={editLora} kind="LoRA" onUpdate={requestEditLoraDownload} />
              {/* Identity strength (sc-11798): the managed edit LoRA is hidden from the manual
                  picker, so expose its apply weight here — threaded into buildEditJobBody's
                  editLoraWeight → the payload edit-LoRA `weight`. Higher = stronger conditioning. */}
              <div className="lora-slot-weight edit-lora-strength">
                <label>
                  <span>Identity strength</span>
                  <span className="lora-slot-weight-value">
                    {editLoraSelection.weightFor(editLora).toFixed(2)}
                  </span>
                </label>
                <input
                  aria-label={`${editLora.name} identity strength`}
                  max="2"
                  min="0"
                  onChange={(event) =>
                    editLoraSelection.setWeight(editLora.id, Number(event.target.value))
                  }
                  step="0.05"
                  type="range"
                  value={editLoraSelection.weightFor(editLora)}
                />
              </div>
            </div>
          ) : (
            <div className="ie-section">
              <p className="ie-note">{editLora.name} is required to edit — the base can’t edit without it.</p>
              <button
                className="ie-btn block"
                disabled={editLoraDownloadRequested}
                onClick={requestEditLoraDownload}
                type="button"
              >
                {editLoraDownloadRequested ? "Downloading…" : `Download ${editLora.name}`}
              </button>
            </div>
          )
        ) : null}

        <EditorLoraPanel
          selectedModel={selectedEditModel}
          selectedLoras={selectedEditLoras}
          selectedLoraIds={editLoraSelection.selectedLoraIds}
          toggleLora={editLoraSelection.toggleLora}
          weightFor={editLoraSelection.weightFor}
          setWeight={editLoraSelection.setWeight}
          availableLoras={editorPickerLoras}
          showIncompatible={showIncompatibleEditLoras}
          setShowIncompatible={setShowIncompatibleEditLoras}
          onUpdateLora={createLoraDownloadJob}
        />

        <div className="ie-section">
          <div className="ie-sec-title">Output</div>
          <div className="ie-field">
            <span className="ie-field-label" style={{ marginBottom: "2px" }}>
              Aspect
            </span>
            <div className="ie-seg wrap four">
              {EDIT_OUTPUT_ASPECTS.map((aspect) => (
                <button
                  className="ie-seg-btn"
                  data-active={editAspect === aspect.key}
                  key={aspect.key}
                  onClick={() => setEditAspect(aspect.key)}
                  type="button"
                >
                  {aspect.label}
                </button>
              ))}
            </div>
          </div>
          {editAspect !== "match" ? (
            <FitModeControl
              inpaintCapable={canMask}
              label="Fill new area"
              onChange={setEditFitMode}
              value={effectiveFitMode(editFitMode, canMask)}
            />
          ) : null}
        </div>

        {canMask ? (
          <div className="ie-section">
            <div className="ie-sec-title">
              Mask
              <button
                className="ie-btn sm ghost"
                data-active={maskMode}
                onClick={() => setMaskMode((on) => !on)}
                style={{ height: "24px" }}
                type="button"
              >
                {maskMode ? "On" : "Off"}
              </button>
            </div>
            {maskMode ? (
              <>
                {smartSelectSupported ? (
                  <div className="ie-seg two" style={{ width: "100%" }}>
                    <button
                      className="ie-seg-btn"
                      data-active={maskSubTool === "brush"}
                      onClick={() => setMaskSubTool("brush")}
                      type="button"
                    >
                      Brush
                    </button>
                    <button
                      className="ie-seg-btn"
                      data-active={maskSubTool === "select"}
                      disabled={aiOp?.label === "smart select"}
                      onClick={() => {
                        setMaskSubTool("select");
                        setMaskErase(false);
                      }}
                      type="button"
                    >
                      {aiOp?.label === "smart select" ? "Segmenting…" : "Smart select"}
                    </button>
                  </div>
                ) : null}
                {!smartSelectSupported || maskSubTool === "brush" ? (
                  <>
                    <div className="ie-field">
                      <div className="ie-field-top">
                        <span className="ie-field-label">Brush size</span>
                        <span className="ie-field-val">{maskBrush} px</span>
                      </div>
                      <input
                        className="ie-range"
                        max={300}
                        min={5}
                        onChange={(event) => setMaskBrush(Number(event.target.value))}
                        step={1}
                        type="range"
                        value={maskBrush}
                      />
                    </div>
                    <button className="ie-btn block" data-active={maskErase} onClick={() => setMaskErase((on) => !on)} type="button">
                      Eraser
                    </button>
                  </>
                ) : (
                  <p className="ie-note">Drag a box around an object on the canvas — SAM3 auto-masks it.</p>
                )}
                <div>
                  <span className="ie-field-label" style={{ display: "block", marginBottom: "7px" }}>
                    Refine selection
                  </span>
                  <div className="ie-field" style={{ marginBottom: "8px" }}>
                    <div className="ie-field-top">
                      <span className="ie-field-label" style={{ fontSize: "11.5px", color: "var(--ie-muted)" }}>
                        Radius
                      </span>
                      <span className="ie-field-val">{maskRefineRadius}px</span>
                    </div>
                    <input
                      className="ie-range"
                      max={40}
                      min={1}
                      onChange={(event) => setMaskRefineRadius(Number(event.target.value))}
                      step={1}
                      type="range"
                      value={maskRefineRadius}
                    />
                  </div>
                  <div className="ie-chip-row">
                    <button
                      className="ie-chip"
                      disabled={!maskHasContent(maskLines) && !maskBaseImage}
                      onClick={() => refineMask("feather")}
                      type="button"
                    >
                      Feather
                    </button>
                    <button
                      className="ie-chip"
                      disabled={!maskHasContent(maskLines) && !maskBaseImage}
                      onClick={() => refineMask("grow")}
                      type="button"
                    >
                      Grow
                    </button>
                    <button
                      className="ie-chip"
                      disabled={!maskHasContent(maskLines) && !maskBaseImage}
                      onClick={() => refineMask("shrink")}
                      type="button"
                    >
                      Shrink
                    </button>
                    <button className="ie-chip" onClick={() => refineMask("invert")} type="button">
                      Invert
                    </button>
                    <button
                      className="ie-chip"
                      disabled={!maskLines.length && !maskBaseImage}
                      onClick={clearMask}
                      type="button"
                    >
                      Clear
                    </button>
                  </div>
                </div>
              </>
            ) : (
              <p className="ie-note">Turn on a mask to confine the edit to a painted or selected region (inpaint).</p>
            )}
          </div>
        ) : null}

        {multiRefCapable ? (
          <div className="ie-section">
            <div className="ie-sec-title">Reference images</div>
            <div className="ie-refs">
              {refAssetIds.map((id) => {
                const asset = imageAssets.find((item) => item.id === id);
                return (
                  <div className="ie-ref" key={id}>
                    {asset ? <img alt="" src={assetDisplayUrl(asset)} /> : <span>?</span>}
                    <button
                      aria-label="Remove reference"
                      className="ie-ref-remove"
                      onClick={() => setRefAssetIds((prev) => prev.filter((other) => other !== id))}
                      type="button"
                    >
                      ✕
                    </button>
                  </div>
                );
              })}
              <button
                className="ie-ref-add"
                disabled={refAssetIds.length >= MAX_EDIT_REFERENCES - 1}
                onClick={() => setRefPickerOpen(true)}
                title="Condition the edit on reference image(s)"
                type="button"
              >
                +
              </button>
            </div>
          </div>
        ) : null}

        <div className="ie-section">
          <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: "10px" }}>
            <div className="ie-field">
              <span className="ie-field-label" style={{ marginBottom: "2px" }}>
                Seed
              </span>
              <input
                className="ie-input"
                min={0}
                onChange={(event) => setEditSeed(event.target.value)}
                placeholder="Random"
                style={{ fontFamily: "var(--ie-mono)" }}
                type="number"
                value={editSeed}
              />
            </div>
            <div className="ie-field">
              <span className="ie-field-label" style={{ marginBottom: "2px" }}>
                Guidance
              </span>
              <input
                className="ie-input"
                min={0}
                onChange={(event) => setEditGuidance(event.target.value)}
                placeholder={guidanceDefaultFromModel(selectedEditModel)?.toString() ?? "Default"}
                step={0.1}
                style={{ fontFamily: "var(--ie-mono)" }}
                type="number"
                value={editGuidance}
              />
            </div>
          </div>
          <button
            className="ie-btn block primary"
            disabled={!editPrompt.trim() || !!aiOp || editLoraRequiredMissing}
            onClick={runEdit}
            type="button"
          >
            {maskActive ? "Inpaint region" : "Generate edit"}
          </button>
        </div>
      </>
    );
  };
  return renderPanel();
}, samePanelProps);

export const ImageEditorBoxesPanel = React.memo(function ImageEditorBoxesPanel({ scope }) {
  const { BOX_PALETTE, MAX_BOX_PALETTE, StudioUpdateBadge, StudioUpdateNotice, addPaletteColor, aiOp, boxColor, boxMetadataGaps, boxes, chooseBoxColor, clearBoxes, composedPrompt, createModelDownloadJob, deleteBox, editModel, editModels, editPrompt, removePaletteColor, runBoxEdit, selectedBox, selectedBoxGaps, selectedBoxId, selectedEditModel, setEditModel, setEditPrompt, setSelectedBoxId, updateBox, updateOptionLabel } = scope;
  const renderPanel = () => (
    <>
      <div className="ie-section">
        <div className="ie-sec-title">Box color</div>
        <div className="ie-swatches">
          {BOX_PALETTE.map((entry) => (
            <button
              aria-label={entry.name}
              className="ie-swatch"
              data-active={boxColor === entry.value}
              key={entry.value}
              onClick={() => chooseBoxColor(entry.value)}
              style={{ background: entry.value }}
              title={entry.name}
              type="button"
            />
          ))}
          <label className="ie-swatch ie-swatch-custom" title="Custom color">
            <input aria-label="Custom box color" onChange={(event) => chooseBoxColor(event.target.value)} type="color" value={boxColor.toLowerCase()} />
          </label>
        </div>
        <p className="ie-note">Drag on the canvas to draw a colored region, then describe what belongs there.</p>
      </div>

      <div className="ie-section">
        <div className="ie-sec-title">Regions ({boxes.length})</div>
        {boxes.length ? (
          <div className="ie-chip-row">
            {boxes.map((box, index) => {
              const incomplete = boxMetadataGaps(box).length > 0;
              return (
                <button
                  className="ie-chip"
                  data-active={selectedBoxId === box.id}
                  key={box.id}
                  onClick={() => setSelectedBoxId(box.id)}
                  title={box.desc ? `${index + 1}: ${box.desc}` : `Box ${index + 1} — needs a description`}
                  type="button"
                >
                  <span className="ie-dot" style={{ background: box.color }} />
                  {index + 1}
                  {incomplete ? <span className="warn">!</span> : null}
                </button>
              );
            })}
          </div>
        ) : (
          <p className="ie-note">Drag on the image to draw a box.</p>
        )}
        {boxes.length ? (
          <div style={{ display: "flex", gap: "8px" }}>
            <button className="ie-btn sm" disabled={!selectedBoxId} onClick={() => deleteBox(selectedBoxId)} type="button">
              Delete
            </button>
            <button className="ie-btn sm" disabled={!boxes.length} onClick={clearBoxes} type="button">
              Clear all
            </button>
          </div>
        ) : null}
      </div>

      {selectedBox ? (
        <div className="ie-section">
          <div className="ie-sec-title">
            Region details
            <span className="ie-field-val">Box {boxes.indexOf(selectedBox) + 1}</span>
          </div>
          <div className="ie-seg two" style={{ width: "100%" }}>
            <button
              className="ie-seg-btn"
              data-active={selectedBox.type === "obj"}
              onClick={() => updateBox(selectedBox.id, { type: "obj" })}
              type="button"
            >
              Object
            </button>
            <button
              className="ie-seg-btn"
              data-active={selectedBox.type === "text"}
              onClick={() => updateBox(selectedBox.id, { type: "text" })}
              type="button"
            >
              Text
            </button>
          </div>
          <div className="ie-field">
            <span className="ie-field-label" style={{ marginBottom: "2px" }}>
              Description
            </span>
            <input
              className="ie-input"
              onChange={(event) => updateBox(selectedBox.id, { desc: event.target.value })}
              placeholder="What is in this region?"
              value={selectedBox.desc ?? ""}
            />
          </div>
          {selectedBox.type === "text" ? (
            <div className="ie-field">
              <span className="ie-field-label" style={{ marginBottom: "2px" }}>
                Literal text
              </span>
              <input
                className="ie-input"
                onChange={(event) => updateBox(selectedBox.id, { text: event.target.value })}
                placeholder="Text to render"
                value={selectedBox.text ?? ""}
              />
            </div>
          ) : null}
          <div className="ie-field">
            <span className="ie-field-label" style={{ marginBottom: "2px" }}>
              Element colors ({(selectedBox.colorPalette ?? []).length}/{MAX_BOX_PALETTE})
            </span>
            <div className="ie-swatches">
              {(selectedBox.colorPalette ?? []).map((color) => (
                <button
                  aria-label={`Remove ${color}`}
                  className="ie-swatch"
                  key={color}
                  onClick={() => updateBox(selectedBox.id, { colorPalette: removePaletteColor(selectedBox.colorPalette, color) })}
                  style={{ background: color }}
                  title={`Remove ${color}`}
                  type="button"
                />
              ))}
              {(selectedBox.colorPalette ?? []).length < MAX_BOX_PALETTE ? (
                <label className="ie-swatch ie-swatch-custom" title="Add color">
                  <input
                    aria-label="Add element color"
                    onChange={(event) => updateBox(selectedBox.id, { colorPalette: addPaletteColor(selectedBox.colorPalette, event.target.value) })}
                    type="color"
                  />
                </label>
              ) : null}
            </div>
          </div>
          {selectedBoxGaps.length ? (
            <p className="ie-note">For Ideogram layout this box still needs {selectedBoxGaps.join(", ")}. The color-keyed edit path only needs a color + description.</p>
          ) : (
            <p className="ie-note" style={{ color: "var(--ie-accent)" }}>
              Ready for Ideogram layout ✓
            </p>
          )}
        </div>
      ) : null}

      {boxes.length ? (
        <div className="ie-section">
          <div className="ie-sec-title">Generate</div>
          {editModels.length ? (
            <>
              <select className="ie-select" onChange={(event) => setEditModel(event.target.value)} value={editModel}>
                {editModels.map((model) => (
                  <option key={model.id} value={model.id}>
                    {updateOptionLabel(model)}
                  </option>
                ))}
              </select>
              <StudioUpdateBadge item={selectedEditModel} />
              <StudioUpdateNotice item={selectedEditModel} onUpdate={createModelDownloadJob} />
              <input
                className="ie-input"
                onChange={(event) => setEditPrompt(event.target.value)}
                onKeyDown={(event) => {
                  if (event.key === "Enter" && !aiOp && editModel) runBoxEdit();
                }}
                placeholder="Prompt (or use Auto-prompt)"
                value={editPrompt}
              />
              <div style={{ display: "grid", gridTemplateColumns: "auto 1fr", gap: "8px" }}>
                <button className="ie-btn" disabled={!composedPrompt} onClick={() => setEditPrompt(composedPrompt)} type="button">
                  Auto-prompt
                </button>
                <button className="ie-btn primary" disabled={!!aiOp || !editModel} onClick={runBoxEdit} type="button">
                  Generate
                </button>
              </div>
            </>
          ) : (
            <p className="ie-note">No edit-capable models installed.</p>
          )}
        </div>
      ) : null}
    </>
  );
  return renderPanel();
}, samePanelProps);

export const ImageEditorToolPanel = React.memo(function ImageEditorToolPanel({ panelKey, scope }) {
  renderObserverForTests?.();
  const { COLOR_ADJUSTMENTS, CROP_RATIOS, CurveEditor, StudioUpdateBadge, StudioUpdateNotice, UPSCALE_ENGINE_DESC, activeGradeIsIdentity, activeLayerOf, actualSize, aiOp, applyColorGrade, applyCrop, availableUpscaleEngines, cancelCrop, channelStroke, chooseRatio, colorAdjust, colorChannel, colorMode, createModelDownloadJob, cropRect, curves, detailCnScale, detailModel, detailModels, detailStrength, endTransformGesture, fitToView, flipActiveLayer, histogramRef, identityTransform, layerCount, levels, onTransformSlider, ratioKey, requestTileControlNetDownload, resetActiveColorMode, resetActiveLayerTransform, resetAdjust, rotated, runDetail, runUpscale, selectedDetailModel, setActiveTransform, setAdjustValue, setColorChannel, setColorMode, setCropDim, setCurves, setDetailCnScale, setDetailModel, setDetailStrength, setLevelsValue, setStraighten, setTool, setUpscaleEngine, setUpscaleFactor, setUpscaleSoftness, straighten, tileControlNet, tileControlNetDownloadRequested, tileControlNetReady, toggleRotate, updateOptionLabel, upscaleEngine, upscaleEngineHasSoftness, upscaleFactor, upscaleFactorsForEngine, upscaleSoftness, working } = scope;
  const renderPanel = (key) => {
    switch (key) {
      case "move":
        return (
          <>
            <div className="ie-section">
              <div className="ie-sec-title">Document</div>
              <div className="ie-readout">
                <span className="ie-readout-k">Dimensions</span>
                <span className="ie-readout-v">
                  {working.width} × {working.height}
                </span>
              </div>
              <div className="ie-readout">
                <span className="ie-readout-k">Layers</span>
                <span className="ie-readout-v">{layerCount}</span>
              </div>
              <p className="ie-note">
                Drag on the canvas to pan. Scroll to zoom. Pick a tool to start editing — each tool&apos;s controls appear
                here.
              </p>
            </div>
            <div className="ie-section">
              <div className="ie-sec-title">Quick actions</div>
              <button className="ie-btn block" onClick={fitToView} type="button">
                Fit to view
              </button>
              <button className="ie-btn block" onClick={actualSize} type="button">
                Actual size (100%)
              </button>
            </div>
          </>
        );
      case "transform": {
        const tLayer = activeLayerOf(working);
        const t = tLayer?.transform ?? identityTransform();
        const scalePct = Math.round(Math.abs(t.scaleX) * 100);
        const signX = t.scaleX < 0 ? -1 : 1;
        const signY = t.scaleY < 0 ? -1 : 1;
        return (
          <>
            <div className="ie-section">
              <div className="ie-sec-title">Position</div>
              <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: "10px" }}>
                <div className="ie-field">
                  <div className="ie-field-top">
                    <span className="ie-field-label">X</span>
                  </div>
                  <input
                    className="ie-input ie-numfield"
                    onChange={(event) => setActiveTransform({ x: Number(event.target.value) || 0 })}
                    type="number"
                    value={Math.round(t.x)}
                  />
                </div>
                <div className="ie-field">
                  <div className="ie-field-top">
                    <span className="ie-field-label">Y</span>
                  </div>
                  <input
                    className="ie-input ie-numfield"
                    onChange={(event) => setActiveTransform({ y: Number(event.target.value) || 0 })}
                    type="number"
                    value={Math.round(t.y)}
                  />
                </div>
              </div>
            </div>
            <div className="ie-section">
              <div className="ie-sec-title">Scale &amp; rotation</div>
              <div className="ie-field">
                <div className="ie-field-top">
                  <span className="ie-field-label">Scale</span>
                  <span className="ie-field-val">{scalePct}%</span>
                </div>
                <input
                  className="ie-range"
                  max={300}
                  min={10}
                  onBlur={endTransformGesture}
                  onChange={(event) => {
                    const pct = Number(event.target.value) / 100;
                    onTransformSlider({ scaleX: signX * pct, scaleY: signY * pct });
                  }}
                  onMouseUp={endTransformGesture}
                  onTouchEnd={endTransformGesture}
                  type="range"
                  value={scalePct}
                />
              </div>
              <div className="ie-field">
                <div className="ie-field-top">
                  <span className="ie-field-label">Rotation</span>
                  <span className="ie-field-val">{Math.round(t.rotation)}°</span>
                </div>
                <input
                  className="ie-range"
                  max={180}
                  min={-180}
                  onBlur={endTransformGesture}
                  onChange={(event) => onTransformSlider({ rotation: Number(event.target.value) })}
                  onMouseUp={endTransformGesture}
                  onTouchEnd={endTransformGesture}
                  type="range"
                  value={Math.round(t.rotation)}
                />
              </div>
              <div className="ie-seg two" style={{ width: "100%" }}>
                <button className="ie-seg-btn" onClick={() => flipActiveLayer("h")} type="button">
                  Flip horizontal
                </button>
                <button className="ie-seg-btn" onClick={() => flipActiveLayer("v")} type="button">
                  Flip vertical
                </button>
              </div>
            </div>
            <div className="ie-section">
              <p className="ie-note">You can also drag the handles on the canvas to move, scale or rotate the layer.</p>
              <button className="ie-btn block" onClick={resetActiveLayerTransform} type="button">
                Reset transform
              </button>
              <button className="ie-btn block primary" onClick={() => setTool("move")} type="button">
                Done
              </button>
            </div>
          </>
        );
      }
      case "crop":
        return (
          <>
            <div className="ie-section">
              <div className="ie-sec-title">Aspect ratio</div>
              <div className="ie-seg wrap" style={{ gridTemplateColumns: "repeat(3, 1fr)" }}>
                {CROP_RATIOS.map((entry) => (
                  <button
                    className="ie-seg-btn"
                    data-active={ratioKey === entry.key}
                    key={entry.key}
                    onClick={() => chooseRatio(entry.key)}
                    type="button"
                  >
                    {entry.label}
                  </button>
                ))}
              </div>
              <button
                className="ie-btn block"
                data-active={rotated}
                disabled={ratioKey === "free" || ratioKey === "1:1"}
                onClick={toggleRotate}
                type="button"
              >
                ⟲ Swap orientation
              </button>
            </div>
            <div className="ie-section">
              <div className="ie-sec-title">Size</div>
              <div style={{ display: "grid", gridTemplateColumns: "1fr auto 1fr", gap: "8px", alignItems: "end" }}>
                <div className="ie-field">
                  <div className="ie-field-top">
                    <span className="ie-field-label">Width</span>
                  </div>
                  <input
                    className="ie-input ie-numfield"
                    onChange={(event) => setCropDim("width", event.target.value)}
                    type="number"
                    value={cropRect ? Math.round(cropRect.width) : ""}
                  />
                </div>
                <span style={{ paddingBottom: "10px", color: "var(--ie-faint)" }}>×</span>
                <div className="ie-field">
                  <div className="ie-field-top">
                    <span className="ie-field-label">Height</span>
                  </div>
                  <input
                    className="ie-input ie-numfield"
                    onChange={(event) => setCropDim("height", event.target.value)}
                    type="number"
                    value={cropRect ? Math.round(cropRect.height) : ""}
                  />
                </div>
              </div>
              <div className="ie-field">
                <div className="ie-field-top">
                  <span className="ie-field-label">Straighten</span>
                  <span className="ie-field-val">
                    {straighten > 0 ? "+" : ""}
                    {straighten}°
                  </span>
                </div>
                <input
                  className="ie-range"
                  max={15}
                  min={-15}
                  onChange={(event) => setStraighten(Number(event.target.value))}
                  type="range"
                  value={straighten}
                />
                <p className="ie-note">Rotates the image within the crop; applied on Apply. Inset the crop so the corners stay filled.</p>
              </div>
            </div>
            <div className="ie-section">
              <button className="ie-btn block primary" onClick={applyCrop} type="button">
                Apply crop
              </button>
              <button className="ie-btn block" onClick={cancelCrop} type="button">
                Cancel
              </button>
            </div>
          </>
        );
      case "upscale":
        return (
          <>
            <div className="ie-section">
              <div className="ie-sec-title">Engine</div>
              <div className="ie-cards">
                {availableUpscaleEngines.map((entry) => (
                  <button
                    className="ie-card"
                    data-active={upscaleEngine === entry.key}
                    key={entry.key}
                    onClick={() => {
                      setUpscaleEngine(entry.key);
                      if (!entry.factors.includes(upscaleFactor)) setUpscaleFactor(entry.factors[0]);
                    }}
                    type="button"
                  >
                    <span className="ie-radio" />
                    <span>
                      <span className="ie-card-name">{entry.label}</span>
                      {UPSCALE_ENGINE_DESC[entry.key] ? (
                        <span className="ie-card-desc">{UPSCALE_ENGINE_DESC[entry.key]}</span>
                      ) : null}
                    </span>
                  </button>
                ))}
              </div>
            </div>
            <div className="ie-section">
              <div className="ie-sec-title">Scale factor</div>
              <div className="ie-seg wrap two">
                {upscaleFactorsForEngine(upscaleEngine).map((value) => (
                  <button
                    className="ie-seg-btn"
                    data-active={upscaleFactor === value}
                    key={value}
                    onClick={() => setUpscaleFactor(value)}
                    type="button"
                  >
                    {value}×
                  </button>
                ))}
              </div>
              {upscaleEngineHasSoftness(upscaleEngine) ? (
                <div className="ie-field">
                  <div className="ie-field-top">
                    <span className="ie-field-label">Detail recovery</span>
                    <span className="ie-field-val">{upscaleSoftness.toFixed(2)}</span>
                  </div>
                  <input
                    className="ie-range"
                    max={1}
                    min={0}
                    onChange={(event) => setUpscaleSoftness(Number(event.target.value))}
                    step={0.05}
                    type="range"
                    value={upscaleSoftness}
                  />
                  <p className="ie-note">Higher restores more texture from a degraded source; 0 stays faithful to the original.</p>
                </div>
              ) : null}
              <div className="ie-readout">
                <span className="ie-readout-k">Output size</span>
                <span className="ie-readout-v">
                  {working.width * upscaleFactor} × {working.height * upscaleFactor}
                </span>
              </div>
            </div>
            <div className="ie-section">
              <button className="ie-btn block primary" disabled={!!aiOp} onClick={runUpscale} type="button">
                Upscale image
              </button>
              <button className="ie-btn block" onClick={() => setTool("move")} type="button">
                Cancel
              </button>
            </div>
          </>
        );
      case "detail":
        return detailModels.length === 0 ? (
          <div className="ie-section">
            <p className="ie-note">No detail-capable models installed.</p>
          </div>
        ) : (
          <>
            <div className="ie-section">
              <div className="ie-sec-title">Backbone</div>
              <select className="ie-select" onChange={(event) => setDetailModel(event.target.value)} value={detailModel}>
                {detailModels.map((model) => (
                  <option key={model.id} value={model.id}>
                    {updateOptionLabel(model)}
                  </option>
                ))}
              </select>
              <StudioUpdateBadge item={selectedDetailModel} />
              <StudioUpdateNotice item={selectedDetailModel} onUpdate={createModelDownloadJob} />
            </div>
            <div className="ie-section">
              <div className="ie-sec-title">Refinement</div>
              <div className="ie-field">
                <div className="ie-field-top">
                  <span className="ie-field-label">Detail amount</span>
                  <span className="ie-field-val">{Math.round(detailStrength * 100)}%</span>
                </div>
                <input
                  className="ie-range"
                  max={0.8}
                  min={0.3}
                  onChange={(event) => setDetailStrength(Number(event.target.value))}
                  step={0.05}
                  type="range"
                  value={detailStrength}
                />
                <p className="ie-note">Higher invents more fine texture.</p>
              </div>
              <div className="ie-field">
                <div className="ie-field-top">
                  <span className="ie-field-label">Structure lock</span>
                  <span className="ie-field-val">{Math.round(detailCnScale * 100)}%</span>
                </div>
                <input
                  className="ie-range"
                  max={1}
                  min={0.4}
                  onChange={(event) => setDetailCnScale(Number(event.target.value))}
                  step={0.05}
                  type="range"
                  value={detailCnScale}
                />
                <p className="ie-note">Higher keeps the result closer to the source composition.</p>
              </div>
            </div>
            {/* Tile ControlNet dependency (sc-2437/sc-2438): Detail can't run without it, but it's a
                separate utility download — surface it as required with a one-click install when missing,
                and gate the run. Mirrors the managed edit-LoRA CTA in the AI Edit panel. */}
            <div className="ie-section">
              <div className="ie-sec-title">Required model</div>
              {tileControlNetReady ? (
                <p className="ie-note">✨ {tileControlNet.name ?? "SDXL Tile ControlNet"} is installed and ready.</p>
              ) : (
                <>
                  <p className="ie-note">
                    The SDXL tile ControlNet (~2.5 GB) is required for Detail enhance and isn’t installed yet.
                  </p>
                  <button
                    className="ie-btn block"
                    disabled={!tileControlNet || !createModelDownloadJob || tileControlNetDownloadRequested}
                    onClick={requestTileControlNetDownload}
                    type="button"
                  >
                    {tileControlNetDownloadRequested
                      ? "Downloading…"
                      : `Download ${tileControlNet?.name ?? "SDXL Tile ControlNet"}`}
                  </button>
                </>
              )}
            </div>
            <div className="ie-section">
              <button
                className="ie-btn block primary"
                disabled={!!aiOp || !detailModel || !tileControlNetReady}
                onClick={runDetail}
                type="button"
              >
                Enhance detail
              </button>
              <button className="ie-btn block" onClick={() => setTool("move")} type="button">
                Cancel
              </button>
            </div>
          </>
        );
      case "color":
        return (
          <>
            <div className="ie-section">
              <div className="ie-seg wrap" style={{ gridTemplateColumns: "repeat(3, 1fr)" }}>
                {[
                  ["adjust", "Adjust"],
                  ["levels", "Levels"],
                  ["curves", "Curves"],
                ].map(([mode, label]) => (
                  <button
                    className="ie-seg-btn"
                    data-active={colorMode === mode}
                    key={mode}
                    onClick={() => setColorMode(mode)}
                    type="button"
                  >
                    {label}
                  </button>
                ))}
              </div>
            </div>
            {colorMode === "adjust" ? (
              <div className="ie-section">
                <div className="ie-sec-title">Tone &amp; color</div>
                {COLOR_ADJUSTMENTS.map(({ key: adjKey, label }) => (
                  <div className="ie-field" key={adjKey}>
                    <div className="ie-field-top">
                      <span className="ie-field-label">{label}</span>
                      <span className="ie-field-val">
                        {colorAdjust[adjKey] > 0 ? "+" : ""}
                        {Math.round(colorAdjust[adjKey] * 100)}
                      </span>
                    </div>
                    <input
                      className="ie-range"
                      max={1}
                      min={-1}
                      onChange={(event) => setAdjustValue(adjKey, Number(event.target.value))}
                      onDoubleClick={() => resetAdjust(adjKey)}
                      step={0.01}
                      type="range"
                      value={colorAdjust[adjKey]}
                    />
                  </div>
                ))}
              </div>
            ) : null}
            {colorMode === "levels" ? (
              <div className="ie-section">
                <div className="ie-sec-title">Channel</div>
                <select className="ie-select" onChange={(event) => setColorChannel(event.target.value)} value={colorChannel}>
                  <option value="master">Master (RGB)</option>
                  <option value="r">Red</option>
                  <option value="g">Green</option>
                  <option value="b">Blue</option>
                </select>
                <canvas className="ie-histo" height={56} ref={histogramRef} width={280} />
                <div className="ie-field">
                  <div className="ie-field-top">
                    <span className="ie-field-label">Black point</span>
                    <span className="ie-field-val">{levels[colorChannel].black}</span>
                  </div>
                  <input
                    className="ie-range"
                    max={254}
                    min={0}
                    onChange={(event) => setLevelsValue("black", Number(event.target.value))}
                    step={1}
                    type="range"
                    value={levels[colorChannel].black}
                  />
                </div>
                <div className="ie-field">
                  <div className="ie-field-top">
                    <span className="ie-field-label">Gamma</span>
                    <span className="ie-field-val">{levels[colorChannel].gamma.toFixed(2)}</span>
                  </div>
                  <input
                    className="ie-range"
                    max={2.5}
                    min={0.1}
                    onChange={(event) => setLevelsValue("gamma", Number(event.target.value))}
                    step={0.01}
                    type="range"
                    value={levels[colorChannel].gamma}
                  />
                </div>
                <div className="ie-field">
                  <div className="ie-field-top">
                    <span className="ie-field-label">White point</span>
                    <span className="ie-field-val">{levels[colorChannel].white}</span>
                  </div>
                  <input
                    className="ie-range"
                    max={255}
                    min={1}
                    onChange={(event) => setLevelsValue("white", Number(event.target.value))}
                    step={1}
                    type="range"
                    value={levels[colorChannel].white}
                  />
                </div>
              </div>
            ) : null}
            {colorMode === "curves" ? (
              <div className="ie-section">
                <div className="ie-sec-title">Tone curve</div>
                <select className="ie-select" onChange={(event) => setColorChannel(event.target.value)} value={colorChannel}>
                  <option value="master">Master (RGB)</option>
                  <option value="r">Red</option>
                  <option value="g">Green</option>
                  <option value="b">Blue</option>
                </select>
                <div className="ie-curvewrap">
                  <CurveEditor
                    onChange={(points) => setCurves((prev) => ({ ...prev, [colorChannel]: points }))}
                    points={curves[colorChannel]}
                    stroke={channelStroke}
                  />
                </div>
                <p className="ie-note">Drag points to reshape the curve. Double-click to add a point.</p>
              </div>
            ) : null}
            <div className="ie-section">
              <button className="ie-btn block" disabled={activeGradeIsIdentity()} onClick={resetActiveColorMode} type="button">
                Reset
              </button>
              <button
                className="ie-btn block primary"
                disabled={activeGradeIsIdentity()}
                onClick={applyColorGrade}
                type="button"
              >
                Apply grade
              </button>
            </div>
          </>
        );
      case "edit":
        return <ImageEditorEditPanel scope={scope} />;
      case "boxes":
        return <ImageEditorBoxesPanel scope={scope} />;
      default:
        return null;
    }
  };
  return renderPanel(panelKey);
}, samePanelProps);
