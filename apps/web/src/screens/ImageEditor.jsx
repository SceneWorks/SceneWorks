import React, { useCallback, useEffect, useRef, useState } from "react";
import { Stage, Layer, Image as KonvaImage, Rect, Transformer } from "react-konva";
import { useAppContext } from "../context/AppContext.js";
import { assetUrl, assetCanRenderAsImage } from "../components/assetMedia.jsx";
import { AssetPickerModal } from "../components/AssetPicker.jsx";

const MIN_SCALE = 0.05;
const MAX_SCALE = 16;
const ZOOM_STEP = 1.2;
const MIN_CROP_PX = 8;

// Tools still to come in epic 2427 — rendered as an inert scaffold so the frame
// (and the next slices' insertion points) are in place. Move + Crop are live.
const UPCOMING_TOOLS = [
  { id: "upscale", label: "Upscale", story: "sc-2433" },
  { id: "edit", label: "AI Edit", story: "sc-2435" },
  { id: "detail", label: "Detail", story: "sc-2438" },
  { id: "color", label: "Color", story: "sc-2439" },
];

// Predefined crop ratios (width / height). Rotate swaps to the transpose; 1:1 and
// Freeform are unaffected.
const CROP_RATIOS = [
  { key: "free", label: "Freeform", ratio: null },
  { key: "1:1", label: "1:1", ratio: 1 },
  { key: "3:4", label: "3:4", ratio: 3 / 4 },
  { key: "5:7", label: "5:7", ratio: 5 / 7 },
  { key: "8:10", label: "8:10", ratio: 8 / 10 },
  { key: "16:9", label: "16:9", ratio: 16 / 9 },
];

const clamp = (value, min, max) => Math.min(max, Math.max(min, value));

// Resolve a ratio key (+ rotate) to a concrete width/height ratio, or null for
// freeform. Rotating transposes non-square ratios (3:4 → 4:3); 1:1 is a no-op.
export function cropRatioForKey(key, rotated) {
  const found = CROP_RATIOS.find((entry) => entry.key === key);
  const base = found ? found.ratio : null;
  if (base == null || base === 1) return base;
  return rotated ? 1 / base : base;
}

// Largest rect of the given ratio that fits in the image, centered. Freeform
// (null ratio) defaults to a centered 80% box. Returns image-pixel coords.
export function centeredCropRect(imgW, imgH, ratio) {
  if (ratio == null) {
    const w = imgW * 0.8;
    const h = imgH * 0.8;
    return { x: (imgW - w) / 2, y: (imgH - h) / 2, width: w, height: h };
  }
  let w = imgW;
  let h = w / ratio;
  if (h > imgH) {
    h = imgH;
    w = h * ratio;
  }
  return { x: (imgW - w) / 2, y: (imgH - h) / 2, width: w, height: h };
}

// The four dim rectangles that mask everything outside the crop rect (image coords).
function cropOverlayRects(imgW, imgH, rect) {
  const right = rect.x + rect.width;
  const bottom = rect.y + rect.height;
  return [
    { x: 0, y: 0, width: imgW, height: rect.y },
    { x: 0, y: bottom, width: imgW, height: imgH - bottom },
    { x: 0, y: rect.y, width: rect.x, height: rect.height },
    { x: right, y: rect.y, width: imgW - right, height: rect.height },
  ];
}

// Decode a blob into an HTMLImageElement via a same-origin object: URL. Asset
// files are served cross-origin from the API in local dev, so loading the bytes
// this way (rather than an <img crossOrigin> against the file URL) guarantees the
// Konva canvas is never tainted — later crop/export (sc-2430/sc-2434) need to read
// pixels back. Resolves { image, objectUrl }; caller owns revoking objectUrl.
function blobToImage(blob) {
  return new Promise((resolve, reject) => {
    const objectUrl = URL.createObjectURL(blob);
    const image = new Image();
    image.onload = () => resolve({ image, objectUrl });
    image.onerror = () => {
      URL.revokeObjectURL(objectUrl);
      reject(new Error("Could not decode image"));
    };
    image.src = objectUrl;
  });
}

export function ImageEditor() {
  const { activeProject, assets, setPreviewAsset } = useAppContext();

  // The working-image session: the single bitmap every tool operates on, plus its
  // provenance. This state is the contract consumed by crop/upscale/save and the
  // later AI tools (epic 2427). `objectUrl` is tracked so we can revoke it.
  const [working, setWorking] = useState(null);
  const [status, setStatus] = useState({ loading: false, error: "" });
  const [pickerOpen, setPickerOpen] = useState(false);
  const [view, setView] = useState({ scale: 1, x: 0, y: 0 });

  // Crop tool (sc-2430): client-side, rasterized into a new working image on Apply.
  const [tool, setTool] = useState("move");
  const [ratioKey, setRatioKey] = useState("free");
  const [rotated, setRotated] = useState(false);
  const [cropRect, setCropRect] = useState(null); // image-pixel coords, or null

  const containerRef = useRef(null);
  const fileInputRef = useRef(null);
  const objectUrlRef = useRef(null);
  const needsFitRef = useRef(false);
  const cropRectRef = useRef(null);
  const transformerRef = useRef(null);
  const [stageSize, setStageSize] = useState({ width: 0, height: 0 });

  const imageAssets = (assets ?? []).filter(assetCanRenderAsImage);

  // Track the container size so the Konva stage fills the available canvas area.
  // Measure once up front (a ResizeObserver alone can miss the first layout) and
  // then observe for later window / layout changes.
  useEffect(() => {
    const el = containerRef.current;
    if (!el) return undefined;
    const measure = () => setStageSize({ width: el.clientWidth, height: el.clientHeight });
    measure();
    if (typeof ResizeObserver === "undefined") return undefined;
    const observer = new ResizeObserver(measure);
    observer.observe(el);
    return () => observer.disconnect();
  }, []);

  // Revoke the live object URL when the editor unmounts.
  useEffect(() => () => {
    if (objectUrlRef.current) URL.revokeObjectURL(objectUrlRef.current);
  }, []);

  const fitToView = useCallback(() => {
    if (!working || !stageSize.width || !stageSize.height) return;
    const scale = clamp(
      Math.min(stageSize.width / working.width, stageSize.height / working.height) * 0.92,
      MIN_SCALE,
      MAX_SCALE,
    );
    setView({
      scale,
      x: (stageSize.width - working.width * scale) / 2,
      y: (stageSize.height - working.height * scale) / 2,
    });
  }, [working, stageSize.width, stageSize.height]);

  // Fit a freshly loaded image once the stage has been measured (the stage may be
  // 0×0 on the first render before the ResizeObserver fires).
  useEffect(() => {
    if (needsFitRef.current && working && stageSize.width && stageSize.height) {
      needsFitRef.current = false;
      fitToView();
    }
  }, [working, stageSize.width, stageSize.height, fitToView]);

  const installWorkingImage = useCallback((image, objectUrl, source) => {
    if (objectUrlRef.current) URL.revokeObjectURL(objectUrlRef.current);
    objectUrlRef.current = objectUrl;
    needsFitRef.current = true;
    setTool("move");
    setCropRect(null);
    setWorking({
      image,
      width: image.naturalWidth,
      height: image.naturalHeight,
      source,
    });
  }, []);

  const openFromBlob = useCallback(
    async (blob, source) => {
      setStatus({ loading: true, error: "" });
      try {
        const { image, objectUrl } = await blobToImage(blob);
        installWorkingImage(image, objectUrl, source);
        setStatus({ loading: false, error: "" });
      } catch (err) {
        setStatus({ loading: false, error: err.message || "Could not open image" });
      }
    },
    [installWorkingImage],
  );

  const openAsset = useCallback(
    async (assetId) => {
      const asset = imageAssets.find((item) => item.id === assetId);
      if (!asset) return;
      const url = assetUrl(asset);
      if (!url) {
        setStatus({ loading: false, error: "Asset has no media file" });
        return;
      }
      setStatus({ loading: true, error: "" });
      try {
        const res = await fetch(url);
        if (!res.ok) throw new Error(`Failed to load asset (${res.status})`);
        const blob = await res.blob();
        await openFromBlob(blob, {
          kind: "asset",
          assetId: asset.id,
          name: asset.displayName ?? asset.id,
        });
      } catch (err) {
        setStatus({ loading: false, error: err.message || "Could not load asset" });
      }
    },
    [imageAssets, openFromBlob],
  );

  const openFile = useCallback(
    (file) => {
      if (!file || !file.type.startsWith("image/")) {
        setStatus({ loading: false, error: "Please choose an image file" });
        return;
      }
      openFromBlob(file, { kind: "upload", name: file.name });
    },
    [openFromBlob],
  );

  function handleDrop(event) {
    event.preventDefault();
    const file = event.dataTransfer?.files?.[0];
    if (file) openFile(file);
  }

  function handleWheel(event) {
    event.evt.preventDefault();
    const stage = event.target.getStage();
    const pointer = stage?.getPointerPosition();
    if (!pointer) return;
    const oldScale = view.scale;
    const newScale = clamp(oldScale * (event.evt.deltaY > 0 ? 1 / ZOOM_STEP : ZOOM_STEP), MIN_SCALE, MAX_SCALE);
    const mouseTo = { x: (pointer.x - view.x) / oldScale, y: (pointer.y - view.y) / oldScale };
    setView({ scale: newScale, x: pointer.x - mouseTo.x * newScale, y: pointer.y - mouseTo.y * newScale });
  }

  function zoomAtCenter(factor) {
    const cx = stageSize.width / 2;
    const cy = stageSize.height / 2;
    const oldScale = view.scale;
    const newScale = clamp(oldScale * factor, MIN_SCALE, MAX_SCALE);
    const mouseTo = { x: (cx - view.x) / oldScale, y: (cy - view.y) / oldScale };
    setView({ scale: newScale, x: cx - mouseTo.x * newScale, y: cy - mouseTo.y * newScale });
  }

  function actualSize() {
    if (!working) return;
    setView({
      scale: 1,
      x: (stageSize.width - working.width) / 2,
      y: (stageSize.height - working.height) / 2,
    });
  }

  // ── Crop ────────────────────────────────────────────────────────────────
  function startCrop() {
    if (!working) return;
    setTool("crop");
    setCropRect(centeredCropRect(working.width, working.height, cropRatioForKey(ratioKey, rotated)));
  }

  function cancelCrop() {
    setTool("move");
    setCropRect(null);
  }

  function chooseRatio(key) {
    setRatioKey(key);
    if (working) setCropRect(centeredCropRect(working.width, working.height, cropRatioForKey(key, rotated)));
  }

  function toggleRotate() {
    const next = !rotated;
    setRotated(next);
    if (working) setCropRect(centeredCropRect(working.width, working.height, cropRatioForKey(ratioKey, next)));
  }

  function clampCropToImage(rect) {
    const width = clamp(rect.width, MIN_CROP_PX, working.width);
    const height = clamp(rect.height, MIN_CROP_PX, working.height);
    return {
      width,
      height,
      x: clamp(rect.x, 0, working.width - width),
      y: clamp(rect.y, 0, working.height - height),
    };
  }

  function handleCropDragEnd() {
    const node = cropRectRef.current;
    if (!node) return;
    const next = clampCropToImage({ ...cropRect, x: node.x(), y: node.y() });
    node.position({ x: next.x, y: next.y });
    setCropRect(next);
  }

  function handleCropTransformEnd() {
    const node = cropRectRef.current;
    if (!node) return;
    const next = clampCropToImage({
      x: node.x(),
      y: node.y(),
      width: node.width() * node.scaleX(),
      height: node.height() * node.scaleY(),
    });
    node.scaleX(1);
    node.scaleY(1);
    node.setAttrs(next);
    setCropRect(next);
  }

  // Apply: rasterize the selected region into a fresh working image. The source
  // bitmap is blob-backed (never tainted), so reading pixels back is safe. The
  // result keeps the same source provenance so lineage survives to Save (sc-2434).
  const applyCrop = useCallback(async () => {
    if (!working || !cropRect) return;
    const sx = clamp(Math.round(cropRect.x), 0, working.width - 1);
    const sy = clamp(Math.round(cropRect.y), 0, working.height - 1);
    const sw = clamp(Math.round(cropRect.width), 1, working.width - sx);
    const sh = clamp(Math.round(cropRect.height), 1, working.height - sy);
    const canvas = document.createElement("canvas");
    canvas.width = sw;
    canvas.height = sh;
    canvas.getContext("2d").drawImage(working.image, sx, sy, sw, sh, 0, 0, sw, sh);
    const blob = await new Promise((resolve) => canvas.toBlob(resolve, "image/png"));
    if (!blob) return;
    const { image, objectUrl } = await blobToImage(blob);
    installWorkingImage(image, objectUrl, working.source);
  }, [working, cropRect, installWorkingImage]);

  // Bind the transformer to the crop rect whenever crop mode is active.
  useEffect(() => {
    const transformer = transformerRef.current;
    const node = cropRectRef.current;
    if (tool === "crop" && transformer && node) {
      transformer.nodes([node]);
      transformer.getLayer()?.batchDraw();
    }
  }, [tool, cropRect]);

  return (
    <section className="main-surface image-editor-surface">
      <div className="surface-header image-editor-header">
        <div className="section-heading">
          <p className="eyebrow">Image Editor</p>
          <h2>{working ? working.source.name : "Edit an image"}</h2>
        </div>
        <div className="image-editor-actions">
          <button onClick={() => setPickerOpen(true)} type="button" disabled={!activeProject}>
            Open from project
          </button>
          <button className="primary" onClick={() => fileInputRef.current?.click()} type="button">
            Upload image
          </button>
          {working ? (
            <button
              onClick={() => working.source.assetId && setPreviewAsset?.(imageAssets.find((a) => a.id === working.source.assetId))}
              type="button"
              disabled={!working.source.assetId}
              title={working.source.assetId ? "Preview the source asset" : "Uploaded image — not yet a project asset"}
            >
              Source
            </button>
          ) : null}
          <input
            accept="image/*"
            hidden
            onChange={(event) => {
              const file = event.target.files?.[0];
              if (file) openFile(file);
              event.target.value = "";
            }}
            ref={fileInputRef}
            type="file"
          />
        </div>
      </div>

      {status.error ? <div className="notice notice-error image-editor-notice">{status.error}</div> : null}

      <div className="image-editor-body">
        <aside className="image-editor-toolbar" aria-label="Editor tools">
          <button
            className={tool === "move" ? "image-editor-tool active" : "image-editor-tool"}
            onClick={cancelCrop}
            title="Move / pan"
            type="button"
          >
            Move
          </button>
          <button
            className={tool === "crop" ? "image-editor-tool active" : "image-editor-tool"}
            disabled={!working}
            onClick={startCrop}
            title="Crop"
            type="button"
          >
            Crop
          </button>
          {UPCOMING_TOOLS.map((upcoming) => (
            <button
              className="image-editor-tool"
              disabled
              key={upcoming.id}
              title={`${upcoming.label} — coming soon (${upcoming.story})`}
              type="button"
            >
              {upcoming.label}
            </button>
          ))}
        </aside>

        <div
          className="image-editor-canvas-wrap"
          onDragOver={(event) => event.preventDefault()}
          onDrop={handleDrop}
          ref={containerRef}
        >
          {working && stageSize.width > 0 && stageSize.height > 0 ? (
            <Stage
              draggable={tool !== "crop"}
              height={stageSize.height}
              onDragEnd={(event) => {
                if (event.target !== event.target.getStage()) return;
                const stage = event.target.getStage();
                setView((prev) => ({ ...prev, x: stage.x(), y: stage.y() }));
              }}
              onWheel={handleWheel}
              scaleX={view.scale}
              scaleY={view.scale}
              width={stageSize.width}
              x={view.x}
              y={view.y}
            >
              <Layer>
                <Rect
                  fill="#ffffff"
                  height={working.height}
                  shadowBlur={12}
                  shadowColor="rgba(0,0,0,0.35)"
                  width={working.width}
                  x={0}
                  y={0}
                />
                <KonvaImage height={working.height} image={working.image} width={working.width} x={0} y={0} />
                {tool === "crop" && cropRect ? (
                  <>
                    {cropOverlayRects(working.width, working.height, cropRect).map((rect, index) => (
                      <Rect
                        key={index}
                        fill="rgba(0,0,0,0.55)"
                        height={rect.height}
                        listening={false}
                        width={rect.width}
                        x={rect.x}
                        y={rect.y}
                      />
                    ))}
                    <Rect
                      draggable
                      fill="rgba(255,255,255,0.01)"
                      height={cropRect.height}
                      onDragEnd={handleCropDragEnd}
                      onTransformEnd={handleCropTransformEnd}
                      ref={cropRectRef}
                      stroke="#ffffff"
                      strokeScaleEnabled={false}
                      strokeWidth={2}
                      width={cropRect.width}
                      x={cropRect.x}
                      y={cropRect.y}
                    />
                    <Transformer
                      anchorSize={8}
                      borderStroke="#ffffff"
                      boundBoxFunc={(oldBox, newBox) =>
                        newBox.width < MIN_CROP_PX || newBox.height < MIN_CROP_PX ? oldBox : newBox
                      }
                      enabledAnchors={
                        ratioKey === "free"
                          ? ["top-left", "top-center", "top-right", "middle-left", "middle-right", "bottom-left", "bottom-center", "bottom-right"]
                          : ["top-left", "top-right", "bottom-left", "bottom-right"]
                      }
                      keepRatio={ratioKey !== "free"}
                      ref={transformerRef}
                      rotateEnabled={false}
                    />
                  </>
                ) : null}
              </Layer>
            </Stage>
          ) : (
            <div className="image-editor-empty">
              {status.loading ? (
                <p>Loading image…</p>
              ) : (
                <>
                  <p className="image-editor-empty-title">Open an image to start editing</p>
                  <p className="image-editor-empty-hint">
                    Drag &amp; drop an image here, upload a file, or open one from this project.
                  </p>
                </>
              )}
            </div>
          )}

          {tool === "crop" && cropRect ? (
            <div className="image-editor-cropbar">
              <div className="image-editor-ratios" role="group" aria-label="Crop ratio">
                {CROP_RATIOS.map((entry) => (
                  <button
                    className={ratioKey === entry.key ? "active" : ""}
                    key={entry.key}
                    onClick={() => chooseRatio(entry.key)}
                    type="button"
                  >
                    {entry.label}
                  </button>
                ))}
              </div>
              <button
                className={rotated ? "active" : ""}
                disabled={ratioKey === "free" || ratioKey === "1:1"}
                onClick={toggleRotate}
                title="Rotate ratio (swap orientation)"
                type="button"
              >
                ⟲ Rotate
              </button>
              <span className="image-editor-cropdims">
                {Math.round(cropRect.width)} × {Math.round(cropRect.height)}
              </span>
              <button className="primary" onClick={applyCrop} type="button">
                Apply
              </button>
              <button onClick={cancelCrop} type="button">
                Cancel
              </button>
            </div>
          ) : null}

          {working ? (
            <div className="image-editor-viewbar">
              <button onClick={() => zoomAtCenter(1 / ZOOM_STEP)} title="Zoom out" type="button">
                −
              </button>
              <span className="image-editor-zoom">{Math.round(view.scale * 100)}%</span>
              <button onClick={() => zoomAtCenter(ZOOM_STEP)} title="Zoom in" type="button">
                +
              </button>
              <button onClick={fitToView} type="button">
                Fit
              </button>
              <button onClick={actualSize} type="button">
                100%
              </button>
              <span className="image-editor-dims">
                {working.width} × {working.height}
              </span>
            </div>
          ) : null}
        </div>
      </div>

      {pickerOpen ? (
        <AssetPickerModal
          assets={imageAssets}
          initialSelectedIds={[]}
          multiple={false}
          onCancel={() => setPickerOpen(false)}
          onConfirm={(ids) => {
            setPickerOpen(false);
            if (ids[0]) openAsset(ids[0]);
          }}
          title="Open image"
        />
      ) : null}
    </section>
  );
}
