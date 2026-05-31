import React, { useCallback, useEffect, useRef, useState } from "react";
import { Stage, Layer, Image as KonvaImage, Rect } from "react-konva";
import { useAppContext } from "../context/AppContext.js";
import { assetUrl, assetCanRenderAsImage } from "../components/assetMedia.jsx";
import { AssetPickerModal } from "../components/AssetPicker.jsx";

const MIN_SCALE = 0.05;
const MAX_SCALE = 16;
const ZOOM_STEP = 1.2;
// Leave room for the page chrome (topbar + surface header) so the canvas grows
// with the viewport without pushing the page into a scroll.
const CANVAS_MIN_HEIGHT = 420;

// Tools that later stories in epic 2427 fill in. Rendered as an inert scaffold
// here so the editor frame (and the next slices' insertion points) are in place.
const UPCOMING_TOOLS = [
  { id: "crop", label: "Crop", story: "sc-2430" },
  { id: "upscale", label: "Upscale", story: "sc-2433" },
  { id: "edit", label: "AI Edit", story: "sc-2435" },
  { id: "detail", label: "Detail", story: "sc-2438" },
  { id: "color", label: "Color", story: "sc-2439" },
];

const clamp = (value, min, max) => Math.min(max, Math.max(min, value));

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

  const containerRef = useRef(null);
  const fileInputRef = useRef(null);
  const objectUrlRef = useRef(null);
  const needsFitRef = useRef(false);
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
          <button className="image-editor-tool active" type="button" title="Move / pan (default)">
            Move
          </button>
          {UPCOMING_TOOLS.map((tool) => (
            <button
              className="image-editor-tool"
              disabled
              key={tool.id}
              title={`${tool.label} — coming soon (${tool.story})`}
              type="button"
            >
              {tool.label}
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
              draggable
              height={stageSize.height}
              onDragEnd={(event) => {
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
