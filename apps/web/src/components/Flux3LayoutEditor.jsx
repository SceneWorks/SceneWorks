import React, { useRef, useState } from "react";
import { rectFromPoints } from "../screens/imageEditor/boxGeometry.js";
import "./Flux3LayoutEditor.css";

const DEFAULT_BOX = [0.25, 0.25, 0.75, 0.75];
const SIDES = ["top", "left", "bottom", "right"];

// Controlled authoring document, deliberately separate from the server's rounded locked table.
// Reference ids identify attachment instances, not ordinals; reordering never rewrites saved data.
export function Flux3LayoutEditor({ value = { version: 1, boxes: [] }, onChange, references = [], width = 1024, height = 1024, disabled = false }) {
  const [selectedId, setSelectedId] = useState(null);
  const gesture = useRef(null);
  const boxes = value.boxes ?? [];
  const selected = boxes.find((box) => box.id === selectedId);
  const source = references.find((ref) => ref.id === selected?.reference_id);
  const isEdit = selected && !Object.hasOwn(selected, "bbox");
  const patch = (changes) => onChange({ ...value, boxes: boxes.map((box) => box.id === selectedId ? { ...box, ...changes } : box) });
  const add = (edit) => {
    const id = `box_${crypto.randomUUID().replaceAll("-", "_")}`;
    const box = edit
      ? { id, desc: "", from: "ref_image_0", reference_id: references[0].id, src_bbox: [...DEFAULT_BOX], tgt_bbox: [...DEFAULT_BOX] }
      : { id, desc: "", bbox: [...DEFAULT_BOX] };
    onChange({ ...value, boxes: [...boxes, box] });
    setSelectedId(id);
  };
  const coordinateKey = (kind) => kind === "source" ? "src_bbox" : isEdit ? "tgt_bbox" : "bbox";
  const updateCoordinates = (kind, index, text) => {
    if (text.trim() === "") return;
    const number = Number(text);
    if (!Number.isFinite(number)) return;
    const key = coordinateKey(kind);
    const coordinates = [...selected[key]];
    coordinates[index] = number;
    patch({ [key]: coordinates });
  };
  const point = (event) => {
    const rect = event.currentTarget.getBoundingClientRect();
    if (!rect.width || !rect.height) return null;
    return { x: (event.clientX - rect.left) / rect.width, y: (event.clientY - rect.top) / rect.height };
  };
  const drawStart = (kind, event) => {
    if (disabled || !selected || (kind === "source" && !isEdit) || (kind === "target" && isEdit && selected.tgt_bbox === null)) return;
    const space = selected[kind === "source" ? "src_space" : isEdit ? "tgt_space" : "bbox_space"];
    if (space && space !== "normalized") return; // never reinterpret existing pixel coordinates
    const start = point(event);
    if (!start) return;
    gesture.current = { kind, start, id: selectedId };
    event.currentTarget.setPointerCapture?.(event.pointerId);
  };
  const drawEnd = (kind, event) => {
    const active = gesture.current;
    gesture.current = null;
    if (!active || active.kind !== kind || active.id !== selectedId || disabled) return;
    const end = point(event);
    if (!end) return;
    const rect = rectFromPoints(active.start, end);
    if (rect.width <= 0 || rect.height <= 0) return;
    patch({ [coordinateKey(kind)]: [rect.y, rect.x, rect.y + rect.height, rect.x + rect.width] });
  };
  const renderCanvas = (kind) => {
    const sourceCanvas = kind === "source";
    const visible = boxes.filter((box) => sourceCanvas
      ? box.reference_id === source?.id && box.src_bbox && (!box.src_space || box.src_space === "normalized")
      : (box.bbox || box.tgt_bbox) && (!((box.bbox ? box.bbox_space : box.tgt_space)) || (box.bbox ? box.bbox_space : box.tgt_space) === "normalized"));
    return (
      <div className="flux3-layout-pane">
        <span className="field-label">{sourceCanvas ? "Selected reference" : "Output"}</span>
        <svg
          aria-label={sourceCanvas ? "Source reference canvas" : "Output layout canvas"}
          role="img" viewBox="0 0 1 1" preserveAspectRatio="none"
          style={{ aspectRatio: sourceCanvas ? `${source?.width || width} / ${source?.height || height}` : `${width} / ${height}` }}
          onPointerDown={(event) => drawStart(kind, event)}
          onPointerUp={(event) => drawEnd(kind, event)}
          onPointerCancel={() => { gesture.current = null; }}
        >
          {sourceCanvas && source?.url && <image href={source.url} width="1" height="1" preserveAspectRatio="none" />}
          {visible.map((box) => {
            const coords = sourceCanvas ? box.src_bbox : box.bbox || box.tgt_bbox;
            return <rect key={box.id} x={coords[1]} y={coords[0]} width={coords[3] - coords[1]} height={coords[2] - coords[0]} className={box.id === selectedId ? "selected" : ""} vectorEffect="non-scaling-stroke"><title>{box.id}: {box.desc}</title></rect>;
          })}
        </svg>
        <span className="muted">{sourceCanvas ? source?.label || "Select a source edit" : `${width} × ${height}`}</span>
      </div>
    );
  };
  const renderCoordinates = (kind) => {
    const key = coordinateKey(kind);
    const coords = selected[key];
    if (!coords) return null;
    const space = selected[kind === "source" ? "src_space" : isEdit ? "tgt_space" : "bbox_space"];
    return <fieldset className="flux3-layout-coordinates" disabled={disabled}>
      <legend>{kind === "source" ? "Source" : "Target"} coordinates ({space || "normalized"}, YXYX)</legend>
      {SIDES.map((side, index) => <label key={side}>{side}<input aria-label={`${kind === "source" ? "Source" : "Target"} ${side}`} type="number" step="any" value={coords[index]} onChange={(event) => updateCoordinates(kind, index, event.target.value)} /></label>)}
    </fieldset>;
  };
  return <section className="flux3-layout-editor" aria-label="FLUX 3 layout editor">
    <div className="flux3-layout-toolbar">
      <button type="button" className="secondary-action" disabled={disabled} onClick={() => add(false)}>Add placement box</button>
      <button type="button" className="secondary-action" disabled={disabled || references.length === 0} onClick={() => add(true)}>Add source edit</button>
    </div>
    <p className="muted">Boxes provide soft spatial guidance, not hard masks or pixel-preserving edits. Select a control, then draw its source or target box, or enter precise coordinates.</p>
    <div className="flux3-layout-canvases">{isEdit && renderCanvas("source")}{renderCanvas("target")}</div>
    <div className="flux3-layout-controls">
      <ol className="flux3-layout-list">{boxes.map((box) => <li key={box.id}><button type="button" className="secondary-action" aria-label={`Select ${box.id}`} aria-pressed={box.id === selectedId} onClick={() => setSelectedId(box.id)}>{box.desc || box.id}<span className="muted">{box.bbox ? "Placement" : box.tgt_bbox === null ? "Removal" : "Move"}</span></button></li>)}</ol>
      {selected && <div className="flux3-layout-inspector">
        <label className="field-label">Box label<input aria-label="Box label" value={selected.desc ?? ""} disabled={disabled} onChange={(event) => patch({ desc: event.target.value })} /></label>
        {isEdit && <>
          <label className="field-label">Source reference<select aria-label="Source reference" value={selected.reference_id || ""} disabled={disabled} onChange={(event) => patch({ reference_id: event.target.value, from: `ref_image_${references.findIndex((ref) => ref.id === event.target.value)}` })}>
            {!source && <option value="">Select an attached reference</option>}
            {references.map((ref, index) => <option key={ref.id} value={ref.id}>{index + 1}. {ref.label || ref.id}</option>)}
          </select></label>
          <div className="flux3-layout-toolbar">
            <button type="button" className="secondary-action" disabled={disabled || selected.tgt_bbox !== null} onClick={() => patch({ tgt_bbox: [...DEFAULT_BOX], ...(selected.tgt_space ? { tgt_space: "normalized" } : {}) })}>Move object to target</button>
            <button type="button" className="secondary-action" disabled={disabled || selected.tgt_bbox === null} onClick={() => patch({ tgt_bbox: null })}>Remove object from output</button>
          </div>
          {renderCoordinates("source")}
        </>}
        {renderCoordinates("target")}
        <button type="button" className="secondary-action" disabled={disabled} onClick={() => { onChange({ ...value, boxes: boxes.filter((box) => box.id !== selectedId) }); setSelectedId(null); }}>Delete control</button>
      </div>}
    </div>
  </section>;
}
