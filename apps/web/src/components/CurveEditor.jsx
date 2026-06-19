import React, { useEffect, useRef } from "react";
import { buildCurveLut, normalizeCurvePoints } from "../colorGrade.js";

// An editable tone-curve widget (sc-6109): a square graph over input→output in
// [0,255] (y up). Drag a control point to reshape the curve, click empty space to
// add a point, double-click an interior point to remove it. The endpoints' x is
// locked (0 and 255); interior points stay ordered (clamped between neighbors), so
// the point list never reorders mid-drag. Pure-math (the LUT + normalization) lives
// in colorGrade.js; this is just the interaction surface. `onChange` receives the
// normalized point list.
const SIZE = 220; // on-screen px (the SVG viewBox is 0..255 in curve space)

export function CurveEditor({ points, onChange, stroke = "var(--accent)" }) {
  const svgRef = useRef(null);
  const dragRef = useRef(null); // index of the point being dragged, or null

  const pts = normalizeCurvePoints(points);
  const lut = buildCurveLut(pts);
  // Curve polyline in SVG space (y flipped: 0 at bottom → 255-y at top).
  const path = Array.from({ length: 256 }, (_, x) => `${x},${255 - lut[x]}`).join(" ");

  // Convert a pointer event to clamped curve coords (x,y in 0..255, y up).
  const toCurve = (event) => {
    const rect = svgRef.current.getBoundingClientRect();
    const x = ((event.clientX - rect.left) / rect.width) * 255;
    const y = 255 - ((event.clientY - rect.top) / rect.height) * 255;
    const clamp = (v) => (v < 0 ? 0 : v > 255 ? 255 : Math.round(v));
    return { x: clamp(x), y: clamp(y) };
  };

  // Drag a point, keeping the list ordered: endpoints move in y only; interior
  // points clamp x strictly between their neighbors so the index stays stable.
  useEffect(() => {
    const onMove = (event) => {
      const i = dragRef.current;
      if (i == null) return;
      const next = pts.map((p) => ({ ...p }));
      const c = toCurve(event);
      if (i === 0) next[i] = { x: 0, y: c.y };
      else if (i === next.length - 1) next[i] = { x: 255, y: c.y };
      else {
        const lo = next[i - 1].x + 1;
        const hi = next[i + 1].x - 1;
        next[i] = { x: Math.min(hi, Math.max(lo, c.x)), y: c.y };
      }
      onChange(next);
    };
    const onUp = () => {
      dragRef.current = null;
    };
    window.addEventListener("pointermove", onMove);
    window.addEventListener("pointerup", onUp);
    return () => {
      window.removeEventListener("pointermove", onMove);
      window.removeEventListener("pointerup", onUp);
    };
  });

  // Click empty space → add a control point there (ignored if it lands on an x that
  // already has a point — normalizeCurvePoints would dedupe it anyway).
  const handleBackgroundPointerDown = (event) => {
    if (event.target.dataset.handle) return; // a handle starts a drag instead
    const c = toCurve(event);
    onChange(normalizeCurvePoints([...pts, c]));
  };

  const removePoint = (index) => {
    if (index === 0 || index === pts.length - 1) return; // keep the endpoints
    onChange(pts.filter((_, i) => i !== index));
  };

  return (
    <svg
      ref={svgRef}
      className="image-editor-curve"
      width={SIZE}
      height={SIZE}
      viewBox="0 0 255 255"
      preserveAspectRatio="none"
      onPointerDown={handleBackgroundPointerDown}
      role="application"
      aria-label="Tone curve editor"
    >
      <rect x={0} y={0} width={255} height={255} className="image-editor-curve-bg" />
      {[64, 128, 191].map((g) => (
        <g key={g}>
          <line x1={g} y1={0} x2={g} y2={255} className="image-editor-curve-grid" />
          <line x1={0} y1={g} x2={255} y2={g} className="image-editor-curve-grid" />
        </g>
      ))}
      <line x1={0} y1={255} x2={255} y2={0} className="image-editor-curve-diagonal" />
      <polyline points={path} fill="none" stroke={stroke} strokeWidth={2} vectorEffect="non-scaling-stroke" />
      {pts.map((p, i) => (
        <circle
          key={i}
          data-handle="1"
          data-index={i}
          cx={p.x}
          cy={255 - p.y}
          r={5}
          className="image-editor-curve-handle"
          stroke={stroke}
          onPointerDown={(event) => {
            event.stopPropagation();
            dragRef.current = i;
          }}
          onDoubleClick={(event) => {
            event.stopPropagation();
            removePoint(i);
          }}
        />
      ))}
    </svg>
  );
}
