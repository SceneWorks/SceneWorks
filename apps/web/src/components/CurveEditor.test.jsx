import React, { act } from "react";
import { createRoot } from "react-dom/client";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { CurveEditor } from "./CurveEditor.jsx";

describe("CurveEditor (sc-6109)", () => {
  let container;
  let root;

  beforeEach(() => {
    global.IS_REACT_ACT_ENVIRONMENT = true;
    container = document.createElement("div");
    document.body.appendChild(container);
    root = createRoot(container);
  });

  afterEach(async () => {
    await act(async () => root.unmount());
    container.remove();
    vi.clearAllMocks();
  });

  const handles = () => [...container.querySelectorAll(".image-editor-curve-handle")];

  it("renders a draggable handle per control point and a 256-point curve polyline", async () => {
    const points = [
      { x: 0, y: 0 },
      { x: 128, y: 180 },
      { x: 255, y: 255 },
    ];
    await act(() => root.render(<CurveEditor points={points} onChange={vi.fn()} />));
    expect(handles()).toHaveLength(3);
    const poly = container.querySelector("polyline");
    expect(poly.getAttribute("points").trim().split(/\s+/)).toHaveLength(256);
    // The middle handle reflects the lifted midtone, drawn y-flipped (255 - y).
    const mid = handles()[1];
    expect(Number(mid.getAttribute("cx"))).toBe(128);
    expect(Number(mid.getAttribute("cy"))).toBe(255 - 180);
  });

  it("double-clicking an interior point removes it; endpoints are kept", async () => {
    const onChange = vi.fn();
    const points = [
      { x: 0, y: 0 },
      { x: 128, y: 180 },
      { x: 255, y: 255 },
    ];
    await act(() => root.render(<CurveEditor points={points} onChange={onChange} />));
    await act(() => handles()[1].dispatchEvent(new MouseEvent("dblclick", { bubbles: true })));
    expect(onChange).toHaveBeenCalledWith([
      { x: 0, y: 0 },
      { x: 255, y: 255 },
    ]);

    onChange.mockClear();
    await act(() => handles()[0].dispatchEvent(new MouseEvent("dblclick", { bubbles: true })));
    expect(onChange).not.toHaveBeenCalled(); // endpoint not removable
  });
});
