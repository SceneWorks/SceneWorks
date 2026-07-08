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

  it("dragging an interior handle moves it in curve space (sc-8940)", async () => {
    const onChange = vi.fn();
    const points = [
      { x: 0, y: 0 },
      { x: 128, y: 128 },
      { x: 255, y: 255 },
    ];
    await act(() => root.render(<CurveEditor points={points} onChange={onChange} />));
    // jsdom returns a zeroed rect, so pin a 255x255 rect at the origin: then a pointer at
    // client (200, 55) maps to curve x=200, y=255-55=200.
    const svg = container.querySelector("svg");
    svg.getBoundingClientRect = () => ({ left: 0, top: 0, width: 255, height: 255 });

    // Begin the drag on the interior handle, then move the window pointer.
    await act(() =>
      handles()[1].dispatchEvent(new PointerEvent("pointerdown", { bubbles: true })),
    );
    await act(() =>
      window.dispatchEvent(new PointerEvent("pointermove", { clientX: 200, clientY: 55 })),
    );
    expect(onChange).toHaveBeenCalledWith([
      { x: 0, y: 0 },
      { x: 200, y: 200 },
      { x: 255, y: 255 },
    ]);

    // pointerup ends the drag: further moves are ignored.
    onChange.mockClear();
    await act(() => window.dispatchEvent(new PointerEvent("pointerup", {})));
    await act(() =>
      window.dispatchEvent(new PointerEvent("pointermove", { clientX: 100, clientY: 100 })),
    );
    expect(onChange).not.toHaveBeenCalled();
  });

  it("uses the latest points/onChange after a re-render without re-subscribing (sc-8940)", async () => {
    // The window drag listeners are registered once; the move handler must read the
    // *current* points and onChange via a ref, not a snapshot from the first render.
    const addSpy = vi.spyOn(window, "addEventListener");
    const firstOnChange = vi.fn();
    const points = [
      { x: 0, y: 0 },
      { x: 128, y: 128 },
      { x: 255, y: 255 },
    ];
    await act(() => root.render(<CurveEditor points={points} onChange={firstOnChange} />));
    const svg = container.querySelector("svg");
    svg.getBoundingClientRect = () => ({ left: 0, top: 0, width: 255, height: 255 });

    const drCount = (type) =>
      addSpy.mock.calls.filter(([evt]) => evt === type).length;
    expect(drCount("pointermove")).toBe(1);
    expect(drCount("pointerup")).toBe(1);

    // Re-render with a NEW onChange; the single subscription must route to it.
    const secondOnChange = vi.fn();
    await act(() => root.render(<CurveEditor points={points} onChange={secondOnChange} />));
    // No churn: still exactly one listener of each type across both renders.
    expect(drCount("pointermove")).toBe(1);
    expect(drCount("pointerup")).toBe(1);

    await act(() =>
      handles()[1].dispatchEvent(new PointerEvent("pointerdown", { bubbles: true })),
    );
    await act(() =>
      window.dispatchEvent(new PointerEvent("pointermove", { clientX: 200, clientY: 55 })),
    );
    expect(firstOnChange).not.toHaveBeenCalled();
    expect(secondOnChange).toHaveBeenCalledWith([
      { x: 0, y: 0 },
      { x: 200, y: 200 },
      { x: 255, y: 255 },
    ]);
    addSpy.mockRestore();
  });
});
