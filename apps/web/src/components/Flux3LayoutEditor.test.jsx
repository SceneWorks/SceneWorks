import React, { useState } from "react";
import { describe, expect, it, vi } from "vitest";
import { fireEvent, render, screen } from "@testing-library/react";
import { Flux3LayoutEditor } from "./Flux3LayoutEditor.jsx";

const references = [{ id: "ref_B", label: "Blue vase", url: "/blue.png" }, { id: "ref_A", label: "Red vase", url: "/red.png" }];
function Harness({ initial = { version: 1, boxes: [] }, refs = references }) {
  const [value, onChange] = useState(initial);
  return <><Flux3LayoutEditor value={value} onChange={onChange} references={refs} width={112} height={80} /><output data-testid="document">{JSON.stringify(value)}</output></>;
}
const documentValue = () => JSON.parse(screen.getByTestId("document").textContent);

describe("FLUX 3 layout authoring", () => {
  it("draws source and target geometry independently through pointer interactions", () => {
    vi.stubGlobal("PointerEvent", MouseEvent);
    try {
      render(<Harness />);
      fireEvent.click(screen.getByRole("button", { name: "Add source edit" }));
      const source = screen.getByRole("img", { name: "Source reference canvas" });
      const target = screen.getByRole("img", { name: "Output layout canvas" });
      // jsdom has no layout; provide only viewport geometry, not authored control state.
      const bounds = { left: 10, top: 20, width: 200, height: 100 };
      vi.spyOn(source, "getBoundingClientRect").mockReturnValue(bounds);
      vi.spyOn(target, "getBoundingClientRect").mockReturnValue(bounds);
      fireEvent.pointerDown(source, { clientX: 10, clientY: 20 });
      fireEvent.pointerUp(source, { clientX: 110, clientY: 70 });
      fireEvent.pointerDown(target, { clientX: 110, clientY: 70 });
      fireEvent.pointerUp(target, { clientX: 210, clientY: 120 });
      expect(documentValue().boxes[0].src_bbox).toEqual([0, 0, 0.5, 0.5]);
      expect(documentValue().boxes[0].tgt_bbox).toEqual([0.5, 0.5, 1, 1]);
      expect(documentValue().boxes[0].reference_id).toBe("ref_B");
    } finally {
      vi.restoreAllMocks();
      vi.unstubAllGlobals();
    }
  });

  it("authors precise labeled placements with stable identifiers", () => {
    render(<Harness />);
    fireEvent.click(screen.getByRole("button", { name: "Add placement box" }));
    const id = documentValue().boxes[0].id;
    fireEvent.change(screen.getByLabelText("Box label"), { target: { value: '猫 "[red]"' } });
    fireEvent.change(screen.getByLabelText("Target left"), { target: { value: "0.123456789012345" } });
    expect(documentValue().boxes[0]).toMatchObject({ id, desc: '猫 "[red]"', bbox: [0.25, 0.123456789012345, 0.75, 0.75] });
    expect(screen.getByRole("img", { name: "Output layout canvas" })).toBeTruthy();
  });

  it("distinguishes moving, null-target removal, and deleting the control", () => {
    render(<Harness />);
    fireEvent.click(screen.getByRole("button", { name: "Add source edit" }));
    const id = documentValue().boxes[0].id;
    expect(documentValue().boxes[0]).toMatchObject({ id, reference_id: "ref_B", from: "ref_image_0", src_bbox: [0.25, 0.25, 0.75, 0.75], tgt_bbox: [0.25, 0.25, 0.75, 0.75] });
    fireEvent.change(screen.getByLabelText("Source reference"), { target: { value: "ref_A" } });
    fireEvent.change(screen.getByLabelText("Target top"), { target: { value: "0" } });
    expect(documentValue().boxes[0]).toMatchObject({ id, reference_id: "ref_A", from: "ref_image_1", tgt_bbox: [0, 0.25, 0.75, 0.75] });
    fireEvent.click(screen.getByRole("button", { name: "Remove object from output" }));
    expect(documentValue().boxes[0]).toMatchObject({ id, reference_id: "ref_A", src_bbox: [0.25, 0.25, 0.75, 0.75], tgt_bbox: null });
    expect(screen.queryByLabelText("Target top")).toBeNull();
    fireEvent.click(screen.getByRole("button", { name: "Move object to target" }));
    expect(documentValue().boxes[0].tgt_bbox).toEqual([0.25, 0.25, 0.75, 0.75]);
    fireEvent.click(screen.getByRole("button", { name: "Delete control" }));
    expect(documentValue().boxes).toEqual([]);
  });

  it("does not rewrite existing data when references reorder or a row is selected", () => {
    const initial = { version: 1, boxes: [{ id: "stable_original", desc: "remove", from: "ref_image_0", reference_id: "ref_A", src_bbox: [-0.1, 0, 0.3, 1], tgt_bbox: null, provenance: { original: true } }] };
    const { rerender } = render(<Flux3LayoutEditor value={initial} onChange={() => { throw new Error("unexpected normalization"); }} references={references} width={112} height={80} />);
    fireEvent.click(screen.getByRole("button", { name: "Select stable_original" }));
    expect(screen.getByLabelText("Source reference").value).toBe("ref_A");
    rerender(<Flux3LayoutEditor value={initial} onChange={() => { throw new Error("unexpected normalization"); }} references={[...references].reverse()} width={112} height={80} />);
    expect(screen.getByLabelText("Source reference").value).toBe("ref_A");
    expect(initial.boxes[0].tgt_bbox).toBeNull();
    expect(initial.boxes[0].from).toBe("ref_image_0");
  });

  it("retains zero, edges, outside and fractional coordinates without clamping", () => {
    render(<Harness initial={{ version: 1, boxes: [{ id: "edge", desc: "edge", bbox: [0, 0, 1, 1] }] }} />);
    fireEvent.click(screen.getByRole("button", { name: "Select edge" }));
    fireEvent.change(screen.getByLabelText("Target left"), { target: { value: "-0.123456789012345" } });
    expect(documentValue().boxes[0].bbox).toEqual([0, -0.123456789012345, 1, 1]);
    expect(screen.getByText(/soft spatial guidance/i)).toBeTruthy();
  });
});
