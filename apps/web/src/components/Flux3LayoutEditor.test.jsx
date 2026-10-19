import React, { act, useState } from "react";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { mountRoot, unmountRoot } from "../testUtils/dom.js";

let root, container;
beforeEach(() => {
  global.IS_REACT_ACT_ENVIRONMENT = true;
  ({ root, container } = mountRoot());
});
afterEach(async () => { await unmountRoot(root, container); });
const render = (element) => {
  act(() => root.render(element));
  return { rerender: (next) => act(() => root.render(next)) };
};
const byLabel = (name) => container.querySelector(`[aria-label="${name}"]`);
const byRole = (role, { name }) => [...container.querySelectorAll(role === "button" ? "button" : `[role="${role}"]`)]
  .find((node) => node.getAttribute("aria-label") === name || node.textContent === name);
const click = (node) => act(() => node.dispatchEvent(new MouseEvent("click", { bubbles: true })));
const change = (node, value) => act(() => {
  if (node.tagName === "INPUT") {
    Object.getOwnPropertyDescriptor(HTMLInputElement.prototype, "value").set.call(node, value);
    node.dispatchEvent(new Event("input", { bubbles: true }));
  } else {
    node.value = value;
    node.dispatchEvent(new Event("change", { bubbles: true }));
  }
});
const pointer = (node, type, options) => act(() => node.dispatchEvent(new MouseEvent(type, { bubbles: true, ...options })));
import { Flux3LayoutEditor } from "./Flux3LayoutEditor.jsx";

const references = [{ id: "ref_B", label: "Blue vase", url: "/blue.png" }, { id: "ref_A", label: "Red vase", url: "/red.png" }];
function Harness({ initial = { version: 1, boxes: [] }, refs = references }) {
  const [value, onChange] = useState(initial);
  return <><Flux3LayoutEditor value={value} onChange={onChange} references={refs} width={112} height={80} /><output data-testid="document">{JSON.stringify(value)}</output></>;
}
const documentValue = () => JSON.parse(container.querySelector('[data-testid="document"]').textContent);

describe("FLUX 3 layout authoring", () => {
  it("draws source and target geometry independently through pointer interactions", () => {
    vi.stubGlobal("PointerEvent", MouseEvent);
    try {
      render(<Harness />);
      click(byRole("button", { name: "Add source edit" }));
      const source = byRole("img", { name: "Source reference canvas" });
      const target = byRole("img", { name: "Output layout canvas" });
      // jsdom has no layout; provide only viewport geometry, not authored control state.
      const bounds = { left: 10, top: 20, width: 200, height: 100 };
      vi.spyOn(source, "getBoundingClientRect").mockReturnValue(bounds);
      vi.spyOn(target, "getBoundingClientRect").mockReturnValue(bounds);
      pointer(source, "pointerdown", { clientX: 10, clientY: 20 });
      pointer(source, "pointerup", { clientX: 110, clientY: 70 });
      pointer(target, "pointerdown", { clientX: 110, clientY: 70 });
      pointer(target, "pointerup", { clientX: 210, clientY: 120 });
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
    click(byRole("button", { name: "Add placement box" }));
    const id = documentValue().boxes[0].id;
    change(byLabel("Box label"), '猫 "[red]"');
    change(byLabel("Target left"), "0.123456789012345");
    expect(documentValue().boxes[0]).toMatchObject({ id, desc: '猫 "[red]"', bbox: [0.25, 0.123456789012345, 0.75, 0.75] });
    expect(byRole("img", { name: "Output layout canvas" })).toBeTruthy();
  });

  it("distinguishes moving, null-target removal, and deleting the control", () => {
    render(<Harness />);
    click(byRole("button", { name: "Add source edit" }));
    const id = documentValue().boxes[0].id;
    expect(documentValue().boxes[0]).toMatchObject({ id, reference_id: "ref_B", from: "ref_image_0", src_bbox: [0.25, 0.25, 0.75, 0.75], tgt_bbox: [0.25, 0.25, 0.75, 0.75] });
    change(byLabel("Source reference"), "ref_A");
    change(byLabel("Target top"), "0");
    expect(documentValue().boxes[0]).toMatchObject({ id, reference_id: "ref_A", from: "ref_image_1", tgt_bbox: [0, 0.25, 0.75, 0.75] });
    click(byRole("button", { name: "Remove object from output" }));
    expect(documentValue().boxes[0]).toMatchObject({ id, reference_id: "ref_A", src_bbox: [0.25, 0.25, 0.75, 0.75], tgt_bbox: null });
    expect(byLabel("Target top")).toBeNull();
    click(byRole("button", { name: "Move object to target" }));
    expect(documentValue().boxes[0].tgt_bbox).toEqual([0.25, 0.25, 0.75, 0.75]);
    click(byRole("button", { name: "Delete control" }));
    expect(documentValue().boxes).toEqual([]);
  });

  it("does not rewrite existing data when references reorder or a row is selected", () => {
    const initial = { version: 1, boxes: [{ id: "stable_original", desc: "remove", from: "ref_image_0", reference_id: "ref_A", src_bbox: [-0.1, 0, 0.3, 1], tgt_bbox: null, provenance: { original: true } }] };
    const { rerender } = render(<Flux3LayoutEditor value={initial} onChange={() => { throw new Error("unexpected normalization"); }} references={references} width={112} height={80} />);
    click(byRole("button", { name: "Select stable_original" }));
    expect(byLabel("Source reference").value).toBe("ref_A");
    rerender(<Flux3LayoutEditor value={initial} onChange={() => { throw new Error("unexpected normalization"); }} references={[...references].reverse()} width={112} height={80} />);
    expect(byLabel("Source reference").value).toBe("ref_A");
    expect(initial.boxes[0].tgt_bbox).toBeNull();
    expect(initial.boxes[0].from).toBe("ref_image_0");
    rerender(<Flux3LayoutEditor value={initial} onChange={() => { throw new Error("unexpected retarget"); }} references={[references[0]]} />);
    expect(byLabel("Source reference").value).toBe("");
    expect(container.querySelector('[role="alert"]').textContent).toMatch(/source attachment is missing/);
    expect(initial.boxes[0].reference_id).toBe("ref_A");
  });

  it("selects a native ordinal-only source without changing the original JSON", () => {
    const initial = { version: 1, boxes: [{ id: "native", desc: "vase", from: "ref_image_1", src_bbox: [0, 0, 1, 1], tgt_bbox: null }] };
    render(<Flux3LayoutEditor value={initial} onChange={() => { throw new Error("unexpected normalization"); }} references={references} />);
    click(byRole("button", { name: "Select native" }));
    expect(byLabel("Source reference").value).toBe("ref_A");
    expect(initial.boxes[0]).not.toHaveProperty("reference_id");
  });

  it("treats bbox:null as a source edit and clears orphaned target units on explicit removal", () => {
    const original = { id: "pixel-edit", desc: "vase", bbox: null, from: "ref_image_0", reference_id: "ref_B",
      src_bbox: [0, 0, 1, 1], tgt_bbox: [0, 10, 80, 100], tgt_space: "output" };
    render(<Harness initial={{ version: 1, boxes: [original] }} />);
    click(byRole("button", { name: "Select pixel-edit" }));
    expect(byLabel("Source reference").value).toBe("ref_B");
    expect(byLabel("Target left").value).toBe("10");
    expect(documentValue().boxes[0]).toEqual(original);
    click(byRole("button", { name: "Remove object from output" }));
    expect(documentValue().boxes[0]).toMatchObject({ bbox: null, src_bbox: original.src_bbox, tgt_bbox: null });
    expect(documentValue().boxes[0]).not.toHaveProperty("tgt_space");
    click(byRole("button", { name: "Move object to target" }));
    expect(documentValue().boxes[0].tgt_bbox).toEqual([0.25, 0.25, 0.75, 0.75]);
    expect(documentValue().boxes[0]).not.toHaveProperty("tgt_space");
  });

  it.each(["normalized", "output"])("edits target-only native rows in their original %s target coordinates", (space) => {
    const original = { id: "target-only", desc: "place vase", from: "target", src_bbox: null,
      tgt_bbox: [0, 0.123456789012345, 1, 1], tgt_space: space };
    render(<Harness initial={{ version: 1, boxes: [original] }} />);
    click(byRole("button", { name: "Select target-only" }));
    expect(byLabel("Source reference")).toBeNull();
    expect(byLabel("Source reference canvas")).toBeNull();
    expect(container.querySelector('[role="alert"]')).toBeNull();
    expect(container.querySelector("legend").textContent).toContain(`(${space}, YXYX)`);
    expect(documentValue().boxes[0]).toEqual(original);
    change(byLabel("Target left"), "0.987654321098765");
    expect(documentValue().boxes[0]).toEqual({ ...original, tgt_bbox: [0, 0.987654321098765, 1, 1] });
    expect(documentValue().boxes[0]).not.toHaveProperty("bbox");
  });

  it("retains zero, edges, outside and fractional coordinates without clamping", () => {
    render(<Harness initial={{ version: 1, boxes: [{ id: "edge", desc: "edge", bbox: [0, 0, 1, 1] }] }} />);
    click(byRole("button", { name: "Select edge" }));
    change(byLabel("Target left"), "-0.123456789012345");
    expect(documentValue().boxes[0].bbox).toEqual([0, -0.123456789012345, 1, 1]);
    expect(container.textContent.match(/soft spatial guidance/i)).toBeTruthy();
  });
});
