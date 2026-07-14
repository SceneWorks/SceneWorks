import React, { act } from "react";
import { createRoot } from "react-dom/client";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";

import { ImageEditSourcePickerField } from "./AssetPicker.jsx";

// The picker was previously "Change"/"Select"-only: once an optional source (img2img reference,
// control image, second edit image) was picked there was no way to un-pick it, so it kept driving
// generations until reload. `clearable` adds an opt-in "Remove" control that resets to "".
describe("ImageEditSourcePickerField clear affordance", () => {
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

  async function render(ui) {
    await act(async () => root.render(ui));
  }

  const buttonByLabel = (label) =>
    [...container.querySelectorAll("button")].find((b) => b.textContent.trim() === label);

  it("renders a Remove control that clears the selection when clearable with a value set", async () => {
    const onChange = vi.fn();
    await render(
      <ImageEditSourcePickerField assets={[]} clearable label="Reference image" onChange={onChange} value="a1" />,
    );
    const remove = buttonByLabel("Remove");
    expect(remove).toBeTruthy();
    await act(async () => remove.dispatchEvent(new MouseEvent("click", { bubbles: true })));
    expect(onChange).toHaveBeenCalledWith("");
  });

  it("omits Remove when clearable but nothing is selected", async () => {
    await render(
      <ImageEditSourcePickerField assets={[]} clearable label="Reference image" onChange={() => {}} value="" />,
    );
    expect(buttonByLabel("Remove")).toBeFalsy();
  });

  it("omits Remove when not clearable even with a value (required edit source)", async () => {
    await render(
      <ImageEditSourcePickerField assets={[]} label="Source image" onChange={() => {}} value="a1" />,
    );
    expect(buttonByLabel("Remove")).toBeFalsy();
  });
});
