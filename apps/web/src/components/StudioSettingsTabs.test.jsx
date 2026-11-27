import React, { act, useState } from "react";
import { createRoot } from "react-dom/client";
import { afterEach, beforeEach, describe, expect, it } from "vitest";
import { StudioSettingsTabs } from "./StudioSettingsTabs.jsx";

// The studio settings strip: one open panel at a time, mounted only while open, the open tab
// closes on a second click, and tabs without content are dropped.

let container;
let root;

beforeEach(() => {
  container = document.createElement("div");
  document.body.appendChild(container);
  root = createRoot(container);
});

afterEach(() => {
  act(() => root.unmount());
  container.remove();
});

function Harness({ initial = "model" }) {
  const [active, setActive] = useState(initial);
  return (
    <StudioSettingsTabs
      active={active}
      onChange={setActive}
      tabs={[
        { id: "model", label: "Model & output", summary: "Krea · 16:9", attention: true, content: <div>model knobs</div> },
        { id: "loras", label: "LoRAs", summary: "2 selected", changed: true, content: <div>lora knobs</div> },
        { id: "empty", label: "Hidden", content: null },
      ]}
    />
  );
}

const tab = (label) =>
  [...container.querySelectorAll('[role="tab"]')].find((node) => node.textContent.includes(label));

describe("StudioSettingsTabs", () => {
  it("renders one tab per group with content and mounts only the open panel", () => {
    act(() => root.render(<Harness />));
    expect(container.querySelectorAll('[role="tab"]')).toHaveLength(2);
    expect(tab("Hidden")).toBeUndefined();
    expect(container.querySelector('[role="tabpanel"]').textContent).toBe("model knobs");
    expect(tab("Model").getAttribute("aria-selected")).toBe("true");
    expect(tab("Model").querySelector(".studio-settings-dot.warn")).toBeTruthy();
    expect(tab("LoRAs").querySelector(".studio-settings-dot:not(.warn)")).toBeTruthy();
    expect(container.textContent).not.toContain("lora knobs");
  });

  it("switches panels and closes the open one on a second click", () => {
    act(() => root.render(<Harness />));
    act(() => tab("LoRAs").click());
    expect(container.querySelector('[role="tabpanel"]').textContent).toBe("lora knobs");
    expect(tab("Model").getAttribute("aria-selected")).toBe("false");
    act(() => tab("LoRAs").click());
    expect(container.querySelector('[role="tabpanel"]')).toBeNull();
    expect(tab("LoRAs").getAttribute("aria-expanded")).toBe("false");
  });

  it("treats an active id that has no content as closed", () => {
    act(() => root.render(<Harness initial="empty" />));
    expect(container.querySelector('[role="tabpanel"]')).toBeNull();
  });
});
