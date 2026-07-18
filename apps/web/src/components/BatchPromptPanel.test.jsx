import React, { act, useState } from "react";
import { createRoot } from "react-dom/client";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";

import BatchPromptPanel from "./BatchPromptPanel.jsx";

// React controlled inputs ignore a direct `.value =`; drive them via the native setter.
function setValue(el, value) {
  const proto =
    el instanceof window.HTMLTextAreaElement
      ? window.HTMLTextAreaElement.prototype
      : window.HTMLInputElement.prototype;
  Object.getOwnPropertyDescriptor(proto, "value").set.call(el, value);
  el.dispatchEvent(new Event("input", { bubbles: true }));
  el.dispatchEvent(new Event("change", { bubbles: true }));
}

// The panel is fully controlled; a tiny stateful harness lets interactions actually
// update prompts/variables/name the way Image Studio wires them.
function Harness({ initialPrompts = "", count = 1, batches = [], onSave = vi.fn(), extra = {} }) {
  const [promptsText, setPromptsText] = useState(initialPrompts);
  const [variableValues, setVariableValues] = useState({});
  const [name, setName] = useState("");
  const [scope, setScope] = useState("global");
  return (
    <BatchPromptPanel
      promptsText={promptsText}
      onPromptsTextChange={setPromptsText}
      variableValues={variableValues}
      onVariableValuesChange={setVariableValues}
      count={count}
      batches={batches}
      projectId={null}
      name={name}
      onNameChange={setName}
      scope={scope}
      onScopeChange={setScope}
      onSave={onSave}
      onLoad={vi.fn()}
      onDelete={vi.fn()}
      onImport={vi.fn()}
      {...extra}
    />
  );
}

describe("BatchPromptPanel (sc-9955)", () => {
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

  const mount = (props) => act(() => root.render(<Harness {...props} />));
  const varKeys = () => [...document.body.querySelectorAll(".batch-var-key")].map((el) => el.textContent);

  it("renders a chip editor per unique {{key}} referenced in the prompts", () => {
    mount({ initialPrompts: "{{name}} with {{hair}} hair\n{{name}} smiling" });
    expect(varKeys()).toEqual(["{{name}}", "{{hair}}"]);
  });

  it("shows the syntax hint when there are no placeholders at all", () => {
    mount({ initialPrompts: "a plain prompt" });
    expect(varKeys()).toEqual([]);
    expect(document.body.querySelector(".batch-hint").textContent).toContain("value box");
  });

  it("shows no chip editors for inline-only prompts, with an auto-expand note", () => {
    mount({ initialPrompts: "a {{red|blue}} {{p:he|she}} {{p:his|her}}" });
    expect(varKeys()).toEqual([]);
    expect(document.body.querySelector(".batch-hint").textContent).toContain("expand automatically");
  });

  it("warns on a mismatched linked group and blocks (via the warning)", () => {
    mount({ initialPrompts: "{{p:he|she|they}} {{p:his|her}}" });
    const warning = [...document.body.querySelectorAll(".batch-warning")].find((el) =>
      /mismatched lengths/.test(el.textContent),
    );
    expect(warning).toBeTruthy();
  });

  it("reflects the live total (prompts × variations)", () => {
    mount({ initialPrompts: "one\ntwo\nthree", count: 4 });
    expect(document.body.querySelector(".batch-total strong").textContent).toBe("12");
  });

  it("commits a typed value live, without pressing Enter", async () => {
    mount({ initialPrompts: "{{name}} portrait", count: 1 });
    await act(async () => setValue(document.body.querySelector(".batch-var-input"), "Alice"));
    expect(document.body.querySelector(".batch-var-count").textContent).toBe("1 value");
  });

  it("auto-expands to accept multiple values as you type", async () => {
    mount({ initialPrompts: "{{c}}" });
    const inputs = () => [...document.body.querySelectorAll(".batch-var-input")];
    expect(inputs()).toHaveLength(1);
    await act(async () => setValue(inputs()[0], "red"));
    expect(inputs()).toHaveLength(2); // a fresh trailing box appears
    await act(async () => setValue(inputs()[1], "blue"));
    expect(document.body.querySelector(".batch-var-count").textContent).toBe("2 values");
  });

  it("disables Save until a name is entered", async () => {
    mount({ initialPrompts: "one" });
    const saveButton = [...document.body.querySelectorAll(".batch-btn")].find((b) => /Save|Update/.test(b.textContent));
    expect(saveButton.disabled).toBe(true);
    const nameInput = document.body.querySelector(".batch-name");
    await act(async () => setValue(nameInput, "My Batch"));
    expect(saveButton.disabled).toBe(false);
  });

  it("previews the first resolved prompt from a live-typed value", async () => {
    mount({ initialPrompts: "{{name}} portrait" });
    await act(async () => setValue(document.body.querySelector(".batch-var-input"), "Alice"));
    expect(document.body.querySelector(".batch-preview-text").textContent).toBe("Alice portrait");
  });

  it("strips a leading [WxH] from the preview and shows it as a size badge", () => {
    mount({ initialPrompts: "[832x1216] a full-body portrait" });
    expect(document.body.querySelector(".batch-preview-text").textContent).toBe("a full-body portrait");
    expect(document.body.querySelector(".batch-preview-res").textContent).toBe("832×1216");
  });

  it("shows Save (not Update) and no New-batch link when no batch is loaded", () => {
    mount({ initialPrompts: "one" });
    const saveButton = [...document.body.querySelectorAll(".batch-btn")].find((b) => /Save|Update/.test(b.textContent));
    expect(saveButton.textContent).toContain("Save");
    expect(document.body.querySelector(".batch-new-link")).toBeNull();
    expect(document.body.querySelector(".batch-save-head .batch-field-label").textContent).toBe("Save this batch");
  });

  it("surfaces a New-batch action once a saved batch is loaded, and Save reads Update", () => {
    mount({ initialPrompts: "one", extra: { loadedBatchId: "b1", onNew: vi.fn() } });
    const saveButton = [...document.body.querySelectorAll(".batch-btn")].find((b) => /Save|Update/.test(b.textContent));
    expect(saveButton.textContent).toContain("Update");
    expect(document.body.querySelector(".batch-save-head .batch-field-label").textContent).toBe("Editing saved batch");
    expect(document.body.querySelector(".batch-new-link")).toBeTruthy();
  });

  it("invokes onNew when the New-batch action is clicked", async () => {
    const onNew = vi.fn();
    mount({ initialPrompts: "one", extra: { loadedBatchId: "b1", onNew } });
    await act(async () => document.body.querySelector(".batch-new-link").click());
    expect(onNew).toHaveBeenCalledTimes(1);
  });

  it("lists saved batches", () => {
    mount({
      batches: [
        { id: "a", name: "Turnaround", scope: "global" },
        { id: "b", name: "Angles", scope: "project" },
      ],
    });
    expect([...document.body.querySelectorAll(".batch-list-load")].map((el) => el.textContent)).toEqual([
      expect.stringContaining("Turnaround"),
      expect.stringContaining("Angles"),
    ]);
  });
});
