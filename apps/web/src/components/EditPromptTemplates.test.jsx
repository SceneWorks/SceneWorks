import React from "react";
import { act } from "react";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";

import { EditPromptTemplates } from "./EditPromptTemplates.jsx";
import { EDIT_PROMPT_TEMPLATES } from "../data/editPromptTemplates.js";
import { click, mountRoot, unmountRoot } from "../testUtils/dom.js";

describe("EditPromptTemplates", () => {
  let container;
  let root;

  beforeEach(() => {
    global.IS_REACT_ACT_ENVIRONMENT = true;
    ({ container, root } = mountRoot());
  });

  afterEach(async () => {
    await unmountRoot(root, container);
  });

  async function render(props) {
    await act(async () => root.render(<EditPromptTemplates onApply={() => {}} {...props} />));
  }

  it("renders one pill per built-in edit recipe", async () => {
    await render();
    const labels = [...container.querySelectorAll("button")].map((button) => button.textContent);
    expect(labels).toEqual(EDIT_PROMPT_TEMPLATES.map((template) => template.label));
  });

  it("applies the template's full instruction, not its label", async () => {
    const onApply = vi.fn();
    await render({ onApply });
    await click(container.querySelectorAll("button")[0]);
    expect(onApply).toHaveBeenCalledWith(EDIT_PROMPT_TEMPLATES[0].prompt);
    expect(EDIT_PROMPT_TEMPLATES[0].prompt).not.toBe(EDIT_PROMPT_TEMPLATES[0].label);
  });

  it("uses each surface's own chip classes", async () => {
    await render({ variant: "studio" });
    expect(container.querySelector(".suggestion-row .suggestion")).toBeTruthy();
    await render({ variant: "editor" });
    expect(container.querySelector(".ie-chip-row .ie-chip")).toBeTruthy();
    await render({ variant: "simple" });
    expect(container.querySelector(".su-edit-templates .su-pill-btn")).toBeTruthy();
  });

  it("keeps every recipe a single self-contained instruction", () => {
    // The row REPLACES the instruction box, so a template that reads like a fragment
    // ("and also sharpen it") would silently produce a nonsense edit.
    for (const template of EDIT_PROMPT_TEMPLATES) {
      expect(template.id).toBeTruthy();
      expect(template.label.length).toBeLessThanOrEqual(20);
      expect(template.prompt.trim()).toBe(template.prompt);
      expect(template.hint).toBeTruthy();
    }
    const ids = EDIT_PROMPT_TEMPLATES.map((template) => template.id);
    expect(new Set(ids).size).toBe(ids.length);
  });
});
