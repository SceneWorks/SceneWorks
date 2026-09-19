import React from "react";
import { act } from "react";
import { createRoot } from "react-dom/client";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";

vi.mock("../../api/films.js", () => ({ addFilmSound: vi.fn() }));

import { FilmSound } from "./FilmSound.jsx";

function draft() {
  return {
    id: "film_1",
    projectId: "project_1",
    revision: 1,
    productionPlan: { sound: { generatedAudio: "mute" }, shots: [] },
    referencePack: { sound: [{ role: "theme", kind: "music", description: "Piano." }] },
  };
}

let container;
let root;

beforeEach(() => {
  global.IS_REACT_ACT_ENVIRONMENT = true;
  container = document.createElement("div");
  document.body.appendChild(container);
});

afterEach(() => {
  act(() => root?.unmount());
  container.remove();
});

async function render(props = {}) {
  root = createRoot(container);
  await act(async () => root.render(
    <FilmSound
      activeProject={{ id: "project_1" }}
      assets={[]}
      disabled={false}
      draft={draft()}
      onChange={vi.fn()}
      onReplaceDraft={vi.fn()}
      saveDraft={vi.fn()}
      setNotice={vi.fn()}
      token="token"
      {...props}
    />,
  ));
}

describe("FilmSound", () => {
  // sc-24028. The sound half of the reference pack is authored here, so its findings are shown
  // here. Between this panel and the References step every pack-level finding has an owner —
  // neither may be the panel that drops it.
  it("shows the pack's sound findings and ignores the rest", async () => {
    await render({
      findings: [
        { field: "referencePack.sound[0].role", message: "duplicate sound role \"theme\"" },
        { field: "referencePack.sound[1].text", message: "a synthesized line needs text" },
        { field: "referencePack.references[0].description", message: "the references step owns this" },
        { shotId: "SH010", field: "audio", message: "the shots step owns this" },
      ],
    });

    const findings = container.querySelector('[aria-label="Sound findings"]');
    expect(findings.textContent).toContain("duplicate sound role \"theme\"");
    expect(findings.textContent).toContain("a synthesized line needs text");
    expect(container.textContent).not.toContain("the references step owns this");
    expect(container.textContent).not.toContain("the shots step owns this");
  });

  it("shows no findings list when the pack's sound is clean", async () => {
    await render({ findings: [{ field: "referencePack.references[0].file", message: "elsewhere" }] });
    expect(container.querySelector('[aria-label="Sound findings"]')).toBeNull();
  });
});
